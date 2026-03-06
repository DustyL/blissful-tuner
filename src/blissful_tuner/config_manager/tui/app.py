"""Blissful Config TUI -- interactive config compiler with ephemeral overrides."""

from __future__ import annotations

import atexit
import logging
import time
from pathlib import Path
from typing import Any

import tomlkit
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: directory discovery
# ---------------------------------------------------------------------------

_DRAFT_MAX_AGE_DAYS = 7


def _find_meta_dir() -> Path:
    """Find configs/meta/ relative to the repo root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        meta = parent / "configs" / "meta"
        if meta.is_dir():
            return meta
    raise FileNotFoundError("Could not find configs/meta/ directory")


def _find_compiled_dir() -> Path:
    """Find configs/compiled/ relative to the repo root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        configs = parent / "configs"
        if configs.is_dir():
            compiled = configs / "compiled"
            compiled.mkdir(parents=True, exist_ok=True)
            return compiled
    raise FileNotFoundError("Could not find configs/ directory")


# ---------------------------------------------------------------------------
# Utility: listing helpers for Select widgets
# ---------------------------------------------------------------------------


def _list_machines(meta_dir: Path) -> list[tuple[str, str]]:
    """List available machine names as (label, value) pairs for Select."""
    machines_dir = meta_dir / "machines"
    if not machines_dir.is_dir():
        return []
    return sorted((p.stem, p.stem) for p in machines_dir.glob("*.toml") if p.stem != "default")


def _list_personas(meta_dir: Path) -> list[tuple[str, str]]:
    """List available persona names as (label, value) pairs for Select."""
    personas_dir = meta_dir / "personas"
    if not personas_dir.is_dir():
        return []
    return sorted((p.stem, p.stem) for p in personas_dir.glob("*.toml"))


def _list_presets(meta_dir: Path) -> list[tuple[str, str]]:
    """List available preset names as (label, value) pairs for Select."""
    presets_dir = meta_dir / "presets"
    if not presets_dir.is_dir():
        return []
    return sorted((p.stem, p.stem) for p in presets_dir.glob("*.toml"))


def _list_archs() -> list[tuple[str, str]]:
    """List available architecture keys as (label, value) pairs for Select."""
    from blissful_tuner.config_manager.registry import ARCH_REGISTRY

    return sorted((f"{arch['display_name']} ({key})", key) for key, arch in ARCH_REGISTRY.items())


# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------

# Tab IDs and their corresponding sections in compile_config() training_toml
_PREVIEW_TABS: list[tuple[str, str]] = [
    ("Model", "model"),
    ("Network", "network"),
    ("Optimizer", "optimizer"),
    ("Training", "training"),
    ("Sampling", "sampling"),
    ("Output", "output"),
    ("Advanced", "advanced"),
]

# Sections that can be overridden through the preset layer
_PRESET_OVERRIDE_SECTIONS = {"network", "optimizer", "training", "sampling"}

_PLACEHOLDER_MSG = "Select all fields to preview"


# ---------------------------------------------------------------------------
# Override dict helpers
# ---------------------------------------------------------------------------


def _build_override_data(ephemeral: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build an override_data dict suitable for compile_config() from ephemeral overrides.

    Only includes sections that can be overridden via preset merging.
    Returns a dict like {"training": {"max_train_steps": 2000}, ...}.
    """
    result: dict[str, dict[str, Any]] = {}
    for section, keys in ephemeral.items():
        if section in _PRESET_OVERRIDE_SECTIONS and keys:
            result[section] = dict(keys)
    return result


# ---------------------------------------------------------------------------
# Draft file management
# ---------------------------------------------------------------------------


def _overrides_dir(meta_dir: Path) -> Path:
    """Return the overrides directory, creating it if needed."""
    d = meta_dir / "overrides"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_override_toml(path: Path, overrides: dict[str, dict[str, Any]]) -> None:
    """Write an override dict as TOML to a file using tomlkit."""
    doc = tomlkit.document()
    for section, keys in sorted(overrides.items()):
        if not keys:
            continue
        table = tomlkit.table()
        for key, value in sorted(keys.items()):
            table.add(key, value)
        doc.add(section, table)
    path.write_text(tomlkit.dumps(doc))


def _list_draft_files(meta_dir: Path) -> list[Path]:
    """List draft override files in the overrides directory."""
    d = _overrides_dir(meta_dir)
    drafts = []
    for p in d.glob(".draft_*.toml"):
        try:
            drafts.append((p, p.stat().st_mtime))
        except OSError:
            continue  # File deleted between glob and stat (TOCTOU)
    drafts.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in drafts]


def _cleanup_old_drafts(meta_dir: Path) -> int:
    """Remove draft files older than _DRAFT_MAX_AGE_DAYS. Returns count removed."""
    cutoff = time.time() - (_DRAFT_MAX_AGE_DAYS * 86400)
    removed = 0
    for draft in _list_draft_files(meta_dir):
        try:
            if draft.stat().st_mtime < cutoff:
                draft.unlink()
                removed += 1
        except OSError:
            pass  # File deleted between stat and unlink (TOCTOU)
    return removed


def _load_draft(path: Path) -> dict[str, dict[str, Any]]:
    """Load a draft TOML file and return it as a section->keys dict."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    # Ensure all values are dicts (section-level)
    result: dict[str, dict[str, Any]] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class QuitConfirmScreen(ModalScreen[str]):
    """Confirm dialog when quitting with unsaved ephemeral overrides.

    Returns "discard", "save", or "cancel".
    """

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    #quit-dialog {
        grid-size: 3;
        grid-gutter: 1 2;
        grid-rows: auto auto;
        padding: 1 2;
        width: 64;
        height: auto;
        border: thick $primary;
        background: $surface;
    }
    #quit-dialog Label {
        column-span: 3;
        content-align: center middle;
        margin-bottom: 1;
    }
    #quit-dialog Button {
        width: 100%;
    }
    """

    def __init__(self, override_count: int) -> None:
        super().__init__()
        self._override_count = override_count

    def compose(self) -> ComposeResult:
        with Grid(id="quit-dialog"):
            yield Label(f"You have {self._override_count} unsaved override(s). What would you like to do?")
            yield Button("Discard & Quit", variant="error", id="quit-discard")
            yield Button("Save & Quit", variant="success", id="quit-save")
            yield Button("Cancel", variant="primary", id="quit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-discard":
            self.dismiss("discard")
        elif event.button.id == "quit-save":
            self.dismiss("save")
        else:
            self.dismiss("cancel")


class SaveOverrideScreen(ModalScreen[str | None]):
    """Modal to prompt for an override file name.

    Returns the name string or None if cancelled.
    """

    DEFAULT_CSS = """
    SaveOverrideScreen {
        align: center middle;
    }
    #save-dialog {
        padding: 1 2;
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
    }
    #save-dialog Label {
        margin-bottom: 1;
    }
    #save-dialog Input {
        margin-bottom: 1;
    }
    #save-btn-row {
        height: 3;
        align: center middle;
    }
    #save-btn-row Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="save-dialog"):
            yield Label("Save overrides as (no extension):")
            yield Input(placeholder="my_overrides", id="save-name-input")
            yield Label("", id="save-error-label")
            with Horizontal(id="save-btn-row"):
                yield Button("Save", variant="success", id="save-confirm")
                yield Button("Cancel", variant="primary", id="save-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-cancel":
            self.dismiss(None)
        elif event.button.id == "save-confirm":
            self._try_save()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._try_save()

    def _try_save(self) -> None:
        name = self.query_one("#save-name-input", Input).value.strip()
        error_label = self.query_one("#save-error-label", Label)
        if not name:
            error_label.update("[bold red]Name cannot be empty[/]")
            return
        # Sanitize: only allow alphanumeric, hyphens, underscores
        sanitized = "".join(c for c in name if c.isalnum() or c in "-_")
        if sanitized != name:
            error_label.update("[bold red]Only alphanumeric, hyphens, underscores allowed[/]")
            return
        self.dismiss(sanitized)


class DraftRestoreScreen(ModalScreen[str]):
    """Offer to restore a draft override file.

    Returns "restore", "delete", or "ignore".
    """

    DEFAULT_CSS = """
    DraftRestoreScreen {
        align: center middle;
    }
    #draft-dialog {
        grid-size: 3;
        grid-gutter: 1 2;
        grid-rows: auto auto;
        padding: 1 2;
        width: 64;
        height: auto;
        border: thick $primary;
        background: $surface;
    }
    #draft-dialog Label {
        column-span: 3;
        content-align: center middle;
        margin-bottom: 1;
    }
    #draft-dialog Button {
        width: 100%;
    }
    """

    def __init__(self, draft_path: Path) -> None:
        super().__init__()
        self._draft_path = draft_path

    def compose(self) -> ComposeResult:
        ts = self._draft_path.stem.replace(".draft_", "")
        with Grid(id="draft-dialog"):
            yield Label(f"Found unsaved draft from {ts}. Restore it?")
            yield Button("Restore", variant="success", id="draft-restore")
            yield Button("Delete Draft", variant="error", id="draft-delete")
            yield Button("Ignore", variant="primary", id="draft-ignore")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "draft-restore":
            self.dismiss("restore")
        elif event.button.id == "draft-delete":
            self.dismiss("delete")
        else:
            self.dismiss("ignore")


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------


class BlissfulConfigApp(App):
    """Blissful Config TUI -- compile layered TOML configs interactively."""

    TITLE = "Blissful Config"
    SUB_TITLE = "Training Config Compiler"

    BINDINGS = [
        ("ctrl+s", "save_override", "Save Override"),
        ("ctrl+q", "request_quit", "Quit"),
    ]

    CSS = """
    #selector-panel {
        height: auto;
        padding: 1 2;
    }

    .selector-row {
        height: 3;
        margin-bottom: 1;
    }

    .selector-label {
        width: 16;
        content-align: right middle;
        padding-right: 1;
    }

    .selector-dropdown {
        width: 1fr;
    }

    #button-row {
        height: 3;
        margin-top: 1;
        padding: 0 2;
        align: center middle;
    }

    #compile-btn {
        margin-right: 1;
    }

    #save-override-btn {
        margin-right: 1;
    }

    #clear-overrides-btn {
        margin-right: 1;
    }

    #preview-panel {
        height: 1fr;
        border-top: solid green;
        padding: 1 2;
    }

    .preview-table {
        height: 1fr;
    }

    #edit-area {
        height: auto;
        padding: 0 2;
        border-top: solid $accent;
    }

    #edit-key-label {
        margin-right: 1;
    }

    #edit-input {
        width: 1fr;
    }

    #edit-error {
        color: $error;
        margin-top: 0;
        height: auto;
    }

    #status-area {
        height: auto;
        min-height: 3;
        padding: 1 2;
        border-top: solid green;
    }

    #status-text {
        width: 1fr;
    }

    .hidden {
        display: none;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            self._meta_dir: Path | None = _find_meta_dir()
        except FileNotFoundError:
            self._meta_dir = None

        # Ephemeral overrides: {section_name: {key: value}}
        self._ephemeral_overrides: dict[str, dict[str, Any]] = {}
        # Track the currently selected key for editing
        self._edit_section: str | None = None
        self._edit_key: str | None = None
        # Track validation errors
        self._has_validation_error: bool = False
        # Track unknown override keys (blocks compile)
        self._has_unknown_keys: bool = False
        # Last compiled training_toml for type checking against existing values
        self._last_training_toml: dict[str, Any] | None = None
        # Baseline training_toml (no overrides) for unknown-key detection
        self._baseline_training_toml: dict[str, Any] | None = None
        # atexit handler registered flag
        self._atexit_registered: bool = False
        # Draft path for the current session (set on atexit write)
        self._draft_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="selector-panel"):
            # Machine selector
            with Horizontal(classes="selector-row"):
                yield Label("Machine:", classes="selector-label")
                machines = _list_machines(self._meta_dir) if self._meta_dir else []
                yield Select(machines, id="machine-select", prompt="Select machine...", classes="selector-dropdown")

            # Architecture selector
            with Horizontal(classes="selector-row"):
                yield Label("Architecture:", classes="selector-label")
                yield Select(_list_archs(), id="arch-select", prompt="Select architecture...", classes="selector-dropdown")

            # Persona selector
            with Horizontal(classes="selector-row"):
                yield Label("Persona:", classes="selector-label")
                personas = _list_personas(self._meta_dir) if self._meta_dir else []
                yield Select(personas, id="persona-select", prompt="Select persona...", classes="selector-dropdown")

            # Preset selector
            with Horizontal(classes="selector-row"):
                yield Label("Preset:", classes="selector-label")
                presets = _list_presets(self._meta_dir) if self._meta_dir else []
                yield Select(presets, id="preset-select", prompt="Select preset...", classes="selector-dropdown")

        with Horizontal(id="button-row"):
            yield Button("Compile", id="compile-btn", variant="primary")
            yield Button("Save Override", id="save-override-btn", variant="warning", classes="hidden")
            yield Button("Clear Overrides", id="clear-overrides-btn", variant="default", classes="hidden")
            yield Button("Quit", id="quit-btn", variant="error")

        # Preview panel with tabbed sections -- DataTable per tab
        with TabbedContent(id="preview-panel"):
            for tab_label, section_key in _PREVIEW_TABS:
                with TabPane(tab_label, id=f"tab-{section_key}"):
                    yield DataTable(id=f"table-{section_key}", cursor_type="row", classes="preview-table")

        # Edit area (initially hidden)
        with Vertical(id="edit-area", classes="hidden"):
            with Horizontal():
                yield Label("Key: (none selected)", id="edit-key-label")
            yield Input(placeholder="Enter value...", id="edit-input")
            yield Label("", id="edit-error")

        with Vertical(id="status-area"):
            yield Static(
                "Ready. Select machine, architecture, persona, and preset to preview. Click a row to edit.",
                id="status-text",
            )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize DataTable columns and check for draft files."""
        for _tab_label, section_key in _PREVIEW_TABS:
            table = self.query_one(f"#table-{section_key}", DataTable)
            table.add_columns("Key", "Value", "Status")

        # Register atexit handler for draft saving
        self._register_atexit()

        # Clean up old drafts and check for recoverable ones
        if self._meta_dir:
            _cleanup_old_drafts(self._meta_dir)
            drafts = _list_draft_files(self._meta_dir)
            if drafts:
                self._offer_draft_restore(drafts[0])

    def _register_atexit(self) -> None:
        """Register an atexit handler to save drafts on abnormal exit."""
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self._atexit_save_draft)

    def _atexit_save_draft(self) -> None:
        """Save ephemeral overrides as a draft file on exit (atexit hook)."""
        if not self._ephemeral_overrides or not self._meta_dir:
            return
        # Only save if there are actual overrides
        total = sum(len(v) for v in self._ephemeral_overrides.values())
        if total == 0:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        draft_path = _overrides_dir(self._meta_dir) / f".draft_{timestamp}.toml"
        try:
            _write_override_toml(draft_path, self._ephemeral_overrides)
            self._draft_path = draft_path
        except Exception:
            pass  # Best-effort on abnormal exit

    def _offer_draft_restore(self, draft_path: Path) -> None:
        """Push the draft restore modal screen."""

        def _handle_draft_result(result: str | None) -> None:
            if result == "restore":
                try:
                    restored = _load_draft(draft_path)
                    # Filter to preset-layer sections only — non-overridable
                    # sections (model, output, etc.) are stale from old drafts.
                    filtered = {k: v for k, v in restored.items() if k in _PRESET_OVERRIDE_SECTIONS}
                    self._ephemeral_overrides = filtered
                    self._update_modified_badge()
                    self._update_preview()
                    status = self.query_one("#status-text", Static)
                    total = sum(len(v) for v in filtered.values())
                    status.update(f"[bold green]Restored {total} override(s) from draft.[/]")
                    # Delete the draft after successful restore
                    draft_path.unlink(missing_ok=True)
                except Exception as e:
                    status = self.query_one("#status-text", Static)
                    status.update(f"[bold red]Failed to restore draft:[/] {e}")
            elif result == "delete":
                draft_path.unlink(missing_ok=True)

        self.push_screen(DraftRestoreScreen(draft_path), _handle_draft_result)

    # ------------------------------------------------------------------
    # Selector change handling
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update preview panel whenever any selector changes.

        Clears ephemeral overrides when the selection changes to avoid
        stale overrides from a different config combination. Defers
        clearing until after the first successful preview so that
        draft-restored overrides survive initial selector setup.
        """
        if self._ephemeral_overrides and self._last_training_toml is not None:
            total = sum(len(v) for v in self._ephemeral_overrides.values())
            if total > 0:
                # Clear ephemeral overrides on navigation
                self._ephemeral_overrides.clear()
                self._clear_edit_area()
                self._update_modified_badge()
        self._update_preview()

    # ------------------------------------------------------------------
    # DataTable row selection -> edit area
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """When a row is selected in any preview table, populate the edit area."""
        table = event.data_table
        # Find which section this table belongs to
        section_key = None
        for _tab_label, sk in _PREVIEW_TABS:
            if table.id == f"table-{sk}":
                section_key = sk
                break

        if section_key is None:
            return

        # Get the row data
        row_data = table.get_row(event.row_key)
        key = str(row_data[0])
        current_value = str(row_data[1])

        self._edit_section = section_key
        self._edit_key = key

        # Show edit area
        edit_area = self.query_one("#edit-area")
        edit_area.remove_class("hidden")

        key_label = self.query_one("#edit-key-label", Label)
        key_label.update(f"[bold]{section_key}.{key}[/]")

        edit_input = self.query_one("#edit-input", Input)
        edit_input.value = current_value
        edit_input.focus()

        error_label = self.query_one("#edit-error", Label)
        error_label.update("")
        self._has_validation_error = False
        self._update_compile_button_state()

    # ------------------------------------------------------------------
    # Edit input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle value submission from the edit input."""
        if event.input.id != "edit-input":
            return
        self._apply_edit(event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live-validate the edit input value."""
        if event.input.id != "edit-input":
            return
        self._validate_edit(event.value)

    def _validate_edit(self, raw_value: str) -> bool:
        """Validate the current edit value. Returns True if valid.

        Shows inline error and updates compile button state.
        """
        error_label = self.query_one("#edit-error", Label)

        if not raw_value.strip():
            error_label.update("")
            self._has_validation_error = False
            self._update_compile_button_state()
            return True

        if self._edit_section is None or self._edit_key is None:
            return True

        # Reject non-preset section edits during live validation
        if self._edit_section not in _PRESET_OVERRIDE_SECTIONS:
            error_label.update(f"[bold red]Cannot override [{self._edit_section}] — computed from machine/persona/arch.[/]")
            self._has_validation_error = True
            self._update_compile_button_state()
            return False

        # Look up existing value for type checking
        existing_value = self._get_existing_value(self._edit_section, self._edit_key)

        try:
            from blissful_tuner.config_manager.value_parser import parse_and_typecheck

            parse_and_typecheck(raw_value, existing_value, self._edit_key)
            error_label.update("")
            self._has_validation_error = False
            self._update_compile_button_state()
            return True
        except (TypeError, ValueError) as e:
            error_label.update(f"[bold red]{e}[/]")
            self._has_validation_error = True
            self._update_compile_button_state()
            return False

    def _apply_edit(self, raw_value: str) -> None:
        """Parse, type-check, and store an ephemeral override."""
        if self._edit_section is None or self._edit_key is None:
            return

        error_label = self.query_one("#edit-error", Label)
        status = self.query_one("#status-text", Static)

        # Reject non-preset section overrides — model, output, advanced, dataset
        # are computed from machine/persona/arch and cannot be overridden.
        if self._edit_section not in _PRESET_OVERRIDE_SECTIONS:
            error_label.update(
                f"[bold red]Cannot override [{self._edit_section}] — "
                f"computed from machine/persona/arch. "
                f"Only {sorted(_PRESET_OVERRIDE_SECTIONS)} can be overridden.[/]"
            )
            self._has_validation_error = True
            self._update_compile_button_state()
            return

        raw_value = raw_value.strip()
        if not raw_value:
            # Empty value = remove override for this key
            if self._edit_section in self._ephemeral_overrides:
                self._ephemeral_overrides[self._edit_section].pop(self._edit_key, None)
                if not self._ephemeral_overrides[self._edit_section]:
                    del self._ephemeral_overrides[self._edit_section]
            self._has_validation_error = False
            self._update_compile_button_state()
            self._update_modified_badge()
            self._update_preview()
            status.update(f"Cleared override for {self._edit_section}.{self._edit_key}")
            return

        existing_value = self._get_existing_value(self._edit_section, self._edit_key)

        try:
            from blissful_tuner.config_manager.value_parser import parse_and_typecheck

            parsed = parse_and_typecheck(raw_value, existing_value, self._edit_key)
        except (TypeError, ValueError) as e:
            error_label.update(f"[bold red]{e}[/]")
            self._has_validation_error = True
            self._update_compile_button_state()
            return

        # Store the override
        if self._edit_section not in self._ephemeral_overrides:
            self._ephemeral_overrides[self._edit_section] = {}
        self._ephemeral_overrides[self._edit_section][self._edit_key] = parsed

        error_label.update("")
        self._has_validation_error = False
        self._update_compile_button_state()
        self._update_modified_badge()
        self._update_preview()
        status.update(f"Override set: {self._edit_section}.{self._edit_key} = {parsed!r}")

    def _get_existing_value(self, section: str, key: str) -> Any:
        """Look up the existing (base) value for a key, for type checking.

        Returns None if unknown.
        """
        if self._last_training_toml is None:
            return None
        section_data = self._last_training_toml.get(section, {})
        return section_data.get(key)

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _update_compile_button_state(self) -> None:
        """Enable/disable compile button based on validation state."""
        btn = self.query_one("#compile-btn", Button)
        btn.disabled = self._has_validation_error or self._has_unknown_keys

    def _update_modified_badge(self) -> None:
        """Update the sub-title to show [MODIFIED] when overrides exist."""
        total = sum(len(v) for v in self._ephemeral_overrides.values())
        if total > 0:
            self.sub_title = f"Training Config Compiler [MODIFIED] ({total} override(s))"
            # Show save/clear buttons
            self.query_one("#save-override-btn").remove_class("hidden")
            self.query_one("#clear-overrides-btn").remove_class("hidden")
        else:
            self.sub_title = "Training Config Compiler"
            self.query_one("#save-override-btn").add_class("hidden")
            self.query_one("#clear-overrides-btn").add_class("hidden")

    def _clear_edit_area(self) -> None:
        """Reset and hide the edit area."""
        self._edit_section = None
        self._edit_key = None
        self._has_validation_error = False
        self._update_compile_button_state()
        edit_area = self.query_one("#edit-area")
        edit_area.add_class("hidden")
        self.query_one("#edit-key-label", Label).update("Key: (none selected)")
        self.query_one("#edit-input", Input).value = ""
        self.query_one("#edit-error", Label).update("")

    def _override_count(self) -> int:
        """Count total ephemeral overrides across all sections."""
        return sum(len(v) for v in self._ephemeral_overrides.values())

    # ------------------------------------------------------------------
    # Preview update
    # ------------------------------------------------------------------

    def _get_selections(self) -> tuple[Any, Any, Any, Any]:
        """Read current dropdown values. Returns (machine, arch, persona, preset)."""
        machine = self.query_one("#machine-select", Select).value
        arch = self.query_one("#arch-select", Select).value
        persona = self.query_one("#persona-select", Select).value
        preset = self.query_one("#preset-select", Select).value
        return machine, arch, persona, preset

    def _update_preview(self) -> None:
        """Recompute the merged config preview from current selections + ephemeral overrides."""
        machine, arch, persona, preset = self._get_selections()

        # If any selection is blank, clear all tables
        if machine is Select.BLANK or arch is Select.BLANK or persona is Select.BLANK or preset is Select.BLANK:
            for _tab_label, section_key in _PREVIEW_TABS:
                table = self.query_one(f"#table-{section_key}", DataTable)
                table.clear()
            self._last_training_toml = None
            return

        if not self._meta_dir:
            return

        # Resolve paths
        machine_path = self._meta_dir / "machines" / f"{machine}.toml"
        persona_path = self._meta_dir / "personas" / f"{persona}.toml"
        preset_path = self._meta_dir / "presets" / f"{preset}.toml"

        try:
            from blissful_tuner.config_manager.compiler import compile_config

            # Always compile a baseline (no overrides) for unknown-key detection
            baseline_result = compile_config(
                machine_path=machine_path,
                arch_key=str(arch),
                persona_path=persona_path,
                preset_path=preset_path,
            )
            self._baseline_training_toml = baseline_result["training_toml"]

            # Build override data from ephemeral overrides (preset-layer sections)
            override_data = _build_override_data(self._ephemeral_overrides) or None

            # Compile with overrides for display (or reuse baseline if none)
            if override_data:
                result = compile_config(
                    machine_path=machine_path,
                    arch_key=str(arch),
                    persona_path=persona_path,
                    preset_path=preset_path,
                    override_data=override_data,
                )
                training_toml = result["training_toml"]
            else:
                training_toml = self._baseline_training_toml

            self._last_training_toml = training_toml

            # Check override keys against baseline sections (same check compile_to_disk uses)
            unknown_keys: list[str] = []
            if override_data and self._baseline_training_toml:
                for section, keys in override_data.items():
                    if not isinstance(keys, dict):
                        continue
                    baseline_section = self._baseline_training_toml.get(section, {})
                    if not isinstance(baseline_section, dict):
                        baseline_section = {}
                    for key in keys:
                        if key not in baseline_section:
                            unknown_keys.append(f"{section}.{key}")

            self._has_unknown_keys = bool(unknown_keys)
            self._update_compile_button_state()

            # Update each tab's DataTable
            for _tab_label, section_key in _PREVIEW_TABS:
                table = self.query_one(f"#table-{section_key}", DataTable)
                table.clear()
                section_data = training_toml.get(section_key, {})
                overrides_for_section = self._ephemeral_overrides.get(section_key, {})

                for key in sorted(section_data.keys()):
                    value = section_data[key]
                    if isinstance(value, dict):
                        # Flatten nested dicts
                        for sub_key, sub_value in sorted(value.items()):
                            flat_key = f"{key}.{sub_key}"
                            status = "[MODIFIED]" if flat_key in overrides_for_section else ""
                            table.add_row(flat_key, str(sub_value), status)
                    else:
                        if key in overrides_for_section:
                            full_key = f"{section_key}.{key}"
                            status = "[UNKNOWN]" if full_key in unknown_keys else "[MODIFIED]"
                        else:
                            status = ""
                        table.add_row(key, str(value), status)

            if unknown_keys:
                status_widget = self.query_one("#status-text", Static)
                status_widget.update(
                    f"[bold yellow]Warning:[/] Unknown key(s) will be rejected on compile: {', '.join(unknown_keys)}"
                )

        except Exception as e:
            # Show error in status
            status_widget = self.query_one("#status-text", Static)
            status_widget.update(f"[bold red]Preview error:[/] {e}")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self.action_request_quit()
        elif event.button.id == "compile-btn":
            self._do_compile()
        elif event.button.id == "save-override-btn":
            self.action_save_override()
        elif event.button.id == "clear-overrides-btn":
            self._clear_all_overrides()

    def _clear_all_overrides(self) -> None:
        """Clear all ephemeral overrides."""
        self._ephemeral_overrides.clear()
        self._clear_edit_area()
        self._update_modified_badge()
        self._update_preview()
        status = self.query_one("#status-text", Static)
        status.update("All ephemeral overrides cleared.")

    def _do_compile(self) -> None:
        """Run the compile pipeline with current selections and ephemeral overrides."""
        status = self.query_one("#status-text", Static)

        # Get current selections
        machine, arch, persona, preset = self._get_selections()

        # Validate all selections made
        if machine is Select.BLANK or arch is Select.BLANK or persona is Select.BLANK or preset is Select.BLANK:
            status.update("[bold red]Error:[/] Please select all four fields before compiling.")
            return

        if not self._meta_dir:
            status.update("[bold red]Error:[/] Could not find configs/meta/ directory.")
            return

        # Resolve paths
        machine_path = self._meta_dir / "machines" / f"{machine}.toml"
        persona_path = self._meta_dir / "personas" / f"{persona}.toml"
        preset_path = self._meta_dir / "presets" / f"{preset}.toml"

        try:
            output_dir = _find_compiled_dir()
        except FileNotFoundError:
            status.update("[bold red]Error:[/] Could not find configs/compiled/ directory.")
            return

        status.update(f"Compiling {persona}/{arch}/{preset} for {machine}...")

        # Build override_data dict directly from ephemeral overrides
        # (no repr+parse round-trip — preserves types exactly)
        override_data = _build_override_data(self._ephemeral_overrides) or None

        try:
            from blissful_tuner.config_manager.compiler import compile_to_disk

            result = compile_to_disk(
                machine_path=machine_path,
                arch_key=str(arch),
                persona_path=persona_path,
                preset_path=preset_path,
                output_dir=output_dir,
                override_data=override_data,
            )

            training_path = result.get("training_toml_path", "?")
            dataset_path = result.get("dataset_toml_path", "?")

            override_note = ""
            if self._override_count() > 0:
                override_note = f" (with {self._override_count()} override(s))"

            status.update(
                f"[bold green]Compiled successfully!{override_note}[/]\n  Training: {training_path}\n  Dataset:  {dataset_path}"
            )
        except Exception as e:
            status.update(f"[bold red]Compile failed:[/] {e}")

    # ------------------------------------------------------------------
    # Save override action
    # ------------------------------------------------------------------

    def action_save_override(self) -> None:
        """Open the save-as-override dialog (Ctrl+S or button)."""
        if self._override_count() == 0:
            status = self.query_one("#status-text", Static)
            status.update("No ephemeral overrides to save.")
            return

        def _handle_save_result(name: str | None) -> None:
            if name is None:
                return
            if not self._meta_dir:
                return
            save_path = _overrides_dir(self._meta_dir) / f"{name}.toml"
            status = self.query_one("#status-text", Static)
            if save_path.exists():
                status.update(f"[bold red]Override file already exists:[/] {save_path.name}")
                return
            try:
                _write_override_toml(save_path, self._ephemeral_overrides)
                status.update(f"[bold green]Saved overrides to {save_path.name}[/]")
            except Exception as e:
                status.update(f"[bold red]Failed to save override:[/] {e}")

        self.push_screen(SaveOverrideScreen(), _handle_save_result)

    # ------------------------------------------------------------------
    # Quit with unsaved-changes check
    # ------------------------------------------------------------------

    def action_request_quit(self) -> None:
        """Handle quit request, with confirm dialog if overrides exist."""
        if self._override_count() == 0:
            self._clean_exit()
            return

        def _handle_quit_result(result: str | None) -> None:
            if result == "discard":
                self._clean_exit()
            elif result == "save":
                # Save then quit
                self.action_save_override()
                # Note: we don't auto-quit after save since the user might cancel
                # the save dialog. The user can quit again after saving.
            # "cancel" or None = do nothing

        self.push_screen(QuitConfirmScreen(self._override_count()), _handle_quit_result)

    def _clean_exit(self) -> None:
        """Exit cleanly, removing the atexit handler's need to save drafts."""
        self._ephemeral_overrides.clear()  # Prevent atexit from saving
        # Clean up any draft from this session
        if self._draft_path and self._draft_path.exists():
            self._draft_path.unlink(missing_ok=True)
        self.exit()


def main():
    """Entry point for bt-tui CLI."""
    app = BlissfulConfigApp()
    app.run()


if __name__ == "__main__":
    main()
