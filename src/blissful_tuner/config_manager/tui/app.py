"""Blissful Config TUI -- interactive config compiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, Select, Static, TabbedContent, TabPane


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


# Tab IDs and their corresponding sections in compile_config() training_toml
_PREVIEW_TABS: list[tuple[str, str]] = [
    ("Model", "model"),
    ("Network", "network"),
    ("Optimizer", "optimizer"),
    ("Training", "training"),
    ("Output", "output"),
    ("Advanced", "advanced"),
]

_PLACEHOLDER_MSG = "Select all fields to preview"


def _format_section(section_name: str, data: dict[str, Any], provenance: dict[str, Any]) -> str:
    """Format a config section as readable key = value lines with provenance header.

    Args:
        section_name: The section name (e.g. "model", "training").
        data: The section dict from compile_config().
        provenance: Provenance metadata from compile_config().

    Returns:
        Formatted string with provenance header and key-value pairs.
    """
    prov_parts = [f"{k}={v}" for k, v in provenance.items()]
    lines = [f"[bold green]\\[{section_name}][/bold green]  [dim]({', '.join(prov_parts)})[/dim]", ""]

    if not data:
        lines.append("[dim]  (empty)[/dim]")
        return "\n".join(lines)

    for key, value in sorted(data.items()):
        if isinstance(value, dict):
            # Nested dict -- show each sub-key indented
            lines.append(f"  [bold]{key}[/bold]:")
            for sub_key, sub_value in sorted(value.items()):
                lines.append(f"    {sub_key} = {sub_value}")
        elif isinstance(value, list):
            lines.append(f"  [bold]{key}[/bold] = {value}")
        else:
            lines.append(f"  [bold]{key}[/bold] = {value}")

    return "\n".join(lines)


class BlissfulConfigApp(App):
    """Blissful Config TUI -- compile layered TOML configs interactively."""

    TITLE = "Blissful Config"
    SUB_TITLE = "Training Config Compiler"

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
        margin-right: 2;
    }

    #preview-panel {
        height: 1fr;
        border-top: solid green;
        padding: 1 2;
    }

    .preview-tab-content {
        height: 1fr;
        overflow-y: auto;
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
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            self._meta_dir: Path | None = _find_meta_dir()
        except FileNotFoundError:
            self._meta_dir = None

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
            yield Button("Quit", id="quit-btn", variant="error")

        # Preview panel with tabbed sections
        with TabbedContent(id="preview-panel"):
            for tab_label, section_key in _PREVIEW_TABS:
                with TabPane(tab_label, id=f"tab-{section_key}"):
                    yield Static(_PLACEHOLDER_MSG, id=f"preview-{section_key}", classes="preview-tab-content")

        with Vertical(id="status-area"):
            yield Static(
                "Ready. Select machine, architecture, persona, and preset to compile.",
                id="status-text",
            )

        yield Footer()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update preview panel whenever any selector changes."""
        self._update_preview()

    def _get_selections(self) -> tuple[Any, Any, Any, Any]:
        """Read current dropdown values. Returns (machine, arch, persona, preset)."""
        machine = self.query_one("#machine-select", Select).value
        arch = self.query_one("#arch-select", Select).value
        persona = self.query_one("#persona-select", Select).value
        preset = self.query_one("#preset-select", Select).value
        return machine, arch, persona, preset

    def _update_preview(self) -> None:
        """Recompute the merged config preview from current selections."""
        machine, arch, persona, preset = self._get_selections()

        # If any selection is blank, show placeholder in all tabs
        if machine is Select.BLANK or arch is Select.BLANK or persona is Select.BLANK or preset is Select.BLANK:
            for _tab_label, section_key in _PREVIEW_TABS:
                widget = self.query_one(f"#preview-{section_key}", Static)
                widget.update(_PLACEHOLDER_MSG)
            return

        if not self._meta_dir:
            for _tab_label, section_key in _PREVIEW_TABS:
                widget = self.query_one(f"#preview-{section_key}", Static)
                widget.update("[bold red]Error:[/] Could not find configs/meta/ directory.")
            return

        # Resolve paths
        machine_path = self._meta_dir / "machines" / f"{machine}.toml"
        persona_path = self._meta_dir / "personas" / f"{persona}.toml"
        preset_path = self._meta_dir / "presets" / f"{preset}.toml"

        try:
            from blissful_tuner.config_manager.compiler import compile_config

            result = compile_config(
                machine_path=machine_path,
                arch_key=str(arch),
                persona_path=persona_path,
                preset_path=preset_path,
            )

            training_toml = result["training_toml"]
            provenance = result["provenance"]

            # Update each tab with its section data
            for _tab_label, section_key in _PREVIEW_TABS:
                widget = self.query_one(f"#preview-{section_key}", Static)
                section_data = training_toml.get(section_key, {})
                formatted = _format_section(section_key, section_data, provenance)
                widget.update(formatted)

        except Exception as e:
            # Show error in all tabs
            error_msg = f"[bold red]Preview error:[/] {e}"
            for _tab_label, section_key in _PREVIEW_TABS:
                widget = self.query_one(f"#preview-{section_key}", Static)
                widget.update(error_msg)

            # Also update status bar
            status = self.query_one("#status-text", Static)
            status.update(f"[bold red]Preview error:[/] {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self.exit()
        elif event.button.id == "compile-btn":
            self._do_compile()

    def _do_compile(self) -> None:
        """Run the compile pipeline with current selections."""
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

        try:
            from blissful_tuner.config_manager.cli import run_compile

            result = run_compile(
                persona=persona,
                arch=arch,
                preset=preset,
                machine_path=machine_path,
                persona_path=persona_path,
                preset_path=preset_path,
                output_dir=output_dir,
            )

            training_path = result.get("training_toml_path", "?")
            dataset_path = result.get("dataset_toml_path", "?")

            status.update(f"[bold green]Compiled successfully![/]\n  Training: {training_path}\n  Dataset:  {dataset_path}")
        except Exception as e:
            status.update(f"[bold red]Compile failed:[/] {e}")


def main():
    """Entry point for bt-tui CLI."""
    app = BlissfulConfigApp()
    app.run()


if __name__ == "__main__":
    main()
