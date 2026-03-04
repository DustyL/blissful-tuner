"""Blissful Config TUI -- interactive config compiler."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, Select, Static


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

        with Vertical(id="status-area"):
            yield Static("Ready. Select machine, architecture, persona, and preset to compile.", id="status-text")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self.exit()
        elif event.button.id == "compile-btn":
            self._do_compile()

    def _do_compile(self) -> None:
        """Run the compile pipeline with current selections."""
        status = self.query_one("#status-text", Static)

        # Get current selections
        machine = self.query_one("#machine-select", Select).value
        arch = self.query_one("#arch-select", Select).value
        persona = self.query_one("#persona-select", Select).value
        preset = self.query_one("#preset-select", Select).value

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
