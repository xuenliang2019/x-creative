"""Main TUI application for target domain manager."""

from __future__ import annotations

try:
    from textual.app import App
    from textual.binding import Binding
except ModuleNotFoundError:
    class App:  # type: ignore[no-redef]
        """Fallback App stub for non-TUI test environments."""

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def install_screen(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def push_screen(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    class Binding:  # type: ignore[no-redef]
        """Fallback Binding stub for non-TUI test environments."""

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

from x_creative.target_manager.services.target_yaml_manager import TargetYAMLManager


class TargetManagerApp(App):
    """Main TUI application for target domain management.

    Manages target domain YAML files: create, view, edit.
    """

    TITLE = "xc-target"
    BINDINGS = [
        Binding("q", "quit", "退出", show=True),
    ]

    CSS = """
    #title {
        text-align: center;
        padding: 2;
    }

    #menu-buttons {
        width: auto;
        height: auto;
        padding: 1;
    }

    #menu-buttons Button {
        width: 40;
        margin: 1;
    }

    #stats {
        text-align: center;
        padding: 2;
        color: $text-muted;
    }

    #main-container {
        height: 100%;
        align: center middle;
    }
    """

    def __init__(self) -> None:
        """Initialize the application."""
        super().__init__()
        self._yaml_manager = TargetYAMLManager()

    @property
    def yaml_manager(self) -> TargetYAMLManager:
        """Get the YAML manager."""
        return self._yaml_manager

    def on_mount(self) -> None:
        """Called when app is mounted."""
        from x_creative.target_manager.screens import (
            CreateTargetScreen,
            EditTargetScreen,
            MainMenuScreen,
            ViewTargetScreen,
        )

        targets = self._yaml_manager.list_target_domains()
        self.install_screen(MainMenuScreen(targets), name="main_menu")
        self.install_screen(CreateTargetScreen(), name="create_target")
        self.install_screen(ViewTargetScreen(), name="view_target")
        self.install_screen(EditTargetScreen(), name="edit_target")
        self.push_screen("main_menu")

    def refresh_targets(self) -> list[dict]:
        """Refresh and return target domain list."""
        return self._yaml_manager.list_target_domains()
