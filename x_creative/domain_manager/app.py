"""Main TUI application for domain manager."""

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

from x_creative.domain_manager.services.yaml_manager import YAMLManager


class DomainManagerApp(App):
    """Main TUI application for domain management.

    Manages source domains within a specific target domain.
    """

    TITLE = "xc-domain"
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

    def __init__(self, target_domain_id: str = "open_source_development") -> None:
        """Initialize the application.

        Args:
            target_domain_id: Target domain to manage.
        """
        super().__init__()
        self._target_domain_id = target_domain_id
        self._yaml_manager = YAMLManager(target_domain_id)
        self._stats = self._yaml_manager.get_stats()

    @property
    def target_domain_id(self) -> str:
        """Get the target domain ID."""
        return self._target_domain_id

    @property
    def target_domain_name(self) -> str:
        """Get the target domain display name."""
        return self._stats.get("target_domain_name", self._target_domain_id)

    def on_mount(self) -> None:
        """Called when app is mounted."""
        from x_creative.domain_manager.screens import (
            AddDomainExploreScreen,
            AddDomainManualScreen,
            ExtendStructuresScreen,
            MainMenuScreen,
        )

        self.install_screen(MainMenuScreen(self._stats), name="main_menu")
        self.install_screen(AddDomainManualScreen(), name="add_domain_manual")
        self.install_screen(AddDomainExploreScreen(), name="add_domain_explore")
        self.install_screen(ExtendStructuresScreen(), name="extend_structures")
        self.push_screen("main_menu")

    @property
    def yaml_manager(self) -> YAMLManager:
        """Get the YAML manager."""
        return self._yaml_manager

    def refresh_stats(self) -> None:
        """Refresh domain statistics."""
        self._stats = self._yaml_manager.get_stats()
