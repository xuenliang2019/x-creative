"""Main menu screen for domain manager."""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static


class MainMenuScreen(Screen):
    """Main menu screen with navigation options."""

    BINDINGS = [
        Binding("1", "add_manual", "添加源域(手动)"),
        Binding("2", "add_explore", "添加源域(探索)"),
        Binding("3", "extend", "扩展结构"),
        Binding("q", "quit", "退出"),
    ]

    def __init__(self, stats: dict[str, Any]) -> None:
        """Initialize with domain statistics."""
        super().__init__()
        self._stats = stats

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        target_name = self._stats.get("target_domain_name", "")
        target_id = self._stats.get("target_domain_id", "")
        domain_count = self._stats.get("domain_count", 0)
        structure_count = self._stats.get("structure_count", 0)
        mapping_count = self._stats.get("mapping_count", 0)

        yield Header()
        yield Vertical(
            Center(
                Static(
                    "[bold]xc-domain - Domain Manager[/bold]\n\n"
                    f"目标域: [cyan]{target_name}[/cyan] ({target_id})",
                    id="title",
                )
            ),
            Center(
                Vertical(
                    Button("添加新源域 (手动指定)", id="btn-manual", variant="primary"),
                    Button("添加新源域 (自动探索)", id="btn-explore", variant="primary"),
                    Button("扩展现有源域的 Structures", id="btn-extend", variant="primary"),
                    Button("退出", id="btn-quit", variant="default"),
                    id="menu-buttons",
                )
            ),
            Center(
                Static(
                    f"当前已有 [cyan]{domain_count}[/cyan] 个源域 | "
                    f"[cyan]{structure_count}[/cyan] 个 structures | "
                    f"[cyan]{mapping_count}[/cyan] 个 mappings",
                    id="stats",
                )
            ),
            id="main-container",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-manual":
            self.action_add_manual()
        elif button_id == "btn-explore":
            self.action_add_explore()
        elif button_id == "btn-extend":
            self.action_extend()
        elif button_id == "btn-quit":
            self.action_quit()

    def action_add_manual(self) -> None:
        """Navigate to add domain (manual) screen."""
        self.app.push_screen("add_domain_manual")

    def action_add_explore(self) -> None:
        """Navigate to add domain (explore) screen."""
        self.app.push_screen("add_domain_explore")

    def action_extend(self) -> None:
        """Navigate to extend structures screen."""
        self.app.push_screen("extend_structures")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
