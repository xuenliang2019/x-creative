"""Main menu screen for target domain manager."""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, OptionList, Static
from textual.widgets.option_list import Option


class MainMenuScreen(Screen):
    """Main menu screen with target domain listing and navigation."""

    BINDINGS = [
        Binding("c", "create", "创建"),
        Binding("v", "view", "查看"),
        Binding("e", "edit", "编辑"),
        Binding("q", "quit", "退出"),
    ]

    CSS = """
    #target-list {
        height: auto;
        max-height: 16;
        width: 70;
        margin: 1 0;
    }

    #menu-action-buttons {
        width: auto;
        height: auto;
        padding: 1;
    }

    #menu-action-buttons Button {
        width: 40;
        margin: 1;
    }

    #summary-stats {
        text-align: center;
        padding: 1;
        color: $text-muted;
    }
    """

    def __init__(self, targets: list[dict[str, Any]]) -> None:
        """Initialize with target domain list.

        Args:
            targets: List of target domain info dicts.
        """
        super().__init__()
        self._targets = targets

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Center(
                Static(
                    "[bold]xc-target - Target Domain Manager[/bold]\n\n"
                    f"共 [cyan]{len(self._targets)}[/cyan] 个 Target Domain",
                    id="title",
                )
            ),
            Center(
                OptionList(id="target-list"),
            ),
            Center(
                Static("", id="summary-stats"),
            ),
            Center(
                Vertical(
                    Button("创建新 Target Domain [C]", id="btn-create", variant="primary"),
                    Button("查看 Target Domain [V]", id="btn-view", variant="primary"),
                    Button("编辑 Target Domain [E]", id="btn-edit", variant="primary"),
                    Button("退出 [Q]", id="btn-quit", variant="default"),
                    id="menu-action-buttons",
                )
            ),
            id="main-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Populate target list on mount."""
        target_list = self.query_one("#target-list", OptionList)
        for target in self._targets:
            label = (
                f"{target['name']} ({target['id']}) - "
                f"{target['domain_count']} 源域, "
                f"{target['constraint_count']} 约束"
            )
            target_list.add_option(Option(label, id=target["id"]))

        if not self._targets:
            target_list.add_option(Option("(无 Target Domain，请先创建)", id="_empty"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-create":
            self.action_create()
        elif button_id == "btn-view":
            self.action_view()
        elif button_id == "btn-edit":
            self.action_edit()
        elif button_id == "btn-quit":
            self.action_quit()

    def _get_selected_target_id(self) -> str | None:
        """Get the currently selected target domain ID."""
        target_list = self.query_one("#target-list", OptionList)
        if target_list.highlighted is not None:
            option = target_list.get_option_at_index(target_list.highlighted)
            if option.id and option.id != "_empty":
                return option.id
        return None

    def action_create(self) -> None:
        """Navigate to create target screen."""
        self.app.push_screen("create_target")

    def action_view(self) -> None:
        """Navigate to view target screen."""
        target_id = self._get_selected_target_id()
        if target_id:
            screen = self.app.get_screen("view_target")
            screen.target_id = target_id
            self.app.push_screen("view_target")
        else:
            self.notify("请先选择一个 Target Domain", severity="warning")

    def action_edit(self) -> None:
        """Navigate to edit target screen."""
        target_id = self._get_selected_target_id()
        if target_id:
            screen = self.app.get_screen("edit_target")
            screen.target_id = target_id
            self.app.push_screen("edit_target")
        else:
            self.notify("请先选择一个 Target Domain", severity="warning")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle double-click / Enter on option list — go to view."""
        if event.option.id and event.option.id != "_empty":
            screen = self.app.get_screen("view_target")
            screen.target_id = event.option.id
            self.app.push_screen("view_target")
