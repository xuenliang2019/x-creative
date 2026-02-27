"""Screen for editing target domain sections."""

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from x_creative.core.plugin import TargetDomainPlugin
from x_creative.target_manager.services.target_generator import TargetGeneratorService


# Editable sections with display labels
_SECTIONS = [
    ("constraints", "Constraints (约束)"),
    ("evaluation_criteria", "Evaluation Criteria (评估指标)"),
    ("anti_patterns", "Anti-Patterns (反模式)"),
    ("terminology", "Terminology (术语表)"),
    ("stale_ideas", "Stale Ideas (陈旧想法)"),
]


class EditTargetScreen(Screen):
    """Screen for editing individual sections of a target domain."""

    BINDINGS = [
        Binding("escape", "back", "返回"),
    ]

    CSS = """
    #edit-container {
        height: 100%;
        padding: 1 2;
    }

    #edit-title {
        text-style: bold;
        text-align: center;
        padding: 1;
    }

    #section-list {
        height: auto;
        max-height: 10;
        width: 50;
        margin: 1 0;
    }

    #section-content {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        margin: 1 0;
    }

    #edit-actions {
        height: auto;
        padding: 1 0;
    }

    #edit-actions Button {
        margin: 0 1;
    }

    #edit-status {
        padding: 1 0;
        color: $text-muted;
    }

    .hidden {
        display: none;
    }
    """

    def __init__(self) -> None:
        """Initialize the screen."""
        super().__init__()
        self.target_id: str = ""
        self._plugin: TargetDomainPlugin | None = None
        self._selected_section: str | None = None
        self._regen_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("", id="edit-title"),
            OptionList(id="section-list"),
            ScrollableContainer(
                Static("", id="section-display"),
                id="section-content",
            ),
            Horizontal(
                Button("Regenerate (LLM 重新生成)", id="btn-regenerate", variant="warning"),
                Button("返回", id="btn-back", variant="default"),
                id="edit-actions",
            ),
            Static("", id="edit-status"),
            id="edit-container",
        )
        yield Footer()

    def on_screen_resume(self) -> None:
        """Refresh data when screen is shown."""
        self._load_data()

    def on_mount(self) -> None:
        """Load data on mount."""
        self._load_data()

    def _load_data(self) -> None:
        """Load target domain data."""
        if not self.target_id:
            return

        try:
            self._plugin = self.app.yaml_manager.load_target_domain(self.target_id)
        except FileNotFoundError:
            self.query_one("#edit-title", Static).update(
                f"[red]Target Domain '{self.target_id}' 未找到[/red]"
            )
            return

        self.query_one("#edit-title", Static).update(
            f"[bold]编辑: {self._plugin.name}[/bold] ({self._plugin.id})"
        )

        # Populate section list
        section_list = self.query_one("#section-list", OptionList)
        section_list.clear_options()
        for section_key, section_label in _SECTIONS:
            count = self._get_section_count(section_key)
            section_list.add_option(
                Option(f"{section_label} ({count} 项)", id=section_key)
            )

    def _get_section_count(self, section: str) -> int:
        """Get the item count for a section."""
        if not self._plugin:
            return 0
        data = getattr(self._plugin, section, None)
        if data is None:
            return 0
        if isinstance(data, dict):
            return len(data)
        if isinstance(data, list):
            return len(data)
        return 0

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Show section content when highlighted."""
        if event.option_list.id == "section-list" and event.option.id:
            self._selected_section = event.option.id
            self._display_section(event.option.id)

    def _display_section(self, section: str) -> None:
        """Display the content of a section."""
        if not self._plugin:
            return

        display = self.query_one("#section-display", Static)
        data = getattr(self._plugin, section, None)

        if section == "constraints":
            if data:
                lines = []
                for c in data:
                    lines.append(f"[bold]{c.name}[/bold] [{c.severity}]")
                    lines.append(f"  {c.description}")
                    if c.check_prompt:
                        lines.append(f"  [dim]Check: {c.check_prompt}[/dim]")
                    lines.append("")
                display.update("\n".join(lines))
            else:
                display.update("(无)")
        elif section == "terminology":
            if data:
                display.update(
                    "\n".join(f"[bold]{k}[/bold]: {v}" for k, v in data.items())
                )
            else:
                display.update("(无)")
        elif isinstance(data, list):
            if data:
                display.update("\n".join(f"- {item}" for item in data))
            else:
                display.update("(无)")
        else:
            display.update("(无)")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-regenerate":
            self._regenerate_section()
        elif event.button.id == "btn-back":
            self.action_back()

    def _regenerate_section(self) -> None:
        """Regenerate the selected section using LLM."""
        if not self._selected_section or not self._plugin:
            self.notify("请先选择一个节", severity="warning")
            return

        if self._regen_task and not self._regen_task.done():
            self._regen_task.cancel()

        status = self.query_one("#edit-status", Static)
        status.update(f"[cyan]正在重新生成 {self._selected_section}...[/cyan]")

        self._regen_task = asyncio.create_task(
            self._do_regenerate(self._selected_section)
        )

    async def _do_regenerate(self, section: str) -> None:
        """Regenerate a section and save."""
        status = self.query_one("#edit-status", Static)
        try:
            async with TargetGeneratorService() as generator:
                name = self._plugin.name
                desc = self._plugin.description

                if section == "constraints":
                    result = await generator.generate_constraints(name, desc)
                    data = [
                        {
                            "name": c.name,
                            "description": c.description,
                            "severity": c.severity,
                            "check_prompt": c.check_prompt,
                        }
                        for c in result
                    ]
                elif section == "evaluation_criteria":
                    data = await generator.generate_evaluation_criteria(name, desc)
                elif section == "anti_patterns":
                    data = await generator.generate_anti_patterns(name, desc)
                elif section == "terminology":
                    data = await generator.generate_terminology(name, desc)
                elif section == "stale_ideas":
                    data = await generator.generate_stale_ideas(name, desc)
                else:
                    status.update(f"[red]不支持重新生成 {section}[/red]")
                    return

            # Save to YAML
            self.app.yaml_manager.update_section(self.target_id, section, data)

            # Reload and refresh display
            self._load_data()
            self._display_section(section)
            status.update(f"[green]{section} 已重新生成并保存！[/green]")

        except asyncio.CancelledError:
            status.update("已取消")
        except Exception as e:
            status.update(f"[red]重新生成失败: {e!s}[/red]")

    def action_back(self) -> None:
        """Return to main menu."""
        if self._regen_task and not self._regen_task.done():
            self._regen_task.cancel()
        self.app.pop_screen()
