"""Screen for manually adding a new domain."""

import asyncio
import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Static,
)

from x_creative.core.types import Domain, DomainStructure, TargetMapping
from x_creative.domain_manager.services import (
    DomainGeneratorService,
    DomainSearchService,
)


class AddDomainManualScreen(Screen):
    """Screen for adding a domain by specifying its name."""

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    CSS = """
    #add-domain-container {
        height: 100%;
        padding: 1 2;
    }

    #input-section {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }

    #input-section Label {
        margin-bottom: 1;
    }

    #domain-input {
        width: 60;
    }

    #button-row {
        height: auto;
        width: auto;
        margin-top: 1;
    }

    #button-row Button {
        margin-right: 1;
    }

    #status-section {
        height: auto;
        padding: 1;
        margin: 1 0;
    }

    #status-text {
        color: $text-muted;
    }

    #loading-section {
        height: auto;
        align: center middle;
        padding: 1;
    }

    #result-section {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        margin: 1 0;
    }

    #result-content {
        height: auto;
        padding: 1;
    }

    #action-buttons {
        height: auto;
        width: auto;
        align: center middle;
        padding: 1;
    }

    #action-buttons Button {
        margin: 0 1;
    }

    .hidden {
        display: none;
    }

    .error-text {
        color: $error;
    }

    .success-text {
        color: $success;
    }

    #preview-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self) -> None:
        """Initialize the screen."""
        super().__init__()
        self._generated_domain: Domain | None = None
        self._generation_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            # Input section
            Vertical(
                Label("请输入要添加的领域名称（中文或英文）:"),
                Input(placeholder="例如：流体力学、博弈论、生态学...", id="domain-input"),
                Horizontal(
                    Button("生成领域", id="btn-generate", variant="primary"),
                    Button("取消", id="btn-cancel", variant="default"),
                    id="button-row",
                ),
                id="input-section",
            ),
            # Status section
            Vertical(
                Static("", id="status-text"),
                id="status-section",
            ),
            # Loading indicator
            Center(
                LoadingIndicator(id="loading"),
                id="loading-section",
                classes="hidden",
            ),
            # Result preview section
            ScrollableContainer(
                Static("", id="result-content"),
                id="result-section",
                classes="hidden",
            ),
            # Action buttons (after generation)
            Horizontal(
                Button("确认添加", id="btn-confirm", variant="success"),
                Button("重新生成", id="btn-regenerate", variant="warning"),
                Button("取消", id="btn-cancel-result", variant="default"),
                id="action-buttons",
                classes="hidden",
            ),
            id="add-domain-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self.query_one("#domain-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-generate":
            self._start_generation()
        elif button_id == "btn-cancel" or button_id == "btn-cancel-result":
            self.action_cancel()
        elif button_id == "btn-confirm":
            self._confirm_add()
        elif button_id == "btn-regenerate":
            self._start_generation()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input."""
        if event.input.id == "domain-input":
            self._start_generation()

    def _start_generation(self) -> None:
        """Start the domain generation process."""
        domain_input = self.query_one("#domain-input", Input)
        domain_name = domain_input.value.strip()

        if not domain_name:
            self._show_status("请输入领域名称", is_error=True)
            return

        # Cancel any existing task
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

        # Reset state
        self._generated_domain = None
        self._hide_results()
        self._show_loading(True)
        self._show_status(f"正在搜索 [{domain_name}] 相关信息...")

        # Start async generation
        self._generation_task = asyncio.create_task(
            self._generate_domain(domain_name)
        )

    async def _generate_domain(self, domain_name: str) -> None:
        """Generate domain content asynchronously."""
        try:
            # Step 1: Search for domain concepts
            self._show_status(f"正在搜索 [{domain_name}] 核心概念...")
            async with DomainSearchService() as search_service:
                search_results = await search_service.search_domain_concepts(
                    domain_name
                )

            if not search_results:
                self._show_status(
                    "搜索未返回结果，将基于 LLM 知识生成",
                    is_error=False,
                )

            # Step 2: Extract structures using LLM
            self._show_status("正在使用 LLM 提取领域结构...")
            async with DomainGeneratorService(self.app.target_domain_name) as generator:
                structures = await generator.extract_structures(
                    domain_name, search_results
                )

                if not structures:
                    self._show_loading(False)
                    self._show_status("未能提取有效结构，请尝试其他领域名称", is_error=True)
                    return

                # Step 3: Generate target mappings for each structure
                target_name = self.app.target_domain_name
                self._show_status(f"正在生成{target_name}映射...")
                mappings: list[TargetMapping] = []
                for structure in structures:
                    mapping_results = await search_service.search_structure_applications(
                        structure.name, domain_name
                    ) if search_results else []
                    mapping = await generator.generate_mapping(
                        structure, mapping_results
                    )
                    mappings.append(mapping)

            # Create domain object
            domain_id = self._generate_domain_id(domain_name)
            self._generated_domain = Domain(
                id=domain_id,
                name=domain_name,
                name_en=self._extract_english_name(domain_name),
                description=f"{domain_name}领域的核心概念和可迁移结构",
                structures=structures,
                target_mappings=mappings,
            )

            # Show results
            self._show_loading(False)
            self._show_results()
            self._show_status("生成完成！请检查预览后确认添加。", is_success=True)

        except asyncio.CancelledError:
            self._show_loading(False)
            self._show_status("已取消")
        except Exception as e:
            self._show_loading(False)
            self._show_status(f"生成失败: {e!s}", is_error=True)

    def _generate_domain_id(self, domain_name: str) -> str:
        """Generate a domain ID from the name."""
        # Simple conversion to snake_case
        # Remove non-alphanumeric characters and replace spaces with underscores
        clean = re.sub(r"[^\w\s]", "", domain_name)
        words = clean.lower().split()
        return "_".join(words) if words else "unnamed_domain"

    def _extract_english_name(self, domain_name: str) -> str | None:
        """Try to extract or generate English name."""
        # If the name is already in English, return it with title case
        if domain_name.isascii():
            return domain_name.title()
        # For Chinese names, return None (can be enhanced later)
        return None

    def _show_status(
        self, message: str, is_error: bool = False, is_success: bool = False
    ) -> None:
        """Update status text."""
        status = self.query_one("#status-text", Static)
        if is_error:
            status.update(f"[red]{message}[/red]")
        elif is_success:
            status.update(f"[green]{message}[/green]")
        else:
            status.update(f"[cyan]{message}[/cyan]")

    def _show_loading(self, show: bool) -> None:
        """Show or hide loading indicator."""
        loading_section = self.query_one("#loading-section")
        if show:
            loading_section.remove_class("hidden")
        else:
            loading_section.add_class("hidden")

    def _show_results(self) -> None:
        """Show the generated domain preview."""
        if not self._generated_domain:
            return

        result_section = self.query_one("#result-section")
        result_content = self.query_one("#result-content", Static)
        action_buttons = self.query_one("#action-buttons")

        # Format preview content
        domain = self._generated_domain
        lines = [
            f"[bold cyan]领域预览[/bold cyan]\n",
            f"[bold]ID:[/bold] {domain.id}",
            f"[bold]名称:[/bold] {domain.name}",
        ]
        if domain.name_en:
            lines.append(f"[bold]英文名:[/bold] {domain.name_en}")
        lines.append(f"[bold]描述:[/bold] {domain.description}")
        lines.append("")
        lines.append(f"[bold cyan]结构 ({len(domain.structures)} 个)[/bold cyan]")
        lines.append("")

        for i, structure in enumerate(domain.structures, 1):
            lines.append(f"[bold]{i}. {structure.name}[/bold] ({structure.id})")
            lines.append(f"   描述: {structure.description}")
            lines.append(f"   关键变量: {', '.join(structure.key_variables)}")
            lines.append(f"   动态规律: {structure.dynamics}")
            lines.append("")

        lines.append(f"[bold cyan]目标映射 ({len(domain.target_mappings)} 个)[/bold cyan]")
        lines.append("")

        for mapping in domain.target_mappings:
            structure = domain.get_structure(mapping.structure)
            structure_name = structure.name if structure else mapping.structure
            lines.append(f"[bold]{structure_name}[/bold] -> {mapping.target}")
            lines.append(f"   可观测指标: {mapping.observable}")
            lines.append("")

        result_content.update("\n".join(lines))

        # Show sections
        result_section.remove_class("hidden")
        action_buttons.remove_class("hidden")

    def _hide_results(self) -> None:
        """Hide the results section."""
        self.query_one("#result-section").add_class("hidden")
        self.query_one("#action-buttons").add_class("hidden")

    def _confirm_add(self) -> None:
        """Confirm and save the generated domain."""
        if not self._generated_domain:
            self._show_status("没有可添加的领域", is_error=True)
            return

        try:
            # Access yaml_manager through app
            self.app.yaml_manager.add_source_domain(self._generated_domain)
            self.app.yaml_manager.save()
            self.app.refresh_stats()

            self._show_status(
                f"已成功添加领域 [{self._generated_domain.name}]！",
                is_success=True,
            )

            # Disable confirm button to prevent double-add
            confirm_btn = self.query_one("#btn-confirm", Button)
            confirm_btn.disabled = True

            # After a short delay, return to main menu
            self.set_timer(1.5, self._return_to_main)

        except Exception as e:
            self._show_status(f"保存失败: {e!s}", is_error=True)

    def _return_to_main(self) -> None:
        """Return to main menu."""
        self.app.pop_screen()

    def action_cancel(self) -> None:
        """Cancel and return to main menu."""
        # Cancel any running task
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        self.app.pop_screen()
