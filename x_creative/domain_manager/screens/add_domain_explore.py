"""Screen for auto-exploring and adding new domains."""

import asyncio
import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
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
    DomainRecommendation,
    DomainSearchService,
)


class AddDomainExploreScreen(Screen):
    """Screen for exploring and adding domains based on research goals."""

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    CSS = """
    #explore-container {
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

    #goal-input {
        width: 80;
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

    #recommendations-section {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        margin: 1 0;
    }

    #recommendations-container {
        height: auto;
        padding: 1;
    }

    .recommendation-item {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
    }

    .recommendation-item Checkbox {
        margin-bottom: 0;
    }

    .recommendation-details {
        margin-left: 3;
        color: $text-muted;
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

    #generation-section {
        height: 1fr;
        padding: 1;
        border: solid $success;
        margin: 1 0;
    }

    #generation-content {
        height: auto;
        padding: 1;
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

    #section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self) -> None:
        """Initialize the screen."""
        super().__init__()
        self._recommendations: list[DomainRecommendation] = []
        self._selected_indices: set[int] = set()
        self._generated_domains: list[Domain] = []
        self._generated_extensions: list[
            tuple[str, str, list[DomainStructure], list[TargetMapping]]
        ] = []  # (domain_id, domain_name, structures, mappings)
        self._is_processing = False
        self._exploration_task: asyncio.Task | None = None
        self._generation_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            # Input section
            Vertical(
                Label("请输入您的研究目标或问题（系统将搜索并推荐相关领域）:"),
                Input(
                    placeholder="例如：寻找提升开源项目用户增长的新视角、发现社区活跃度的领先指标...",
                    id="goal-input",
                ),
                Horizontal(
                    Button("探索领域", id="btn-explore", variant="primary"),
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
            # Recommendations section
            ScrollableContainer(
                Static("[bold cyan]推荐领域[/bold cyan]", id="section-title"),
                Vertical(id="recommendations-container"),
                id="recommendations-section",
                classes="hidden",
            ),
            # Action buttons for recommendations
            Horizontal(
                Button("生成选中领域", id="btn-generate", variant="success"),
                Button("重新探索", id="btn-re-explore", variant="warning"),
                Button("取消", id="btn-cancel-recommend", variant="default"),
                id="action-buttons",
                classes="hidden",
            ),
            # Generation results section
            ScrollableContainer(
                Static("", id="generation-content"),
                id="generation-section",
                classes="hidden",
            ),
            id="explore-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self.query_one("#goal-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-explore":
            self._start_exploration()
        elif button_id == "btn-cancel" or button_id == "btn-cancel-recommend":
            self.action_cancel()
        elif button_id == "btn-generate":
            self._start_generation()
        elif button_id == "btn-re-explore" or button_id == "btn-re-explore-save":
            self._reset_and_explore()
        elif button_id == "btn-save":
            self._save_domains()
        elif button_id == "btn-cancel-save":
            self.action_cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input."""
        if event.input.id == "goal-input":
            self._start_exploration()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox selection changes."""
        checkbox_id = event.checkbox.id
        if checkbox_id and checkbox_id.startswith("rec-"):
            try:
                index = int(checkbox_id.replace("rec-", ""))
                if event.value:
                    self._selected_indices.add(index)
                else:
                    self._selected_indices.discard(index)

                # Update generate button state
                generate_btn = self.query_one("#btn-generate", Button)
                generate_btn.disabled = len(self._selected_indices) == 0
            except ValueError:
                pass

    def _start_exploration(self) -> None:
        """Start the domain exploration process."""
        goal_input = self.query_one("#goal-input", Input)
        research_goal = goal_input.value.strip()

        if not research_goal:
            self._show_status("请输入研究目标", is_error=True)
            return

        if self._is_processing:
            return

        # Cancel any existing task
        if self._exploration_task and not self._exploration_task.done():
            self._exploration_task.cancel()

        # Reset state
        self._recommendations = []
        self._selected_indices.clear()
        self._generated_domains = []
        self._generated_extensions = []
        self._hide_all_results()
        self._show_loading(True)
        self._is_processing = True
        self._show_status(f"正在搜索与 [{research_goal}] 相关的领域...")

        # Start async exploration
        self._exploration_task = asyncio.create_task(
            self._explore_domains(research_goal)
        )

    async def _explore_domains(self, research_goal: str) -> None:
        """Explore domains based on research goal."""
        try:
            # Step 1: Search for related domains
            self._show_status("正在搜索相关领域...")
            async with DomainSearchService() as search_service:
                search_results = await search_service.search_related_domains(
                    research_goal
                )

            if not search_results:
                self._show_status(
                    "搜索未返回结果，将基于 LLM 知识推荐领域",
                    is_error=False,
                )

            # Step 2: Get LLM recommendations
            self._show_status("正在使用 LLM 分析并推荐领域...")
            async with DomainGeneratorService(self.app.target_domain_name) as generator:
                self._recommendations = await generator.recommend_domains(
                    research_goal, search_results
                )

            if not self._recommendations:
                self._show_loading(False)
                self._is_processing = False
                self._show_status("未能获取推荐领域，请尝试其他研究目标", is_error=True)
                return

            # Step 3: Check similarity with existing domains
            self._show_status("正在检查与现有领域的重合度...")
            existing_domains = [
                (d.id, d.name, d.name_en or "")
                for d in self.app.yaml_manager.domains
            ]

            checked_recommendations = []
            for rec in self._recommendations:
                checked_rec = await generator.check_domain_similarity(
                    rec, existing_domains
                )
                checked_recommendations.append(checked_rec)

            self._recommendations = checked_recommendations

            # Display recommendations
            self._show_loading(False)
            self._is_processing = False
            self._display_recommendations()

            # Count new vs extensions
            new_count = sum(1 for r in self._recommendations if not r.is_extension)
            ext_count = sum(1 for r in self._recommendations if r.is_extension)
            status_parts = []
            if new_count > 0:
                status_parts.append(f"{new_count} 个新领域")
            if ext_count > 0:
                status_parts.append(f"{ext_count} 个可扩展现有领域")
            self._show_status(
                f"发现 {' + '.join(status_parts)}，请选择要处理的领域",
                is_success=True,
            )

        except asyncio.CancelledError:
            self._show_loading(False)
            self._is_processing = False
            self._show_status("已取消")
        except Exception as e:
            self._show_loading(False)
            self._is_processing = False
            self._show_status(f"探索失败: {e!s}", is_error=True)

    def _display_recommendations(self) -> None:
        """Display the recommendation checkboxes."""
        container = self.query_one("#recommendations-container", Vertical)

        # Clear existing items
        container.remove_children()

        # Add recommendation items
        for i, rec in enumerate(self._recommendations):
            # Different display for extensions vs new domains
            if rec.is_extension:
                label = f"[yellow]⊕[/yellow] {rec.domain_name} → 扩展 [{rec.existing_domain_name}]"
                type_indicator = "[yellow]（扩展现有领域）[/yellow]"
            else:
                label = f"[green]✚[/green] {rec.domain_name} ({rec.domain_name_en})"
                type_indicator = "[green]（新领域）[/green]"

            checkbox = Checkbox(label, id=f"rec-{i}")
            details = Static(
                f"{type_indicator}\n"
                f"[dim]关联原因: {rec.relevance_reason}\n"
                f"可能的结构: {', '.join(rec.potential_structures[:3])}[/dim]",
                classes="recommendation-details",
            )
            item = Vertical(checkbox, details, classes="recommendation-item")
            container.mount(item)

        # Show sections
        self.query_one("#recommendations-section").remove_class("hidden")
        self.query_one("#action-buttons").remove_class("hidden")

        # Disable generate button initially
        generate_btn = self.query_one("#btn-generate", Button)
        generate_btn.disabled = True

    def _reset_and_explore(self) -> None:
        """Reset state and start new exploration."""
        self._hide_all_results()
        self._start_exploration()

    def _start_generation(self) -> None:
        """Start generating selected domains."""
        if not self._selected_indices:
            self._show_status("请至少选择一个领域", is_error=True)
            return

        if self._is_processing:
            return

        # Cancel any existing task
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

        self._generated_domains = []
        self._generated_extensions = []
        self._show_loading(True)
        self._is_processing = True

        selected_count = len(self._selected_indices)
        self._show_status(f"正在生成 {selected_count} 个领域的详细内容...")

        # Start async generation
        self._generation_task = asyncio.create_task(
            self._generate_selected_domains()
        )

    async def _generate_selected_domains(self) -> None:
        """Generate full domain content for selected recommendations."""
        try:
            selected_recs = [
                self._recommendations[i]
                for i in sorted(self._selected_indices)
                if i < len(self._recommendations)
            ]

            total = len(selected_recs)
            for idx, rec in enumerate(selected_recs, 1):
                if rec.is_extension:
                    self._show_status(
                        f"正在扩展 [{rec.existing_domain_name}] ({idx}/{total})..."
                    )
                else:
                    self._show_status(f"正在生成 [{rec.domain_name}] ({idx}/{total})...")

                # Search for domain concepts
                async with DomainSearchService() as search_service:
                    search_results = await search_service.search_domain_concepts(
                        rec.domain_name
                    )

                # Extract structures
                async with DomainGeneratorService(self.app.target_domain_name) as generator:
                    structures = await generator.extract_structures(
                        rec.domain_name, search_results
                    )

                    if not structures:
                        self._show_status(
                            f"[{rec.domain_name}] 未能提取结构，跳过...",
                            is_error=False,
                        )
                        continue

                    # For extensions, filter out duplicate structures
                    if rec.is_extension and rec.existing_domain_id:
                        existing_domain = self.app.yaml_manager.get_domain(
                            rec.existing_domain_id
                        )
                        if existing_domain:
                            self._show_status(
                                f"正在过滤与 [{rec.existing_domain_name}] 重复的结构..."
                            )
                            structures = await generator.filter_duplicate_structures(
                                structures, existing_domain.structures
                            )
                            if not structures:
                                self._show_status(
                                    f"[{rec.domain_name}] 的所有结构都已存在于 [{rec.existing_domain_name}]，跳过...",
                                    is_error=False,
                                )
                                continue

                    # Generate mappings
                    mappings: list[TargetMapping] = []
                    for structure in structures:
                        if search_results:
                            mapping_results = await search_service.search_structure_applications(
                                structure.name, rec.domain_name
                            )
                        else:
                            mapping_results = []
                        mapping = await generator.generate_mapping(
                            structure, mapping_results
                        )
                        mappings.append(mapping)

                # Handle extension vs new domain
                if rec.is_extension and rec.existing_domain_id:
                    self._generated_extensions.append((
                        rec.existing_domain_id,
                        rec.existing_domain_name or rec.domain_name,
                        structures,
                        mappings,
                    ))
                else:
                    # Create new domain
                    domain_id = self._generate_domain_id(rec.domain_name)
                    domain = Domain(
                        id=domain_id,
                        name=rec.domain_name,
                        name_en=rec.domain_name_en or None,
                        description=f"{rec.domain_name}领域的核心概念和可迁移结构。{rec.relevance_reason}",
                        structures=structures,
                        target_mappings=mappings,
                    )
                    self._generated_domains.append(domain)

            if not self._generated_domains and not self._generated_extensions:
                self._show_loading(False)
                self._is_processing = False
                self._show_status("未能生成任何内容，请重新选择", is_error=True)
                return

            # Show generation results
            self._show_loading(False)
            self._is_processing = False
            self._display_generation_results()

            # Build status message
            status_parts = []
            if self._generated_domains:
                status_parts.append(f"{len(self._generated_domains)} 个新领域")
            if self._generated_extensions:
                status_parts.append(f"{len(self._generated_extensions)} 个领域扩展")
            self._show_status(
                f"已生成 {' + '.join(status_parts)}，请确认保存",
                is_success=True,
            )

        except asyncio.CancelledError:
            self._show_loading(False)
            self._is_processing = False
            self._show_status("已取消")
        except Exception as e:
            self._show_loading(False)
            self._is_processing = False
            self._show_status(f"生成失败: {e!s}", is_error=True)

    def _display_generation_results(self) -> None:
        """Display the generated domains for confirmation."""
        generation_section = self.query_one("#generation-section")
        generation_content = self.query_one("#generation-content", Static)

        lines = ["[bold cyan]生成结果预览[/bold cyan]\n"]

        # Display new domains
        if self._generated_domains:
            lines.append("[bold green]━━━ 新领域 ━━━[/bold green]\n")
            for domain in self._generated_domains:
                lines.append(f"[bold]✚ 领域: {domain.name}[/bold] ({domain.id})")
                if domain.name_en:
                    lines.append(f"   英文名: {domain.name_en}")
                lines.append(f"   描述: {domain.description[:100]}...")
                lines.append(f"   结构数: {len(domain.structures)}")
                lines.append(f"   映射数: {len(domain.target_mappings)}")
                lines.append("")

                for i, structure in enumerate(domain.structures, 1):
                    lines.append(f"   [{i}] {structure.name}")
                    lines.append(f"       {structure.description[:60]}...")
                lines.append("")
                lines.append("-" * 50)
                lines.append("")

        # Display extensions
        if self._generated_extensions:
            lines.append("[bold yellow]━━━ 扩展现有领域 ━━━[/bold yellow]\n")
            for domain_id, domain_name, structures, mappings in self._generated_extensions:
                lines.append(f"[bold]⊕ 扩展: {domain_name}[/bold] ({domain_id})")
                lines.append(f"   新增结构数: {len(structures)}")
                lines.append(f"   新增映射数: {len(mappings)}")
                lines.append("")

                for i, structure in enumerate(structures, 1):
                    lines.append(f"   [+{i}] {structure.name}")
                    lines.append(f"        {structure.description[:60]}...")
                lines.append("")
                lines.append("-" * 50)
                lines.append("")

        generation_content.update("\n".join(lines))

        # Hide recommendations, show generation results
        self.query_one("#recommendations-section").add_class("hidden")
        self.query_one("#action-buttons").add_class("hidden")
        generation_section.remove_class("hidden")

        # Update action buttons for save (use different IDs to avoid conflicts)
        action_buttons = self.query_one("#action-buttons", Horizontal)
        action_buttons.remove_children()
        action_buttons.mount(
            Button("保存所有领域", id="btn-save", variant="success"),
            Button("重新探索", id="btn-re-explore-save", variant="warning"),
            Button("取消", id="btn-cancel-save", variant="default"),
        )
        action_buttons.remove_class("hidden")

    def _save_domains(self) -> None:
        """Save all generated domains and extensions."""
        if not self._generated_domains and not self._generated_extensions:
            self._show_status("没有可保存的内容", is_error=True)
            return

        try:
            new_domain_count = 0
            extension_count = 0
            structure_count = 0

            # Save new domains
            for domain in self._generated_domains:
                self.app.yaml_manager.add_source_domain(domain)
                new_domain_count += 1

            # Save extensions (add structures to existing domains)
            for domain_id, _, structures, mappings in self._generated_extensions:
                for structure, mapping in zip(structures, mappings):
                    self.app.yaml_manager.add_structure(domain_id, structure, mapping)
                    structure_count += 1
                extension_count += 1

            self.app.yaml_manager.save()
            self.app.refresh_stats()

            # Build status message
            status_parts = []
            if new_domain_count > 0:
                status_parts.append(f"{new_domain_count} 个新领域")
            if extension_count > 0:
                status_parts.append(f"{extension_count} 个领域扩展（共 {structure_count} 个新结构）")

            self._show_status(
                f"已成功保存 {', '.join(status_parts)}！",
                is_success=True,
            )

            # Disable save button
            save_btn = self.query_one("#btn-save", Button)
            save_btn.disabled = True

            # Return after delay
            self.set_timer(1.5, self._return_to_main)

        except Exception as e:
            self._show_status(f"保存失败: {e!s}", is_error=True)

    def _generate_domain_id(self, domain_name: str) -> str:
        """Generate a domain ID from the name."""
        clean = re.sub(r"[^\w\s]", "", domain_name)
        words = clean.lower().split()
        return "_".join(words) if words else "unnamed_domain"

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

    def _hide_all_results(self) -> None:
        """Hide all result sections."""
        self.query_one("#recommendations-section").add_class("hidden")
        self.query_one("#action-buttons").add_class("hidden")
        self.query_one("#generation-section").add_class("hidden")

        # Clear recommendations container
        container = self.query_one("#recommendations-container", Vertical)
        container.remove_children()

    def _return_to_main(self) -> None:
        """Return to main menu."""
        self.app.pop_screen()

    def action_cancel(self) -> None:
        """Cancel and return to main menu."""
        # Cancel any running tasks
        if self._exploration_task and not self._exploration_task.done():
            self._exploration_task.cancel()
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        self.app.pop_screen()
