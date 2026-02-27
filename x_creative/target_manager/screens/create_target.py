"""Screen for creating a new target domain with multi-step wizard."""

import asyncio
from typing import Any

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
    ProgressBar,
    RadioButton,
    RadioSet,
    Static,
    TabbedContent,
    TabPane,
)

from x_creative.core.types import Domain, DomainStructure, TargetMapping
from x_creative.domain_manager.services.yaml_manager import (
    YAMLManager as SourceYAMLManager,
)
from x_creative.target_manager.services.target_generator import TargetGeneratorService
from x_creative.target_manager.services.target_yaml_manager import TargetYAMLManager


class CreateTargetScreen(Screen):
    """Multi-step wizard for creating a new target domain."""

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    CSS = """
    #create-container {
        height: 100%;
        padding: 1 2;
    }

    .step-section {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }

    .step-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #input-fields Label {
        margin-bottom: 1;
    }

    #input-fields Input {
        width: 60;
        margin-bottom: 1;
    }

    #step-nav {
        height: auto;
        width: auto;
        margin-top: 1;
        dock: bottom;
        padding: 1;
    }

    #step-nav Button {
        margin: 0 1;
    }

    #status-text {
        color: $text-muted;
        padding: 1 0;
    }

    #loading-section {
        height: auto;
        align: center middle;
        padding: 1;
    }

    #progress-section {
        height: auto;
        padding: 1;
    }

    #metadata-review {
        height: 1fr;
        padding: 1;
        border: solid $primary;
    }

    #base-target-selection {
        height: auto;
        max-height: 12;
        padding: 1;
    }

    #source-domain-selection {
        height: 1fr;
        padding: 1;
        border: solid $primary;
    }

    #preview-content {
        height: 1fr;
        padding: 1;
        border: solid $primary;
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

    #select-all-row {
        height: auto;
        padding: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        """Initialize the screen."""
        super().__init__()
        self._current_step = 1
        self._generation_task: asyncio.Task | None = None
        self._rewrite_task: asyncio.Task | None = None
        self._metadata: dict[str, Any] = {}
        self._base_target_id: str | None = None
        self._base_target_name: str = ""
        self._selected_source_domains: list[Domain] = []
        self._rewritten_domains: list[Domain] = []
        self._all_source_domains: list[Domain] = []

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        with Vertical(id="create-container"):
            # Step 1: Basic Info
            with Vertical(id="step-1", classes="step-section"):
                yield Static("[bold]步骤 1/7: 基础信息[/bold]", classes="step-title")
                with Vertical(id="input-fields"):
                    yield Label("Target ID (snake_case):")
                    yield Input(placeholder="例如：drug_discovery", id="input-id")
                    yield Label("显示名称:")
                    yield Input(placeholder="例如：药物发现", id="input-name")
                    yield Label("描述:")
                    yield Input(placeholder="简要描述该目标领域的核心目标", id="input-description")
            # Step 2: LLM Generate Metadata
            with Vertical(id="step-2", classes="step-section hidden"):
                yield Static("[bold]步骤 2/7: LLM 生成元数据[/bold]", classes="step-title")
                yield Static("", id="status-text")
                with Center(id="loading-section", classes="hidden"):
                    yield LoadingIndicator(id="loading")
                with Vertical(id="progress-section", classes="hidden"):
                    yield ProgressBar(total=5, id="gen-progress")
                    yield Static("", id="progress-text")
            # Step 3: Review Metadata
            with Vertical(id="step-3", classes="step-section hidden"):
                yield Static("[bold]步骤 3/7: 审阅元数据[/bold]", classes="step-title")
                with ScrollableContainer(id="metadata-review"):
                    with TabbedContent(id="metadata-tabs"):
                        with TabPane("Constraints", id="tab-constraints"):
                            yield Static("", id="review-constraints")
                        with TabPane("Evaluation", id="tab-evaluation"):
                            yield Static("", id="review-evaluation")
                        with TabPane("Anti-Patterns", id="tab-anti-patterns"):
                            yield Static("", id="review-anti-patterns")
                        with TabPane("Terminology", id="tab-terminology"):
                            yield Static("", id="review-terminology")
                        with TabPane("Stale Ideas", id="tab-stale-ideas"):
                            yield Static("", id="review-stale-ideas")
            # Step 4: Select Base Target
            with Vertical(id="step-4", classes="step-section hidden"):
                yield Static("[bold]步骤 4/7: 选择基准 Target Domain (可跳过)[/bold]", classes="step-title")
                yield Static(
                    "选择一个已有的 Target Domain 作为基准，复制其 source_domains。\n"
                    "如果跳过，将创建不含 source_domains 的空 target。",
                )
                yield RadioSet(id="base-target-radio")
            # Step 5: Filter Source Domains
            with Vertical(id="step-5", classes="step-section hidden"):
                yield Static("[bold]步骤 5/7: 筛选 Source Domains[/bold]", classes="step-title")
                with Horizontal(id="select-all-row"):
                    yield Button("全选", id="btn-select-all", variant="default")
                    yield Button("全不选", id="btn-deselect-all", variant="default")
                with ScrollableContainer(id="source-domain-selection"):
                    yield Vertical(id="source-domain-checkboxes")
            # Step 6: Rewrite Mappings
            with Vertical(id="step-6", classes="step-section hidden"):
                yield Static("[bold]步骤 6/7: 重写 Target Mappings[/bold]", classes="step-title")
                yield Static("", id="rewrite-status")
                with Vertical(id="rewrite-progress-section", classes="hidden"):
                    yield ProgressBar(total=100, id="rewrite-progress")
                    yield Static("", id="rewrite-progress-text")
            # Step 7: Preview + Save
            with Vertical(id="step-7", classes="step-section hidden"):
                yield Static("[bold]步骤 7/7: 预览 + 保存[/bold]", classes="step-title")
                with ScrollableContainer(id="preview-content"):
                    yield Static("", id="preview-text")
            # Navigation
            with Horizontal(id="step-nav"):
                yield Button("上一步", id="btn-prev", variant="default")
                yield Button("下一步", id="btn-next", variant="primary")
                yield Button("跳过步骤 4-6", id="btn-skip", variant="warning", classes="hidden")
                yield Button("保存", id="btn-save", variant="success", classes="hidden")
                yield Button("取消", id="btn-cancel", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the first input."""
        self.query_one("#input-id", Input).focus()
        self._show_step(1)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-next":
            self._next_step()
        elif button_id == "btn-prev":
            self._prev_step()
        elif button_id == "btn-skip":
            self._skip_to_save()
        elif button_id == "btn-save":
            self._save_target()
        elif button_id == "btn-cancel":
            self.action_cancel()
        elif button_id == "btn-select-all":
            self._toggle_all_checkboxes(True)
        elif button_id == "btn-deselect-all":
            self._toggle_all_checkboxes(False)

    def _show_step(self, step: int) -> None:
        """Show the specified step and hide others."""
        for i in range(1, 8):
            section = self.query_one(f"#step-{i}")
            if i == step:
                section.remove_class("hidden")
            else:
                section.add_class("hidden")

        self._current_step = step

        # Update navigation visibility
        prev_btn = self.query_one("#btn-prev", Button)
        next_btn = self.query_one("#btn-next", Button)
        skip_btn = self.query_one("#btn-skip", Button)
        save_btn = self.query_one("#btn-save", Button)

        prev_btn.disabled = step == 1
        next_btn.remove_class("hidden")
        save_btn.add_class("hidden")

        if step == 4:
            skip_btn.remove_class("hidden")
        else:
            skip_btn.add_class("hidden")

        if step == 7:
            next_btn.add_class("hidden")
            save_btn.remove_class("hidden")

    def _next_step(self) -> None:
        """Advance to the next step."""
        if self._current_step == 1:
            if not self._validate_basic_info():
                return
            self._show_step(2)
            self._start_metadata_generation()
        elif self._current_step == 2:
            if not self._metadata:
                self._show_status("请等待元数据生成完成", is_error=True)
                return
            self._populate_review_tabs()
            self._show_step(3)
        elif self._current_step == 3:
            self._populate_base_target_selection()
            self._show_step(4)
        elif self._current_step == 4:
            if self._base_target_id:
                self._populate_source_domain_selection()
                self._show_step(5)
            else:
                self._skip_to_save()
        elif self._current_step == 5:
            self._collect_selected_domains()
            if self._selected_source_domains:
                self._show_step(6)
                self._start_rewrite()
            else:
                self._prepare_preview()
                self._show_step(7)
        elif self._current_step == 6:
            if not self._rewritten_domains:
                self._show_status("请等待重写完成", is_error=True)
                return
            self._prepare_preview()
            self._show_step(7)

    def _prev_step(self) -> None:
        """Go back to the previous step."""
        if self._current_step > 1:
            # Cancel running tasks if going back
            if self._current_step == 2 and self._generation_task and not self._generation_task.done():
                self._generation_task.cancel()
            if self._current_step == 6 and self._rewrite_task and not self._rewrite_task.done():
                self._rewrite_task.cancel()
            self._show_step(self._current_step - 1)

    def _skip_to_save(self) -> None:
        """Skip steps 4-6 and go directly to save."""
        self._base_target_id = None
        self._selected_source_domains = []
        self._rewritten_domains = []
        self._prepare_preview()
        self._show_step(7)

    def _validate_basic_info(self) -> bool:
        """Validate step 1 inputs."""
        target_id = self.query_one("#input-id", Input).value.strip()
        name = self.query_one("#input-name", Input).value.strip()
        description = self.query_one("#input-description", Input).value.strip()

        if not target_id:
            self.notify("请输入 Target ID", severity="error")
            return False
        if not name:
            self.notify("请输入显示名称", severity="error")
            return False
        if not description:
            self.notify("请输入描述", severity="error")
            return False

        is_valid, msg = self.app.yaml_manager.validate_target_id(target_id)
        if not is_valid:
            self.notify(msg, severity="error")
            return False

        return True

    def _show_status(self, message: str, is_error: bool = False, is_success: bool = False) -> None:
        """Update status text."""
        status = self.query_one("#status-text", Static)
        if is_error:
            status.update(f"[red]{message}[/red]")
        elif is_success:
            status.update(f"[green]{message}[/green]")
        else:
            status.update(f"[cyan]{message}[/cyan]")

    def _start_metadata_generation(self) -> None:
        """Start LLM metadata generation."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

        self._metadata = {}
        loading = self.query_one("#loading-section")
        loading.remove_class("hidden")
        progress_section = self.query_one("#progress-section")
        progress_section.remove_class("hidden")

        self._show_status("正在使用 LLM 并行生成元数据...")

        name = self.query_one("#input-name", Input).value.strip()
        description = self.query_one("#input-description", Input).value.strip()

        self._generation_task = asyncio.create_task(
            self._generate_metadata(name, description)
        )

    async def _generate_metadata(self, name: str, description: str) -> None:
        """Generate all metadata sections via LLM."""
        try:
            async with TargetGeneratorService() as generator:
                progress_bar = self.query_one("#gen-progress", ProgressBar)
                progress_text = self.query_one("#progress-text", Static)

                # Wrap each coroutine into a named task so we can track progress
                task_names = [
                    "constraints", "evaluation_criteria",
                    "anti_patterns", "terminology", "stale_ideas",
                ]
                coros = [
                    generator.generate_constraints(name, description),
                    generator.generate_evaluation_criteria(name, description),
                    generator.generate_anti_patterns(name, description),
                    generator.generate_terminology(name, description),
                    generator.generate_stale_ideas(name, description),
                ]

                # Create asyncio tasks
                async_tasks = [asyncio.create_task(c) for c in coros]

                # Wait for each to complete and update progress
                all_results: list = [None] * 5
                completed = 0
                for task_future in asyncio.as_completed(async_tasks):
                    await task_future
                    completed += 1
                    progress_bar.update(progress=completed)
                    progress_text.update(f"已完成 {completed}/5 个部分")

                # Gather results in original order
                for i, t in enumerate(async_tasks):
                    all_results[i] = t.result()

                constraints_dicts = [
                    {
                        "name": c.name,
                        "description": c.description,
                        "severity": c.severity,
                        "check_prompt": c.check_prompt,
                    }
                    for c in all_results[0]
                ]

                self._metadata = {
                    "constraints": constraints_dicts,
                    "evaluation_criteria": all_results[1],
                    "anti_patterns": all_results[2],
                    "terminology": all_results[3],
                    "stale_ideas": all_results[4],
                }

            self.query_one("#loading-section").add_class("hidden")
            self._show_status("元数据生成完成！请点击 '下一步' 审阅。", is_success=True)

        except asyncio.CancelledError:
            self.query_one("#loading-section").add_class("hidden")
            self._show_status("已取消")
        except Exception as e:
            self.query_one("#loading-section").add_class("hidden")
            self._show_status(f"生成失败: {e!s}", is_error=True)

    def _populate_review_tabs(self) -> None:
        """Populate the review tabs with generated metadata."""
        # Constraints
        constraints = self._metadata.get("constraints", [])
        lines = []
        for c in constraints:
            lines.append(f"[bold]{c['name']}[/bold] [{c['severity']}]")
            lines.append(f"  {c['description']}")
            if c.get("check_prompt"):
                lines.append(f"  [dim]Check: {c['check_prompt']}[/dim]")
            lines.append("")
        self.query_one("#review-constraints", Static).update(
            "\n".join(lines) or "(无)"
        )

        # Evaluation criteria
        criteria = self._metadata.get("evaluation_criteria", [])
        self.query_one("#review-evaluation", Static).update(
            "\n".join(f"- {c}" for c in criteria) or "(无)"
        )

        # Anti-patterns
        anti_patterns = self._metadata.get("anti_patterns", [])
        self.query_one("#review-anti-patterns", Static).update(
            "\n".join(f"- {a}" for a in anti_patterns) or "(无)"
        )

        # Terminology
        terminology = self._metadata.get("terminology", {})
        term_lines = [f"[bold]{k}[/bold]: {v}" for k, v in terminology.items()]
        self.query_one("#review-terminology", Static).update(
            "\n".join(term_lines) or "(无)"
        )

        # Stale ideas
        stale_ideas = self._metadata.get("stale_ideas", [])
        self.query_one("#review-stale-ideas", Static).update(
            "\n".join(f"- {s}" for s in stale_ideas) or "(无)"
        )

    def _populate_base_target_selection(self) -> None:
        """Populate base target radio buttons."""
        radio_set = self.query_one("#base-target-radio", RadioSet)

        # Clear existing children
        radio_set.remove_children()

        targets = self.app.yaml_manager.list_target_domains()
        if targets:
            radio_set.mount(RadioButton("(不选择基准 - 跳过)", id="base-none"))
            for target in targets:
                label = f"{target['name']} ({target['id']}) - {target['domain_count']} 源域"
                radio_set.mount(RadioButton(label, id=f"base-{target['id']}"))
        else:
            radio_set.mount(RadioButton("(无可用的 Target Domain)", id="base-none"))

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle base target selection."""
        if event.radio_set.id == "base-target-radio":
            btn_id = event.pressed.id or ""
            if btn_id.startswith("base-") and btn_id != "base-none":
                self._base_target_id = btn_id.removeprefix("base-")
                # Load the base target name
                targets = self.app.yaml_manager.list_target_domains()
                for t in targets:
                    if t["id"] == self._base_target_id:
                        self._base_target_name = t["name"]
                        break
            else:
                self._base_target_id = None
                self._base_target_name = ""

    def _populate_source_domain_selection(self) -> None:
        """Populate source domain checkboxes from base target."""
        container = self.query_one("#source-domain-checkboxes")
        container.remove_children()
        self._all_source_domains = []

        if not self._base_target_id:
            return

        try:
            # Load source domains from base target
            mgr = SourceYAMLManager(self._base_target_id)
            self._all_source_domains = mgr.load_domains()

            for domain in self._all_source_domains:
                label = (
                    f"{domain.name} ({domain.id}) - "
                    f"{len(domain.structures)} 结构"
                )
                container.mount(
                    Checkbox(label, value=True, id=f"sd-{domain.id}")
                )
        except Exception as e:
            container.mount(Static(f"[red]加载失败: {e!s}[/red]"))

    def _toggle_all_checkboxes(self, value: bool) -> None:
        """Toggle all source domain checkboxes."""
        for checkbox in self.query("Checkbox"):
            if isinstance(checkbox, Checkbox) and checkbox.id and checkbox.id.startswith("sd-"):
                checkbox.value = value

    def _collect_selected_domains(self) -> None:
        """Collect selected source domains."""
        self._selected_source_domains = []
        for checkbox in self.query("Checkbox"):
            if (
                isinstance(checkbox, Checkbox)
                and checkbox.id
                and checkbox.id.startswith("sd-")
                and checkbox.value
            ):
                domain_id = checkbox.id.removeprefix("sd-")
                for domain in self._all_source_domains:
                    if domain.id == domain_id:
                        self._selected_source_domains.append(domain)
                        break

    def _start_rewrite(self) -> None:
        """Start target mapping rewrite."""
        if self._rewrite_task and not self._rewrite_task.done():
            self._rewrite_task.cancel()

        self._rewritten_domains = []
        progress_section = self.query_one("#rewrite-progress-section")
        progress_section.remove_class("hidden")

        rewrite_status = self.query_one("#rewrite-status", Static)
        rewrite_status.update("[cyan]正在使用 LLM 重写 target mappings...[/cyan]")

        progress_bar = self.query_one("#rewrite-progress", ProgressBar)
        progress_bar.update(total=len(self._selected_source_domains), progress=0)

        self._rewrite_task = asyncio.create_task(self._rewrite_mappings())

    async def _rewrite_mappings(self) -> None:
        """Rewrite target mappings for selected source domains."""
        try:
            name = self.query_one("#input-name", Input).value.strip()
            description = self.query_one("#input-description", Input).value.strip()
            progress_bar = self.query_one("#rewrite-progress", ProgressBar)
            progress_text = self.query_one("#rewrite-progress-text", Static)

            def on_progress(current: int, total: int) -> None:
                progress_bar.update(progress=current)
                progress_text.update(f"已完成 {current}/{total} 个源域")

            async with TargetGeneratorService() as generator:
                self._rewritten_domains = await generator.batch_rewrite_source_domains(
                    self._selected_source_domains,
                    new_target_name=name,
                    new_target_description=description,
                    base_target_name=self._base_target_name,
                    progress_callback=on_progress,
                )

            rewrite_status = self.query_one("#rewrite-status", Static)
            rewrite_status.update(
                f"[green]重写完成！共 {len(self._rewritten_domains)} 个源域。"
                f"请点击 '下一步' 预览。[/green]"
            )

        except asyncio.CancelledError:
            self.query_one("#rewrite-status", Static).update("已取消")
        except Exception as e:
            self.query_one("#rewrite-status", Static).update(
                f"[red]重写失败: {e!s}[/red]"
            )

    def _prepare_preview(self) -> None:
        """Prepare the preview content for step 7."""
        target_id = self.query_one("#input-id", Input).value.strip()
        name = self.query_one("#input-name", Input).value.strip()
        description = self.query_one("#input-description", Input).value.strip()

        lines = [
            f"[bold cyan]Target Domain 预览[/bold cyan]\n",
            f"[bold]ID:[/bold] {target_id}",
            f"[bold]Name:[/bold] {name}",
            f"[bold]Description:[/bold] {description}",
            "",
        ]

        # Constraints
        constraints = self._metadata.get("constraints", [])
        lines.append(f"[bold cyan]Constraints ({len(constraints)})[/bold cyan]")
        for c in constraints:
            lines.append(f"  - {c['name']} [{c['severity']}]: {c['description']}")
        lines.append("")

        # Evaluation criteria
        criteria = self._metadata.get("evaluation_criteria", [])
        lines.append(f"[bold cyan]Evaluation Criteria ({len(criteria)})[/bold cyan]")
        for c in criteria:
            lines.append(f"  - {c}")
        lines.append("")

        # Anti-patterns
        anti_patterns = self._metadata.get("anti_patterns", [])
        lines.append(f"[bold cyan]Anti-Patterns ({len(anti_patterns)})[/bold cyan]")
        for a in anti_patterns:
            lines.append(f"  - {a}")
        lines.append("")

        # Terminology
        terminology = self._metadata.get("terminology", {})
        lines.append(f"[bold cyan]Terminology ({len(terminology)})[/bold cyan]")
        for k, v in terminology.items():
            lines.append(f"  - {k}: {v}")
        lines.append("")

        # Stale ideas
        stale_ideas = self._metadata.get("stale_ideas", [])
        lines.append(f"[bold cyan]Stale Ideas ({len(stale_ideas)})[/bold cyan]")
        for s in stale_ideas:
            lines.append(f"  - {s}")
        lines.append("")

        # Source domains
        domains = self._rewritten_domains or []
        lines.append(f"[bold cyan]Source Domains ({len(domains)})[/bold cyan]")
        for d in domains:
            lines.append(f"  - {d.name} ({d.id}): {len(d.structures)} 结构, {len(d.target_mappings)} 映射")
        lines.append("")

        self.query_one("#preview-text", Static).update("\n".join(lines))

    def _save_target(self) -> None:
        """Save the new target domain."""
        target_id = self.query_one("#input-id", Input).value.strip()
        name = self.query_one("#input-name", Input).value.strip()
        description = self.query_one("#input-description", Input).value.strip()

        # Prepare source_domains as dicts
        source_domains_data = None
        if self._rewritten_domains:
            source_domains_data = [
                {
                    "id": d.id,
                    "name": d.name,
                    "name_en": d.name_en,
                    "description": d.description,
                    "structures": [
                        {
                            "id": s.id,
                            "name": s.name,
                            "description": s.description,
                            "key_variables": list(s.key_variables),
                            "dynamics": s.dynamics,
                        }
                        for s in d.structures
                    ],
                    "target_mappings": [
                        {
                            "structure": m.structure,
                            "target": m.target,
                            "observable": m.observable,
                        }
                        for m in d.target_mappings
                    ],
                }
                for d in self._rewritten_domains
            ]

        try:
            path = self.app.yaml_manager.create_target_domain(
                target_id=target_id,
                name=name,
                description=description,
                constraints=self._metadata.get("constraints"),
                evaluation_criteria=self._metadata.get("evaluation_criteria"),
                anti_patterns=self._metadata.get("anti_patterns"),
                terminology=self._metadata.get("terminology"),
                stale_ideas=self._metadata.get("stale_ideas"),
                source_domains=source_domains_data,
            )

            self.notify(
                f"Target Domain '{name}' 已保存到 {path}",
                severity="information",
            )

            # Return to main menu after short delay
            self.set_timer(1.5, self._return_to_main)

        except Exception as e:
            self.notify(f"保存失败: {e!s}", severity="error")

    def _return_to_main(self) -> None:
        """Return to main menu with refreshed data."""
        self.app.pop_screen()

    def action_cancel(self) -> None:
        """Cancel and return to main menu."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        if self._rewrite_task and not self._rewrite_task.done():
            self._rewrite_task.cancel()
        self.app.pop_screen()
