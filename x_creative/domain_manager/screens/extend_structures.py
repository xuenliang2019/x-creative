"""Screen for extending structures of an existing domain."""

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Label,
    LoadingIndicator,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option

from x_creative.core.types import Domain, DomainStructure, TargetMapping
from x_creative.domain_manager.services import (
    DomainGeneratorService,
    DomainSearchService,
)


class ExtendStructuresScreen(Screen):
    """Screen for extending structures of an existing domain."""

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    CSS = """
    #extend-container {
        height: 100%;
        padding: 1 2;
    }

    #domain-select-section {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }

    #domain-select-section Label {
        margin-bottom: 1;
    }

    #domain-list {
        height: 10;
        width: 60;
        border: solid $primary;
    }

    #current-domain-info {
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $surface;
    }

    #current-structures-section {
        height: auto;
        max-height: 15;
        padding: 1;
        margin: 1 0;
        border: solid $secondary;
    }

    #current-structures-list {
        height: auto;
        padding: 0 1;
    }

    .structure-item {
        margin: 0;
        color: $text-muted;
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

    #suggestions-section {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        margin: 1 0;
    }

    #suggestions-container {
        height: auto;
        padding: 1;
    }

    .suggestion-item {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
    }

    .suggestion-item Checkbox {
        margin-bottom: 0;
    }

    .suggestion-details {
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
        self._selected_domain: Domain | None = None
        self._suggested_structures: list[DomainStructure] = []
        self._suggested_mappings: list[TargetMapping] = []
        self._selected_indices: set[int] = set()
        self._is_processing = False
        self._search_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            # Domain selection section
            Vertical(
                Label("请选择要扩展的领域:"),
                OptionList(id="domain-list"),
                id="domain-select-section",
            ),
            # Current domain info
            Vertical(
                Static("", id="domain-info-text"),
                id="current-domain-info",
                classes="hidden",
            ),
            # Current structures section
            ScrollableContainer(
                Static("[bold cyan]现有结构[/bold cyan]", id="section-title"),
                Vertical(id="current-structures-list"),
                id="current-structures-section",
                classes="hidden",
            ),
            # Button row
            Horizontal(
                Button("搜索建议", id="btn-search", variant="primary", disabled=True),
                Button("返回", id="btn-cancel", variant="default"),
                id="button-row",
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
            # Suggestions section
            ScrollableContainer(
                Static("[bold cyan]建议的新结构[/bold cyan]", id="suggestions-title"),
                Vertical(id="suggestions-container"),
                id="suggestions-section",
                classes="hidden",
            ),
            # Action buttons for suggestions
            Horizontal(
                Button("添加选中", id="btn-add", variant="success", disabled=True),
                Button("重新搜索", id="btn-re-search", variant="warning"),
                Button("取消", id="btn-cancel-add", variant="default"),
                id="action-buttons",
                classes="hidden",
            ),
            id="extend-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Populate domain list when mounted."""
        self._populate_domain_list()

    def _populate_domain_list(self) -> None:
        """Populate the domain list with available domains."""
        domain_list = self.query_one("#domain-list", OptionList)
        domains = self.app.yaml_manager.domains

        if not domains:
            self._show_status("没有可用的领域，请先添加领域", is_error=True)
            return

        for domain in domains:
            structure_count = len(domain.structures)
            label = f"{domain.name} ({structure_count} 个结构)"
            domain_list.add_option(Option(label, id=domain.id))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle domain selection."""
        if event.option_list.id != "domain-list":
            return

        domain_id = event.option.id
        if domain_id is None:
            return

        self._selected_domain = self.app.yaml_manager.get_domain(domain_id)
        if self._selected_domain:
            self._display_current_structures()
            # Enable search button
            search_btn = self.query_one("#btn-search", Button)
            search_btn.disabled = False
            self._show_status(f"已选择领域: {self._selected_domain.name}")

    def _display_current_structures(self) -> None:
        """Display current structures of the selected domain."""
        if not self._selected_domain:
            return

        # Update domain info
        info_text = self.query_one("#domain-info-text", Static)
        info_text.update(
            f"[bold]{self._selected_domain.name}[/bold]\n"
            f"[dim]{self._selected_domain.description[:100]}...[/dim]"
        )
        self.query_one("#current-domain-info").remove_class("hidden")

        # Display current structures
        structures_list = self.query_one("#current-structures-list", Vertical)
        structures_list.remove_children()

        for i, structure in enumerate(self._selected_domain.structures, 1):
            item = Static(
                f"  {i}. {structure.name} - {structure.description[:50]}...",
                classes="structure-item",
            )
            structures_list.mount(item)

        self.query_one("#current-structures-section").remove_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-search":
            self._start_search()
        elif button_id == "btn-cancel" or button_id == "btn-cancel-add":
            self.action_cancel()
        elif button_id == "btn-add":
            self._add_selected()
        elif button_id == "btn-re-search":
            self._reset_and_search()

    def _start_search(self) -> None:
        """Start searching for suggested structures."""
        if not self._selected_domain:
            self._show_status("请先选择一个领域", is_error=True)
            return

        if self._is_processing:
            return

        # Cancel any existing task
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()

        # Reset state
        self._suggested_structures = []
        self._suggested_mappings = []
        self._selected_indices.clear()
        self._hide_suggestions()
        self._show_loading(True)
        self._is_processing = True
        self._show_status(f"正在搜索 [{self._selected_domain.name}] 领域的新结构...")

        # Start async search
        self._search_task = asyncio.create_task(self._search_suggestions())

    async def _search_suggestions(self) -> None:
        """Search for additional structures."""
        if not self._selected_domain:
            return

        try:
            domain_name = self._selected_domain.name
            existing_structures = [s.name for s in self._selected_domain.structures]

            # Step 1: Search for additional structures
            self._show_status("正在搜索相关概念...")
            async with DomainSearchService() as search_service:
                search_results = await search_service.search_additional_structures(
                    domain_name, existing_structures
                )

            if not search_results:
                self._show_status(
                    "搜索未返回结果，将基于 LLM 知识推荐",
                    is_error=False,
                )

            # Step 2: Extract structures using LLM
            self._show_status("正在使用 LLM 提取新结构...")
            async with DomainGeneratorService(self.app.target_domain_name) as generator:
                all_structures = await generator.extract_structures(
                    domain_name, search_results
                )

                # Filter out existing structures
                new_structures = [
                    s for s in all_structures
                    if s.name not in existing_structures
                    and s.id not in [es.id for es in self._selected_domain.structures]
                ]

                if not new_structures:
                    self._show_loading(False)
                    self._is_processing = False
                    self._show_status(
                        "未发现新结构，该领域的结构可能已经比较完整",
                        is_error=False,
                    )
                    return

                # Generate mappings for new structures
                target_name = self.app.target_domain_name
                self._show_status(f"正在为 {len(new_structures)} 个新结构生成{target_name}映射...")
                self._suggested_structures = new_structures
                self._suggested_mappings = []

                for structure in new_structures:
                    async with DomainSearchService() as search_service:
                        mapping_results = await search_service.search_structure_applications(
                            structure.name, domain_name
                        )
                    mapping = await generator.generate_mapping(structure, mapping_results)
                    self._suggested_mappings.append(mapping)

            # Display suggestions
            self._show_loading(False)
            self._is_processing = False
            self._display_suggestions()
            self._show_status(
                f"发现 {len(self._suggested_structures)} 个新结构，请选择要添加的结构",
                is_success=True,
            )

        except asyncio.CancelledError:
            self._show_loading(False)
            self._is_processing = False
            self._show_status("已取消")
        except Exception as e:
            self._show_loading(False)
            self._is_processing = False
            self._show_status(f"搜索失败: {e!s}", is_error=True)

    def _display_suggestions(self) -> None:
        """Display the suggested structures with checkboxes."""
        container = self.query_one("#suggestions-container", Vertical)
        container.remove_children()

        for i, structure in enumerate(self._suggested_structures):
            mapping = self._suggested_mappings[i] if i < len(self._suggested_mappings) else None
            checkbox = Checkbox(
                f"{structure.name}",
                id=f"sug-{i}",
            )
            details_text = (
                f"[dim]描述: {structure.description[:80]}...\n"
                f"关键变量: {', '.join(structure.key_variables[:3])}"
            )
            if mapping:
                details_text += f"\n映射目标: {mapping.target}[/dim]"
            else:
                details_text += "[/dim]"

            details = Static(details_text, classes="suggestion-details")
            item = Vertical(checkbox, details, classes="suggestion-item")
            container.mount(item)

        # Show sections
        self.query_one("#suggestions-section").remove_class("hidden")
        self.query_one("#action-buttons").remove_class("hidden")

        # Disable add button initially
        add_btn = self.query_one("#btn-add", Button)
        add_btn.disabled = True

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox selection changes."""
        checkbox_id = event.checkbox.id
        if checkbox_id and checkbox_id.startswith("sug-"):
            try:
                index = int(checkbox_id.replace("sug-", ""))
                if event.value:
                    self._selected_indices.add(index)
                else:
                    self._selected_indices.discard(index)

                # Update add button state
                add_btn = self.query_one("#btn-add", Button)
                add_btn.disabled = len(self._selected_indices) == 0
            except ValueError:
                pass

    def _add_selected(self) -> None:
        """Add selected structures to the domain."""
        if not self._selected_domain:
            self._show_status("未选择领域", is_error=True)
            return

        if not self._selected_indices:
            self._show_status("请至少选择一个结构", is_error=True)
            return

        try:
            added_count = 0
            for idx in sorted(self._selected_indices):
                if idx < len(self._suggested_structures):
                    structure = self._suggested_structures[idx]
                    mapping = (
                        self._suggested_mappings[idx]
                        if idx < len(self._suggested_mappings)
                        else TargetMapping(
                            structure=structure.id,
                            target="待定义",
                            observable="待定义",
                        )
                    )

                    self.app.yaml_manager.add_structure(
                        self._selected_domain.id,
                        structure,
                        mapping,
                    )
                    added_count += 1

            # Save changes
            self.app.yaml_manager.save()
            self.app.refresh_stats()

            self._show_status(
                f"已成功添加 {added_count} 个结构到 [{self._selected_domain.name}]!",
                is_success=True,
            )

            # Disable add button
            add_btn = self.query_one("#btn-add", Button)
            add_btn.disabled = True

            # Return after delay
            self.set_timer(1.5, self._return_to_main)

        except Exception as e:
            self._show_status(f"添加失败: {e!s}", is_error=True)

    def _reset_and_search(self) -> None:
        """Reset state and start new search."""
        self._hide_suggestions()
        self._start_search()

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

    def _hide_suggestions(self) -> None:
        """Hide suggestion sections."""
        self.query_one("#suggestions-section").add_class("hidden")
        self.query_one("#action-buttons").add_class("hidden")

        # Clear suggestions container
        container = self.query_one("#suggestions-container", Vertical)
        container.remove_children()

    def _return_to_main(self) -> None:
        """Return to main menu."""
        self.app.pop_screen()

    def action_cancel(self) -> None:
        """Cancel and return to main menu."""
        # Cancel any running tasks
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()
        self.app.pop_screen()
