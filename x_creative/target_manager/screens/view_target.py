"""Screen for viewing target domain details."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane


class ViewTargetScreen(Screen):
    """Screen for viewing target domain details with tabbed content."""

    BINDINGS = [
        Binding("escape", "back", "返回"),
    ]

    CSS = """
    #view-container {
        height: 100%;
        padding: 1 2;
    }

    #view-title {
        text-style: bold;
        text-align: center;
        padding: 1;
    }

    #view-tabs {
        height: 1fr;
    }

    .tab-content {
        height: auto;
        padding: 1;
    }
    """

    def __init__(self) -> None:
        """Initialize the screen."""
        super().__init__()
        self.target_id: str = ""

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        with Vertical(id="view-container"):
            yield Static("", id="view-title")
            with ScrollableContainer():
                with TabbedContent(id="view-tabs"):
                    with TabPane("Overview", id="tab-overview"):
                        yield Static("", id="tab-overview-content")
                    with TabPane("Constraints", id="tab-constraints"):
                        yield Static("", id="tab-constraints-content")
                    with TabPane("Evaluation", id="tab-evaluation"):
                        yield Static("", id="tab-evaluation-content")
                    with TabPane("Anti-Patterns", id="tab-anti-patterns"):
                        yield Static("", id="tab-anti-patterns-content")
                    with TabPane("Terminology", id="tab-terminology"):
                        yield Static("", id="tab-terminology-content")
                    with TabPane("Stale Ideas", id="tab-stale-ideas"):
                        yield Static("", id="tab-stale-ideas-content")
                    with TabPane("Source Domains", id="tab-source-domains"):
                        yield Static("", id="tab-source-domains-content")
        yield Footer()

    def on_screen_resume(self) -> None:
        """Refresh data when screen is shown."""
        self._load_data()

    def on_mount(self) -> None:
        """Load data on mount."""
        self._load_data()

    def _load_data(self) -> None:
        """Load and display target domain data."""
        if not self.target_id:
            return

        try:
            plugin = self.app.yaml_manager.load_target_domain(self.target_id)
            stats = self.app.yaml_manager.get_stats(self.target_id)
        except FileNotFoundError:
            self.query_one("#view-title", Static).update(
                f"[red]Target Domain '{self.target_id}' 未找到[/red]"
            )
            return

        # Title
        self.query_one("#view-title", Static).update(
            f"[bold]{plugin.name}[/bold] ({plugin.id})\n{plugin.description}"
        )

        # Overview tab
        overview_lines = [
            f"[bold]ID:[/bold] {plugin.id}",
            f"[bold]Name:[/bold] {plugin.name}",
            f"[bold]Description:[/bold] {plugin.description}",
            "",
            "[bold cyan]统计[/bold cyan]",
            f"  Constraints: {stats['constraint_count']}",
            f"  Evaluation Criteria: {stats['evaluation_criteria_count']}",
            f"  Anti-Patterns: {stats['anti_pattern_count']}",
            f"  Terminology: {stats['terminology_count']}",
            f"  Stale Ideas: {stats['stale_idea_count']}",
            f"  Source Domains: {stats['domain_count']}",
            f"  Structures: {stats['structure_count']}",
        ]
        self.query_one("#tab-overview-content", Static).update("\n".join(overview_lines))

        # Constraints tab
        if plugin.constraints:
            constraint_lines = []
            for c in plugin.constraints:
                constraint_lines.append(
                    f"[bold]{c.name}[/bold] [{c.severity}]"
                )
                constraint_lines.append(f"  {c.description}")
                if c.check_prompt:
                    constraint_lines.append(f"  [dim]Check: {c.check_prompt}[/dim]")
                constraint_lines.append("")
            self.query_one("#tab-constraints-content", Static).update(
                "\n".join(constraint_lines)
            )
        else:
            self.query_one("#tab-constraints-content", Static).update("(无)")

        # Evaluation criteria tab
        if plugin.evaluation_criteria:
            self.query_one("#tab-evaluation-content", Static).update(
                "\n".join(f"- {c}" for c in plugin.evaluation_criteria)
            )
        else:
            self.query_one("#tab-evaluation-content", Static).update("(无)")

        # Anti-patterns tab
        if plugin.anti_patterns:
            self.query_one("#tab-anti-patterns-content", Static).update(
                "\n".join(f"- {a}" for a in plugin.anti_patterns)
            )
        else:
            self.query_one("#tab-anti-patterns-content", Static).update("(无)")

        # Terminology tab
        if plugin.terminology:
            term_lines = [f"[bold]{k}[/bold]: {v}" for k, v in plugin.terminology.items()]
            self.query_one("#tab-terminology-content", Static).update(
                "\n".join(term_lines)
            )
        else:
            self.query_one("#tab-terminology-content", Static).update("(无)")

        # Stale ideas tab
        if plugin.stale_ideas:
            self.query_one("#tab-stale-ideas-content", Static).update(
                "\n".join(f"- {s}" for s in plugin.stale_ideas)
            )
        else:
            self.query_one("#tab-stale-ideas-content", Static).update("(无)")

        # Source domains tab
        if plugin.source_domains:
            domain_lines = []
            for d in plugin.source_domains:
                structures = d.get("structures", [])
                mappings = d.get("target_mappings", [])
                domain_lines.append(
                    f"[bold]{d.get('name', d.get('id', '?'))}[/bold] "
                    f"({d.get('id', '?')})"
                )
                if d.get("name_en"):
                    domain_lines.append(f"  English: {d['name_en']}")
                domain_lines.append(f"  {d.get('description', '')}")
                domain_lines.append(
                    f"  Structures: {len(structures)} | "
                    f"Mappings: {len(mappings)}"
                )
                domain_lines.append("")
            self.query_one("#tab-source-domains-content", Static).update(
                "\n".join(domain_lines)
            )
        else:
            self.query_one("#tab-source-domains-content", Static).update("(无)")

    def action_back(self) -> None:
        """Return to main menu."""
        self.app.pop_screen()
