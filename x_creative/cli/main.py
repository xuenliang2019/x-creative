"""Main CLI entry point for x-creative."""

import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from x_creative import __version__
from x_creative.cli import answer as answer_cli
from x_creative.cli import hkg as hkg_cli
from x_creative.cli import run as run_cli
from x_creative.cli import session as session_cli
from x_creative.cli import show as show_cli
from x_creative.cli.concept_space_cli import concept_space_app
from x_creative.config.settings import USER_CONFIG_FILE, get_settings, init_user_config
from x_creative.core.plugin import load_target_domain
from x_creative.core.domain_loader import DomainLibrary
from x_creative.core.types import Hypothesis, ProblemFrame, SearchConfig
from x_creative.answer.source_selector import SourceDomainSelector
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.coordinator import SAGACoordinator

app = typer.Typer(
    name="x-creative",
    help="A creativity-driven cross-domain research agent workflow system.",
    add_completion=False,
)
app.add_typer(session_cli.app, name="session")
app.add_typer(run_cli.app, name="run")
app.add_typer(show_cli.app, name="show")
app.add_typer(hkg_cli.app, name="hkg")
app.add_typer(answer_cli.app, name="answer")
app.add_typer(concept_space_app, name="concept-space")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"x-creative version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """X-Creative: Creativity-driven cross-domain research."""
    pass


# =============================================================================
# Generate Command
# =============================================================================


@app.command()
def generate(
    problem: Annotated[str, typer.Argument(help="Research problem description")],
    num_hypotheses: Annotated[
        int, typer.Option("--num-hypotheses", "-n", help="Number of hypotheses")
    ] = 50,
    search_depth: Annotated[
        int, typer.Option("--search-depth", "-d", help="Search depth")
    ] = 3,
    search_breadth: Annotated[
        int, typer.Option("--search-breadth", "-b", help="Search breadth")
    ] = 5,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output file (JSON)")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Show top N results")] = 10,
) -> None:
    """Generate creative hypotheses for a research problem."""
    problem_frame = ProblemFrame(
        description=problem,
    )

    settings = get_settings()
    runtime_profile = (
        "research"
        if str(settings.runtime_profile).lower() == "research"
        else "interactive"
    )
    config = SearchConfig(
        num_hypotheses=num_hypotheses,
        search_depth=search_depth,
        search_breadth=search_breadth,
        enable_extreme=settings.enable_extreme,
        enable_blending=settings.enable_blending,
        enable_transform_space=settings.enable_transform_space,
        max_blend_pairs=settings.max_blend_pairs,
        max_transform_hypotheses=settings.max_transform_hypotheses,
        runtime_profile=runtime_profile,
        blend_expand_budget_per_round=settings.blend_expand_budget_per_round,
        transform_space_budget_per_round=settings.transform_space_budget_per_round,
        hyperpath_expand_topN=settings.hyperpath_expand_topN,
    )

    console.print(Panel(f"[bold]Problem:[/bold] {problem}", title="X-Creative"))

    with console.status("[bold green]Generating hypotheses..."):
        hypotheses = asyncio.run(_generate_async(problem_frame, config))

    if not hypotheses:
        console.print("[yellow]No hypotheses generated.[/yellow]")
        return

    # Display results
    _display_hypotheses(hypotheses[:top])

    # Save to file if requested
    if output:
        _save_hypotheses(hypotheses, output)
        console.print(f"\n[green]Saved {len(hypotheses)} hypotheses to {output}[/green]")


async def _generate_async(problem: ProblemFrame, config: SearchConfig) -> list[Hypothesis]:
    """Run generation asynchronously."""
    engine = CreativityEngine()
    coordinator = SAGACoordinator(engine=engine)
    try:
        source_domains = None
        target_plugin = load_target_domain(problem.target_domain)
        if target_plugin is not None:
            selector = SourceDomainSelector()
            filtered_domains = await selector.filter_by_mapping_feasibility(
                frame=problem,
                target=target_plugin,
            )
            if filtered_domains:
                source_domains = filtered_domains

        result = await coordinator.run(
            problem,
            config,
            source_domains=source_domains,
        )
        return result.hypotheses
    finally:
        await engine.close()


def _display_hypotheses(hypotheses: list[Hypothesis]) -> None:
    """Display hypotheses in a formatted table."""
    console.print(f"\n[bold]Top {len(hypotheses)} Hypotheses[/bold]\n")

    for i, hyp in enumerate(hypotheses, 1):
        score = hyp.composite_score() if hyp.scores else 0.0

        panel_content = f"""[bold]{hyp.description}[/bold]

[dim]Source:[/dim] {hyp.source_domain}/{hyp.source_structure}

[dim]Analogy:[/dim] {hyp.analogy_explanation[:200]}...

[dim]Observable:[/dim] {hyp.observable}
"""

        if hyp.scores:
            panel_content += f"""
[dim]Scores:[/dim] D={hyp.scores.divergence:.1f} T={hyp.scores.testability:.1f} R={hyp.scores.rationale:.1f} Rb={hyp.scores.robustness:.1f}"""

        console.print(
            Panel(
                panel_content,
                title=f"#{i} [Score: {score:.1f}]",
                border_style="blue" if score >= 7.0 else "dim",
            )
        )


def _save_hypotheses(hypotheses: list[Hypothesis], path: Path) -> None:
    """Save hypotheses to JSON file."""
    data = [h.model_dump() for h in hypotheses]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =============================================================================
# Domains Command
# =============================================================================

domains_app = typer.Typer(help="Manage source domain library")
app.add_typer(domains_app, name="domains")


@domains_app.command("list")
def domains_list(
    target_domain: Annotated[
        str, typer.Option("--target", "-t", help="Target domain ID")
    ] = "open_source_development",
) -> None:
    """List all available source domains for a target domain."""
    library = DomainLibrary.from_target_domain(target_domain)

    table = Table(title=f"Source Domain Library ({target_domain})")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Name (EN)")
    table.add_column("Structures", justify="right")
    table.add_column("Mappings", justify="right")

    for domain in library:
        table.add_row(
            domain.id,
            domain.name,
            domain.name_en or "-",
            str(len(domain.structures)),
            str(len(domain.target_mappings)),
        )

    console.print(table)


@domains_app.command("show")
def domains_show(
    domain_id: Annotated[str, typer.Argument(help="Domain ID to show")],
    target_domain: Annotated[
        str, typer.Option("--target", "-t", help="Target domain ID")
    ] = "open_source_development",
) -> None:
    """Show details of a specific source domain."""
    library = DomainLibrary.from_target_domain(target_domain)
    domain = library.get(domain_id)

    if domain is None:
        console.print(f"[red]Domain '{domain_id}' not found in target domain '{target_domain}'.[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]{domain.name}[/bold] ({domain.name_en or domain.id})", title="Domain"))
    console.print(f"\n{domain.description}\n")

    # Structures
    console.print("[bold]Structures:[/bold]")
    for s in domain.structures:
        console.print(f"  - [cyan]{s.id}[/cyan] ({s.name})")
        console.print(f"    {s.description}")
        console.print(f"    [dim]Variables:[/dim] {', '.join(s.key_variables)}")
        console.print(f"    [dim]Dynamics:[/dim] {s.dynamics}")
        console.print()

    # Mappings
    if domain.target_mappings:
        console.print("[bold]Target Mappings:[/bold]")
        for m in domain.target_mappings:
            console.print(f"  - {m.structure} -> {m.target}")
            console.print(f"    [dim]Observable:[/dim] {m.observable}")


# =============================================================================
# Config Command
# =============================================================================

config_app = typer.Typer(help="Manage configuration")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    settings = get_settings()

    def _mask_key(raw_key: str) -> str:
        if not raw_key:
            return "[red]Not set[/red]"
        return raw_key[:10] + "..." + raw_key[-4:] if len(raw_key) > 14 else "***"

    console.print(Panel("[bold]X-Creative Configuration[/bold]"))

    console.print("\n[bold]Config Files:[/bold]")
    console.print(f"  User config: {USER_CONFIG_FILE}")
    console.print(f"  Exists: {USER_CONFIG_FILE.exists()}")

    console.print("\n[bold]General:[/bold]")
    console.print(f"  Default Provider: {settings.default_provider}")
    console.print(f"  Default Hypotheses: {settings.default_num_hypotheses}")
    console.print(f"  Default Search Depth: {settings.default_search_depth}")

    console.print("\n[bold]OpenRouter:[/bold]")
    console.print(f"  Base URL: {settings.openrouter.base_url}")
    console.print(f"  API Key: {_mask_key(settings.openrouter.api_key.get_secret_value())}")

    console.print("\n[bold]Yunwu:[/bold]")
    console.print(f"  Base URL: {settings.yunwu.base_url}")
    console.print(f"  API Key: {_mask_key(settings.yunwu.api_key.get_secret_value())}")

    console.print("\n[bold]Score Weights:[/bold]")
    console.print(f"  Divergence: {settings.score_weight_divergence}")
    console.print(f"  Testability: {settings.score_weight_testability}")
    console.print(f"  Rationale: {settings.score_weight_rationale}")
    console.print(f"  Robustness: {settings.score_weight_robustness}")

    console.print("\n[bold]Model Routing:[/bold]")
    routing = settings.task_routing
    tasks = ["creativity", "analogical_mapping", "structured_search", "hypothesis_scoring"]
    for task in tasks:
        config = getattr(routing, task)
        console.print(f"  {task}:")
        console.print(f"    Model: {config.model}")
        console.print(f"    Temperature: {config.temperature}")
        if config.fallback:
            console.print(f"    Fallbacks: {', '.join(config.fallback)}")


@config_app.command("init")
def config_init() -> None:
    """Initialize user configuration file.

    Creates ~/.config/x-creative/config.yaml with a template.
    """
    config_path = init_user_config()
    console.print(f"[green]Configuration file created at:[/green] {config_path}")
    console.print("\nEdit this file to set your API key and other options.")
    console.print("Environment variables will override settings in this file.")


@config_app.command("path")
def config_path() -> None:
    """Show the path to the user configuration file."""
    console.print(f"User config file: {USER_CONFIG_FILE}")
    if USER_CONFIG_FILE.exists():
        console.print("[green]File exists[/green]")
    else:
        console.print("[yellow]File does not exist. Run 'x-creative config init' to create it.[/yellow]")


@config_app.command("check")
def config_check(
    quick: Annotated[
        bool, typer.Option("--quick", "-q", help="Static check only, no API calls")
    ] = False,
) -> None:
    """Check .env configuration validity.

    Runs three stages:
    1. Static Check — validates .env fields, model format, score weights
    2. Connectivity Check — verifies API keys work
    3. Model Availability — tests each configured model

    Use --quick to only run static checks (no API calls).
    """
    from x_creative.config.checker import (
        CheckResult,
        CheckStatus,
        check_connectivity,
        check_models,
        check_static,
    )

    def _render(result: CheckResult) -> None:
        lines: list[str] = []
        for item in result.items:
            if item.status == CheckStatus.PASS:
                icon = "[green]\u2713[/green]"
            elif item.status == CheckStatus.WARN:
                icon = "[yellow]\u26a0[/yellow]"
            else:
                icon = "[red]\u2717[/red]"
            msg = f" ({item.message})" if item.message else ""
            lines.append(f" {icon} {item.label}{msg}")
        console.print(Panel("\n".join(lines), title=f"Stage: {result.stage}"))

    has_failure = False

    # Stage 1
    static_result = check_static()
    _render(static_result)
    if not static_result.all_passed:
        has_failure = True
        console.print("\n[red]Static check failed. Skipping remaining stages.[/red]")
        raise typer.Exit(1)

    if quick:
        console.print("\n[green]Static check passed.[/green]")
        raise typer.Exit(0)

    # Stage 2
    settings = get_settings()
    provider = settings.default_provider
    if provider == "openrouter":
        api_key = settings.openrouter.api_key.get_secret_value()
        base_url = settings.openrouter.base_url
    else:
        api_key = settings.yunwu.api_key.get_secret_value()
        base_url = settings.yunwu.base_url

    brave_key = settings.brave_search.api_key.get_secret_value() or None

    conn_result = asyncio.run(check_connectivity(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        brave_api_key=brave_key,
    ))
    _render(conn_result)
    if not conn_result.all_passed:
        has_failure = True

    # Stage 3
    routing = settings.task_routing
    all_models: list[str] = []
    for task_name in type(routing).model_fields:
        config = getattr(routing, task_name)
        all_models.append(config.model)

    models_result = asyncio.run(check_models(
        models=all_models,
        api_key=api_key,
        base_url=base_url,
    ))
    _render(models_result)
    if not models_result.all_passed:
        has_failure = True

    # Summary
    if has_failure:
        console.print("\n[red]Some checks failed.[/red]")
        raise typer.Exit(1)
    else:
        console.print("\n[bold green]All checks passed.[/bold green]")


if __name__ == "__main__":
    app()
