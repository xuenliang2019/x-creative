"""CLI commands for viewing stage results."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from x_creative.core.types import Hypothesis, ProblemFrame
from x_creative.session import SessionManager

app = typer.Typer(help="View stage results")
console = Console()


def get_data_dir() -> Path:
    """Get the data directory from environment or default."""
    env_dir = os.environ.get("X_CREATIVE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("local_data")


def get_manager() -> SessionManager:
    """Get the session manager."""
    return SessionManager(data_dir=get_data_dir())


def get_session(manager: SessionManager, session_id: str | None = None):
    """Get the session to operate on."""
    if session_id:
        session = manager.load_session(session_id)
        if session is None:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            raise typer.Exit(1)
        return session

    session = manager.get_current_session()
    if session is None:
        console.print("[red]No current session.[/red]")
        raise typer.Exit(1)
    return session


@app.command("problem")
def show_problem(
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw JSON")] = False,
) -> None:
    """Show problem definition."""
    manager = get_manager()
    session = get_session(manager, session_id)

    data = manager.load_stage_data(session.id, "problem")
    if data is None:
        console.print("[yellow]Problem not defined yet.[/yellow]")
        console.print("Run 'x-creative run problem' first.")
        raise typer.Exit(0)

    if raw:
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    problem = ProblemFrame.model_validate(data)

    # Build display content
    content = f"[bold]{problem.description}[/bold]\n\n"
    content += f"Target Domain: {problem.target_domain}\n"

    if problem.context:
        content += f"Context: {json.dumps(problem.context, ensure_ascii=False)}\n"

    if problem.constraints:
        content += "\nConstraints:\n"
        for c in problem.constraints:
            content += f"  • {c}\n"

    console.print(Panel(
        content.rstrip(),
        title="Problem Definition",
    ))


@app.command("biso")
def show_biso(
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Show top N")] = 10,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw JSON")] = False,
) -> None:
    """Show BISO stage results."""
    manager = get_manager()
    session = get_session(manager, session_id)

    data = manager.load_stage_data(session.id, "biso")
    if data is None:
        console.print("[yellow]BISO not run yet.[/yellow]")
        console.print("Run 'x-creative run biso' first.")
        raise typer.Exit(0)

    if raw:
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    hypotheses = [Hypothesis.model_validate(h) for h in data["hypotheses"]]

    console.print(f"[bold]BISO Results: {len(hypotheses)} hypotheses[/bold]\n")

    for i, h in enumerate(hypotheses[:top], 1):
        description_display = h.description[:100] if len(h.description) > 100 else h.description
        observable_display = h.observable[:80] if len(h.observable) > 80 else h.observable

        console.print(Panel(
            f"[bold]{description_display}[/bold]\n\n"
            f"[dim]Domain:[/dim] {h.source_domain}/{h.source_structure}\n"
            f"[dim]Observable:[/dim] {observable_display}...",
            title=f"#{i}",
        ))

    if len(hypotheses) > top:
        console.print(f"\n[dim]... and {len(hypotheses) - top} more[/dim]")


@app.command("search")
def show_search(
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Show top N")] = 10,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw JSON")] = False,
) -> None:
    """Show SEARCH stage results."""
    manager = get_manager()
    session = get_session(manager, session_id)

    data = manager.load_stage_data(session.id, "search")
    if data is None:
        console.print("[yellow]SEARCH not run yet.[/yellow]")
        console.print("Run 'x-creative run search' first.")
        raise typer.Exit(0)

    if raw:
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    hypotheses = [Hypothesis.model_validate(h) for h in data["hypotheses"]]

    console.print(f"[bold]SEARCH Results: {len(hypotheses)} hypotheses[/bold]\n")

    # Group by generation
    by_gen: dict[int, int] = {}
    for h in hypotheses:
        by_gen[h.generation] = by_gen.get(h.generation, 0) + 1

    table = Table(title="Generation Summary")
    table.add_column("Generation")
    table.add_column("Count")

    for gen in sorted(by_gen.keys()):
        table.add_row(str(gen), str(by_gen[gen]))

    console.print(table)


@app.command("verify")
def show_verify(
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    top: Annotated[int, typer.Option("--top", "-t", help="Show top N")] = 10,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw JSON")] = False,
) -> None:
    """Show VERIFY stage results."""
    manager = get_manager()
    session = get_session(manager, session_id)

    data = manager.load_stage_data(session.id, "verify")
    if data is None:
        console.print("[yellow]VERIFY not run yet.[/yellow]")
        console.print("Run 'x-creative run verify' first.")
        raise typer.Exit(0)

    if raw:
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    hypotheses = [Hypothesis.model_validate(h) for h in data["hypotheses"]]

    console.print(f"[bold]Final Results: {len(hypotheses)} verified hypotheses[/bold]\n")

    for i, h in enumerate(hypotheses[:top], 1):
        score = h.composite_score()
        score_style = "green" if score >= 7.0 else "yellow" if score >= 5.0 else "dim"

        content = f"[bold]{h.description}[/bold]\n\n"
        content += f"[dim]Source:[/dim] {h.source_domain}/{h.source_structure}\n"

        observable_display = h.observable[:100] if len(h.observable) > 100 else h.observable
        content += f"[dim]Observable:[/dim] {observable_display}...\n"

        if h.scores:
            content += (
                f"\n[dim]Scores:[/dim] D={h.scores.divergence:.1f} "
                f"T={h.scores.testability:.1f} R={h.scores.rationale:.1f} "
                f"Rb={h.scores.robustness:.1f} F={h.scores.feasibility:.1f}"
            )

        console.print(Panel(
            content,
            title=f"#{i} [{score_style}]Score: {score:.1f}[/{score_style}]",
            border_style=score_style,
        ))

    if len(hypotheses) > top:
        console.print(f"\n[dim]... and {len(hypotheses) - top} more[/dim]")


@app.command("report")
def show_report(
    stage: Annotated[str, typer.Argument(help="Stage name (problem, biso, search, verify)")],
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Save to file instead of printing")] = None,
) -> None:
    """Show the Markdown report for a stage."""
    manager = get_manager()
    session = get_session(manager, session_id)

    report_path = manager.data_dir / session.id / f"{stage}.md"
    if not report_path.exists():
        console.print(f"[yellow]Report for '{stage}' not found.[/yellow]")
        raise typer.Exit(0)

    content = report_path.read_text()

    if output:
        output_path = Path(output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.print(f"[red]Cannot create output directory {output_path.parent}: {e}[/red]")
            raise typer.Exit(1)
        output_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Report saved to {output_path}[/green]")
    else:
        console.print(Markdown(content))
