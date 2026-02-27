"""CLI commands for Answer Engine (single-entry deep research)."""

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from x_creative.answer.engine import AnswerEngine
from x_creative.answer.constraint_preflight import UserConstraintConflictError
from x_creative.answer.types import AnswerConfig
from x_creative.cli.progress import AnswerProgress
from x_creative.saga.constraint_compliance import UserConstraintComplianceError

console = Console()
app = typer.Typer(help="Single-entry deep research")


@app.callback(invoke_without_command=True)
def answer(
    question: Annotated[str, typer.Option("-q", "--question", help="The question to research")],
    budget: Annotated[int, typer.Option("--budget", help="Cognitive budget units")] = 60,
    target: Annotated[str, typer.Option("--target", help="Target domain ID (auto=auto-detect)")] = "auto",
    depth: Annotated[int, typer.Option("--depth", help="SEARCH depth")] = 3,
    breadth: Annotated[int, typer.Option("--breadth", help="SEARCH breadth")] = 5,
    mode: Annotated[str, typer.Option("--mode", help="quick|deep_research|exhaustive")] = "deep_research",
    no_hkg: Annotated[bool, typer.Option("--no-hkg", help="Disable HKG")] = False,
    no_saga: Annotated[bool, typer.Option("--no-saga", help="Disable SAGA supervision")] = False,
    fresh: Annotated[bool, typer.Option("--fresh", help="Skip pre-defined domains, generate from scratch")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path")] = None,
) -> None:
    """Single-entry deep research: input a question, get a creative answer."""
    try:
        config = AnswerConfig(
            budget=budget,
            target_domain=target,
            search_depth=depth,
            search_breadth=breadth,
            mode=mode,
            hkg_enabled=not no_hkg,
            saga_enabled=not no_saga,
            fresh=fresh,
        )
    except ValueError as e:
        console.print(f"[red]Invalid configuration: {e}[/red]")
        raise typer.Exit(1)

    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.print(f"[red]Cannot create output directory {output.parent}: {e}[/red]")
            raise typer.Exit(1)

    engine = AnswerEngine(config=config)

    def _run_with_progress(q: str):
        with AnswerProgress() as progress:
            return asyncio.run(engine.answer(q, progress_callback=progress.callback))

    try:
        result = _run_with_progress(question)
    except UserConstraintConflictError as e:
        report = e.report or {}
        console.print(f"[red]{report.get('message', 'User constraints conflict')}[/red]")
        constraints = report.get("constraints", []) or []
        pairs = report.get("conflict_pairs", []) or []
        if constraints:
            console.print("\n[bold]Constraints:[/bold]")
            for item in constraints:
                cid = item.get("id", "")
                text = item.get("text", "")
                console.print(f"- {cid}: {text}")
        if pairs:
            console.print("\n[bold]Conflicts:[/bold]")
            for pair in pairs:
                left_id = pair.get("left_id", "")
                right_id = pair.get("right_id", "")
                left_text = pair.get("left_text", "")
                right_text = pair.get("right_text", "")
                console.print(f"- {left_id} <-> {right_id}")
                console.print(f"  - {left_text}")
                console.print(f"  - {right_text}")
        raise typer.Exit(1)
    except UserConstraintComplianceError as e:
        report = e.report or {}
        console.print(f"[red]{report.get('message', 'User constraint compliance failed')}[/red]")
        audit = report.get("audit_report") or {}
        items = audit.get("items", []) if isinstance(audit, dict) else []
        if items:
            console.print("\n[bold]Audit Items:[/bold]")
            for item in items:
                cid = item.get("id", "")
                text = item.get("text", "")
                verdict = item.get("verdict", "")
                rationale = item.get("rationale", "")
                fix = item.get("suggested_fix", "")
                console.print(f"- {cid}: verdict={verdict} | {text}")
                if rationale:
                    console.print(f"  - rationale: {rationale}")
                if fix:
                    console.print(f"  - suggested_fix: {fix}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Answer engine failed: {e}[/red]")
        raise typer.Exit(1)

    if result.needs_clarification:
        console.print(f"[yellow]Need clarification:[/yellow] {result.clarification_question}")
        followup = typer.prompt("Please provide more context")
        try:
            result = _run_with_progress(f"{question}\nContext: {followup}")
        except Exception as e:
            console.print(f"[red]Answer engine failed: {e}[/red]")
            raise typer.Exit(1)

    console.print(Markdown(result.answer_md))

    if output:
        output.write_text(result.answer_md, encoding="utf-8")
        console.print(f"\n[green]Saved to {output}[/green]")
