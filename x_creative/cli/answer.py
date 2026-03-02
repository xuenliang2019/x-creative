"""CLI commands for Answer Engine (single-entry deep research)."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import structlog
import typer
from rich.console import Console
from rich.markdown import Markdown

from x_creative.answer.engine import AnswerEngine
from x_creative.answer.constraint_preflight import UserConstraintConflictError
from x_creative.answer.types import AnswerConfig, AnswerPack
from x_creative.cli.progress import AnswerProgress
from x_creative.config.settings import get_settings
from x_creative.saga.constraint_compliance import UserConstraintComplianceError

console = Console()
app = typer.Typer(help="Single-entry deep research")
_summary_logger = structlog.get_logger("answer.summary")


def _format_duration(seconds: float) -> str:
    """Format seconds into '小时分秒' human-readable string."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}小时")
    if m:
        parts.append(f"{m}分")
    parts.append(f"{s}秒")
    return "".join(parts)


def _log_run_summary(result: AnswerPack) -> None:
    """Log token usage and timing summary after an answer run."""
    metadata = result.answer_json.get("metadata", {})
    duration_sec = metadata.get("duration_seconds", 0)
    duration_str = _format_duration(duration_sec)

    summary = result.token_summary
    if not summary:
        _summary_logger.info("run_summary", duration=duration_str, total_tokens=0)
        return

    total_tokens = summary.get("total_tokens", 0)

    # Build stage/task table lines (descending by tokens)
    by_stage_task: dict[str, Any] = summary.get("by_stage_task", {})
    stage_lines: list[str] = []
    for key, val in sorted(by_stage_task.items(), key=lambda x: x[1]["tokens"], reverse=True):
        stage_lines.append(f"  {key}: {val['tokens']:,} tokens ({val['calls']} calls)")

    # Build model table lines (descending by tokens)
    by_model: dict[str, Any] = summary.get("by_model", {})
    model_lines: list[str] = []
    for model, val in sorted(by_model.items(), key=lambda x: x[1]["tokens"], reverse=True):
        model_lines.append(f"  {model}: {val['tokens']:,} tokens ({val['calls']} calls)")

    _summary_logger.info(
        "run_summary",
        duration=duration_str,
        total_tokens=total_tokens,
        by_stage_task="\n" + "\n".join(stage_lines) if stage_lines else "(none)",
        by_model="\n" + "\n".join(model_lines) if model_lines else "(none)",
    )


def _run_json_mode_preflight() -> None:
    """Verify that all json_mode=True models support response_format.

    Exits with code 1 if any model fails the check.
    """
    from x_creative.config.checker import check_json_mode, CheckStatus

    settings = get_settings()
    routing = settings.task_routing

    json_mode_models: list[str] = []
    for task_name in type(routing).model_fields:
        config = getattr(routing, task_name)
        if config.json_mode:
            json_mode_models.append(config.model)
            json_mode_models.extend(config.fallback)

    if not json_mode_models:
        return

    provider = settings.default_provider
    if provider == "openrouter":
        api_key = settings.openrouter.api_key.get_secret_value()
        base_url = settings.openrouter.base_url
    else:
        api_key = settings.yunwu.api_key.get_secret_value()
        base_url = settings.yunwu.base_url

    result = asyncio.run(check_json_mode(
        models=json_mode_models,
        api_key=api_key,
        base_url=base_url,
    ))

    if not result.all_passed:
        console.print("\n[red]JSON mode preflight check failed:[/red]")
        for item in result.items:
            if item.status == CheckStatus.FAIL:
                console.print(f"  [red]\u2717[/red] {item.label}: {item.message}")
        console.print(
            "\n[yellow]Hint: set JSON_MODE=false for affected tasks in .env, "
            "or switch to a model that supports response_format.[/yellow]"
        )
        raise typer.Exit(1)


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

    # Preflight: verify JSON mode support
    _run_json_mode_preflight()

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
        if sys.stdin.isatty():
            followup = typer.prompt("Please provide more context")
            try:
                result = _run_with_progress(f"{question}\nContext: {followup}")
            except Exception as e:
                console.print(f"[red]Answer engine failed: {e}[/red]")
                raise typer.Exit(1)
        else:
            console.print("[yellow]Non-interactive mode — skipping clarification, proceeding with original question.[/yellow]")
            try:
                result = _run_with_progress(question)
            except Exception as e:
                console.print(f"[red]Answer engine failed: {e}[/red]")
                raise typer.Exit(1)

    console.print(Markdown(result.answer_md))
    _log_run_summary(result)

    if output:
        output.write_text(result.answer_md, encoding="utf-8")
        console.print(f"\n[green]Saved to {output}[/green]")
