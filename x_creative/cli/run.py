"""CLI commands for running pipeline stages."""

import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel

from x_creative.config.settings import get_settings
from x_creative.core.plugin import load_target_domain
from x_creative.core.types import Hypothesis, ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.belief import UserQuestion
from x_creative.saga.reasoner import QualityAuditRejected, ReasonerFatalError
from x_creative.saga.solve import TalkerReasonerSolver
from x_creative.session import ReportGenerator, SessionManager, StageStatus

app = typer.Typer(help="Run pipeline stages")
console = Console()

# Backward-compatible hook for tests that monkeypatch run.SAGASolver.
SAGASolver = TalkerReasonerSolver


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
        console.print("Use 'x-creative session new <topic>' to create one.")
        raise typer.Exit(1)
    return session


def save_report(manager: SessionManager, session_id: str, stage: str, content: str) -> None:
    """Save a markdown report for a stage."""
    report_path = manager.data_dir / session_id / f"{stage}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_context(value: str | None) -> dict[str, Any]:
    """Parse JSON context string into dict."""
    if not value:
        return {}
    try:
        result = json.loads(value)
        if not isinstance(result, dict):
            console.print("[red]Context must be a JSON object, not an array.[/red]")
            raise typer.Exit(1)
        return result
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in context: {e}[/red]")
        raise typer.Exit(1)


def _normalize_solve_result(raw_result: Any) -> dict[str, Any]:
    """Convert solver output to a serializable dict."""
    if isinstance(raw_result, dict):
        return raw_result
    if hasattr(raw_result, "model_dump"):
        return raw_result.model_dump(mode="json")
    raise ValueError(f"Unsupported solve result type: {type(raw_result)}")


async def _score_missing_mapping_quality(
    hypotheses: list[Hypothesis],
    *,
    router: Any,
) -> None:
    """Populate mapping_quality for hypotheses that have mapping_table but no score."""
    from x_creative.verify.mapping_scorer import MappingScorer

    pending = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.mapping_quality is None and hypothesis.mapping_table
    ]
    if not pending:
        return

    scorer = MappingScorer(router=router)

    async def _score_one(hypothesis: Hypothesis) -> None:
        try:
            result = await scorer.score(hypothesis)
            hypothesis.mapping_quality = result.score
        except Exception:
            # Keep compatibility with legacy data and scorer outages.
            return

    await asyncio.gather(*(_score_one(hypothesis) for hypothesis in pending))


@app.command("problem")
def run_problem(
    description: Annotated[str | None, typer.Option("--description", "-d", help="Problem description")] = None,
    target_domain: Annotated[str, typer.Option("--target-domain", "-t", help="Target domain ID")] = "general",
    context: Annotated[str | None, typer.Option("--context", "-c", help="Domain context (JSON)")] = None,
    constraint: Annotated[list[str] | None, typer.Option("--constraint", help="Constraint (can be used multiple times)")] = None,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID (overrides current)")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force re-run even if completed")] = False,
) -> None:
    """Define the research problem."""
    manager = get_manager()
    session = get_session(manager, session_id)

    # Check if already completed
    if session.is_stage_completed("problem") and not force:
        console.print("[yellow]Problem already defined. Use --force to re-run.[/yellow]")
        raise typer.Exit(0)

    # Interactive mode if description not provided
    if description is None:
        description = typer.prompt("Enter problem description")

    context_dict = parse_context(context)
    constraints = constraint or []

    problem = ProblemFrame(
        description=description,
        target_domain=target_domain,
        context=context_dict,
        constraints=constraints,
    )

    # Update status to running
    manager.update_stage_status(session.id, "problem", StageStatus.RUNNING)

    # Save data
    manager.save_stage_data(session.id, "problem", problem.model_dump())

    # Generate and save report
    report = ReportGenerator.problem_report(problem)
    save_report(manager, session.id, "problem", report)

    # Update status to completed
    manager.update_stage_status(session.id, "problem", StageStatus.COMPLETED)

    console.print(Panel(
        f"[bold]Problem defined successfully[/bold]\n\n"
        f"{description[:100]}{'...' if len(description) > 100 else ''}",
        title="Problem Stage Complete",
        border_style="green",
    ))


@app.command("biso")
def run_biso(
    num_per_domain: Annotated[int, typer.Option("--num-per-domain", "-n", help="Hypotheses per domain")] = 3,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force re-run")] = False,
) -> None:
    """Run BISO stage to generate analogies from distant domains."""
    manager = get_manager()
    session = get_session(manager, session_id)

    # Check dependencies
    if not session.can_run_stage("biso"):
        console.print("[red]Cannot run BISO: problem stage not completed.[/red]")
        console.print("Run 'x-creative run problem' first.")
        raise typer.Exit(1)

    if session.is_stage_completed("biso") and not force:
        console.print("[yellow]BISO already completed. Use --force to re-run.[/yellow]")
        raise typer.Exit(0)

    # Load problem
    problem_data = manager.load_stage_data(session.id, "problem")
    if problem_data is None:
        console.print("[red]Problem data not found.[/red]")
        raise typer.Exit(1)

    problem = ProblemFrame.model_validate(problem_data)

    # Update status
    manager.update_stage_status(session.id, "biso", StageStatus.RUNNING)

    console.print("[bold]Running BISO stage...[/bold]")

    try:
        from x_creative.answer.source_selector import SourceDomainSelector
        from x_creative.creativity.biso import BISOModule
        from x_creative.llm.router import ModelRouter

        async def run_biso_async():
            router = ModelRouter()
            biso = BISOModule(router=router)
            try:
                source_domains = None
                target_plugin = load_target_domain(problem.target_domain)
                if target_plugin is not None:
                    selector = SourceDomainSelector(router=router)
                    filtered_domains = await selector.filter_by_mapping_feasibility(
                        frame=problem,
                        target=target_plugin,
                    )
                    if filtered_domains:
                        source_domains = filtered_domains

                hypotheses = await biso.generate_all_analogies(
                    problem=problem,
                    num_per_domain=num_per_domain,
                    source_domains=source_domains,
                )
                await _score_missing_mapping_quality(hypotheses, router=router)
                return hypotheses
            finally:
                await router.close()

        hypotheses = asyncio.run(run_biso_async())

        # Save data
        data = [h.model_dump() for h in hypotheses]
        manager.save_stage_data(session.id, "biso", {"hypotheses": data})

        # Generate and save report
        report = ReportGenerator.biso_report(hypotheses)
        save_report(manager, session.id, "biso", report)

        # Update status
        manager.update_stage_status(session.id, "biso", StageStatus.COMPLETED)

        console.print(Panel(
            f"[bold]BISO completed[/bold]\n\n"
            f"Generated {len(hypotheses)} hypotheses from distant domains.",
            title="BISO Stage Complete",
            border_style="green",
        ))

    except Exception as e:
        manager.update_stage_status(session.id, "biso", StageStatus.FAILED, error=str(e))
        console.print(f"[red]BISO failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("search")
def run_search(
    depth: Annotated[int, typer.Option("--depth", "-d", help="Search depth")] = 3,
    breadth: Annotated[int, typer.Option("--breadth", "-b", help="Search breadth")] = 5,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force re-run")] = False,
) -> None:
    """Run SEARCH stage to expand hypothesis space."""
    manager = get_manager()
    session = get_session(manager, session_id)

    if not session.can_run_stage("search"):
        console.print("[red]Cannot run SEARCH: BISO stage not completed.[/red]")
        console.print("Run 'x-creative run biso' first.")
        raise typer.Exit(1)

    if session.is_stage_completed("search") and not force:
        console.print("[yellow]SEARCH already completed. Use --force to re-run.[/yellow]")
        raise typer.Exit(0)

    # Load BISO results
    biso_data = manager.load_stage_data(session.id, "biso")
    if biso_data is None:
        console.print("[red]BISO data not found.[/red]")
        raise typer.Exit(1)

    from x_creative.core.types import Hypothesis

    hypotheses = [Hypothesis.model_validate(h) for h in biso_data["hypotheses"]]

    manager.update_stage_status(session.id, "search", StageStatus.RUNNING)

    console.print("[bold]Running SEARCH stage...[/bold]")

    try:
        from x_creative.creativity.search import SearchModule
        from x_creative.llm.router import ModelRouter

        settings = get_settings()
        runtime_profile = (
            "research"
            if str(settings.runtime_profile).lower() == "research"
            else "interactive"
        )
        config = SearchConfig(
            search_depth=depth,
            search_breadth=breadth,
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
        mapping_quality_gate = (
            settings.mapping_quality_gate_threshold
            if settings.mapping_quality_gate_enabled
            else None
        )

        async def run_search_async():
            router = ModelRouter()
            search = SearchModule(
                router=router,
                mapping_quality_gate=mapping_quality_gate,
            )
            try:
                await _score_missing_mapping_quality(hypotheses, router=router)
                expanded = await search.run_search(
                    initial_hypotheses=hypotheses,
                    config=config,
                )
                return expanded
            finally:
                await router.close()

        expanded = asyncio.run(run_search_async())

        # Save data
        data = [h.model_dump() for h in expanded]
        manager.save_stage_data(session.id, "search", {"hypotheses": data})

        # Generate and save report
        report = ReportGenerator.search_report(expanded)
        save_report(manager, session.id, "search", report)

        manager.update_stage_status(session.id, "search", StageStatus.COMPLETED)

        console.print(Panel(
            f"[bold]SEARCH completed[/bold]\n\n"
            f"Expanded to {len(expanded)} hypotheses.",
            title="SEARCH Stage Complete",
            border_style="green",
        ))

    except Exception as e:
        manager.update_stage_status(session.id, "search", StageStatus.FAILED, error=str(e))
        console.print(f"[red]SEARCH failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("verify")
def run_verify(
    threshold: Annotated[float, typer.Option("--threshold", "-t", help="Minimum score threshold")] = 5.0,
    top: Annotated[int, typer.Option("--top", help="Output top N hypotheses")] = 50,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force re-run")] = False,
) -> None:
    """Run VERIFY stage to score and filter hypotheses."""
    manager = get_manager()
    session = get_session(manager, session_id)

    if not session.can_run_stage("verify"):
        console.print("[red]Cannot run VERIFY: SEARCH stage not completed.[/red]")
        console.print("Run 'x-creative run search' first.")
        raise typer.Exit(1)

    if session.is_stage_completed("verify") and not force:
        console.print("[yellow]VERIFY already completed. Use --force to re-run.[/yellow]")
        raise typer.Exit(0)

    # Load SEARCH results
    search_data = manager.load_stage_data(session.id, "search")
    if search_data is None:
        console.print("[red]SEARCH data not found.[/red]")
        raise typer.Exit(1)

    problem_data = manager.load_stage_data(session.id, "problem")
    if problem_data is None:
        console.print("[red]Problem data not found.[/red]")
        raise typer.Exit(1)

    from x_creative.core.types import Hypothesis

    hypotheses = [Hypothesis.model_validate(h) for h in search_data["hypotheses"]]
    problem = ProblemFrame.model_validate(problem_data)

    manager.update_stage_status(session.id, "verify", StageStatus.RUNNING)

    console.print("[bold]Running VERIFY stage...[/bold]")

    try:
        async def run_verify_async():
            engine = CreativityEngine()
            try:
                scored = await engine.score_and_verify_batch(
                    hypotheses,
                    problem_frame=problem,
                )
                filtered = engine.filter_by_threshold(scored, threshold=threshold)
                sorted_hypotheses = engine.sort_by_score(filtered)
                return sorted_hypotheses[:top]
            finally:
                await engine.close()

        final = asyncio.run(run_verify_async())

        # Save data
        data = [h.model_dump() for h in final]
        manager.save_stage_data(session.id, "verify", {"hypotheses": data})

        # Generate and save report
        report = ReportGenerator.verify_report(final, top_n=top)
        save_report(manager, session.id, "verify", report)

        manager.update_stage_status(session.id, "verify", StageStatus.COMPLETED)

        console.print(Panel(
            f"[bold]VERIFY completed[/bold]\n\n"
            f"Final: {len(final)} hypotheses (threshold: {threshold})",
            title="VERIFY Stage Complete",
            border_style="green",
        ))

    except Exception as e:
        manager.update_stage_status(session.id, "verify", StageStatus.FAILED, error=str(e))
        console.print(f"[red]VERIFY failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("all")
def run_all(
    description: Annotated[str | None, typer.Option("--description", "-d", help="Problem description")] = None,
    target_domain: Annotated[str, typer.Option("--target-domain", "-t", help="Target domain ID")] = "general",
    context: Annotated[str | None, typer.Option("--context", "-c", help="Domain context (JSON)")] = None,
    constraint: Annotated[list[str] | None, typer.Option("--constraint", help="Constraint (can be used multiple times)")] = None,
    num_per_domain: Annotated[int, typer.Option("--num-per-domain", help="Hypotheses per domain")] = 3,
    depth: Annotated[int, typer.Option("--depth", help="Search depth")] = 3,
    breadth: Annotated[int, typer.Option("--breadth", help="Search breadth")] = 5,
    threshold: Annotated[float, typer.Option("--threshold", help="Score threshold")] = 5.0,
    top: Annotated[int, typer.Option("--top", help="Output top N")] = 50,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
) -> None:
    """Run all pipeline stages in sequence."""
    run_problem(
        description=description,
        target_domain=target_domain,
        context=context,
        constraint=constraint,
        session_id=session_id,
        force=True,
    )

    run_biso(
        num_per_domain=num_per_domain,
        session_id=session_id,
        force=True,
    )

    run_search(
        depth=depth,
        breadth=breadth,
        session_id=session_id,
        force=True,
    )

    run_verify(
        threshold=threshold,
        top=top,
        session_id=session_id,
        force=True,
    )

    console.print("\n[bold green]All stages completed![/bold green]")


@app.command("solve")
def run_solve(
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Session ID")] = None,
    max_ideas: Annotated[int, typer.Option("--max-ideas", help="Max verify ideas to use")] = 8,
    max_web_results: Annotated[int, typer.Option("--max-web-results", help="Max web results per round")] = 8,
    force: Annotated[bool, typer.Option("--force", help="Force overwrite existing solve artifacts")] = False,
    no_interactive: Annotated[
        bool, typer.Option("--no-interactive", help="禁用推理过程中的交互式提问")
    ] = False,
    auto_refine: Annotated[
        bool,
        typer.Option("--auto-refine/--no-auto-refine", help="启用自适应风险精炼循环"),
    ] = True,
    inner_max: Annotated[
        int, typer.Option("--inner-max", help="内层循环最大轮次 (Step 5↔6 迭代)")
    ] = 3,
    outer_max: Annotated[
        int, typer.Option("--outer-max", help="外层循环最大轮次 (提取新约束后重跑)")
    ] = 2,
) -> None:
    """Run Talker-Reasoner deep reasoning solve based on verify outputs."""
    manager = get_manager()
    session = get_session(manager, session_id)

    if not session.is_stage_completed("verify"):
        console.print("[red]Cannot run SOLVE: VERIFY stage not completed.[/red]")
        console.print("Run 'x-creative run verify' first.")
        raise typer.Exit(1)

    session_dir = manager.data_dir / session.id
    solve_md_path = session_dir / "solve.md"
    solve_json_path = session_dir / "solve.json"

    if solve_md_path.exists() and not force:
        console.print("[yellow]solve.md already exists. Use --force to re-run.[/yellow]")
        raise typer.Exit(0)

    problem_data = manager.load_stage_data(session.id, "problem")
    verify_data = manager.load_stage_data(session.id, "verify")
    verify_md_path = session_dir / "verify.md"

    if problem_data is None:
        console.print("[red]Problem data not found.[/red]")
        raise typer.Exit(1)
    if verify_data is None:
        console.print("[red]VERIFY data not found.[/red]")
        raise typer.Exit(1)
    if not verify_md_path.exists():
        console.print("[red]verify.md not found. Re-run VERIFY stage first.[/red]")
        raise typer.Exit(1)

    problem = ProblemFrame.model_validate(problem_data)
    raw_hypotheses = verify_data.get("hypotheses", [])
    hypotheses = [Hypothesis.model_validate(h) for h in raw_hypotheses]
    if not hypotheses:
        console.print("[red]No verified hypotheses available for solve stage.[/red]")
        raise typer.Exit(1)

    verify_markdown = verify_md_path.read_text(encoding="utf-8")
    saga_session_dir = session_dir / "saga"

    if auto_refine:
        console.print(
            f"[bold]Running Talker-Reasoner solve stage with auto-refine "
            f"(inner={inner_max}, outer={outer_max})...[/bold]"
        )
    else:
        console.print("[bold]Running Talker-Reasoner solve stage...[/bold]")

    async def user_question_handler(question: UserQuestion) -> str:
        """Handle interactive questions from the Reasoner."""
        console.print(f"\n[bold yellow][Reasoner][/bold yellow] {question.question}")
        if question.context:
            # Use Panel for long context (e.g. detailed risk analysis)
            if len(question.context) > 200:
                console.print(Panel(question.context, title="详细信息", border_style="yellow"))
            else:
                console.print(f"[dim]{question.context}[/dim]")
        if question.options:
            for i, opt in enumerate(question.options, 1):
                console.print(f"  {i}. {opt}")
        default_hint = ""
        if question.default and question.options:
            # Show which number corresponds to the default
            for i, opt in enumerate(question.options, 1):
                if opt == question.default:
                    default_hint = str(i)
                    break
        response = await asyncio.to_thread(
            typer.prompt, "请选择", default=default_hint or question.default or ""
        )
        return response

    async def run_solve_async() -> dict[str, Any]:
        async def progress_logger(event: str, payload: dict[str, Any]) -> None:
            def _fmt_seconds(value: Any) -> str:
                try:
                    return f"{float(value):.1f}s"
                except (TypeError, ValueError):
                    return "n/a"

            if event == "run_started":
                console.print(
                    "[cyan][Solve][/cyan] started "
                    f"(target={payload.get('target_domain')}, "
                    f"candidates={payload.get('candidate_hypotheses')}, "
                    f"max_ideas={payload.get('max_ideas')})"
                )
            elif event == "reasoner_step":
                step = payload.get("step", "?")
                total = payload.get("total_steps", 7)
                phase = payload.get("phase", "")
                elapsed = _fmt_seconds(payload.get("elapsed_seconds"))
                summary = payload.get("summary", "")
                console.print(
                    f"[cyan][Reasoner][/cyan] Step {step}/{total}: {phase}"
                )
                if summary:
                    console.print(f"  [green]✓[/green] {summary} ({elapsed})")
            elif event == "reasoner_evidence":
                idx = payload.get("idea_index", "?")
                total = payload.get("idea_total", "?")
                refs = payload.get("references", 0)
                console.print(
                    f"[cyan][Reasoner][/cyan] 证据收集 ({idx}/{total}), "
                    f"refs={refs}"
                )
            elif event == "talker_generating":
                phase = payload.get("phase", "")
                if phase == "start":
                    console.print("[cyan][Talker][/cyan] 基于信念状态生成详细方案...")
                elif phase == "done":
                    tokens = payload.get("tokens", 0)
                    citations = payload.get("citations", 0)
                    console.print(
                        f"  [green]✓[/green] {tokens:,} tokens, "
                        f"{citations} citations"
                    )
            elif event == "refine_inner_round":
                r = payload.get("round", "?")
                mx = payload.get("max_rounds", "?")
                hr = payload.get("high_risks_before", 0)
                console.print(
                    f"[magenta][Refine][/magenta] 内层 Round {r}/{mx}"
                    f" (上轮 high-risk: {hr})"
                )
            elif event == "refine_outer_round":
                r = payload.get("round", "?")
                mx = payload.get("max_rounds", "?")
                nc = payload.get("constraints_count", 0)
                console.print(
                    f"[magenta][Refine][/magenta] 外层 Round {r}/{mx}"
                    f" (当前约束数: {nc})"
                )
                new_c = payload.get("new_constraints", [])
                for c in new_c:
                    console.print(f"  [dim]+ {c}[/dim]")
            elif event == "refine_new_constraints":
                count = payload.get("count", 0)
                constraints = payload.get("constraints", [])
                console.print(
                    f"[magenta][Refine][/magenta] 提取 {count} 条新约束:"
                )
                for c in constraints:
                    console.print(f"  [yellow]+[/yellow] {c}")
            elif event == "refine_converged":
                r = payload.get("outer_round", "?")
                added = payload.get("total_constraints_added", 0)
                console.print(
                    f"[green][Refine][/green] 收敛！"
                    f"(外层第 {r} 轮, 共追加 {added} 条约束)"
                )
            elif event == "refine_not_converged":
                remaining = payload.get("remaining_high_risks", 0)
                added = payload.get("total_constraints_added", 0)
                console.print(
                    f"[yellow][Refine][/yellow] 未完全收敛"
                    f" (剩余 {remaining} 个 high-risk,"
                    f" 共追加 {added} 条约束)"
                )
            elif event == "run_completed":
                console.print(
                    f"[green][Solve][/green] completed "
                    f"(elapsed={payload.get('elapsed_seconds')}s, "
                    f"llm_calls={payload.get('total_llm_calls')}, "
                    f"tokens={payload.get('total_tokens_used', 0):,}, "
                    f"confidence={payload.get('confidence', 0):.2f})"
                )

        # auto-refine implies non-interactive
        effective_no_interactive = no_interactive or auto_refine
        user_cb = None if effective_no_interactive else user_question_handler

        solver = SAGASolver(
            session_dir=saga_session_dir,
            progress_callback=progress_logger,
            user_callback=user_cb,
        )
        try:
            run_kwargs: dict[str, Any] = {
                "problem": problem,
                "verify_markdown": verify_markdown,
                "hypotheses": hypotheses,
                "max_ideas": max_ideas,
                "max_web_results": max_web_results,
                "auto_refine": auto_refine,
                "inner_max": inner_max,
                "outer_max": outer_max,
            }

            # Compatibility: allow legacy solver signatures in tests/plugins.
            try:
                accepted = inspect.signature(solver.run).parameters
                run_kwargs = {k: v for k, v in run_kwargs.items() if k in accepted}
            except (TypeError, ValueError):
                pass

            raw_result = await solver.run(**run_kwargs)
            return _normalize_solve_result(raw_result)
        finally:
            close_fn = getattr(solver, "close", None)
            if callable(close_fn):
                maybe_result = close_fn()
                if asyncio.iscoroutine(maybe_result):
                    await maybe_result

    try:
        result = asyncio.run(run_solve_async())
    except QualityAuditRejected as e:
        console.print(f"\n[yellow]⚠ 质量审查终止: {e}[/yellow]")
        console.print("[dim]提示: 可调整约束条件或假设后重新运行 solve[/dim]")
        raise typer.Exit(0)
    except ReasonerFatalError as e:
        console.print(Panel(
            str(e),
            title="Reasoner 致命错误",
            border_style="red",
        ))
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]SOLVE failed: {e}[/red]")
        raise typer.Exit(1)

    solution_markdown = str(result.get("solution_markdown", "")).strip()
    if not solution_markdown:
        solution_markdown = "# Final Solution\n\nNo solution content generated.\n"

    solve_md_path.write_text(solution_markdown + "\n", encoding="utf-8")
    with open(solve_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    metrics = result.get("metrics", {})
    belief_state = result.get("belief_state", {})
    reasoning_steps = belief_state.get("reasoning_steps", [])
    evidence = result.get("evidence", [])
    user_clarifications = belief_state.get("user_clarifications", [])

    # Refinement summary
    refine_info = ""
    refine_trace = belief_state.get("refinement_trace", {})
    if refine_trace and refine_trace.get("inner_rounds"):
        inner_count = len(refine_trace.get("inner_rounds", []))
        outer_count = refine_trace.get("outer_rounds", 0)
        converged = refine_trace.get("converged", False)
        added = refine_trace.get("total_constraints_added", [])
        final_hr = refine_trace.get("final_high_risk_count", 0)

        status = "[green]converged[/green]" if converged else f"[yellow]{final_hr} high-risk remaining[/yellow]"
        refine_info = (
            f"\n- Refinement: inner={inner_count}, outer={outer_count}, {status}\n"
            f"- Constraints added: {len(added)}"
        )

    console.print(
        Panel(
            "[bold]Talker-Reasoner solve completed[/bold]\n\n"
            f"- Reasoning steps: {len(reasoning_steps)}\n"
            f"- Evidence items: {len(evidence)}\n"
            f"- LLM calls: {metrics.get('total_llm_calls', '-')}\n"
            f"- Tokens used: {metrics.get('total_tokens_used', 0):,}\n"
            f"- User interactions: {len(user_clarifications)}"
            f"{refine_info}\n\n"
            f"Saved: {solve_md_path.name}, {solve_json_path.name}",
            title="Solve Complete",
            border_style="green",
        )
    )
