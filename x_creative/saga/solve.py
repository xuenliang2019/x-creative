"""Talker-Reasoner solve workflow built on top of verify-stage artifacts.

Replaces the original SAGA dual-process solver with a true Talker-Reasoner
architecture: Reasoner (System 2) builds structured belief state through
multi-step reasoning, Talker (System 1) generates detailed solution conditioned
on the belief state.
"""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog
from structlog import contextvars
from pydantic import BaseModel, Field

from x_creative.core.types import ConstraintSpec, Hypothesis, ProblemFrame
from x_creative.config.settings import get_settings
from x_creative.creativity.utils import extract_json_object
from x_creative.llm.router import ModelRouter
from x_creative.saga.events import EventBus, EventType, FastAgentEvent
from x_creative.saga.belief import BeliefState, EvidenceItem, TalkerReasonerResult, UserQuestion
from x_creative.saga.reasoner import Reasoner, UserQuestionCallback
from x_creative.saga.talker import Talker
from x_creative.verify.search import SearchValidator

logger = structlog.get_logger()

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class TalkerReasonerSolver:
    """Talker-Reasoner solver using verify artifacts + multi-step reasoning."""

    def __init__(
        self,
        router: ModelRouter | None = None,
        session_dir: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        user_callback: UserQuestionCallback | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._router = router or ModelRouter()
        self._owns_router = router is None
        self._session_dir = session_dir
        self._progress_callback = progress_callback
        self._user_callback = user_callback
        self._event_bus = event_bus

    async def close(self) -> None:
        """Close owned resources."""
        if self._owns_router:
            await self._router.close()

    async def run(
        self,
        problem: ProblemFrame,
        verify_markdown: str,
        hypotheses: list[Hypothesis],
        max_ideas: int = 8,
        max_web_results: int = 8,
        auto_refine: bool = False,
        inner_max: int = 3,
        outer_max: int = 2,
    ) -> TalkerReasonerResult:
        """Run Talker-Reasoner solve workflow and return grounded plan.

        Args:
            auto_refine: Enable adaptive risk refinement loop.
            inner_max: Max inner-loop iterations (Step 5↔6) per solve run.
            outer_max: Max outer-loop iterations (re-run verify + solve with
                       new constraints extracted from unresolved risks).
        """
        with contextvars.bound_contextvars(pipeline_stage="solve"):
            if auto_refine:
                return await self._run_with_refinement(
                    problem=problem,
                    verify_markdown=verify_markdown,
                    hypotheses=hypotheses,
                    max_ideas=max_ideas,
                    max_web_results=max_web_results,
                    inner_max=inner_max,
                    outer_max=outer_max,
                )
            return await self._run_once(
                problem=problem,
                verify_markdown=verify_markdown,
                hypotheses=hypotheses,
                max_ideas=max_ideas,
                max_web_results=max_web_results,
            )

    async def _run_once(
        self,
        problem: ProblemFrame,
        verify_markdown: str,
        hypotheses: list[Hypothesis],
        max_ideas: int = 8,
        max_web_results: int = 8,
        inner_max: int = 1,
        outer_round: int = 1,
        outer_max: int = 1,
    ) -> TalkerReasonerResult:
        """Single solve run (original behavior or inner-loop-only refinement)."""
        started_at = time.time()
        await self._report_progress(
            "run_started",
            {
                "target_domain": problem.target_domain,
                "candidate_hypotheses": len(hypotheses),
                "max_ideas": max_ideas,
                "max_web_results": max_web_results,
            },
        )

        # 1. Reasoner builds belief state
        reasoner = Reasoner(
            router=self._router,
            progress_callback=self._progress_callback,
            user_callback=self._user_callback,
            max_web_results=max_web_results,
        )
        belief = await reasoner.reason(
            problem=problem,
            verify_markdown=verify_markdown,
            hypotheses=hypotheses,
            max_ideas=max_ideas,
            inner_max=inner_max,
        )

        # 2. Persist reasoning trace
        self._persist_reasoning_trace(belief)

        # 3. Talker generates solution
        talker = Talker(
            router=self._router,
            progress_callback=self._progress_callback,
        )
        solution = await talker.generate(belief, problem)

        # 3.5 Strict user-constraint compliance audit + bounded revise loop.
        # User constraints are treated as hard; on persistent failure we abort.
        from x_creative.saga.constraint_compliance import (
            UserConstraintComplianceError,
            audit_user_constraints,
        )

        max_revision_rounds = 2
        audit_report = await audit_user_constraints(
            router=self._router,
            problem=problem,
            solution_markdown=solution,
        )
        if not audit_report.overall_pass:
            last_report = audit_report
            for round_idx in range(1, max_revision_rounds + 1):
                failing = [item for item in last_report.items if item.verdict != "pass"]
                failures_text = "\n".join(
                    [
                        f"- {item.id}: {item.text}\n  - verdict={item.verdict}\n  - rationale={item.rationale}\n  - fix={item.suggested_fix}"
                        for item in failing
                    ]
                )
                revision_prompt = (
                    "Revise the solution markdown to satisfy ALL user hard constraints.\n"
                    "Rules:\n"
                    "- Make the minimal necessary changes; keep any correct parts.\n"
                    "- Keep citations in [E#] form and keep the References section consistent.\n"
                    "- Output FULL revised markdown only.\n\n"
                    f"Problem:\n{problem.description}\n\n"
                    f"Unmet constraints (from audit):\n{failures_text}\n\n"
                    f"Current solution markdown:\n{solution}\n"
                )
                revised = await self._router.complete(
                    task="constraint_compliance_revision",
                    messages=[{"role": "user", "content": revision_prompt}],
                    temperature=0.2,
                    max_tokens=8192,
                )
                belief.total_llm_calls += 1
                belief.total_tokens_used += int(getattr(revised, "prompt_tokens", 0)) + int(
                    getattr(revised, "completion_tokens", 0)
                )
                revised_text = str(getattr(revised, "content", "") or "").strip()
                if revised_text:
                    solution = revised_text

                last_report = await audit_user_constraints(
                    router=self._router,
                    problem=problem,
                    solution_markdown=solution,
                )
                if last_report.overall_pass:
                    break

            if not last_report.overall_pass:
                raise UserConstraintComplianceError(
                    {
                        "message": (
                            "User constraint compliance failed after "
                            f"{max_revision_rounds} revision rounds"
                        ),
                        "audit_report": last_report.model_dump(mode="json"),
                    }
                )

        # 4. Build result
        elapsed = round(time.time() - started_at, 2)
        citation_count = len(re.findall(r"\[E\d+\]", solution))

        result = TalkerReasonerResult(
            solution_markdown=solution,
            belief_state=belief,
            evidence=belief.evidence,
            metrics={
                "elapsed_seconds": elapsed,
                "reasoning_steps": len(belief.reasoning_steps),
                "evidence_items": len(belief.evidence),
                "total_llm_calls": belief.total_llm_calls,
                "total_tokens_used": belief.total_tokens_used,
                "citation_count": citation_count,
                "confidence": belief.confidence,
                "user_interactions": len(belief.user_clarifications),
            },
        )

        high_risk_count = len(
            [
                risk for risk in belief.quality_assessment.risks
                if risk.get("severity") == "high"
            ]
        )
        await self._emit_saga_event(
            event_type=EventType.SOLVE_ROUND_COMPLETED,
            payload={
                "outer_round": outer_round,
                "outer_max": outer_max,
                "inner_max": inner_max,
                "high_risk_count": high_risk_count,
                "reasoning_steps": len(belief.reasoning_steps),
                "confidence": belief.confidence,
            },
            metrics={
                "high_risk_count": float(high_risk_count),
                "reasoning_steps": float(len(belief.reasoning_steps)),
                "confidence": float(belief.confidence),
            },
        )

        await self._report_progress(
            "run_completed",
            {
                "elapsed_seconds": elapsed,
                "reasoning_steps": len(belief.reasoning_steps),
                "evidence_items": len(belief.evidence),
                "total_llm_calls": belief.total_llm_calls,
                "total_tokens_used": belief.total_tokens_used,
                "citation_count": citation_count,
                "confidence": belief.confidence,
                "user_interactions": len(belief.user_clarifications),
            },
        )

        return result

    async def _run_with_refinement(
        self,
        problem: ProblemFrame,
        verify_markdown: str,
        hypotheses: list[Hypothesis],
        max_ideas: int,
        max_web_results: int,
        inner_max: int,
        outer_max: int,
    ) -> TalkerReasonerResult:
        """Outer refinement loop: re-run solve with new constraints when
        inner loop fails to converge."""

        # Work on a structured copy of constraints so we don't mutate the original.
        current_constraints: list[ConstraintSpec] = []
        seen_constraint_norm: set[str] = set()

        def _append_constraint(spec: ConstraintSpec) -> None:
            text = spec.text.strip()
            if not text:
                return
            normalized = Reasoner._normalize_constraint_text(text)
            if normalized in seen_constraint_norm:
                return
            seen_constraint_norm.add(normalized)
            current_constraints.append(spec.model_copy(update={"text": text}))

        for spec in problem.structured_constraints or []:
            _append_constraint(spec)
        for text in problem.constraints or []:
            _append_constraint(
                ConstraintSpec(
                    text=str(text),
                    origin="user",
                )
            )
        all_added_constraints: list[str] = []
        best_result: TalkerReasonerResult | None = None
        risk_history: list[list[dict[str, Any]]] = []
        try:
            settings = get_settings()
            constraint_similarity_threshold = float(settings.constraint_similarity_threshold)
            max_constraints = int(settings.max_constraints)
        except Exception:
            constraint_similarity_threshold = 0.6
            max_constraints = 15

        for outer_round in range(1, outer_max + 1):
            await self._report_progress(
                "refine_outer_round",
                {
                    "round": outer_round,
                    "max_rounds": outer_max,
                    "constraints_count": len(current_constraints),
                    "new_constraints": all_added_constraints[-3:] if all_added_constraints else [],
                },
            )

            # Build problem with updated constraints
            refined_problem = problem.model_copy(
                update={
                    "constraints": [spec.text for spec in current_constraints],
                    "structured_constraints": [spec.model_copy() for spec in current_constraints],
                }
            )

            # Run solve with inner-loop refinement
            result = await self._run_once(
                problem=refined_problem,
                verify_markdown=verify_markdown,
                hypotheses=hypotheses,
                max_ideas=max_ideas,
                max_web_results=max_web_results,
                inner_max=inner_max,
                outer_round=outer_round,
                outer_max=outer_max,
            )
            best_result = result

            # Check if converged
            belief = result.belief_state
            high_risks = [
                r for r in belief.quality_assessment.risks
                if r.get("severity") == "high"
            ]
            risk_history.append(high_risks)

            from x_creative.saga.constraint_checker import audit_constraint_coverage

            coverage_audit = audit_constraint_coverage(
                constraints=current_constraints,
                risks=high_risks,
                similarity_threshold=constraint_similarity_threshold,
            )
            if coverage_audit.has_violations:
                await self._report_progress(
                    EventType.CONSTRAINT_VIOLATION_FOUND.value,
                    {
                        "uncovered_risk_count": len(coverage_audit.uncovered_risks),
                        "uncovered_risks": coverage_audit.uncovered_risks[:5],
                    },
                )
                await self._emit_saga_event(
                    event_type=EventType.CONSTRAINT_VIOLATION_FOUND,
                    payload={
                        "uncovered_risk_count": len(coverage_audit.uncovered_risks),
                        "uncovered_risks": coverage_audit.uncovered_risks[:5],
                    },
                )

            if len(risk_history) >= 2:
                from x_creative.saga.constraint_checker import detect_repeated_risks

                repeated = detect_repeated_risks(
                    risk_history,
                    similarity_threshold=constraint_similarity_threshold,
                )
                if repeated.has_repeats:
                    await self._report_progress(
                        EventType.CONSTRAINT_REFRAME_TRIGGERED.value,
                        {
                            "reason": "repeated_high_risks",
                            "repeated_pairs": repeated.repeated_pairs[:5],
                        },
                    )
                    await self._emit_saga_event(
                        event_type=EventType.CONSTRAINT_REFRAME_TRIGGERED,
                        payload={
                            "reason": "repeated_high_risks",
                            "repeated_pairs": repeated.repeated_pairs[:5],
                        },
                    )

            if not high_risks:
                belief.refinement_trace.converged = True
                belief.refinement_trace.outer_rounds = outer_round
                belief.refinement_trace.total_constraints_added = all_added_constraints
                logger.info(
                    "Refinement converged",
                    outer_round=outer_round,
                    total_constraints_added=len(all_added_constraints),
                )
                await self._report_progress(
                    "refine_converged",
                    {
                        "outer_round": outer_round,
                        "total_constraints_added": len(all_added_constraints),
                    },
                )
                return result

            # Not converged: extract new constraints from unresolved risks
            if outer_round < outer_max:
                reasoner = Reasoner(
                    router=self._router,
                    progress_callback=self._progress_callback,
                    max_web_results=max_web_results,
                )
                new_constraints = await reasoner.risks_to_constraints(
                    risks=high_risks,
                    existing_constraints=current_constraints,
                )

                if new_constraints:
                    for spec in new_constraints:
                        _append_constraint(spec)
                    all_added_constraints.extend([spec.text for spec in new_constraints])
                    await self._report_progress(
                        EventType.CONSTRAINT_PATCH_APPLIED.value,
                        {
                            "count": len(new_constraints),
                            "constraints": [spec.text for spec in new_constraints[:5]],
                        },
                    )
                    await self._emit_saga_event(
                        event_type=EventType.CONSTRAINT_PATCH_APPLIED,
                        payload={
                            "count": len(new_constraints),
                            "constraints": [spec.text for spec in new_constraints[:5]],
                        },
                    )

                    # Apply constraint budget to prevent inflation
                    from x_creative.saga.constraint_checker import apply_constraint_budget

                    if len(current_constraints) > max_constraints:
                        await self._report_progress(
                            EventType.CONSTRAINT_SET_GROWTH_ALERT.value,
                            {
                                "constraint_count": len(current_constraints),
                                "max_constraints": max_constraints,
                            },
                        )
                        await self._emit_saga_event(
                            event_type=EventType.CONSTRAINT_SET_GROWTH_ALERT,
                            payload={
                                "constraint_count": len(current_constraints),
                                "max_constraints": max_constraints,
                            },
                        )
                        current_constraints = apply_constraint_budget(
                            current_constraints,
                            max_constraints=max_constraints,
                        )
                        seen_constraint_norm = {
                            Reasoner._normalize_constraint_text(spec.text)
                            for spec in current_constraints
                        }

                    from x_creative.saga.constraint_checker import (
                        detect_conflicting_constraints,
                        resolve_conflicting_constraints,
                    )

                    conflicts = detect_conflicting_constraints(
                        [spec.text for spec in current_constraints],
                        similarity_threshold=constraint_similarity_threshold,
                    )
                    if conflicts.has_conflicts:
                        await self._report_progress(
                            EventType.CONSTRAINT_CONFLICT_DETECTED.value,
                            {"conflict_pairs": conflicts.conflict_pairs[:5]},
                        )
                        await self._emit_saga_event(
                            event_type=EventType.CONSTRAINT_CONFLICT_DETECTED,
                            payload={"conflict_pairs": conflicts.conflict_pairs[:5]},
                        )
                        # Resolve conflicts without ever dropping user constraints.
                        current_constraints = resolve_conflicting_constraints(
                            current_constraints,
                            conflicts.conflict_pairs,
                        )
                        seen_constraint_norm = {
                            Reasoner._normalize_constraint_text(spec.text)
                            for spec in current_constraints
                        }

                    await self._report_progress(
                        "refine_new_constraints",
                        {
                            "count": len(new_constraints),
                            "constraints": [spec.text for spec in new_constraints],
                        },
                    )
                else:
                    # LLM couldn't generate new constraints; stop outer loop
                    logger.warning(
                        "No new constraints generated, stopping outer loop",
                        outer_round=outer_round,
                    )
                    break

        # Did not fully converge
        if best_result is not None:
            belief = best_result.belief_state
            belief.refinement_trace.outer_rounds = outer_max
            belief.refinement_trace.total_constraints_added = all_added_constraints
            belief.refinement_trace.converged = False
            belief.refinement_trace.final_high_risk_count = len(
                [r for r in belief.quality_assessment.risks if r.get("severity") == "high"]
            )
            await self._report_progress(
                "refine_not_converged",
                {
                    "outer_rounds": outer_max,
                    "remaining_high_risks": belief.refinement_trace.final_high_risk_count,
                    "total_constraints_added": len(all_added_constraints),
                },
            )
            return best_result

        # Should never reach here, but satisfy type checker
        raise RuntimeError("Refinement loop produced no result")

    def _persist_reasoning_trace(self, belief: BeliefState) -> None:
        """Write reasoning steps to reasoning_trace.jsonl."""
        if not self._session_dir:
            return

        self._session_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self._session_dir / "reasoning_trace.jsonl"

        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                for step in belief.reasoning_steps:
                    f.write(step.model_dump_json() + "\n")
            logger.info("Reasoning trace persisted", path=str(trace_path), steps=len(belief.reasoning_steps))
        except Exception as exc:
            logger.warning("Failed to persist reasoning trace", error=str(exc))

    async def _report_progress(self, event: str, payload: dict[str, Any]) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        try:
            maybe_result = callback(event, payload)
            if asyncio.iscoroutine(maybe_result):
                await maybe_result
        except Exception as exc:
            logger.warning("Progress callback failed", event=event, error=str(exc))

    async def _emit_saga_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Emit solve-stage event to SAGA EventBus when available."""
        if self._event_bus is None:
            return
        try:
            event = FastAgentEvent(
                event_type=event_type,
                stage="solve",
                payload=payload,
                metrics=metrics or {},
            )
            await self._event_bus.emit_event(event)
        except Exception as exc:
            logger.warning("Solve event emission failed", event_type=event_type.value, error=str(exc))


class SolveResult(BaseModel):
    """Backward-compatible solve result for legacy SAGA solver flow."""

    solution_markdown: str
    applied_constraints: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    audit_report: dict[str, Any] = Field(default_factory=dict)


class SAGASolver:
    """Compatibility solver that preserves the pre-Talker/Reasoner contract."""

    def __init__(
        self,
        router: ModelRouter | None = None,
        session_dir: Path | None = None,  # noqa: ARG002
        progress_callback: ProgressCallback | None = None,
        user_callback: UserQuestionCallback | None = None,  # noqa: ARG002
    ) -> None:
        self._router = router or ModelRouter()
        self._owns_router = router is None
        self._progress_callback = progress_callback

    async def close(self) -> None:
        if self._owns_router:
            await self._router.close()

    @staticmethod
    def _hypothesis_rank_score(hypothesis: Hypothesis) -> float:
        """Rank hypotheses by final_score first, then composite score."""
        if hypothesis.final_score is not None:
            return float(hypothesis.final_score)
        return float(hypothesis.composite_score())

    async def run(
        self,
        problem: ProblemFrame,
        verify_markdown: str,
        hypotheses: list[Hypothesis],
        max_ideas: int = 8,
        max_web_results: int = 8,
    ) -> SolveResult:
        stage_seconds: dict[str, float] = {}
        event_count = 0
        started = time.perf_counter()

        async def _emit(event: str, payload: dict[str, Any]) -> None:
            nonlocal event_count
            event_count += 1
            callback = self._progress_callback
            if callback is None:
                return
            maybe_result = callback(event, payload)
            if asyncio.iscoroutine(maybe_result):
                await maybe_result

        await _emit(
            "run_started",
            {
                "target_domain": problem.target_domain,
                "candidate_hypotheses": len(hypotheses),
                "max_ideas": max_ideas,
                "max_web_results": max_web_results,
            },
        )

        # Step 1: select top ideas.
        t0 = time.perf_counter()
        ranked = sorted(
            hypotheses,
            key=self._hypothesis_rank_score,
            reverse=True,
        )
        selected = ranked[: max(1, max_ideas)] if ranked else []
        stage_seconds["idea_selection"] = round(time.perf_counter() - t0, 3)
        await _emit(
            "ideas_selected",
            {
                "ideas_selected": len(selected),
                "idea_total": len(selected),
                "stage_seconds": stage_seconds["idea_selection"],
            },
        )

        # Step 2: build idea evidence from verify artifacts.
        t1 = time.perf_counter()
        evidence: list[dict[str, Any]] = []
        for idx, hypothesis in enumerate(selected, start=1):
            evidence.append(
                {
                    "id": f"E{idx}",
                    "hypothesis_id": hypothesis.id,
                    "source": "verify",
                    "summary": hypothesis.description,
                    "observable": hypothesis.observable,
                }
            )
            await _emit(
                "idea_evidence_collected",
                {
                    "idea_index": idx,
                    "idea_total": len(selected),
                    "progress_percent": round((idx / max(len(selected), 1)) * 100.0, 1),
                    "stage_seconds": round(time.perf_counter() - t1, 3),
                },
            )
        stage_seconds["idea_evidence"] = round(time.perf_counter() - t1, 3)

        # Step 3: gather web evidence via SearchValidator.
        t2 = time.perf_counter()
        try:
            async with SearchValidator() as validator:
                for hypothesis in selected:
                    preliminary = (
                        float(hypothesis.final_score)
                        if hypothesis.final_score is not None
                        else float(hypothesis.composite_score())
                    )
                    verdict = await validator.validate(
                        hypothesis=hypothesis,
                        preliminary_score=preliminary,
                    )
                    for i, work in enumerate(verdict.similar_works[:max_web_results], start=1):
                        evidence.append(
                            {
                                "id": f"W{hypothesis.id}_{i}",
                                "hypothesis_id": hypothesis.id,
                                "source": work.source,
                                "url": work.url,
                                "title": work.title,
                                "summary": work.difference_summary,
                            }
                        )
        except Exception as exc:
            logger.warning("Legacy SAGA web evidence collection failed", error=str(exc))
        stage_seconds["web_evidence"] = round(time.perf_counter() - t2, 3)
        await _emit(
            "web_evidence_collected",
            {
                "evidence_items": len(evidence),
                "stage_seconds": stage_seconds["web_evidence"],
            },
        )

        # Step 4: generate draft and enforce citation constraint when needed.
        t3 = time.perf_counter()
        applied_constraints: list[str] = []
        draft = await self._generate_draft(
            problem=problem,
            verify_markdown=verify_markdown,
            selected=selected,
            evidence=evidence,
            applied_constraints=applied_constraints,
        )

        if not re.search(r"\[E\d+\]", draft):
            applied_constraints.append("all claims must cite evidence ids")
            draft = await self._generate_draft(
                problem=problem,
                verify_markdown=verify_markdown,
                selected=selected,
                evidence=evidence,
                applied_constraints=applied_constraints,
            )

        stage_seconds["draft_generation"] = round(time.perf_counter() - t3, 3)
        await _emit(
            "draft_generated",
            {
                "citations": len(re.findall(r"\[E\d+\]", draft)),
                "stage_seconds": stage_seconds["draft_generation"],
            },
        )

        elapsed_seconds = round(time.perf_counter() - started, 3)
        metrics = {
            "ideas_used": len(selected),
            "evidence_items": len(evidence),
            "elapsed_seconds": elapsed_seconds,
            "stage_seconds": stage_seconds,
        }
        audit_report = {
            "events_processed": event_count,
            "constraints_applied": len(applied_constraints),
        }

        await _emit(
            "run_completed",
            {
                "ideas_used": len(selected),
                "evidence_items": len(evidence),
                "elapsed_seconds": elapsed_seconds,
                "stage_seconds": stage_seconds,
            },
        )

        return SolveResult(
            solution_markdown=draft,
            applied_constraints=applied_constraints,
            evidence=evidence,
            metrics=metrics,
            audit_report=audit_report,
        )

    async def _generate_draft(
        self,
        problem: ProblemFrame,
        verify_markdown: str,
        selected: list[Hypothesis],
        evidence: list[dict[str, Any]],
        applied_constraints: list[str],
    ) -> str:
        selected_text = "\n".join(
            [
                f"- {h.id}: {h.description} (observable: {h.observable})"
                for h in selected
            ]
        )
        evidence_text = "\n".join(
            [
                f"- [{item.get('id', 'E?')}] {item.get('summary', '')}"
                for item in evidence[:20]
            ]
        )
        constraints_text = "\n".join(applied_constraints) if applied_constraints else "none"

        prompt = (
            "You are generating the final solution markdown.\n"
            f"Problem: {problem.description}\n"
            f"Target domain: {problem.target_domain}\n"
            f"Applied constraints: {constraints_text}\n\n"
            "Selected ideas:\n"
            f"{selected_text or '- none'}\n\n"
            "Evidence snippets:\n"
            f"{evidence_text or '- none'}\n\n"
            "Verification notes:\n"
            f"{verify_markdown[:2500]}\n\n"
            "Output markdown with a clear actionable plan."
        )

        if applied_constraints:
            prompt += (
                "\nIMPORTANT: all claims must cite evidence ids using bracket form like [E1]."
            )

        completion = await self._router.complete(
            task="talker_output",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        content = str(getattr(completion, "content", "")).strip()
        if not content:
            return "# Final Solution\n\nNo solution content generated.\n"
        return content


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from text."""
    return extract_json_object(text)


# Backward-compatible alias for callers that explicitly need Talker-Reasoner.
TalkerReasonerLegacyResult = TalkerReasonerResult
