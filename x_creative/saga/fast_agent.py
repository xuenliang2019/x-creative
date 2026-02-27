"""SAGA Fast Agent — event-aware pipeline wrapper.

Wraps the existing CreativityEngine without modifying its internals.
Injects event emission at key pipeline points and checks for Slow Agent
directives at inter-stage checkpoints.
"""

import asyncio
import inspect
import re
import time
from collections.abc import Awaitable, Callable
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import structlog
from structlog import contextvars

from x_creative.config.settings import get_settings
from x_creative.core.types import Domain, Hypothesis, ProblemFrame, SearchConfig, VerifyStatus
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.constraint_checker import detect_conflicting_constraints
from x_creative.saga.events import (
    DirectiveType,
    EventBus,
    EventType,
    FastAgentEvent,
    SlowAgentDirective,
)
from x_creative.saga.state import GenerationMetrics, SharedCognitionState

logger = structlog.get_logger()

_GATE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "no_lookahead_bias": [
        re.compile(r"\bt\+1\b", re.IGNORECASE),
        re.compile(r"\bfuture\b", re.IGNORECASE),
        re.compile(r"\blook[\s-]?ahead\b", re.IGNORECASE),
        re.compile(r"未来"),
    ],
    "no_survivorship_bias": [
        re.compile(r"\bsurvivor(?:ship)?\b", re.IGNORECASE),
        re.compile(r"幸存者偏差"),
    ],
}


class FastAgent:
    """Fast Agent — event-aware wrapper around CreativityEngine.

    Core principle: does NOT modify CreativityEngine / xc-domain internals.
    Only inserts event emission and directive checking at key points.

    Responsibilities:
    1. Execute BISO → SEARCH → VERIFY pipeline
    2. Emit events to EventBus at each stage
    3. Check Slow Agent directives at checkpoints
    4. Support runtime parameter hot-updates

    Args:
        engine: The CreativityEngine instance to wrap.
        event_bus: Event bus for communication with Slow Agent.
        state: Shared cognition state.
    """

    def __init__(
        self,
        engine: CreativityEngine,
        event_bus: EventBus,
        state: SharedCognitionState,
        post_verify_directive_grace_s: float = 0.0,
        post_verify_directive_poll_s: float = 0.05,
        checkpoint_callback: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> None:
        self._engine = engine
        self._event_bus = event_bus
        self._state = state
        self._paused = False
        self._weight_overrides: dict[str, float] | None = None
        self._active_config: SearchConfig | None = None
        self._rescore_requested = False
        self._critical_rejected_ids: set[str] = set()
        self._post_verify_directive_grace_s = max(0.0, float(post_verify_directive_grace_s))
        self._post_verify_directive_poll_s = max(0.01, float(post_verify_directive_poll_s))
        self._checkpoint_callback = checkpoint_callback
        settings = getattr(engine, "_settings", None)
        if settings is None:
            try:
                settings = get_settings()
            except Exception:
                settings = None
        self._max_constraints = int(getattr(settings, "max_constraints", 15))
        self._constraint_similarity_threshold = float(
            getattr(settings, "constraint_similarity_threshold", 0.6)
        )

    async def run_pipeline(
        self,
        problem: ProblemFrame,
        config: SearchConfig | None = None,
        source_domains: list[Domain] | None = None,
    ) -> list[Hypothesis]:
        """Execute the full pipeline with event emission and directive response.

        Emits events at each stage and checks for Slow Agent directives
        at inter-stage checkpoints:
        - After BISO, before SEARCH
        - After each SEARCH round
        - After VERIFY scoring, before filtering
        - After pipeline completion

        Args:
            problem: The research problem framing.
            config: Search configuration. Uses engine defaults if not provided.

        Returns:
            List of scored and sorted hypotheses.
        """
        if config is None:
            config = SearchConfig()

        self._active_config = config
        self._rescore_requested = False
        self._critical_rejected_ids = set()
        self._state.adversarial_challenges = []

        self._state.current_stage = "biso"
        self._state.target_domain_id = problem.target_domain

        logger.info(
            "Fast Agent pipeline started",
            target_domain=problem.target_domain,
        )

        # === Phase 1: BISO ===
        with contextvars.bound_contextvars(pipeline_stage="biso"):
            num_per_domain = max(3, config.num_hypotheses // 15)
            start_payload: dict[str, Any] = {"num_per_domain": num_per_domain}
            if source_domains is not None:
                start_payload["total_domains"] = len(source_domains)
            await self._emit_event(EventType.BISO_STARTED, "biso", payload=start_payload)

            raw_hypotheses = await self._engine._biso.generate_all_analogies(
                problem=problem,
                num_per_domain=num_per_domain,
                source_domains=source_domains,
                on_domain_complete=self._on_biso_domain_complete,
            )
            score_mapping = getattr(self._engine, "_score_mapping_quality_with_events", None)
            if callable(score_mapping):
                await score_mapping(raw_hypotheses, on_event=self._on_mapping_scorer_event)
            else:
                fallback_scoring = getattr(self._engine, "_score_mapping_quality", None)
                if callable(fallback_scoring):
                    await fallback_scoring(raw_hypotheses)

            await self._emit_event(
                EventType.BISO_COMPLETED,
                "biso",
                payload={"hypothesis_count": len(raw_hypotheses)},
                metrics=self._extract_event_metrics(raw_hypotheses),
            )
            await self._emit_mapping_quality_events(raw_hypotheses)

            self._state.hypotheses_pool = [
                h.model_dump() for h in raw_hypotheses
            ]

            # Checkpoint: check directives before SEARCH
            await self._check_directives()
            await self._notify_checkpoint("cp4_biso_completed")
            raw_hypotheses = self._drop_critical_rejected(raw_hypotheses)

            if not raw_hypotheses:
                logger.warning("BISO produced no hypotheses")
                self._state.current_stage = "completed"
                await self._emit_event(EventType.PIPELINE_COMPLETED, "completed")
                return []

        # === Phase 2: SEARCH ===
        with contextvars.bound_contextvars(pipeline_stage="search"):
            self._state.current_stage = "search"
            await self._emit_event(
                EventType.SEARCH_STARTED,
                "search",
                payload={"total_rounds": int(getattr(config, "search_depth", 0) or 0)},
            )

            # Keep SEARCH/HKG behavior aligned with CreativityEngine.generate().
            prepare_search_context = getattr(self._engine, "_prepare_search_context", None)
            if callable(prepare_search_context):
                prepare_search_context(problem, config)
            elif hasattr(self._engine._search, "_problem_frame"):
                self._engine._search._problem_frame = problem

            expanded = await self._run_search_with_round_events(raw_hypotheses, config)

            await self._emit_event(
                EventType.SEARCH_COMPLETED,
                "search",
                payload={"hypothesis_count": len(expanded)},
                metrics=self._extract_event_metrics(expanded),
            )

            self._state.hypotheses_pool = [h.model_dump() for h in expanded]

            # Checkpoint: check directives before VERIFY
            await self._check_directives()
            await self._notify_checkpoint("cp5_search_completed")
            expanded = self._drop_critical_rejected(expanded)

        # === Phase 3: VERIFY ===
        with contextvars.bound_contextvars(pipeline_stage="verify"):
            self._state.current_stage = "verify"
            await self._emit_event(
                EventType.VERIFY_STARTED,
                "verify",
                payload={"score_total": len(expanded)},
            )

            scored = await self._engine.score_and_verify_batch(
                expanded,
                problem_frame=problem,
                score_progress_callback=self._on_verify_score_progress,
                dual_verify_progress_callback=self._on_verify_dual_progress,
            )
            scored = self._apply_adversarial_adjustments(scored)
            self._state.hypotheses_pool = [h.model_dump() for h in scored]

            await self._emit_event(
                EventType.VERIFY_BATCH_SCORED,
                "verify",
                payload={"scored_count": len(scored)},
                metrics=self._extract_event_metrics(scored),
            )
            await self._emit_verify_status_events(scored)

            # Checkpoint: check directives after scoring, before filtering
            await self._check_directives()
            await self._notify_checkpoint("cp6_verify_batch_scored")
            scored = self._drop_critical_rejected(scored)

            # Re-score batch when requested by Slow Agent challenge flow.
            if self._rescore_requested:
                self._rescore_requested = False
                scored = await self._engine.score_and_verify_batch(
                    expanded,
                    problem_frame=problem,
                    score_progress_callback=self._on_verify_score_progress,
                    dual_verify_progress_callback=self._on_verify_dual_progress,
                )
                scored = self._apply_adversarial_adjustments(scored)
                scored = self._drop_critical_rejected(scored)
                self._state.hypotheses_pool = [h.model_dump() for h in scored]
                await self._emit_event(
                    EventType.VERIFY_BATCH_SCORED,
                    "verify",
                    payload={"scored_count": len(scored), "rescored": True},
                    metrics=self._extract_event_metrics(scored),
                )

            # Apply filtering and sorting (with potential weight overrides)
            if any(h.scores is not None for h in scored):
                filtered = self._engine.filter_by_threshold(
                    scored, threshold=config.prune_threshold
                )
            else:
                # Compatibility path for externally pre-scored hypotheses.
                filtered = list(scored)

            sorted_hypotheses = self._sort_with_adjusted_weights(filtered)

            # Apply hard gate constraints from ADD_CONSTRAINT directives
            sorted_hypotheses = self._apply_hard_gates(sorted_hypotheses)

            if self._critical_rejected_ids:
                sorted_hypotheses = [
                    h for h in sorted_hypotheses if h.id not in self._critical_rejected_ids
                ]

            final = sorted_hypotheses[: config.num_hypotheses]

            # Update shared state
            self._state.hypotheses_pool = [h.model_dump() for h in final]
            self._state.generation_metrics = GenerationMetrics(
                **self._calculate_metrics(final)
            ) if final else None

            self._state.current_stage = "completed"
            await self._emit_event(
                EventType.VERIFY_COMPLETED,
                "verify",
                payload={"final_count": len(final)},
            )
            await self._notify_checkpoint("cp6_verify_completed")

            # Give Slow Agent's VERIFY_COMPLETED checkpoint a short response window.
            # Any critical FLAG directives arriving here must still be applied
            # before final output is returned.
            final = await self._apply_post_verify_directives(final)
            self._state.hypotheses_pool = [h.model_dump() for h in final]
            self._state.generation_metrics = GenerationMetrics(
                **self._calculate_metrics(final)
            ) if final else None

            await self._emit_event(EventType.PIPELINE_COMPLETED, "completed")

            logger.info(
                "Fast Agent pipeline completed",
                final_count=len(final),
            )

            return final

    async def _on_mapping_scorer_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Bridge MappingScorer audit events into FastAgent event bus."""
        if event_type != EventType.MAPPING_PADDING_SUSPECTED.value:
            return
        await self._emit_event(
            EventType.MAPPING_PADDING_SUSPECTED,
            "biso",
            payload=payload,
        )

    async def _notify_checkpoint(self, checkpoint_id: str) -> None:
        """Notify optional checkpoint callback (used by CKCoordinator)."""
        callback = self._checkpoint_callback
        if callback is None:
            return
        try:
            maybe_awaitable = callback(checkpoint_id)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as exc:
            logger.debug(
                "Checkpoint callback failed",
                checkpoint_id=checkpoint_id,
                error=str(exc),
            )

    async def _check_directives(self) -> None:
        """Check and execute pending Slow Agent directives.

        Handles:
        - ADJUST_WEIGHTS: update scoring weights
        - ADJUST_SEARCH_PARAMS: modify SearchConfig parameters
        - PAUSE_PIPELINE: wait until RESUME_PIPELINE is received
        - FLAG_HYPOTHESIS: mark hypothesis in shared state
        - ADD_CONSTRAINT: add dynamic constraint
        """
        directives = await self._event_bus.poll_directives()

        for directive in directives:
            logger.info(
                "Fast Agent executing directive",
                directive_type=directive.directive_type.value,
                reason=directive.reason,
            )
            await self._execute_directive(directive)

    async def _apply_post_verify_directives(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[Hypothesis]:
        """Apply directives that arrive right after VERIFY_COMPLETED.

        This closes the race window where critical FLAG directives can be emitted
        from Slow Agent's VERIFY_COMPLETED checkpoint after Fast Agent has already
        produced a final list.
        """
        if self._post_verify_directive_grace_s <= 0.0:
            return hypotheses

        deadline = time.monotonic() + self._post_verify_directive_grace_s
        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            directives = await self._event_bus.poll_directives()
            if directives:
                for directive in directives:
                    logger.info(
                        "Fast Agent executing post-verify directive",
                        directive_type=directive.directive_type.value,
                        reason=directive.reason,
                    )
                    await self._execute_directive(directive)
                hypotheses = self._drop_critical_rejected(hypotheses)
                continue

            remaining = deadline - now
            await asyncio.sleep(min(self._post_verify_directive_poll_s, remaining))

        return hypotheses

    async def _execute_directive(self, directive: SlowAgentDirective) -> None:
        """Execute a single Slow Agent directive.

        Args:
            directive: The directive to execute.
        """
        dt = directive.directive_type

        if dt == DirectiveType.ADJUST_WEIGHTS:
            self._weight_overrides = directive.payload.get("weights")
            logger.info(
                "Weights adjusted by Slow Agent",
                weights=self._weight_overrides,
            )

        elif dt == DirectiveType.PAUSE_PIPELINE:
            logger.info("Pipeline paused by Slow Agent", reason=directive.reason)
            self._paused = True
            # Wait for RESUME directive
            while self._paused:
                resume_directives = await self._event_bus.poll_directives()
                for rd in resume_directives:
                    if rd.directive_type == DirectiveType.RESUME_PIPELINE:
                        self._paused = False
                        logger.info("Pipeline resumed by Slow Agent")
                        break
                if self._paused:
                    await asyncio.sleep(0.1)

        elif dt == DirectiveType.FLAG_HYPOTHESIS:
            hypothesis_id = directive.payload.get("hypothesis_id")
            if hypothesis_id:
                self._state.adversarial_challenges.append({
                    "hypothesis_id": hypothesis_id,
                    "reason": directive.reason,
                    "timestamp": time.time(),
                })
                violations = directive.payload.get("violations", [])
                if any(str(v).startswith("critical:") for v in violations):
                    self._critical_rejected_ids.add(str(hypothesis_id))
                    self._state.hypotheses_pool = [
                        item
                        for item in self._state.hypotheses_pool
                        if not (
                            isinstance(item, dict)
                            and str(item.get("id", "")) == str(hypothesis_id)
                        )
                    ]

        elif dt == DirectiveType.INJECT_CHALLENGE:
            self._state.adversarial_challenges.append({
                "hypothesis_id": directive.payload.get("hypothesis_id"),
                "challenge_type": directive.payload.get("challenge_type"),
                "severity": directive.payload.get("severity", "medium"),
                "reason": directive.reason,
                "timestamp": time.time(),
            })
            self._rescore_requested = True

        elif dt == DirectiveType.ADJUST_SEARCH_PARAMS:
            if self._active_config is not None:
                for key, value in directive.payload.items():
                    if hasattr(self._active_config, key):
                        setattr(self._active_config, key, value)
            logger.info(
                "Search params adjustment requested",
                params=directive.payload,
            )

        elif dt == DirectiveType.RESCORE_BATCH:
            self._rescore_requested = True

        elif dt == DirectiveType.ADD_CONSTRAINT:
            constraint = directive.payload.get("constraint")
            if constraint:
                adjustments = self._state.evaluation_adjustments
                if adjustments.hard_gates is None:
                    adjustments.hard_gates = []
                existing = list(adjustments.hard_gates)
                similar_to = [
                    item
                    for item in existing
                    if SequenceMatcher(
                        None,
                        str(item).lower(),
                        str(constraint).lower(),
                    ).ratio()
                    >= self._constraint_similarity_threshold
                ]
                if similar_to:
                    await self._emit_event(
                        EventType.CONSTRAINT_REFRAME_TRIGGERED,
                        self._state.current_stage or "pipeline",
                        payload={
                            "constraint": str(constraint),
                            "similar_to": [str(item) for item in similar_to[:5]],
                            "threshold": self._constraint_similarity_threshold,
                        },
                    )

                adjustments.hard_gates.append(str(constraint))
                conflicts = detect_conflicting_constraints(
                    adjustments.hard_gates,
                    similarity_threshold=self._constraint_similarity_threshold,
                )
                if conflicts.has_conflicts:
                    await self._emit_event(
                        EventType.CONSTRAINT_CONFLICT_DETECTED,
                        self._state.current_stage or "pipeline",
                        payload={
                            "count": len(conflicts.conflict_pairs),
                            "conflict_pairs": conflicts.conflict_pairs[:5],
                        },
                    )

                if len(adjustments.hard_gates) > self._max_constraints:
                    overflow = len(adjustments.hard_gates) - self._max_constraints
                    await self._emit_event(
                        EventType.CONSTRAINT_SET_GROWTH_ALERT,
                        self._state.current_stage or "pipeline",
                        payload={
                            "constraint_count": len(adjustments.hard_gates),
                            "max_constraints": self._max_constraints,
                            "overflow_count": overflow,
                        },
                    )
                    adjustments.hard_gates = adjustments.hard_gates[-self._max_constraints:]

        else:
            logger.debug(
                "Unhandled directive type",
                directive_type=dt.value,
            )

    async def _run_search_with_round_events(
        self,
        raw_hypotheses: list[Hypothesis],
        config: SearchConfig,
    ) -> list[Hypothesis]:
        """Run SEARCH and emit round-level progress events when supported."""
        async def _on_round_complete(
            round_idx: int,
            pool: list[Hypothesis],
            new_count: int,
        ) -> None:
            await self._emit_event(
                EventType.SEARCH_ROUND_COMPLETED,
                "search",
                payload={
                    "round_index": round_idx,
                    "total_rounds": int(getattr(config, "search_depth", 0) or 0),
                    "hypothesis_count": len(pool),
                    "new_count": new_count,
                },
                metrics=self._extract_event_metrics(pool),
            )
            # SEARCH checkpoint: apply Slow Agent directives between rounds.
            await self._check_directives()
            await self._notify_checkpoint("cp5_search_round_completed")
            if self._critical_rejected_ids:
                pool[:] = self._drop_critical_rejected(pool)

        async def _on_hkg_event(
            event_type: str,
            payload: dict[str, Any],
        ) -> None:
            mapping = {
                "hkg_path_found": EventType.HKG_PATH_FOUND,
                "hkg_path_not_found": EventType.HKG_PATH_NOT_FOUND,
                "hkg_expansion_created": EventType.HKG_EXPANSION_CREATED,
                "blend_expand_completed": EventType.BLEND_EXPAND_COMPLETED,
                "transform_proposed": EventType.TRANSFORM_PROPOSED,
                "transform_accepted_or_rejected": EventType.TRANSFORM_ACCEPTED_OR_REJECTED,
                "transform_space_applied": EventType.TRANSFORM_SPACE_APPLIED,
            }
            mapped = mapping.get(event_type)
            if mapped is None:
                return
            await self._emit_event(
                mapped,
                "search",
                payload=payload,
            )

        run_search = self._engine._search.run_search

        try:
            signature = inspect.signature(run_search)
            supports_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            kwargs: dict[str, Any] = {
                "initial_hypotheses": raw_hypotheses,
                "config": config,
            }
            if "on_round_complete" in signature.parameters or supports_kwargs:
                kwargs["on_round_complete"] = _on_round_complete
            if "on_hkg_event" in signature.parameters or supports_kwargs:
                kwargs["on_hkg_event"] = _on_hkg_event
            return await run_search(**kwargs)
        except (TypeError, ValueError):
            pass

        try:
            return await run_search(
                initial_hypotheses=raw_hypotheses,
                config=config,
                on_round_complete=_on_round_complete,
                on_hkg_event=_on_hkg_event,
            )
        except TypeError as exc:
            if (
                "on_round_complete" not in str(exc)
                and "on_hkg_event" not in str(exc)
            ):
                raise
            try:
                return await run_search(
                    initial_hypotheses=raw_hypotheses,
                    config=config,
                    on_round_complete=_on_round_complete,
                )
            except TypeError as exc_round:
                if "on_round_complete" not in str(exc_round):
                    raise
                return await run_search(
                    initial_hypotheses=raw_hypotheses,
                    config=config,
                )

    def _calculate_metrics(self, hypotheses: list[Hypothesis]) -> dict:
        """Calculate full generation metrics from a list of hypotheses.

        Used to construct GenerationMetrics for shared state.

        Args:
            hypotheses: The hypotheses to compute metrics for.

        Returns:
            Dict suitable for GenerationMetrics construction.
        """
        if not hypotheses:
            return {
                "source_domain_distribution": {},
                "unique_structure_count": 0,
                "expansion_type_distribution": {},
                "score_mean": 0.0,
                "score_std": 0.0,
                "score_min": 0.0,
                "score_max": 0.0,
                "dimension_correlations": {},
                "generation_depth_vs_quality": [],
                "deduplication_ratio": 0.0,
            }

        # Domain distribution
        domain_dist = dict(Counter(h.source_domain for h in hypotheses))
        structures = {f"{h.source_domain}::{h.source_structure}" for h in hypotheses}

        # Expansion type distribution
        expansion_dist = dict(Counter(
            h.expansion_type or "original" for h in hypotheses
        ))

        # Score statistics
        scores = [h.composite_score() for h in hypotheses]
        n = len(scores)
        mean = sum(scores) / n if n > 0 else 0.0
        variance = sum((s - mean) ** 2 for s in scores) / n if n > 0 else 0.0
        std = variance ** 0.5

        return {
            "source_domain_distribution": domain_dist,
            "unique_structure_count": len(structures),
            "expansion_type_distribution": expansion_dist,
            "score_mean": round(mean, 4),
            "score_std": round(std, 4),
            "score_min": round(min(scores) if scores else 0.0, 4),
            "score_max": round(max(scores) if scores else 0.0, 4),
            "dimension_correlations": {},
            "generation_depth_vs_quality": [],
            "deduplication_ratio": 0.0,
        }

    def _extract_event_metrics(self, hypotheses: list[Hypothesis]) -> dict[str, float]:
        """Extract numeric-only metrics for event emission.

        FastAgentEvent.metrics requires dict[str, float], so this extracts
        only the scalar numeric values from the full metrics.

        Args:
            hypotheses: The hypotheses to compute metrics for.

        Returns:
            Dict of scalar metric names to float values.
        """
        full = self._calculate_metrics(hypotheses)
        return {
            "score_mean": float(full["score_mean"]),
            "score_std": float(full["score_std"]),
            "score_min": float(full["score_min"]),
            "score_max": float(full["score_max"]),
            "unique_structure_count": float(full["unique_structure_count"]),
            "hypothesis_count": float(len(hypotheses)),
            "deduplication_ratio": float(full["deduplication_ratio"]),
        }

    def _apply_hard_gates(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Filter hypotheses that violate hard gate constraints."""
        gates = self._state.evaluation_adjustments.hard_gates
        if not gates:
            return hypotheses

        passed: list[Hypothesis] = []
        for hyp in hypotheses:
            text = f"{hyp.description} {hyp.observable or ''} {hyp.analogy_explanation or ''}"
            violations = self._check_gate_violations(text, gates)
            if violations:
                self._critical_rejected_ids.add(hyp.id)
                logger.info(
                    "Hard gate rejected hypothesis",
                    hypothesis_id=hyp.id,
                    violations=violations,
                )
            else:
                passed.append(hyp)
        return passed

    def _drop_critical_rejected(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Drop hypotheses flagged by critical directives immediately."""
        if not self._critical_rejected_ids:
            return hypotheses
        filtered = [h for h in hypotheses if h.id not in self._critical_rejected_ids]
        if len(filtered) != len(hypotheses):
            logger.info(
                "Dropped critical-flagged hypotheses",
                dropped=len(hypotheses) - len(filtered),
            )
        self._state.hypotheses_pool = [h.model_dump() for h in filtered]
        return filtered

    @staticmethod
    def _check_gate_violations(text: str, gates: list[str]) -> list[str]:
        """Check text against hard gate constraints."""
        violations: list[str] = []
        for gate in gates:
            patterns = _GATE_PATTERNS.get(gate, [])
            if patterns:
                if any(p.search(text) for p in patterns):
                    violations.append(f"critical:{gate}")
            else:
                if gate.replace("_", " ").lower() in text.lower():
                    violations.append(f"critical:{gate}")
        return violations

    def _sort_with_adjusted_weights(
        self, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Sort hypotheses using potentially adjusted weights.

        If Slow Agent has issued weight overrides, uses those instead
        of the default settings weights.

        Args:
            hypotheses: Hypotheses to sort.

        Returns:
            Sorted list (descending by composite score).
        """
        if self._weight_overrides:
            return sorted(
                hypotheses,
                key=lambda h: h.composite_score(
                    w_divergence=self._weight_overrides.get("divergence", 0.21),
                    w_testability=self._weight_overrides.get("testability", 0.26),
                    w_rationale=self._weight_overrides.get("rationale", 0.21),
                    w_robustness=self._weight_overrides.get("robustness", 0.17),
                    w_feasibility=self._weight_overrides.get("feasibility", 0.15),
                ),
                reverse=True,
            )
        return self._engine.sort_by_score(hypotheses)

    @staticmethod
    def _penalize_scores(hypothesis: Hypothesis, penalty: float) -> Hypothesis:
        """Apply adversarial penalty to score fields while preserving schema."""
        updates: dict[str, Any] = {}
        if hypothesis.final_score is not None:
            updates["final_score"] = max(0.0, float(hypothesis.final_score) - penalty)

        if hypothesis.scores is not None:
            dimension_penalty = min(2.0, max(0.0, penalty * 0.5))
            updates["scores"] = hypothesis.scores.model_copy(
                update={
                    "divergence": max(0.0, float(hypothesis.scores.divergence) - dimension_penalty),
                    "testability": max(0.0, float(hypothesis.scores.testability) - dimension_penalty),
                    "rationale": max(0.0, float(hypothesis.scores.rationale) - dimension_penalty),
                    "robustness": max(0.0, float(hypothesis.scores.robustness) - dimension_penalty),
                    "feasibility": max(0.0, float(hypothesis.scores.feasibility) - dimension_penalty),
                }
            )

        if not updates:
            return hypothesis
        return hypothesis.model_copy(update=updates)

    def _challenge_penalties(self) -> dict[str, float]:
        """Aggregate challenge penalties by hypothesis id."""
        penalties: dict[str, float] = {}
        for challenge in self._state.adversarial_challenges:
            hypothesis_id = challenge.get("hypothesis_id")
            if not hypothesis_id:
                continue
            challenge_type_raw = challenge.get("challenge_type")
            if not challenge_type_raw:
                continue
            severity = str(challenge.get("severity", "medium")).lower()
            base = {"low": 0.2, "medium": 0.4, "high": 0.7}.get(severity, 0.4)

            challenge_type = str(challenge_type_raw).lower()
            if challenge_type in {"counterexample", "causal_reversal"}:
                base += 0.1

            penalties[str(hypothesis_id)] = min(2.5, penalties.get(str(hypothesis_id), 0.0) + base)
        return penalties

    def _apply_adversarial_adjustments(
        self, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Make challenge directives effective by adjusting rescored hypotheses."""
        penalties = self._challenge_penalties()
        if not penalties:
            return hypotheses

        adjusted: list[Hypothesis] = []
        for hypothesis in hypotheses:
            penalty = penalties.get(hypothesis.id, 0.0)
            if penalty <= 0.0:
                adjusted.append(hypothesis)
                continue
            adjusted.append(self._penalize_scores(hypothesis, penalty))
        return adjusted

    async def _on_biso_domain_complete(
        self,
        domain_id: str,
        completed: int,
        total: int,
        generated: int,
    ) -> None:
        """Bridge per-domain BISO completion into the EventBus for progress UI."""
        await self._emit_event(
            EventType.BISO_DOMAIN_COMPLETED,
            "biso",
            payload={
                "domain_id": domain_id,
                "completed": completed,
                "total": total,
                "generated": generated,
            },
        )

    async def _on_verify_score_progress(
        self,
        completed: int,
        total: int,
        hypothesis_id: str,
    ) -> None:
        """Bridge VERIFY scoring progress into the EventBus for progress UI."""
        await self._emit_event(
            EventType.VERIFY_HYPOTHESIS_SCORED,
            "verify",
            payload={
                "phase": "scoring",
                "completed": completed,
                "total": total,
                "hypothesis_id": hypothesis_id,
            },
        )

    async def _on_verify_dual_progress(
        self,
        completed: int,
        total: int,
        hypothesis_id: str,
    ) -> None:
        """Bridge dual-model VERIFY progress into the EventBus for progress UI."""
        await self._emit_event(
            EventType.VERIFY_HYPOTHESIS_SCORED,
            "verify",
            payload={
                "phase": "dual_verify",
                "completed": completed,
                "total": total,
                "hypothesis_id": hypothesis_id,
            },
        )

    async def _emit_event(
        self,
        event_type: EventType,
        stage: str,
        payload: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Emit an event to the EventBus.

        Args:
            event_type: Type of event.
            stage: Current pipeline stage.
            payload: Event payload data.
            metrics: Statistical metrics.
        """
        event = FastAgentEvent(
            event_type=event_type,
            stage=stage,
            payload=payload or {},
            metrics=metrics or {},
        )
        await self._event_bus.emit_event(event)

    async def _emit_mapping_quality_events(self, hypotheses: list[Hypothesis]) -> None:
        """Emit mapping-quality related events after BISO generation."""
        missing = [h.id for h in hypotheses if not h.mapping_table]
        if missing:
            await self._emit_event(
                EventType.MAPPING_TABLE_MISSING,
                "biso",
                payload={"count": len(missing), "hypothesis_ids": missing[:10]},
            )

        low_systematicity: list[str] = []
        for hypothesis in hypotheses:
            if not hypothesis.mapping_table:
                continue
            group_sizes = Counter(item.systematicity_group_id for item in hypothesis.mapping_table)
            if max(group_sizes.values(), default=0) < 3:
                low_systematicity.append(hypothesis.id)
        if low_systematicity:
            await self._emit_event(
                EventType.MAPPING_LOW_SYSTEMATICITY,
                "biso",
                payload={"count": len(low_systematicity), "hypothesis_ids": low_systematicity[:10]},
            )

    async def _emit_verify_status_events(self, hypotheses: list[Hypothesis]) -> None:
        """Emit VERIFY confidence/status events after scoring."""
        escalated = [
            h.id for h in hypotheses
            if h.verify_status in {VerifyStatus.ESCALATED, VerifyStatus.ESCALATED.value}
        ]
        if escalated:
            await self._emit_event(
                EventType.VERIFY_ESCALATED,
                "verify",
                payload={"count": len(escalated), "hypothesis_ids": escalated[:10]},
            )

        abstained = [
            h.id for h in hypotheses
            if h.verify_status in {VerifyStatus.ABSTAINED, VerifyStatus.ABSTAINED.value}
        ]
        if abstained:
            await self._emit_event(
                EventType.VERIFY_ABSTAINED,
                "verify",
                payload={"count": len(abstained), "hypothesis_ids": abstained[:10]},
            )

        inconsistent = [h.id for h in hypotheses if h.position_consistency is False]
        if inconsistent:
            await self._emit_event(
                EventType.VERIFY_POSITION_INCONSISTENT,
                "verify",
                payload={"count": len(inconsistent), "hypothesis_ids": inconsistent[:10]},
            )

        injection_hits = [h.id for h in hypotheses if h.injection_detected is True]
        if injection_hits:
            await self._emit_event(
                EventType.VERIFY_INJECTION_DETECTED,
                "verify",
                payload={"count": len(injection_hits), "hypothesis_ids": injection_hits[:10]},
            )
