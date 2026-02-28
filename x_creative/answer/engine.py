"""AnswerEngine - single-entry orchestrator for the full creativity pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import inspect
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from structlog import contextvars

from x_creative.answer.pack_builder import AnswerPackBuilder
from x_creative.answer.problem_frame import ProblemFrameBuilder
from x_creative.answer.source_selector import SourceDomainSelector
from x_creative.answer.target_resolver import TargetDomainResolver
from x_creative.answer.types import AnswerConfig, AnswerPack, PipelineStageError
from x_creative.answer.constraint_preflight import preflight_user_constraints
from x_creative.config.settings import get_settings
from x_creative.core.types import ConstraintSpec, ProblemFrame
from x_creative.core.types import SearchConfig
from x_creative.creativity.engine import CreativityEngine
from x_creative.llm.router import ModelRouter
from x_creative.saga.coordinator import SAGACoordinator
from x_creative.saga.events import DirectiveType
from x_creative.saga.solve import TalkerReasonerSolver
from x_creative.session import SessionManager, StageStatus
from x_creative.session.report import ReportGenerator

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class AnswerEngine:
    """Single-entry orchestrator: question -> AnswerPack."""

    def __init__(self, config: AnswerConfig | None = None):
        self.config = config or AnswerConfig()
        self.session_manager = SessionManager()
        self._session = None

    async def answer(
        self,
        question: str,
        config: AnswerConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnswerPack:
        cfg = config or self.config
        settings = get_settings()
        start_time = time.time()
        stage_total = 8

        # 1. Create session
        session = self.session_manager.create_session(topic=question[:80])
        self._session = session
        session_dir = Path(self.session_manager.data_dir) / session.id
        await self._report_progress(
            progress_callback,
            "answer_started",
            {
                "session_id": session.id,
                "stage_total": stage_total,
                "budget_total": int(cfg.budget),
            },
        )

        # 2. Build ProblemFrame
        builder = ProblemFrameBuilder()
        await self._report_progress(
            progress_callback,
            "answer_stage_started",
            {"stage": "problem", "stage_index": 1, "stage_total": stage_total},
        )
        with contextvars.bound_contextvars(pipeline_stage="problem"):
            frame_result = await builder.build(question)
        if frame_result.needs_clarification:
            return AnswerPack.clarification_needed(
                frame_result.partial_frame,
                frame_result.clarification_question,
            )
        problem_frame = preflight_user_constraints(frame_result.frame)
        cp1 = self._cp1_constraint_audit(problem_frame)
        self._save_stage("problem", {
            "problem_frame": problem_frame.model_dump(),
            "cp1_constraint_audit": cp1,
        })
        await self._report_progress(
            progress_callback,
            "answer_stage_completed",
            {
                "stage": "problem",
                "stage_index": 1,
                "stage_total": stage_total,
                "constraint_count": len(problem_frame.constraints or []),
            },
        )

        # 3. Resolve target domain
        resolver = TargetDomainResolver()
        await self._report_progress(
            progress_callback,
            "answer_stage_started",
            {"stage": "target", "stage_index": 2, "stage_total": stage_total},
        )
        with contextvars.bound_contextvars(pipeline_stage="target"):
            target_plugin = await resolver.resolve(problem_frame, cfg.target_domain, fresh=cfg.fresh)
        problem_frame.target_domain = target_plugin.id
        cp2 = self._cp2_target_domain_audit(problem_frame, target_plugin)
        self._save_stage("biso", {
            "target_resolved": {
                "id": target_plugin.id,
                "name": target_plugin.name,
                "description": target_plugin.description,
            },
            "cp2_target_domain_audit": cp2,
        })
        await self._report_progress(
            progress_callback,
            "answer_stage_completed",
            {
                "stage": "target",
                "stage_index": 2,
                "stage_total": stage_total,
                "target_domain_id": target_plugin.id,
            },
        )

        # 4. Select source domains
        selector = SourceDomainSelector()
        await self._report_progress(
            progress_callback,
            "answer_stage_started",
            {"stage": "sources", "stage_index": 3, "stage_total": stage_total},
        )
        with contextvars.bound_contextvars(pipeline_stage="sources"):
            source_domains = await selector.select(
                problem_frame, target_plugin,
                min_domains=18, max_domains=30,
            )
        await self._stage_gate(
            "sources", source_domains, critical=True,
            reason="Source domain selection returned 0 domains",
            context={"target_domain": target_plugin.id},
            progress_callback=progress_callback,
        )
        cp3 = self._cp3_source_bias_audit(source_domains, target_plugin.id)
        mode_depth, mode_breadth = self._resolve_mode_search_params(cfg)
        runtime_profile = (
            "research"
            if str(settings.runtime_profile).lower() == "research"
            else "interactive"
        )
        search_config = SearchConfig(
            search_depth=mode_depth,
            search_breadth=mode_breadth,
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
        checkpoint_directives = self._derive_checkpoint_directives(cp1, cp2, cp3)
        problem_frame, search_config = self._apply_checkpoint_directives(
            problem_frame,
            search_config,
            checkpoint_directives,
        )
        self._save_stage("search", {
            "sources_selected": {
                "count": len(source_domains),
                "domain_ids": [d.id for d in source_domains],
            },
            "cp3_source_bias_audit": cp3,
            "checkpoint_directives": checkpoint_directives,
        })
        await self._report_progress(
            progress_callback,
            "answer_stage_completed",
            {
                "stage": "sources",
                "stage_index": 3,
                "stage_total": stage_total,
                "source_domain_count": len(source_domains),
            },
        )

        # 5. BISO -> SEARCH -> VERIFY
        router = ModelRouter()
        hypotheses: list[Any]
        saga_directives: list[dict[str, Any]] = list(checkpoint_directives)
        budget_used = 0
        search_rounds = search_config.search_depth
        coordinator: SAGACoordinator | None = None
        creativity_engine = CreativityEngine(router=router)
        if not cfg.hkg_enabled:
            creativity_engine.set_hkg_enabled(False)

        if cfg.saga_enabled:
            saga_budget = self._build_saga_budget(float(cfg.budget))
            coordinator = SAGACoordinator(
                engine=creativity_engine,
                budget=saga_budget,
                session_dir=session_dir,
                enable_ck=settings.ck_enabled,
            )
            saga_result = await coordinator.run(
                problem=problem_frame,
                config=search_config,
                source_domains=source_domains,
                initial_directives=checkpoint_directives,
                progress_callback=progress_callback,
                target_plugin=target_plugin,
            )
            hypotheses = saga_result.hypotheses
            saga_directives.extend(list(saga_result.intervention_log))
            budget_used = int(round(float(saga_result.budget_spent)))
            if isinstance(saga_result.metrics, dict):
                search_rounds = int(saga_result.metrics.get("search_rounds", search_rounds))
        else:
            try:
                hypotheses = await creativity_engine.generate(
                    problem=problem_frame,
                    config=search_config,
                    source_domains=source_domains,
                    progress_callback=progress_callback,
                )
            finally:
                await creativity_engine.close()

        await self._stage_gate(
            "generation", hypotheses, critical=True,
            reason="BISO/SEARCH/VERIFY produced 0 hypotheses",
            context={"saga_enabled": cfg.saga_enabled},
            progress_callback=progress_callback,
        )

        # Filter by threshold
        verified_above = [
            h for h in hypotheses
            if (h.final_score or h.composite_score() or 0) >= cfg.verify_threshold
        ]

        # 6. SOLVE
        solve_result = None
        if verified_above and cfg.auto_refine:
            solve_event_bus = None
            solve_slow_agent = None
            solve_slow_task = None

            if cfg.saga_enabled and coordinator is not None:
                try:
                    from x_creative.saga.budget import CognitiveBudget
                    from x_creative.saga.events import EventBus
                    from x_creative.saga.slow_agent import SlowAgent
                    from x_creative.saga.state import SharedCognitionState

                    solve_event_bus = EventBus(
                        session_dir=session_dir,
                        progress_callback=progress_callback,
                    )
                    solve_state = SharedCognitionState(
                        target_domain_id=problem_frame.target_domain,
                        target_plugin=target_plugin,
                        current_stage="solve",
                        hypotheses_pool=[
                            h.model_dump() for h in verified_above[:cfg.max_ideas]
                        ],
                    )
                    solve_budget = CognitiveBudget(
                        total_budget=max(10.0, float(cfg.budget) * 0.2),
                    )
                    solve_slow_agent = SlowAgent(
                        event_bus=solve_event_bus,
                        state=solve_state,
                        budget=solve_budget,
                        detectors=coordinator._create_detectors(),
                        auditors=coordinator._create_auditors(),
                        evaluators=coordinator._create_evaluators(),
                        router=getattr(creativity_engine, "_router", None),
                    )
                    solve_slow_task = asyncio.create_task(solve_slow_agent.run())
                except Exception as exc:
                    logger.warning("solve_stage_saga_monitor_unavailable: %s", exc)
                    solve_event_bus = None
                    solve_slow_agent = None
                    solve_slow_task = None

            solver = TalkerReasonerSolver(
                session_dir=session_dir,
                event_bus=solve_event_bus,
                progress_callback=progress_callback,
            )
            try:
                verify_md = ReportGenerator.verify_report(verified_above[:cfg.max_ideas])
                solve_result = await solver.run(
                    problem=problem_frame,
                    verify_markdown=verify_md,
                    hypotheses=verified_above[:cfg.max_ideas],
                    max_ideas=cfg.max_ideas,
                    auto_refine=cfg.auto_refine,
                    inner_max=cfg.inner_max,
                    outer_max=cfg.outer_max,
                )
            finally:
                await solver.close()
                if solve_slow_agent is not None:
                    solve_slow_agent.request_stop()
                    if solve_slow_task is not None:
                        try:
                            await asyncio.wait_for(solve_slow_task, timeout=5.0)
                        except asyncio.TimeoutError:
                            solve_slow_task.cancel()
                            try:
                                await asyncio.wait_for(solve_slow_task, timeout=1.0)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass
                        except Exception:
                            pass
                    if solve_slow_agent.intervention_log:
                        saga_directives.extend(list(solve_slow_agent.intervention_log))

        # 7. Build AnswerPack
        solve_dict = None
        if solve_result is not None:
            if hasattr(solve_result, "model_dump"):
                solve_dict = solve_result.model_dump(mode="json")
            elif isinstance(solve_result, dict):
                solve_dict = solve_result

        duration = time.time() - start_time

        pack = AnswerPackBuilder.build(
            session=session,
            problem_frame=problem_frame,
            target_plugin=target_plugin,
            source_domains=source_domains,
            verified_hypotheses=verified_above,
            solve_result=solve_dict,
            question=question,
            budget_used=budget_used,
            budget_total=cfg.budget,
            saga_directives=saga_directives,
            search_rounds=search_rounds,
            cp1_result=cp1,
            cp2_result=cp2,
            cp3_result=cp3,
            duration_seconds=duration,
        )
        self._save_answer_pack(session_dir, pack)
        await self._report_progress(
            progress_callback,
            "answer_completed",
            {
                "session_id": session.id,
                "duration_seconds": round(duration, 2),
                "verified_hypotheses": len(verified_above),
            },
        )
        return pack

    @staticmethod
    async def _report_progress(
        callback: ProgressCallback | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            maybe = callback(event, payload)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception as exc:
            logger.debug("progress_callback_failed: %s", exc)

    async def _stage_gate(
        self,
        stage: str,
        result: Any,
        *,
        critical: bool = True,
        reason: str = "",
        context: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Validate that a pipeline stage produced non-empty output.

        Args:
            stage: Stage identifier (e.g. "sources", "generation").
            result: The stage output to check.
            critical: If True, raise on empty; if False, log warning only.
            reason: Human-readable failure description.
            context: Diagnostic data included in exception and progress event.
            progress_callback: Optional callback to emit stage_failed event.
        """
        is_empty = result is None or (isinstance(result, (list, dict)) and len(result) == 0)
        if not is_empty:
            return

        msg = reason or f"Stage '{stage}' produced no output"
        ctx = context or {}

        if critical:
            logger.error("pipeline_stage_failed: stage=%s reason=%s context=%s", stage, msg, ctx)
            await self._report_progress(
                progress_callback, "stage_failed",
                {"stage": stage, "reason": msg, **ctx},
            )
            raise PipelineStageError(stage, msg, ctx)
        else:
            logger.warning("pipeline_stage_warning: stage=%s reason=%s context=%s", stage, msg, ctx)

    def _save_stage(self, stage: str, data: Any) -> None:
        if self._session:
            self.session_manager.save_stage_data(self._session.id, stage, data)

    @staticmethod
    def _resolve_mode_search_params(cfg: AnswerConfig) -> tuple[int, int]:
        """Resolve effective SEARCH depth/breadth from Answer mode."""
        depth = max(0, int(cfg.search_depth))
        breadth = max(1, int(cfg.search_breadth))

        if cfg.mode == "quick":
            depth = min(depth, 2)
            breadth = min(breadth, 3)
        elif cfg.mode == "exhaustive":
            depth = max(depth, 4)
            breadth = max(breadth, 8)

        return depth, breadth

    @staticmethod
    def _build_saga_budget(total_budget: float):
        """Build CognitiveBudget from AnswerConfig budget + settings allocation."""
        from x_creative.saga.budget import CognitiveBudget

        settings = get_settings()
        alloc = settings.saga_cognitive_budget_allocation
        reserve_ratio = max(0.0, min(1.0, float(alloc.emergency_reserve) / 100.0))
        stage_total = max(1e-6, 1.0 - reserve_ratio)
        stage_allocation = {
            "domain_audit": float(alloc.domain_audit) / (100.0 * stage_total),
            "biso_monitoring": float(alloc.biso_monitor) / (100.0 * stage_total),
            "search_monitoring": float(alloc.search_monitor) / (100.0 * stage_total),
            "verify_monitoring": float(alloc.verify_monitor) / (100.0 * stage_total),
            "adversarial": float(alloc.adversarial) / (100.0 * stage_total),
            "global_review": float(alloc.global_review) / (100.0 * stage_total),
        }

        # Guard against rounding drift in user config percentages.
        ratio_sum = sum(stage_allocation.values())
        if ratio_sum <= 0:
            stage_allocation = {
                "domain_audit": 0.10,
                "biso_monitoring": 0.15,
                "search_monitoring": 0.15,
                "verify_monitoring": 0.20,
                "adversarial": 0.25,
                "global_review": 0.15,
            }
        elif abs(ratio_sum - 1.0) > 1e-6:
            stage_allocation = {
                key: value / ratio_sum for key, value in stage_allocation.items()
            }

        return CognitiveBudget(
            total_budget=max(1.0, float(total_budget)),
            reserve_ratio=reserve_ratio,
            stage_allocation=stage_allocation,
        )

    @staticmethod
    def _save_answer_pack(session_dir: Path, pack: AnswerPack) -> None:
        """Persist answer markdown and JSON payload into session directory."""
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "answer.md").write_text(pack.answer_md, encoding="utf-8")
            (session_dir / "answer.json").write_text(
                json.dumps(pack.answer_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("failed_to_save_answer_pack: %s", exc)

    @staticmethod
    def _cp1_constraint_audit(problem_frame: ProblemFrame) -> dict[str, Any]:
        """CP-1: decidability/conflict audit for constraint set."""
        from x_creative.saga.constraint_checker import detect_conflicting_constraints

        settings = get_settings()
        constraints = [c.strip() for c in problem_frame.constraints if c.strip()]
        constraints.extend(
            c.text.strip()
            for c in problem_frame.structured_constraints
            if c.text.strip()
        )
        lowered = [c.lower() for c in constraints]
        duplicates = sorted({text for text in lowered if lowered.count(text) > 1})
        conflicts = detect_conflicting_constraints(
            constraints,
            similarity_threshold=settings.constraint_similarity_threshold,
        )
        vague_markers = (
            "good",
            "better",
            "best",
            "reasonable",
            "appropriate",
            "optimize",
            "improve",
            "高质量",
            "更好",
            "优化",
        )
        decidable_pattern = re.compile(
            r"(must|should|avoid|no|not|only|within|at least|at most|>=|<=|>|<|必须|不得|禁止|避免|至少|最多|\d)",
            re.IGNORECASE,
        )
        undecidable = [
            c for c in constraints
            if any(marker in c.lower() for marker in vague_markers)
            and not decidable_pattern.search(c)
        ]
        over_budget = len(constraints) > settings.max_constraints
        return {
            "constraint_count": len(constraints),
            "duplicate_constraints": duplicates,
            "conflict_pairs": conflicts.conflict_pairs,
            "undecidable_constraints": undecidable,
            "over_budget": over_budget,
            "max_constraints": settings.max_constraints,
            "passed": (
                not duplicates
                and not conflicts.has_conflicts
                and not undecidable
                and not over_budget
            ),
        }

    @staticmethod
    def _cp2_target_domain_audit(problem_frame: ProblemFrame, target_plugin: Any) -> dict[str, Any]:
        """CP-2: target-domain decision consistency and rationale audit."""
        target_domain_id = str(getattr(target_plugin, "id", "")).strip() or "general"
        target_desc = " ".join([
            str(getattr(target_plugin, "name", "")),
            str(getattr(target_plugin, "description", "")),
        ]).lower()
        hint_id = None
        hint_conf = None
        if problem_frame.domain_hint:
            hint_id = problem_frame.domain_hint.get("domain_id")
            hint_conf = problem_frame.domain_hint.get("confidence")

        text = " ".join([
            problem_frame.description or "",
            problem_frame.objective or "",
            " ".join(problem_frame.constraints or []),
        ]).lower()
        tokens = {
            token
            for token in re.findall(r"[a-z][a-z_]{2,}", text)
            if token not in {"with", "from", "that", "this", "have", "will"}
        }
        plugin_tokens = {
            token
            for token in re.findall(r"[a-z][a-z_]{2,}", target_desc)
            if token not in {"with", "from", "that", "this", "have", "will"}
        }
        matched_tokens = sorted(tokens & plugin_tokens)
        keyword_alignment_score = (
            round(len(matched_tokens) / max(1, len(plugin_tokens)), 3)
            if plugin_tokens
            else 0.0
        )

        matches_hint = hint_id is None or hint_id == target_domain_id
        high_conflict = bool(
            hint_id is not None
            and hint_id != target_domain_id
            and (hint_conf or 0.0) >= 0.7
        )
        warnings: list[str] = []
        if high_conflict:
            warnings.append("high_confidence_hint_mismatch")
        if keyword_alignment_score < 0.05 and target_domain_id != "general":
            warnings.append("low_keyword_alignment")
        return {
            "hint_domain_id": hint_id,
            "hint_confidence": hint_conf,
            "resolved_domain_id": target_domain_id,
            "matches_hint": matches_hint,
            "keyword_alignment_score": keyword_alignment_score,
            "matched_tokens": matched_tokens[:20],
            "warnings": warnings,
            "passed": not warnings,
        }

    @staticmethod
    def _cp3_source_bias_audit(source_domains: list[Any], target_domain_id: str) -> dict[str, Any]:
        """CP-3: source-domain diversity and bias audit."""
        ids = [str(getattr(domain, "id", "")).strip() for domain in source_domains]
        ids = [domain_id for domain_id in ids if domain_id]
        counts = Counter(ids)
        selected_count = len(ids)
        unique_count = len(set(ids))
        dominant_domain = ""
        dominant_ratio = 0.0
        if counts:
            dominant_domain, dominant_count = counts.most_common(1)[0]
            dominant_ratio = dominant_count / selected_count

        unique_ratio = unique_count / selected_count if selected_count else 1.0
        entropy = 0.0
        if selected_count > 0 and unique_count > 1:
            probs = [count / selected_count for count in counts.values()]
            entropy_raw = -sum(p * math.log2(p) for p in probs if p > 0)
            entropy = entropy_raw / math.log2(unique_count)

        bias_flags: list[str] = []
        if selected_count and dominant_ratio > 0.5:
            bias_flags.append("dominant_domain_bias")
        if selected_count >= 4 and unique_ratio < 0.6:
            bias_flags.append("insufficient_diversity")
        if target_domain_id in counts and selected_count:
            if counts[target_domain_id] / selected_count > 0.4:
                bias_flags.append("target_domain_overrepresented")

        return {
            "selected_count": selected_count,
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 3),
            "dominant_domain": dominant_domain,
            "dominant_ratio": round(dominant_ratio, 3),
            "normalized_entropy": round(entropy, 3),
            "bias_flags": bias_flags,
            "passed": len(bias_flags) == 0,
        }

    @staticmethod
    def _derive_checkpoint_directives(
        cp1_result: dict[str, Any],
        cp2_result: dict[str, Any],
        cp3_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Map CP-1/2/3 audit findings into Slow-Agent-style directives."""
        directives: list[dict[str, Any]] = []

        if cp1_result.get("conflict_pairs"):
            directives.append(
                {
                    "directive_type": DirectiveType.ADD_CONSTRAINT.value,
                    "reason": "CP-1 detected conflicting constraints",
                    "confidence": 0.9,
                    "priority": 2,
                    "checkpoint": "cp1",
                    "payload": {
                        "constraint": "resolve conflicting constraints before final ranking",
                        "conflict_pairs": cp1_result.get("conflict_pairs", [])[:5],
                    },
                }
            )
        if cp1_result.get("undecidable_constraints"):
            directives.append(
                {
                    "directive_type": DirectiveType.ADD_CONSTRAINT.value,
                    "reason": "CP-1 found undecidable constraints",
                    "confidence": 0.75,
                    "priority": 4,
                    "checkpoint": "cp1",
                    "payload": {
                        "constraint": "all constraints must be operationally testable",
                        "undecidable_constraints": cp1_result.get(
                            "undecidable_constraints", []
                        )[:5],
                    },
                }
            )

        warnings = list(cp2_result.get("warnings", []))
        if "high_confidence_hint_mismatch" in warnings:
            directives.append(
                {
                    "directive_type": DirectiveType.ADJUST_SEARCH_PARAMS.value,
                    "reason": "CP-2 high-confidence domain-hint mismatch",
                    "confidence": 0.8,
                    "priority": 3,
                    "checkpoint": "cp2",
                    "payload": {"search_breadth": 7},
                }
            )

        bias_flags = list(cp3_result.get("bias_flags", []))
        if bias_flags:
            directives.append(
                {
                    "directive_type": DirectiveType.ADJUST_SEARCH_PARAMS.value,
                    "reason": "CP-3 detected source-domain bias",
                    "confidence": 0.8,
                    "priority": 3,
                    "checkpoint": "cp3",
                    "payload": {"search_breadth": 7},
                }
            )

        return directives

    @staticmethod
    def _apply_checkpoint_directives(
        problem_frame: ProblemFrame,
        search_config: SearchConfig,
        directives: list[dict[str, Any]],
    ) -> tuple[ProblemFrame, SearchConfig]:
        """Apply actionable CP directives before entering BISO/SEARCH/VERIFY."""
        constraints = list(problem_frame.constraints or [])
        structured_constraints = list(problem_frame.structured_constraints or [])
        normalized_constraints = {
            re.sub(r"\s+", " ", text.strip().lower())
            for text in constraints
            if text.strip()
        }
        normalized_constraints.update(
            re.sub(r"\s+", " ", c.text.strip().lower())
            for c in structured_constraints
            if c.text.strip()
        )

        for directive in directives:
            dtype = str(directive.get("directive_type", ""))
            payload = directive.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}

            if dtype == DirectiveType.ADD_CONSTRAINT.value:
                constraint = str(payload.get("constraint", "")).strip()
                normalized = re.sub(r"\s+", " ", constraint.lower()) if constraint else ""
                if constraint and normalized not in normalized_constraints:
                    constraints.append(constraint)
                    structured_constraints.append(
                        ConstraintSpec(
                            text=constraint,
                            origin="risk_refinement",
                        )
                    )
                    normalized_constraints.add(normalized)
                continue

            if dtype == DirectiveType.ADJUST_SEARCH_PARAMS.value:
                for key, value in payload.items():
                    if hasattr(search_config, key):
                        setattr(search_config, key, value)

        if (
            constraints != list(problem_frame.constraints or [])
            or structured_constraints != list(problem_frame.structured_constraints or [])
        ):
            problem_frame = problem_frame.model_copy(
                update={
                    "constraints": constraints,
                    "structured_constraints": structured_constraints,
                }
            )
        return problem_frame, search_config
