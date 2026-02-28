"""SAGA Coordinator — orchestrates Fast and Slow Agent lifecycle.

Manages initialization, parallel execution, and result merging
for the dual-process cognitive architecture.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import structlog

from x_creative.config.settings import get_settings
from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Domain, Hypothesis, ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.budget import CognitiveBudget
from x_creative.saga.ck_coordinator import CKCoordinator
from x_creative.saga.events import (
    DirectiveType,
    EventBus,
    EventType,
    FastAgentEvent,
    ProgressCallback,
    SlowAgentDirective,
)
from x_creative.saga.fast_agent import FastAgent
from x_creative.saga.detectors import (
    DimensionCollinearityDetector,
    ScoreCompressionDetector,
    ShallowRewriteDetector,
    SourceDomainBiasDetector,
    StructureCollapseDetector,
)
from x_creative.saga.auditors import DomainConstraintAuditor
from x_creative.saga.evaluation import (
    AdversarialChallengeEvaluator,
    PatternMemoryEvaluator,
)
from x_creative.saga.memory import PatternMemory
from x_creative.saga.slow_agent import BaseAuditor, BaseDetector, BaseEvaluator, SlowAgent
from x_creative.saga.state import CognitionAlert, SharedCognitionState

logger = structlog.get_logger()


class AuditReport(BaseModel):
    """Lightweight audit report from Slow Agent."""

    events_processed: int = Field(default=0, description="Total events processed")
    alerts_raised: int = Field(default=0, description="Total alerts raised")
    alert_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Alert counts by severity:type",
    )
    interventions: int = Field(default=0, description="Total interventions made")
    intervention_log: list[dict] = Field(
        default_factory=list,
        description="Detailed intervention records",
    )
    budget_spent: float = Field(default=0.0, description="Cognitive budget consumed")
    budget_remaining: float = Field(default=0.0, description="Cognitive budget remaining")


class SAGAResult(BaseModel):
    """Complete result of a SAGA-driven creative generation run."""

    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Final hypothesis list (potentially re-ranked by Slow Agent)",
    )
    audit_report: AuditReport = Field(
        default_factory=AuditReport,
        description="Slow Agent audit report",
    )
    alerts: list[CognitionAlert] = Field(
        default_factory=list,
        description="All alerts raised during execution",
    )
    intervention_log: list[dict] = Field(
        default_factory=list,
        description="All Slow Agent intervention records",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metrics",
    )
    budget_spent: float = Field(default=0.0, description="Total cognitive budget spent")
    budget_total: float = Field(default=0.0, description="Total cognitive budget allocated")
    elapsed_seconds: float = Field(default=0.0, description="Total execution time")


class SAGACoordinator:
    """SAGA orchestrator managing Fast/Slow Agent lifecycle.

    Responsibilities:
    1. Initialize shared state, event bus, cognitive budget
    2. Launch Fast Agent and Slow Agent in parallel
    3. Wait for execution to complete
    4. Merge final output (pipeline results + audit report)
    5. Persist audit logs to session directory

    Args:
        engine: CreativityEngine instance (creates one if not provided).
        budget: Cognitive budget (uses default if not provided).
        session_dir: Path for persisting events and reports.
    """

    def __init__(
        self,
        engine: CreativityEngine | None = None,
        budget: CognitiveBudget | None = None,
        session_dir: Any | None = None,
        enable_ck: bool = False,
    ) -> None:
        self._engine = engine or CreativityEngine()
        self._budget = budget or CognitiveBudget()
        self._session_dir = session_dir

        # C-K Coordinator (optional)
        self._ck_coordinator: CKCoordinator | None = None
        if enable_ck:
            # Deferred: event_bus and state are created per-run in run(),
            # so we store the flag and instantiate in run().
            pass
        self._enable_ck = enable_ck
        self._last_mome_coverage: float = 0.0
        self._coverage_stagnation_count: int = 0

    @property
    def ck_coordinator(self) -> CKCoordinator | None:
        """Return the CKCoordinator instance, or None if C-K is disabled."""
        return self._ck_coordinator

    async def run(
        self,
        problem: ProblemFrame,
        config: SearchConfig | None = None,
        source_domains: list[Domain] | None = None,
        initial_directives: list[dict[str, Any]] | None = None,
        progress_callback: ProgressCallback | None = None,
        target_plugin: TargetDomainPlugin | None = None,
    ) -> SAGAResult:
        """Execute a SAGA-driven creative generation.

        Initializes state and event bus, then runs Fast Agent and
        Slow Agent in parallel. Waits for Fast Agent to complete,
        signals Slow Agent to stop, and merges results.

        Args:
            problem: The research problem framing.
            config: Search configuration (uses defaults if not provided).
            initial_directives: Optional directives pre-seeded before Fast/Slow
                agents start (used by AnswerEngine CP checkpoints).

        Returns:
            SAGAResult containing hypotheses, audit report, and metrics.
        """
        start_time = time.time()

        logger.info(
            "SAGA Coordinator starting",
            target_domain=problem.target_domain,
        )

        # 1. Initialize shared state
        state = SharedCognitionState(
            target_domain_id=problem.target_domain,
            target_plugin=target_plugin,
        )

        # 2. Initialize event bus (with optional persistence)
        event_bus = EventBus(
            session_dir=self._session_dir,
            progress_callback=progress_callback,
        )

        # 3. Initialize budget
        budget = CognitiveBudget(
            total_budget=self._budget.total_budget,
            strategy=self._budget.strategy,
            stage_allocation=self._budget.stage_allocation,
            reserve_ratio=self._budget.reserve_ratio,
        )

        # 3b. Initialize C-K Coordinator (optional)
        if self._enable_ck:
            from x_creative.config.settings import get_settings as _get_settings
            ck_settings = _get_settings()
            self._ck_coordinator = CKCoordinator(
                event_bus=event_bus,
                state=state,
                budget=budget,
                min_phase_duration_s=ck_settings.ck_min_phase_duration_s,
                max_k_expansion_per_session=ck_settings.ck_max_k_expansion_per_session,
                coverage_plateau_threshold=ck_settings.ck_coverage_plateau_threshold,
                evidence_gap_threshold=ck_settings.ck_evidence_gap_threshold,
            )

        # 4. Seed initial directives (e.g., AnswerEngine CP-1/2/3 checkpoints)
        for raw in initial_directives or []:
            directive = self._build_initial_directive(raw)
            if directive is not None:
                await event_bus.emit_directive(directive)

        # 5. Create agents
        fast_agent = FastAgent(
            engine=self._engine,
            event_bus=event_bus,
            state=state,
            post_verify_directive_grace_s=2.0,
            checkpoint_callback=(
                self._on_fast_agent_checkpoint
                if self._enable_ck
                else None
            ),
        )
        slow_agent = SlowAgent(
            event_bus=event_bus,
            state=state,
            budget=budget,
            detectors=self._create_detectors(),
            auditors=self._create_auditors(),
            evaluators=self._create_evaluators(),
            router=getattr(self._engine, "_router", None),
        )

        # 4b. Validate cross-model family routing for adversarial evaluation
        self._validate_cross_family_routing()

        # 6. Launch both agents in parallel
        fast_task = asyncio.create_task(
            fast_agent.run_pipeline(problem, config, source_domains=source_domains)
        )
        slow_task = asyncio.create_task(
            slow_agent.run()
        )

        # 7. Wait for Fast Agent to complete
        hypotheses: list[Hypothesis] = []
        try:
            hypotheses = await fast_task
        except Exception as e:
            logger.error(
                "Fast Agent failed",
                error=str(e),
            )

        # 8. Signal Slow Agent to stop and wait
        slow_agent.request_stop()
        try:
            await asyncio.wait_for(slow_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Slow Agent did not stop within timeout")
            slow_task.cancel()
            try:
                await asyncio.wait_for(slow_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        except Exception as e:
            logger.warning("Slow Agent error on shutdown", error=str(e))

        # 9. Build audit report
        report_data = slow_agent.generate_report()
        audit_report = AuditReport(**report_data)

        elapsed = time.time() - start_time

        # 10. Merge results
        result = SAGAResult(
            hypotheses=hypotheses,
            audit_report=audit_report,
            alerts=state.active_alerts,
            intervention_log=slow_agent.intervention_log,
            metrics={
                "events_processed": report_data.get("events_processed", 0),
                "hypothesis_count": len(hypotheses),
                "stage": state.current_stage,
            },
            budget_spent=budget.spent,
            budget_total=budget.total_budget,
            elapsed_seconds=round(elapsed, 2),
        )

        logger.info(
            "SAGA Coordinator completed",
            hypothesis_count=len(hypotheses),
            alerts=len(state.active_alerts),
            interventions=len(slow_agent.intervention_log),
            elapsed_seconds=round(elapsed, 2),
        )

        try:
            await self._engine.close()
        except Exception as exc:
            logger.warning("Failed to close engine after SAGA run", error=str(exc))

        return result

    @staticmethod
    def _build_initial_directive(raw: dict[str, Any]) -> SlowAgentDirective | None:
        """Normalize AnswerEngine checkpoint directives into SlowAgentDirective."""
        dtype_raw = str(raw.get("directive_type", "")).strip()
        if not dtype_raw:
            return None
        try:
            dtype = DirectiveType(dtype_raw)
        except ValueError:
            return None

        payload = raw.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        reason = str(raw.get("reason", "checkpoint directive")).strip() or "checkpoint directive"
        try:
            confidence = float(raw.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))
        try:
            priority = int(raw.get("priority", 5))
        except (TypeError, ValueError):
            priority = 5
        priority = max(1, min(10, priority))
        return SlowAgentDirective(
            directive_type=dtype,
            reason=reason,
            confidence=confidence,
            payload=payload,
            priority=priority,
        )

    def _evaluate_ck_checkpoint(self) -> None:
        """Evaluate CK phase transition at a pipeline checkpoint.

        Checks MOME coverage stagnation and calls evaluate_transition().
        """
        if self._ck_coordinator is None:
            return

        # Check MOME coverage stagnation
        mome_archive = getattr(self._engine._search, "_mome_archive", None)
        if mome_archive is not None:
            coverage = mome_archive.coverage_ratio

            if coverage <= self._last_mome_coverage and self._last_mome_coverage > 0.0:
                self._coverage_stagnation_count += 1
                from x_creative.saga.ck_coordinator import CKTrigger
                self._ck_coordinator.add_trigger(
                    CKTrigger(
                        trigger_type="coverage_plateau",
                        severity=float(self._coverage_stagnation_count),
                        details=f"Coverage stagnant at {coverage:.2%} for {self._coverage_stagnation_count} checkpoint(s)",
                    )
                )
            else:
                self._coverage_stagnation_count = 0
            self._last_mome_coverage = coverage

        # Evaluate transition
        new_phase = self._ck_coordinator.evaluate_transition()
        if new_phase is not None:
            logger.info(
                "ck_phase_transition_in_saga",
                new_phase=new_phase.value,
            )

    async def _on_fast_agent_checkpoint(self, checkpoint_id: str) -> None:
        """Run CK checkpoint evaluation on each Fast-Agent checkpoint."""
        self._evaluate_ck_checkpoint()

    def _create_detectors(self) -> list[BaseDetector]:
        """Create baseline detector instances."""
        settings = get_settings()
        return [
            ScoreCompressionDetector(
                std_threshold=settings.score_compression_threshold,
            ),
            StructureCollapseDetector(),
            DimensionCollinearityDetector(
                corr_threshold=settings.dimension_colinearity_threshold,
            ),
            SourceDomainBiasDetector(),
            ShallowRewriteDetector(),
        ]

    def _create_auditors(self) -> list[BaseAuditor]:
        """Create baseline auditor instances."""
        router = getattr(self._engine, "_router", None)
        return [DomainConstraintAuditor(router=router)]

    def _create_evaluators(self) -> list[BaseEvaluator]:
        """Create baseline evaluator instances."""
        router = getattr(self._engine, "_router", None)

        # Build SemanticHasher if embedding API is available
        hasher = self._create_semantic_hasher()

        memory_path = self._resolve_pattern_memory_path()
        memory = PatternMemory(storage_path=memory_path, hasher=hasher)

        return [
            AdversarialChallengeEvaluator(router=router),
            PatternMemoryEvaluator(memory=memory),
        ]

    def _resolve_pattern_memory_path(self) -> Path:
        """Resolve persistent path for cross-session pattern memory."""
        if isinstance(self._session_dir, (str, Path)):
            return Path(self._session_dir) / "pattern_memory.json"

        try:
            from x_creative.config.settings import get_settings

            settings = get_settings()
            return settings.cache_dir / "saga" / "pattern_memory.json"
        except Exception:
            # Fallback for environments where settings cannot be loaded.
            return Path("local_data") / "saga" / "pattern_memory.json"

    def _create_semantic_hasher(self) -> Any:
        """Create SemanticHasher from settings if embedding API is available."""
        try:
            from x_creative.config.settings import get_settings
            from x_creative.hkg.embeddings import EmbeddingClient
            from x_creative.saga.memory.semantic_hash import SemanticHasher

            settings = get_settings()
            api_key = settings.openrouter.api_key.get_secret_value()
            if not api_key:
                logger.warning(
                    "SemanticHasher unavailable — no embedding API key; "
                    "PatternMemory will use local semantic hash fallback"
                )
                return None
            client = EmbeddingClient(
                api_key=api_key,
                base_url=settings.openrouter.base_url,
            )
            return SemanticHasher(embedding_service=client)
        except Exception as exc:
            logger.warning(
                "SemanticHasher unavailable — PatternMemory will use local semantic hash fallback",
                error=str(exc),
            )
            return None

    @staticmethod
    def _model_family(model_id: str) -> str:
        """Extract model family prefix (e.g. 'anthropic' from 'anthropic/claude-sonnet-4')."""
        return model_id.split("/")[0].lower() if "/" in model_id else model_id.lower()

    def _validate_cross_family_routing(self) -> None:
        """Enforce cross-family routing for adversarial evaluation.

        Theory §3.4 requires the adversarial red team to use a different model
        family than Fast Agent to prevent reward hacking.
        """
        router = getattr(self._engine, "_router", None)
        if router is None:
            return
        try:
            gen_model = router.get_model("creativity")
            adv_model = router.get_model("saga_adversarial")
            gen_family = self._model_family(gen_model)
            adv_family = self._model_family(adv_model)
            if gen_family == adv_family:
                raise ValueError(
                    "Invalid SAGA routing: saga_adversarial must use a different "
                    "model family than creativity"
                )
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(
                "Failed to validate cross-family routing; skipping strict check",
                error=str(exc),
            )
