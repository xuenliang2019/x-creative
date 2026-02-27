"""SAGA Slow Agent — metacognitive overseer.

Runs asynchronously in parallel with Fast Agent, subscribing to the
event stream and applying detection, auditing, and evaluation layers:

1. Detectors (lightweight): run on every event, identify anomalies
2. Auditors (medium): triggered by critical alerts
3. Evaluators (deep): triggered at checkpoint events, consume cognitive budget
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from x_creative.saga.budget import BudgetPolicy, CognitiveBudget
from x_creative.saga.events import (
    EventBus,
    EventType,
    FastAgentEvent,
    SlowAgentDirective,
)
from x_creative.saga.state import CognitionAlert, SharedCognitionState

logger = structlog.get_logger()


def _safe_float(value: Any) -> float | None:
    """Parse numeric score values safely."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _composite_score_from_item(item: dict[str, Any]) -> float | None:
    """Compute track-1 composite score from dict payload when available."""
    scores = item.get("scores")
    if not isinstance(scores, dict):
        return None
    divergence = _safe_float(scores.get("divergence"))
    testability = _safe_float(scores.get("testability"))
    rationale = _safe_float(scores.get("rationale"))
    robustness = _safe_float(scores.get("robustness"))
    feasibility = _safe_float(scores.get("feasibility"))
    if None in {divergence, testability, rationale, robustness, feasibility}:
        return None
    return (
        0.21 * float(divergence)
        + 0.26 * float(testability)
        + 0.21 * float(rationale)
        + 0.17 * float(robustness)
        + 0.15 * float(feasibility)
    )


def hypothesis_rank_score(item: dict[str, Any]) -> tuple[int, float]:
    """Rank key: prefer final_score, fallback to composite score, then quick_score."""
    final_score = _safe_float(item.get("final_score"))
    if final_score is not None:
        return (3, final_score)
    composite_score = _composite_score_from_item(item)
    if composite_score is not None:
        return (2, composite_score)
    quick_score = _safe_float(item.get("quick_score"))
    if quick_score is not None:
        return (1, quick_score)
    return (0, 0.0)


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors.

    Detectors are lightweight checks run on every event. They should
    be fast (no LLM calls) and return alerts when anomalies are found.

    Subclass in saga/detectors/ for Phase 2+ implementations.
    """

    @abstractmethod
    async def detect(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,
    ) -> list[CognitionAlert]:
        """Run detection on an event.

        Args:
            event: The Fast Agent event to analyze.
            state: Current shared cognition state.

        Returns:
            List of alerts (empty if no anomalies detected).
        """
        ...


class BaseAuditor(ABC):
    """Abstract base class for auditors.

    Auditors perform medium-depth checks, typically triggered when
    critical alerts are raised. May involve lightweight LLM calls.

    Subclass in saga/auditors/ for Phase 2+ implementations.
    """

    @abstractmethod
    async def audit(
        self,
        event: FastAgentEvent,
        alerts: list[CognitionAlert],
        state: SharedCognitionState,
    ) -> list[SlowAgentDirective]:
        """Run audit in response to alerts.

        Args:
            event: The triggering event.
            alerts: Alerts that triggered this audit.
            state: Current shared cognition state.

        Returns:
            List of directives to send to Fast Agent.
        """
        ...


class BaseEvaluator(ABC):
    """Abstract base class for evaluators.

    Evaluators perform deep review at checkpoint events, consuming
    cognitive budget. Typically involve strong-model LLM calls.

    Subclass in saga/evaluation/ for Phase 2+ implementations.
    """

    @abstractmethod
    async def evaluate(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,
        budget: CognitiveBudget,
    ) -> list[SlowAgentDirective]:
        """Run deep evaluation at a checkpoint.

        Args:
            event: The checkpoint event.
            state: Current shared cognition state.
            budget: Cognitive budget to consume from.

        Returns:
            List of directives to send to Fast Agent.
        """
        ...


class SlowAgent:
    """Metacognitive overseer that monitors Fast Agent's execution.

    Subscribes to the event stream and applies three layers of processing:
    1. Detectors — lightweight statistical anomaly detection
    2. Auditors — medium-depth audit on critical alerts
    3. Evaluators — deep review at checkpoint events

    Model strategy (mixed):
    - Checkpoints: lightweight model (Claude Haiku / GPT-4o-mini)
    - Deep audit: strong model (Claude Opus / o3)
    - Adversarial: cross-family model (Gemini 3 Pro)

    Args:
        event_bus: Event bus for receiving events and sending directives.
        state: Shared cognition state.
        budget: Cognitive budget tracker.
        detectors: List of anomaly detectors.
        auditors: List of auditors.
        evaluators: List of evaluators.
        router: Model router for LLM calls (optional in Phase 1).
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: SharedCognitionState,
        budget: CognitiveBudget,
        detectors: list[BaseDetector] | None = None,
        auditors: list[BaseAuditor] | None = None,
        evaluators: list[BaseEvaluator] | None = None,
        router: Any | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._state = state
        self._budget = budget
        self._detectors = detectors or []
        self._auditors = auditors or []
        self._evaluators = evaluators or []
        self._router = router
        self._budget_policy = BudgetPolicy()
        self._stop_requested = False
        self._events_processed = 0
        self.intervention_log: list[dict] = []

    async def run(self) -> None:
        """Main monitoring loop.

        Consumes events from EventBus and applies the three-layer
        processing pipeline until stop is requested.
        """
        logger.info("Slow Agent started")

        async for event in self._event_bus.subscribe_events():
            self._events_processed += 1
            logger.debug(
                "Slow Agent processing event",
                event_type=event.event_type.value,
                stage=event.stage,
                events_processed=self._events_processed,
            )

            try:
                # Layer 1: Lightweight detection (all events)
                alerts = await self._run_detectors(event)

                audited = False
                if alerts:
                    self._state.active_alerts.extend(alerts)
                    logger.info(
                        "Detectors raised alerts",
                        alert_count=len(alerts),
                        severities=[a.severity for a in alerts],
                    )

                    # Layer 2: Audit on critical alerts
                    if any(a.severity == "critical" for a in alerts):
                        await self._run_auditors(event, alerts)
                        audited = True

                # Layer 3: Deep evaluation at checkpoints
                if self._is_checkpoint_event(event):
                    # Proactive audit at checkpoints (skip if already audited)
                    if not audited:
                        await self._run_auditors(event, alerts)
                    await self._run_evaluators(event)

            except Exception as e:
                logger.warning(
                    "Slow Agent error processing event",
                    event_type=event.event_type.value,
                    error=str(e),
                )

        logger.info(
            "Slow Agent stopped",
            events_processed=self._events_processed,
            interventions=len(self.intervention_log),
        )

    async def _run_detectors(
        self, event: FastAgentEvent
    ) -> list[CognitionAlert]:
        """Run all detectors on an event.

        Args:
            event: The event to analyze.

        Returns:
            Combined list of alerts from all detectors.
        """
        all_alerts: list[CognitionAlert] = []
        for detector in self._detectors:
            try:
                alerts = await detector.detect(event, self._state)
                all_alerts.extend(alerts)
            except Exception as e:
                logger.warning(
                    "Detector failed",
                    detector=type(detector).__name__,
                    error=str(e),
                )
        return all_alerts

    async def _run_auditors(
        self,
        event: FastAgentEvent,
        alerts: list[CognitionAlert],
    ) -> None:
        """Run auditors in response to critical alerts.

        Args:
            event: The triggering event.
            alerts: The alerts that triggered auditing.
        """
        critical_alert = any(a.severity == "critical" for a in alerts)
        for auditor in self._auditors:
            stage_key = "domain_audit"
            cost = 1.0
            if not self._budget.spend_stage(
                cost=cost,
                stage=stage_key,
                reason=f"auditor:{type(auditor).__name__}",
                allow_reserve=critical_alert,
            ):
                logger.debug(
                    "Skipping auditor due to stage budget",
                    auditor=type(auditor).__name__,
                    stage=stage_key,
                )
                continue
            try:
                directives = await auditor.audit(event, alerts, self._state)
                for directive in directives:
                    await self._event_bus.emit_directive(directive)
                    self._state.intervention_count += 1
                    self.intervention_log.append({
                        "type": "audit",
                        "directive": directive.directive_type.value,
                        "reason": directive.reason,
                        "timestamp": time.time(),
                    })
            except Exception as e:
                logger.warning(
                    "Auditor failed",
                    auditor=type(auditor).__name__,
                    error=str(e),
                )

    async def _run_evaluators(self, event: FastAgentEvent) -> None:
        """Run evaluators at checkpoint events, gated by budget policy."""
        top_hyp = self._get_top_hypothesis()
        hyp_score = 0.0
        novelty_score = 0.0
        if top_hyp:
            final_score = _safe_float(top_hyp.get("final_score"))
            if final_score is not None:
                hyp_score = final_score
            else:
                hyp_score = (
                    _composite_score_from_item(top_hyp)
                    or _safe_float(top_hyp.get("quick_score"))
                    or 0.0
                )
            novelty_score = _safe_float(top_hyp.get("novelty_score")) or 0.0
        anomaly_detected = any(
            a.severity == "critical" for a in self._state.active_alerts
        )

        for evaluator in self._evaluators:
            stage_key = self._stage_bucket_for_evaluator(evaluator, event)
            should_run, cost = self._budget_policy.should_deep_review(
                hypothesis_score=hyp_score,
                novelty_score=novelty_score,
                anomaly_detected=anomaly_detected,
                budget=self._budget,
            )

            if not should_run:
                logger.debug(
                    "Budget policy declined deep review",
                    evaluator=type(evaluator).__name__,
                    strategy=self._budget.strategy.value,
                )
                continue

            if not self._budget.spend_stage(
                cost=cost,
                stage=stage_key,
                reason=f"evaluator:{type(evaluator).__name__}",
                allow_reserve=anomaly_detected,
            ):
                logger.warning(
                    "Insufficient budget for evaluator stage",
                    evaluator=type(evaluator).__name__,
                    cost=cost,
                    stage=stage_key,
                )
                continue

            try:
                directives = await evaluator.evaluate(
                    event, self._state, self._budget
                )
                for directive in directives:
                    await self._event_bus.emit_directive(directive)
                    self._state.intervention_count += 1
                    self.intervention_log.append({
                        "type": "evaluation",
                        "directive": directive.directive_type.value,
                        "reason": directive.reason,
                        "timestamp": time.time(),
                    })
            except Exception as e:
                logger.warning(
                    "Evaluator failed",
                    evaluator=type(evaluator).__name__,
                    error=str(e),
                )

    @staticmethod
    def _stage_bucket_for_event_stage(stage: str) -> str:
        """Map pipeline stage name to budget stage bucket."""
        normalized = (stage or "").strip().lower()
        if normalized == "biso":
            return "biso_monitoring"
        if normalized == "search":
            return "search_monitoring"
        if normalized == "verify":
            return "verify_monitoring"
        if normalized == "solve":
            return "global_review"
        return "global_review"

    def _stage_bucket_for_evaluator(
        self,
        evaluator: BaseEvaluator,
        event: FastAgentEvent,
    ) -> str:
        """Map evaluator type to stage budget bucket."""
        name = type(evaluator).__name__.lower()
        if "adversarial" in name:
            return "adversarial"
        if "patternmemory" in name:
            return "global_review"
        return self._stage_bucket_for_event_stage(event.stage)

    def _get_top_hypothesis(self) -> dict | None:
        """Get the highest-scoring hypothesis from the pool."""
        candidates = [
            item for item in self._state.hypotheses_pool
            if isinstance(item, dict)
        ]
        if not candidates:
            return None
        return max(candidates, key=hypothesis_rank_score)

    def _is_checkpoint_event(self, event: FastAgentEvent) -> bool:
        """Check if an event is a checkpoint that triggers deep evaluation.

        Args:
            event: The event to check.

        Returns:
            True if this is a checkpoint event.
        """
        return event.event_type in {
            EventType.BISO_COMPLETED,
            EventType.SEARCH_COMPLETED,
            EventType.VERIFY_BATCH_SCORED,
            EventType.VERIFY_COMPLETED,
            EventType.SOLVE_ROUND_COMPLETED,
        }

    def request_stop(self) -> None:
        """Request the Slow Agent to stop after processing remaining events."""
        self._stop_requested = True
        self._event_bus.stop()
        logger.info("Slow Agent stop requested")

    def generate_report(self) -> dict:
        """Generate a summary audit report.

        Returns:
            Dict containing alert summary, interventions, and budget usage.
        """
        alert_summary: dict[str, int] = {}
        for alert in self._state.active_alerts:
            key = f"{alert.severity}:{alert.alert_type}"
            alert_summary[key] = alert_summary.get(key, 0) + 1

        return {
            "events_processed": self._events_processed,
            "alerts_raised": len(self._state.active_alerts),
            "alert_summary": alert_summary,
            "interventions": len(self.intervention_log),
            "intervention_log": self.intervention_log,
            "budget_spent": self._budget.spent,
            "budget_remaining": self._budget.remaining,
        }
