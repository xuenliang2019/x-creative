"""CKCoordinator: C-K Theory dual-space scheduling.

Manages transitions between Concept expansion (C) and Knowledge expansion (K)
phases. Embedded in SAGACoordinator to provide macro-level exploration guidance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from x_creative.saga.budget import CognitiveBudget
    from x_creative.saga.events import EventBus
    from x_creative.saga.state import SharedCognitionState

logger = structlog.get_logger()


class CKPhase(str, Enum):
    """Current phase of the C-K dual-space process."""
    CONCEPT_EXPANSION = "concept_expansion"
    KNOWLEDGE_EXPANSION = "knowledge_expansion"


@dataclass
class CKTrigger:
    """A trigger that may cause phase transition."""
    trigger_type: str
    severity: float = 0.0
    details: str = ""


@dataclass
class CKDirective:
    """A directive from CKCoordinator to the pipeline."""
    action: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CKCoordinator:
    """C-K Theory dual-space coordinator.

    Tracks concept/knowledge expansion phases and triggers transitions:
    - C->K: when concepts have evidence gaps or are non-operationalizable
    - K->C: when new knowledge/mechanisms enable new concept generation

    Anti-oscillation: minimum phase duration prevents rapid switching.
    """

    def __init__(
        self,
        event_bus: "EventBus",
        state: "SharedCognitionState",
        budget: "CognitiveBudget",
        min_phase_duration_s: float = 10.0,
        max_k_expansion_per_session: int = 5,
        coverage_plateau_threshold: int = 2,
        evidence_gap_threshold: float = 5.0,
    ) -> None:
        self._event_bus = event_bus
        self._state = state
        self._budget = budget
        self._min_phase_duration_s = min_phase_duration_s
        self._max_k_expansion = max_k_expansion_per_session
        self._coverage_plateau_threshold = coverage_plateau_threshold
        self._evidence_gap_threshold = evidence_gap_threshold

        self._phase = CKPhase.CONCEPT_EXPANSION
        self._phase_start_time = time.monotonic()
        self._k_expansion_count = 0
        self._triggers: list[CKTrigger] = []
        self._pending_directive: CKDirective | None = None

    @property
    def phase(self) -> CKPhase:
        return self._phase

    def evaluate_transition(self) -> CKPhase | None:
        """Evaluate whether a phase transition should occur.

        Returns the new phase if transitioning, None otherwise.
        """
        elapsed = time.monotonic() - self._phase_start_time
        if elapsed < self._min_phase_duration_s:
            return None

        if not self._triggers:
            return None

        if self._phase == CKPhase.CONCEPT_EXPANSION:
            # Check for evidence gaps or non-operationalizable concepts
            gap_triggers = [
                t for t in self._triggers
                if t.trigger_type in ("evidence_gap", "non_operationalizable")
                and t.severity >= self._evidence_gap_threshold
            ]
            if gap_triggers and self._k_expansion_count < self._max_k_expansion:
                return self._transition_to(CKPhase.KNOWLEDGE_EXPANSION)

            # Check for MOME coverage plateau (stagnation)
            plateau_triggers = [
                t for t in self._triggers
                if t.trigger_type == "coverage_plateau"
                and t.severity >= self._coverage_plateau_threshold
            ]
            if plateau_triggers and self._k_expansion_count < self._max_k_expansion:
                return self._transition_to(CKPhase.KNOWLEDGE_EXPANSION)

        elif self._phase == CKPhase.KNOWLEDGE_EXPANSION:
            # Check for new knowledge that enables concept generation
            k_triggers = [
                t for t in self._triggers
                if t.trigger_type in ("new_mechanism", "new_evidence", "structure_found")
            ]
            if k_triggers:
                return self._transition_to(CKPhase.CONCEPT_EXPANSION)

        return None

    def _transition_to(self, new_phase: CKPhase) -> CKPhase:
        old_phase = self._phase
        self._phase = new_phase
        self._phase_start_time = time.monotonic()
        self._triggers.clear()

        if new_phase == CKPhase.KNOWLEDGE_EXPANSION:
            self._k_expansion_count += 1

        logger.info(
            "ck_phase_transition",
            from_phase=old_phase.value,
            to_phase=new_phase.value,
            k_expansion_count=self._k_expansion_count,
        )

        return new_phase

    def get_directive(self) -> CKDirective | None:
        """Get the current directive (if any) for the pipeline."""
        return self._pending_directive

    def add_trigger(self, trigger: CKTrigger) -> None:
        """Add a transition trigger."""
        self._triggers.append(trigger)
        logger.debug("ck_trigger_added", trigger_type=trigger.trigger_type, severity=trigger.severity)
