# tests/unit/test_ck_coordinator.py
"""Tests for CKCoordinator."""

import pytest
from unittest.mock import MagicMock

from x_creative.saga.ck_coordinator import CKCoordinator, CKPhase, CKTrigger
from x_creative.saga.events import EventBus
from x_creative.saga.state import SharedCognitionState
from x_creative.saga.budget import CognitiveBudget


class TestCKCoordinator:
    def test_initial_phase_is_concept_expansion(self) -> None:
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
        )
        assert ck.phase == CKPhase.CONCEPT_EXPANSION

    def test_evaluate_no_transition_initially(self) -> None:
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
        )
        result = ck.evaluate_transition()
        assert result is None

    def test_get_directive_none_initially(self) -> None:
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
        )
        directive = ck.get_directive()
        assert directive is None

    def test_min_phase_duration_prevents_rapid_switching(self) -> None:
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
            min_phase_duration_s=1000.0,
        )
        ck._triggers.append(MagicMock(trigger_type="evidence_gap", severity=1.0))
        result = ck.evaluate_transition()
        assert result is None or ck.phase == CKPhase.CONCEPT_EXPANSION


class TestCoveragePlateauTrigger:
    def test_coverage_plateau_triggers_k_expansion(self) -> None:
        """coverage_plateau trigger with severity >= threshold should cause C->K transition."""
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
            min_phase_duration_s=0.0,  # disable anti-oscillation for test
            coverage_plateau_threshold=2,
        )
        assert ck.phase == CKPhase.CONCEPT_EXPANSION

        ck.add_trigger(CKTrigger(trigger_type="coverage_plateau", severity=3))
        result = ck.evaluate_transition()
        assert result == CKPhase.KNOWLEDGE_EXPANSION

    def test_coverage_plateau_below_threshold_no_transition(self) -> None:
        """coverage_plateau with severity below threshold should NOT trigger transition."""
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
            min_phase_duration_s=0.0,
            coverage_plateau_threshold=5,
        )
        ck.add_trigger(CKTrigger(trigger_type="coverage_plateau", severity=3))
        result = ck.evaluate_transition()
        assert result is None

    def test_coverage_plateau_respects_max_k_expansion(self) -> None:
        """coverage_plateau should NOT trigger if max K expansions reached."""
        ck = CKCoordinator(
            event_bus=EventBus(),
            state=SharedCognitionState(target_domain_id="test"),
            budget=CognitiveBudget(),
            min_phase_duration_s=0.0,
            coverage_plateau_threshold=1,
            max_k_expansion_per_session=0,  # already exhausted
        )
        ck.add_trigger(CKTrigger(trigger_type="coverage_plateau", severity=5))
        result = ck.evaluate_transition()
        assert result is None
