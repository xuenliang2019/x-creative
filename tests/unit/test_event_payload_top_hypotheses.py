"""Tests that event payloads include top_hypotheses in both SAGA and non-SAGA paths."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from x_creative.core.types import Hypothesis, ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.events import EventBus, EventType
from x_creative.saga.fast_agent import FastAgent
from x_creative.saga.state import SharedCognitionState


def _hyp(hid: str, final_score: float | None = None) -> Hypothesis:
    return Hypothesis(
        id=hid,
        description=f"Hypothesis {hid}",
        source_domain="test_domain",
        source_structure="struct",
        analogy_explanation="analogy",
        observable="obs",
        final_score=final_score,
    )


@pytest.fixture
def problem() -> ProblemFrame:
    return ProblemFrame(
        description="Test question",
        target_domain="test_domain",
    )


# ---------------------------------------------------------------------------
# SAGA path (FastAgent)
# ---------------------------------------------------------------------------


class TestSAGAEventPayloads:
    """Verify FastAgent emits top_hypotheses in key event payloads."""

    @pytest.mark.asyncio
    async def test_pipeline_events_contain_top_hypotheses(self, problem: ProblemFrame) -> None:
        engine = CreativityEngine()
        event_bus = EventBus()
        state = SharedCognitionState()
        agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

        raw = [_hyp("h1"), _hyp("h2")]
        scored = [_hyp("h1", final_score=7.0), _hyp("h2", final_score=6.5)]

        emitted_payloads: dict[str, dict[str, Any]] = {}
        original_emit = event_bus.emit_event

        async def _capture(event):
            emitted_payloads[event.event_type.value] = dict(event.payload or {})
            await original_emit(event)

        with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
             patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
             patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)), \
             patch.object(event_bus, "emit_event", side_effect=_capture):
            await agent.run_pipeline(problem, SearchConfig(search_depth=1))

        # BISO_COMPLETED
        assert "top_hypotheses" in emitted_payloads.get(EventType.BISO_COMPLETED.value, {}), \
            f"BISO_COMPLETED missing top_hypotheses. payload={emitted_payloads.get(EventType.BISO_COMPLETED.value)}"

        # SEARCH_COMPLETED
        assert "top_hypotheses" in emitted_payloads.get(EventType.SEARCH_COMPLETED.value, {}), \
            f"SEARCH_COMPLETED missing top_hypotheses"

        # VERIFY_BATCH_SCORED
        assert "top_hypotheses" in emitted_payloads.get(EventType.VERIFY_BATCH_SCORED.value, {}), \
            f"VERIFY_BATCH_SCORED missing top_hypotheses"

        # VERIFY_COMPLETED
        assert "top_hypotheses" in emitted_payloads.get(EventType.VERIFY_COMPLETED.value, {}), \
            f"VERIFY_COMPLETED missing top_hypotheses"

    @pytest.mark.asyncio
    async def test_top_hypotheses_are_lists_of_dicts(self, problem: ProblemFrame) -> None:
        engine = CreativityEngine()
        event_bus = EventBus()
        state = SharedCognitionState()
        agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

        raw = [_hyp("h1")]
        scored = [_hyp("h1", final_score=7.0)]

        emitted_payloads: dict[str, dict[str, Any]] = {}

        async def _capture(event):
            emitted_payloads[event.event_type.value] = dict(event.payload or {})

        with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
             patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
             patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)), \
             patch.object(event_bus, "emit_event", side_effect=_capture):
            await agent.run_pipeline(problem, SearchConfig(search_depth=1))

        for event_name in [
            EventType.BISO_COMPLETED.value,
            EventType.SEARCH_COMPLETED.value,
            EventType.VERIFY_BATCH_SCORED.value,
            EventType.VERIFY_COMPLETED.value,
        ]:
            top = emitted_payloads.get(event_name, {}).get("top_hypotheses")
            assert isinstance(top, list), f"{event_name}: top_hypotheses is not a list"
            if top:
                assert isinstance(top[0], dict), f"{event_name}: items are not dicts"
                assert "id" in top[0], f"{event_name}: item missing 'id'"
                assert "score" in top[0], f"{event_name}: item missing 'score'"


# ---------------------------------------------------------------------------
# Non-SAGA path (CreativityEngine.generate)
# ---------------------------------------------------------------------------


class TestNonSAGAEventPayloads:
    """Verify CreativityEngine.generate emits top_hypotheses via _report callback."""

    @pytest.mark.asyncio
    async def test_generate_reports_contain_top_hypotheses(self, problem: ProblemFrame) -> None:
        engine = CreativityEngine()
        raw = [_hyp("h1"), _hyp("h2")]
        scored = [_hyp("h1", final_score=7.0), _hyp("h2", final_score=6.5)]

        reported: dict[str, dict[str, Any]] = {}

        async def _capture_callback(event: str, payload: dict[str, Any]) -> None:
            reported[event] = dict(payload)

        with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
             patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
             patch.object(engine._verify, "score_batch", AsyncMock(return_value=scored)), \
             patch.object(engine, "_verify_batch_dual_model", AsyncMock(return_value=[])), \
             patch.object(engine, "_merge_dual_verification", return_value=scored):
            await engine.generate(
                problem=problem,
                config=SearchConfig(search_depth=1),
                progress_callback=_capture_callback,
            )

        # biso_completed
        assert "top_hypotheses" in reported.get("biso_completed", {}), \
            f"biso_completed missing top_hypotheses"

        # search_completed
        assert "top_hypotheses" in reported.get("search_completed", {}), \
            f"search_completed missing top_hypotheses"

        # verify_completed
        assert "top_hypotheses" in reported.get("verify_completed", {}), \
            f"verify_completed missing top_hypotheses"

    @pytest.mark.asyncio
    async def test_search_round_completed_contains_top_hypotheses(self, problem: ProblemFrame) -> None:
        engine = CreativityEngine()
        raw = [_hyp("h1")]
        scored = [_hyp("h1", final_score=7.0)]

        reported: dict[str, dict[str, Any]] = {}

        async def _capture_callback(event: str, payload: dict[str, Any]) -> None:
            reported[event] = dict(payload)

        # Make run_search invoke on_round_complete callback
        async def _fake_search(initial_hypotheses, config, on_round_complete=None, **kw):
            if on_round_complete:
                await on_round_complete(1, initial_hypotheses, 0)
            return initial_hypotheses

        with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
             patch.object(engine._search, "run_search", side_effect=_fake_search), \
             patch.object(engine._verify, "score_batch", AsyncMock(return_value=scored)), \
             patch.object(engine, "_verify_batch_dual_model", AsyncMock(return_value=[])), \
             patch.object(engine, "_merge_dual_verification", return_value=scored):
            await engine.generate(
                problem=problem,
                config=SearchConfig(search_depth=1),
                progress_callback=_capture_callback,
            )

        assert "top_hypotheses" in reported.get("search_round_completed", {}), \
            f"search_round_completed missing top_hypotheses"
