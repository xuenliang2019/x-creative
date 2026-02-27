"""Tests for SAGA FastAgent directive handling and dual-verify flow."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.core.types import Hypothesis, HypothesisScores, ProblemFrame, SearchConfig
from x_creative.core.types import VerifyStatus
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.events import DirectiveType, EventBus, EventType, SlowAgentDirective
from x_creative.saga.fast_agent import FastAgent
from x_creative.saga.state import SharedCognitionState


def _hypothesis(hypothesis_id: str, final_score: float | None = None) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        description=f"Hypothesis {hypothesis_id}",
        source_domain="queueing_theory",
        source_structure="queue_dynamics",
        analogy_explanation="queue analogy",
        observable="queue_depth / trade_rate",
        final_score=final_score,
    )


@pytest.fixture
def sample_problem() -> ProblemFrame:
    return ProblemFrame(
        description="How to design a viral open source tool?",
        target_domain="open_source_development",
    )


@pytest.mark.asyncio
async def test_adjust_search_params_directive_updates_config(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1")]
    scored = [_hypothesis("h1", final_score=7.2)]

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)) as mock_search, \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.ADJUST_SEARCH_PARAMS,
                reason="broaden search breadth",
                confidence=0.8,
                payload={"search_breadth": 7},
            )
        )

        cfg = SearchConfig(search_depth=1, search_breadth=2)
        await agent.run_pipeline(sample_problem, cfg)

        called_config = mock_search.call_args.kwargs["config"]
        assert called_config.search_breadth == 7


@pytest.mark.asyncio
async def test_inject_challenge_and_rescore_directives(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]

    def _scored_batch() -> list[Hypothesis]:
        return [_hypothesis("h1", final_score=7.0), _hypothesis("h2", final_score=6.5)]

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(
             engine,
             "score_and_verify_batch",
             AsyncMock(side_effect=[_scored_batch(), _scored_batch()]),
         ) as mock_score:
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.INJECT_CHALLENGE,
                reason="red-team checkpoint",
                confidence=0.75,
                payload={
                    "hypothesis_id": "h1",
                    "challenge_type": "counterexample",
                    "severity": "high",
                },
            )
        )
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.RESCORE_BATCH,
                reason="re-score after challenge",
                confidence=0.7,
            )
        )

        final = await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

        assert mock_score.call_count == 2
        assert len(state.adversarial_challenges) >= 1
        score_by_id = {h.id: float(h.final_score or 0.0) for h in final}
        assert score_by_id["h1"] < 7.0


@pytest.mark.asyncio
async def test_verify_batch_event_sees_scored_state(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("raw_1")]
    scored = [_hypothesis("scored_1", final_score=7.4)]
    seen_pool_ids: list[list[str]] = []

    original_emit = event_bus.emit_event

    async def spy_emit(event):  # noqa: ANN001
        if event.event_type == EventType.VERIFY_BATCH_SCORED:
            seen_pool_ids.append(
                [str(item.get("id")) for item in state.hypotheses_pool if isinstance(item, dict)]
            )
        await original_emit(event)

    with patch.object(event_bus, "emit_event", side_effect=spy_emit), \
         patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    assert seen_pool_ids
    assert seen_pool_ids[0] == ["scored_1"]


@pytest.mark.asyncio
async def test_search_round_completed_events_emitted(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h1", final_score=7.1), _hypothesis("h2", final_score=6.8)]

    async def fake_run_search(initial_hypotheses, config, on_round_complete=None):  # noqa: ANN001
        pool = list(initial_hypotheses)
        if on_round_complete is not None:
            await on_round_complete(1, pool, 0)
            await on_round_complete(2, pool, 0)
        return pool

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=2, search_breadth=1))

    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    round_events = [
        event for event in events if event.event_type == EventType.SEARCH_ROUND_COMPLETED
    ]
    assert len(round_events) == 2


@pytest.mark.asyncio
async def test_round_checkpoint_processes_directives(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1")]
    scored = [_hypothesis("h1", final_score=7.1)]

    async def fake_run_search(initial_hypotheses, config, on_round_complete=None):  # noqa: ANN001
        pool = list(initial_hypotheses)
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.ADJUST_SEARCH_PARAMS,
                reason="tighten threshold from SEARCH checkpoint",
                confidence=0.8,
                payload={"prune_threshold": 7.0},
            )
        )
        if on_round_complete is not None:
            await on_round_complete(1, pool, 0)
        # Directive should be applied at SEARCH round checkpoint.
        assert config.prune_threshold == 7.0
        return pool

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))


@pytest.mark.asyncio
async def test_hkg_events_are_emitted(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h1", final_score=7.1), _hypothesis("h2", final_score=6.8)]

    async def fake_run_search(  # noqa: ANN001
        initial_hypotheses,
        config,
        on_round_complete=None,
        on_hkg_event=None,
    ):
        pool = list(initial_hypotheses)
        if on_round_complete is not None:
            await on_round_complete(1, pool, 0)
        if on_hkg_event is not None:
            await on_hkg_event("hkg_path_found", {"path_count": 2})
            await on_hkg_event("hkg_expansion_created", {"new_count": 1})
        return pool

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    event_types = {event.event_type for event in events}
    assert EventType.HKG_PATH_FOUND in event_types
    assert EventType.HKG_EXPANSION_CREATED in event_types


@pytest.mark.asyncio
async def test_mapping_and_verify_status_events_are_emitted(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [
        _hypothesis("h1").model_copy(update={"mapping_table": []}),
    ]
    scored = [
        _hypothesis("h1", final_score=0.0).model_copy(
            update={
                "verify_status": VerifyStatus.ESCALATED,
                "position_consistency": False,
                "injection_detected": True,
            }
        )
    ]

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    event_types = {event.event_type for event in events}
    assert EventType.MAPPING_TABLE_MISSING in event_types
    assert EventType.VERIFY_ESCALATED in event_types
    assert EventType.VERIFY_POSITION_INCONSISTENT in event_types
    assert EventType.VERIFY_INJECTION_DETECTED in event_types


@pytest.mark.asyncio
async def test_mapping_quality_scoring_runs_before_search_and_emits_padding_event(
    sample_problem: ProblemFrame,
) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1")]
    scored = [_hypothesis("h1", final_score=7.0)]

    async def fake_score_mapping(hypotheses, on_event=None):  # noqa: ANN001
        hypotheses[0].mapping_quality = 7.5
        if on_event is not None:
            await on_event(
                EventType.MAPPING_PADDING_SUSPECTED.value,
                {"hypothesis_id": hypotheses[0].id},
            )

    async def fake_run_search(initial_hypotheses, config, **kwargs):  # noqa: ANN001, ANN003
        assert initial_hypotheses[0].mapping_quality == 7.5
        return list(initial_hypotheses)

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(
             engine,
             "_score_mapping_quality_with_events",
             AsyncMock(side_effect=fake_score_mapping),
         ) as mock_map_score, \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    assert mock_map_score.await_count == 1
    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    event_types = {event.event_type for event in events}
    assert EventType.MAPPING_PADDING_SUSPECTED in event_types


@pytest.mark.asyncio
async def test_blend_transform_events_are_emitted(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h1", final_score=7.1), _hypothesis("h2", final_score=6.8)]

    async def fake_run_search(  # noqa: ANN001
        initial_hypotheses,
        config,
        on_round_complete=None,
        on_hkg_event=None,
    ):
        pool = list(initial_hypotheses)
        if on_round_complete is not None:
            await on_round_complete(1, pool, 0)
        if on_hkg_event is not None:
            await on_hkg_event("blend_expand_completed", {"new_count": 1})
            await on_hkg_event("transform_proposed", {"hypothesis_id": "h1"})
            await on_hkg_event(
                "transform_accepted_or_rejected",
                {"hypothesis_id": "h1", "transform_status": "ACCEPTED"},
            )
            await on_hkg_event("transform_space_applied", {"new_count": 1})
        return pool

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    event_types = {event.event_type for event in events}
    assert EventType.BLEND_EXPAND_COMPLETED in event_types
    assert EventType.TRANSFORM_PROPOSED in event_types
    assert EventType.TRANSFORM_ACCEPTED_OR_REJECTED in event_types
    assert EventType.TRANSFORM_SPACE_APPLIED in event_types


@pytest.mark.asyncio
async def test_constraint_management_events_are_emitted(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h1", final_score=7.1), _hypothesis("h2", final_score=6.8)]

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        # repeated/similar => reframe signal
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.ADD_CONSTRAINT,
                reason="refine risk",
                confidence=0.8,
                payload={"constraint": "must use daily frequency data"},
            )
        )
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.ADD_CONSTRAINT,
                reason="refine risk",
                confidence=0.8,
                payload={"constraint": "must use daily data"},
            )
        )

        # explicit conflict => conflict signal
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.ADD_CONSTRAINT,
                reason="red-team conflict",
                confidence=0.9,
                payload={"constraint": "do not use daily frequency data"},
            )
        )

        # force budget growth alert (> default max_constraints=15)
        for i in range(20):
            await event_bus.emit_directive(
                SlowAgentDirective(
                    directive_type=DirectiveType.ADD_CONSTRAINT,
                    reason=f"constraint {i}",
                    confidence=0.6,
                    payload={"constraint": f"must include check {i}"},
                )
            )

        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    event_types = {event.event_type for event in events}
    assert EventType.CONSTRAINT_REFRAME_TRIGGERED in event_types
    assert EventType.CONSTRAINT_CONFLICT_DETECTED in event_types
    assert EventType.CONSTRAINT_SET_GROWTH_ALERT in event_types


@pytest.mark.asyncio
async def test_critical_flagged_hypotheses_are_rejected(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h1", final_score=8.5), _hypothesis("h2", final_score=6.7)]

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await event_bus.emit_directive(
            SlowAgentDirective(
                directive_type=DirectiveType.FLAG_HYPOTHESIS,
                reason="critical constraint violated",
                confidence=0.9,
                payload={
                    "hypothesis_id": "h1",
                    "violations": ["critical:no_lookahead_bias"],
                },
            )
        )
        final = await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    assert [hyp.id for hyp in final] == ["h2"]


@pytest.mark.asyncio
async def test_critical_flagged_hypothesis_is_dropped_before_verify(
    sample_problem: ProblemFrame,
) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h2", final_score=6.7)]

    async def fake_run_search(initial_hypotheses, config, on_round_complete=None):  # noqa: ANN001
        pool = list(initial_hypotheses)
        if on_round_complete is not None:
            await event_bus.emit_directive(
                SlowAgentDirective(
                    directive_type=DirectiveType.FLAG_HYPOTHESIS,
                    reason="critical constraint violated",
                    confidence=0.9,
                    payload={
                        "hypothesis_id": "h1",
                        "violations": ["critical:no_lookahead_bias"],
                    },
                )
            )
            await on_round_complete(1, pool, 0)
        return pool

    async def fake_score_and_verify_batch(expanded, problem_frame, **_kwargs):  # noqa: ANN001
        ids = [h.id for h in expanded]
        assert "h1" not in ids
        return scored

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(
             engine,
             "score_and_verify_batch",
             AsyncMock(side_effect=fake_score_and_verify_batch),
         ):
        final = await agent.run_pipeline(
            sample_problem,
            SearchConfig(search_depth=1, search_breadth=2),
        )

    assert [hyp.id for hyp in final] == ["h2"]


@pytest.mark.asyncio
async def test_verify_completed_critical_flag_is_applied_before_return(
    sample_problem: ProblemFrame,
) -> None:
    """Critical FLAG emitted after VERIFY_COMPLETED should still reject immediately."""
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(
        engine=engine,
        event_bus=event_bus,
        state=state,
        post_verify_directive_grace_s=0.2,
    )

    raw = [_hypothesis("h1"), _hypothesis("h2")]
    scored = [_hypothesis("h1", final_score=8.5), _hypothesis("h2", final_score=6.7)]

    async def emit_late_critical_flag() -> None:
        async for event in event_bus.subscribe_events():
            if event.event_type == EventType.VERIFY_COMPLETED:
                await event_bus.emit_directive(
                    SlowAgentDirective(
                        directive_type=DirectiveType.FLAG_HYPOTHESIS,
                        reason="late critical constraint violation",
                        confidence=0.9,
                        payload={
                            "hypothesis_id": "h1",
                            "violations": ["critical:no_lookahead_bias"],
                        },
                    )
                )
                break

    watcher = asyncio.create_task(emit_late_critical_flag())

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        final = await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    event_bus.stop()
    try:
        await asyncio.wait_for(watcher, timeout=1.0)
    except Exception:  # pragma: no cover - defensive for async teardown
        watcher.cancel()

    assert [hyp.id for hyp in final] == ["h2"]


@pytest.mark.asyncio
async def test_verify_stage_passes_problem_frame_to_dual_verify(
    sample_problem: ProblemFrame,
) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1")]
    scored = [_hypothesis("h1", final_score=7.3)]

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)) as mock_score:
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))

    assert mock_score.call_count >= 1
    assert mock_score.call_args.kwargs.get("problem_frame") == sample_problem


@pytest.mark.asyncio
async def test_search_stage_sets_problem_frame_for_hkg(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1")]
    scored = [_hypothesis("h1", final_score=7.3)]

    async def fake_run_search(initial_hypotheses, config, **kwargs):  # noqa: ANN001, ANN003
        assert engine._search._problem_frame == sample_problem
        return list(initial_hypotheses)

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", side_effect=fake_run_search), \
         patch.object(engine, "score_and_verify_batch", AsyncMock(return_value=scored)):
        await agent.run_pipeline(sample_problem, SearchConfig(search_depth=1, search_breadth=1))


@pytest.mark.asyncio
async def test_verify_filter_prefers_final_score_threshold(sample_problem: ProblemFrame) -> None:
    engine = CreativityEngine()
    event_bus = EventBus()
    state = SharedCognitionState()
    agent = FastAgent(engine=engine, event_bus=event_bus, state=state)

    raw = [_hypothesis("h1")]
    low_final_high_composite = Hypothesis(
        id="h1",
        description="low final high composite",
        source_domain="queueing_theory",
        source_structure="queue_dynamics",
        analogy_explanation="queue analogy",
        observable="queue_depth / trade_rate",
        final_score=4.0,
        scores=HypothesisScores(
            divergence=9.0,
            testability=9.0,
            rationale=9.0,
            robustness=9.0,
            feasibility=9.0,
        ),
    )

    with patch.object(engine._biso, "generate_all_analogies", AsyncMock(return_value=raw)), \
         patch.object(engine._search, "run_search", AsyncMock(return_value=raw)), \
         patch.object(
             engine,
             "score_and_verify_batch",
             AsyncMock(return_value=[low_final_high_composite]),
         ):
        final = await agent.run_pipeline(
            sample_problem,
            SearchConfig(search_depth=1, search_breadth=1, prune_threshold=5.0),
        )

    assert final == []


class TestHardGateFiltering:
    """#7: ADD_CONSTRAINT hard_gates must actually filter hypotheses."""

    def _make_agent(self, hard_gates: list[str] | None = None) -> FastAgent:
        from x_creative.saga.events import EventBus
        from x_creative.saga.state import SharedCognitionState
        bus = EventBus()
        state = SharedCognitionState()
        if hard_gates:
            state.evaluation_adjustments.hard_gates = hard_gates
        engine = MagicMock()
        return FastAgent(engine=engine, event_bus=bus, state=state)

    def _make_hyp(self, hyp_id: str, description: str) -> Hypothesis:
        return Hypothesis(
            id=hyp_id, description=description,
            source_domain="test", source_structure="test",
            analogy_explanation="test", observable="safe_obs",
        )

    def test_hard_gate_rejects_violating_hypothesis(self) -> None:
        agent = self._make_agent(["no_lookahead_bias"])
        violating = self._make_hyp("bad", "predict using future t+1 returns")
        clean = self._make_hyp("good", "use historical momentum signal")
        result = agent._apply_hard_gates([violating, clean])
        ids = [h.id for h in result]
        assert "bad" not in ids
        assert "good" in ids

    def test_hard_gate_passes_clean_hypothesis(self) -> None:
        agent = self._make_agent(["no_lookahead_bias"])
        h1 = self._make_hyp("h1", "use historical returns as momentum proxy")
        h2 = self._make_hyp("h2", "entropy-based volatility indicator")
        result = agent._apply_hard_gates([h1, h2])
        assert len(result) == 2

    def test_multiple_hard_gates_all_checked(self) -> None:
        agent = self._make_agent(["no_lookahead_bias", "no_survivorship_bias"])
        h1 = self._make_hyp("h1", "uses look-ahead information")
        h2 = self._make_hyp("h2", "clean hypothesis with no issues")
        result = agent._apply_hard_gates([h1, h2])
        assert len(result) == 1
        assert result[0].id == "h2"

    def test_no_gates_passes_all(self) -> None:
        agent = self._make_agent(None)
        h1 = self._make_hyp("h1", "any description")
        result = agent._apply_hard_gates([h1])
        assert len(result) == 1
