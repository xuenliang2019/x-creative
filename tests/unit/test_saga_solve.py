"""Tests for SAGA solve workflow."""

import pytest

from x_creative.core.types import Hypothesis, HypothesisScores, NoveltyVerdict, ProblemFrame


class _DummyCompletion:
    def __init__(self, content: str):
        self.content = content


class _DummyRouter:
    def __init__(self):
        self.calls = 0

    async def complete(self, task, messages, temperature=None, max_tokens=None, **kwargs):  # noqa: ANN001
        self.calls += 1
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)

        if "all claims must cite evidence ids" in prompt_text.lower():
            return _DummyCompletion(
                "# Final Solution\n\n"
                "Use queue-based prioritization [E1].\n\n"
                "## References\n- [E1] https://example.com/evidence-1\n"
            )

        return _DummyCompletion("# Final Solution\n\nUse queue-based prioritization.\n")

    async def close(self):
        return None


class _DummySearchValidator:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        return None

    async def validate(self, hypothesis: Hypothesis, preliminary_score: float) -> NoveltyVerdict:
        return NoveltyVerdict(
            score=min(10.0, preliminary_score + 0.5),
            searched=True,
            similar_works=[],
            novelty_analysis=f"Validated for {hypothesis.id}",
        )


@pytest.mark.asyncio
async def test_saga_solver_collects_evidence_and_generates_solution(monkeypatch):
    from x_creative.saga.solve import SAGASolver

    monkeypatch.setattr("x_creative.saga.solve.SearchValidator", _DummySearchValidator)

    problem = ProblemFrame(
        description="Design a practical plan to reduce support backlog",
        target_domain="general",
    )
    hypotheses = [
        Hypothesis(
            id="hyp_001",
            description="Queueing-inspired ticket prioritization",
            source_domain="queueing_theory",
            source_structure="priority_queue",
            analogy_explanation="Map queue disciplines to support ticket routing",
            observable="weighted_ticket_resolution_time",
            scores=HypothesisScores(
                divergence=6.0,
                testability=7.0,
                rationale=7.0,
                robustness=6.5,
                feasibility=7.5,
            ),
        ),
    ]

    solver = SAGASolver(router=_DummyRouter())
    result = await solver.run(
        problem=problem,
        verify_markdown="# Verification Results\n\nIdea: prioritize high-risk tickets.\n",
        hypotheses=hypotheses,
        max_ideas=3,
        max_web_results=5,
    )

    assert "Final Solution" in result.solution_markdown
    assert result.metrics["ideas_used"] == 1
    assert result.metrics["evidence_items"] >= 1
    assert result.audit_report["events_processed"] > 0


@pytest.mark.asyncio
async def test_saga_solver_applies_constraint_when_citations_missing(monkeypatch):
    from x_creative.saga.solve import SAGASolver

    monkeypatch.setattr("x_creative.saga.solve.SearchValidator", _DummySearchValidator)

    problem = ProblemFrame(
        description="Create a feasible implementation roadmap",
        target_domain="general",
    )
    hypotheses = [
        Hypothesis(
            id="hyp_002",
            description="Flow-control strategy for work intake",
            source_domain="networking",
            source_structure="congestion_control",
            analogy_explanation="Map congestion windows to intake controls",
            observable="intake_rate_vs_resolution_rate",
            scores=HypothesisScores(
                divergence=6.5,
                testability=7.0,
                rationale=6.8,
                robustness=6.1,
                feasibility=7.2,
            ),
        ),
    ]

    router = _DummyRouter()
    solver = SAGASolver(router=router)
    result = await solver.run(
        problem=problem,
        verify_markdown="# Verification Results\n\nIdea: control intake rate by queue pressure.\n",
        hypotheses=hypotheses,
        max_ideas=3,
        max_web_results=5,
    )

    assert router.calls >= 2
    assert result.applied_constraints
    assert "all claims must cite evidence ids" in " ".join(result.applied_constraints).lower()
    assert "[E1]" in result.solution_markdown


@pytest.mark.asyncio
async def test_saga_solver_reports_progress_events(monkeypatch):
    from x_creative.saga.solve import SAGASolver

    monkeypatch.setattr("x_creative.saga.solve.SearchValidator", _DummySearchValidator)

    progress_events: list[tuple[str, dict]] = []

    async def _progress(event: str, payload: dict):
        progress_events.append((event, payload))

    problem = ProblemFrame(
        description="Provide an actionable delivery plan",
        target_domain="general",
    )
    hypotheses = [
        Hypothesis(
            id="hyp_003",
            description="Kanban-style flow limits for delivery work",
            source_domain="operations",
            source_structure="flow_limit",
            analogy_explanation="Map WIP limits to project task flow",
            observable="wip_limit_breach_rate",
            scores=HypothesisScores(
                divergence=6.2,
                testability=7.1,
                rationale=6.7,
                robustness=6.2,
                feasibility=7.6,
            ),
        ),
    ]

    solver = SAGASolver(router=_DummyRouter(), progress_callback=_progress)
    await solver.run(
        problem=problem,
        verify_markdown="# Verification Results\n\nIdea: limit WIP for throughput.\n",
        hypotheses=hypotheses,
        max_ideas=3,
        max_web_results=5,
    )

    event_names = [name for name, _ in progress_events]
    assert "run_started" in event_names
    assert "ideas_selected" in event_names
    assert "idea_evidence_collected" in event_names
    assert "web_evidence_collected" in event_names
    assert "draft_generated" in event_names
    assert "run_completed" in event_names

    idea_progress_payload = next(
        payload for name, payload in progress_events if name == "idea_evidence_collected"
    )
    assert idea_progress_payload["idea_index"] == 1
    assert idea_progress_payload["idea_total"] == 1
    assert idea_progress_payload["progress_percent"] == 100.0
    assert idea_progress_payload["stage_seconds"] >= 0.0

    run_completed_payload = next(
        payload for name, payload in progress_events if name == "run_completed"
    )
    stage_seconds = run_completed_payload.get("stage_seconds", {})
    assert stage_seconds["idea_selection"] >= 0.0
    assert stage_seconds["web_evidence"] >= 0.0
    assert stage_seconds["draft_generation"] >= 0.0


@pytest.mark.asyncio
async def test_talker_reasoner_solver_emits_solve_round_event() -> None:
    from x_creative.saga.events import EventBus, EventType
    from x_creative.saga.solve import TalkerReasonerSolver

    event_bus = EventBus()
    solver = TalkerReasonerSolver(router=_DummyRouter(), event_bus=event_bus)

    await solver._emit_saga_event(
        event_type=EventType.SOLVE_ROUND_COMPLETED,
        payload={"outer_round": 1, "high_risk_count": 0},
        metrics={"high_risk_count": 0.0},
    )

    event_bus.stop()
    events = [event async for event in event_bus.subscribe_events()]
    assert len(events) == 1
    assert events[0].event_type == EventType.SOLVE_ROUND_COMPLETED
    assert events[0].stage == "solve"


class TestSolveConstraintBudget:
    """Test that constraint budget functions are available and work correctly."""

    def test_constraint_budget_applied(self) -> None:
        """Verify constraints don't exceed budget during refinement."""
        from x_creative.saga.constraint_checker import apply_constraint_budget
        from x_creative.core.types import ConstraintSpec

        constraints = [
            ConstraintSpec(text=f"C{i}", origin="risk_refinement", weight=0.3)
            for i in range(20)
        ]
        result = apply_constraint_budget(constraints, max_constraints=10)
        assert len(result) <= 10

    def test_constraint_budget_in_settings(self) -> None:
        """Verify max_constraints setting exists."""
        from x_creative.config.settings import Settings
        s = Settings()
        assert hasattr(s, 'max_constraints')
        assert s.max_constraints == 15
