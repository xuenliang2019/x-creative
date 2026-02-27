"""Tests for strict constraint compliance audit + revise loop in TalkerReasonerSolver."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from x_creative.core.types import ConstraintSpec, Hypothesis, ProblemFrame
from x_creative.saga.belief import BeliefState


class _DummyCompletion:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish_reason = "stop"
        self.model = "dummy/model"


class _ScriptedRouter:
    """Router that returns scripted responses per call for deterministic tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, task, messages, temperature=None, max_tokens=None, **kwargs):  # noqa: ANN001, ANN002, ARG002
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        self.calls.append((str(task), prompt))
        if not self._responses:
            raise RuntimeError("No scripted responses left")
        return _DummyCompletion(self._responses.pop(0))

    async def close(self):  # noqa: D401
        return None


class _DummyReasoner:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass

    async def reason(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return BeliefState()


class _DummyTalker:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass

    async def generate(self, belief, problem):  # noqa: ANN001, ARG002
        return "# Solution v1\n\nMissing required part.\n"


@pytest.mark.asyncio
async def test_solver_revises_until_constraints_pass() -> None:
    from x_creative.saga.solve import TalkerReasonerSolver

    # Script: audit(fail) -> revise(v2) -> audit(pass)
    router = _ScriptedRouter(
        responses=[
            '{"overall_pass": false, "items": [{"id":"C1","text":"must do X","verdict":"fail","rationale":"missing","suggested_fix":"add X"}]}',
            "# Solution v2\n\nNow includes X.\n",
            '{"overall_pass": true, "items": [{"id":"C1","text":"must do X","verdict":"pass","rationale":"present"}]}',
        ]
    )

    problem = ProblemFrame(
        description="test",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(text="must do X", origin="user", type="hard", priority="critical", weight=1.0),
        ],
    )

    solver = TalkerReasonerSolver(router=router)

    with (
        patch("x_creative.saga.solve.Reasoner", _DummyReasoner),
        patch("x_creative.saga.solve.Talker", _DummyTalker),
    ):
        result = await solver.run(
            problem=problem,
            verify_markdown="",
            hypotheses=[
                Hypothesis(
                    id="h1",
                    description="d",
                    source_domain="s",
                    source_structure="st",
                    analogy_explanation="a",
                    observable="o",
                )
            ],
            auto_refine=False,
        )

    assert "Solution v2" in result.solution_markdown
    # Ensure we did at least one revise call.
    assert any(task == "constraint_compliance_revision" for task, _ in router.calls)


@pytest.mark.asyncio
async def test_solver_raises_when_constraints_never_pass() -> None:
    from x_creative.saga.constraint_compliance import UserConstraintComplianceError
    from x_creative.saga.solve import TalkerReasonerSolver

    # Script: audit(fail) -> revise -> audit(fail) -> revise -> audit(fail) ...
    router = _ScriptedRouter(
        responses=[
            '{"overall_pass": false, "items": [{"id":"C1","text":"must do X","verdict":"fail","rationale":"missing","suggested_fix":"add X"}]}',
            "# Solution v2\n\nStill missing.\n",
            '{"overall_pass": false, "items": [{"id":"C1","text":"must do X","verdict":"fail","rationale":"still missing","suggested_fix":"add X"}]}',
            "# Solution v3\n\nStill missing.\n",
            '{"overall_pass": false, "items": [{"id":"C1","text":"must do X","verdict":"fail","rationale":"still missing","suggested_fix":"add X"}]}',
        ]
    )

    problem = ProblemFrame(
        description="test",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(text="must do X", origin="user", type="hard", priority="critical", weight=1.0),
        ],
    )

    solver = TalkerReasonerSolver(router=router)
    with (
        patch("x_creative.saga.solve.Reasoner", _DummyReasoner),
        patch("x_creative.saga.solve.Talker", _DummyTalker),
    ):
        with pytest.raises(UserConstraintComplianceError):
            await solver.run(
                problem=problem,
                verify_markdown="",
                hypotheses=[
                    Hypothesis(
                        id="h1",
                        description="d",
                        source_domain="s",
                        source_structure="st",
                        analogy_explanation="a",
                        observable="o",
                    )
                ],
                auto_refine=False,
            )
