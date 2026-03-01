"""Tests for strict constraint compliance audit + revise loop in TalkerReasonerSolver."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from x_creative.core.types import ConstraintSpec, Hypothesis, ProblemFrame
from x_creative.saga.belief import BeliefState
from x_creative.saga.constraint_compliance import _normalise_item


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


class TestNormaliseItem:
    """Tests for _normalise_item verdict/field normalisation."""

    @pytest.mark.parametrize("raw_verdict,expected", [
        ("pass", "pass"),
        ("Pass", "pass"),
        ("PASS", "pass"),
        ("passed", "pass"),
        ("satisfied", "pass"),
        ("compliant", "pass"),
        ("fail", "fail"),
        ("Failed", "fail"),
        ("violated", "fail"),
        ("non-compliant", "fail"),
        ("unknown", "unknown"),
        ("partial", "unknown"),
        ("unclear", "unknown"),
        ("gibberish", "unknown"),
    ])
    def test_verdict_normalisation(self, raw_verdict: str, expected: str) -> None:
        item = {"id": "C1", "text": "test", "verdict": raw_verdict}
        result = _normalise_item(item)
        assert result["verdict"] == expected

    def test_constraint_text_alias(self) -> None:
        item = {"id": "C1", "constraint_text": "must do X", "verdict": "pass"}
        result = _normalise_item(item)
        assert result["text"] == "must do X"

    def test_description_alias(self) -> None:
        item = {"id": "C1", "description": "must do X", "verdict": "fail"}
        result = _normalise_item(item)
        assert result["text"] == "must do X"

    def test_missing_text_falls_back_to_id(self) -> None:
        item = {"id": "C1", "verdict": "pass"}
        result = _normalise_item(item)
        assert result["text"] == "C1"

    def test_existing_text_preserved(self) -> None:
        item = {"id": "C1", "text": "original", "constraint_text": "alt", "verdict": "pass"}
        result = _normalise_item(item)
        assert result["text"] == "original"


@pytest.mark.asyncio
async def test_audit_tolerates_non_canonical_verdicts() -> None:
    """audit_user_constraints succeeds when LLM returns uppercase/synonym verdicts."""
    from x_creative.saga.constraint_compliance import audit_user_constraints

    router = _ScriptedRouter(
        responses=[
            '{"overall_pass": true, "items": [{"id":"C1","constraint_text":"must do X","verdict":"Passed","rationale":"ok"}]}',
        ]
    )
    problem = ProblemFrame(
        description="test",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(text="must do X", origin="user", type="hard", priority="critical", weight=1.0),
        ],
    )
    report = await audit_user_constraints(router, problem, "# Solution\n")
    assert report.overall_pass is True
    assert report.items[0].verdict == "pass"


@pytest.mark.asyncio
async def test_audit_extracts_json_from_markdown_wrapped_response() -> None:
    """audit_user_constraints handles JSON wrapped in markdown fences with trailing commentary."""
    from x_creative.saga.constraint_compliance import audit_user_constraints

    response = (
        "Here is my analysis:\n\n```json\n"
        '{"overall_pass": true, "items": [{"id":"C1","text":"must do X","verdict":"pass","rationale":"ok"}]}\n'
        "```\n\nThe strategy {mentioned above} fully complies."
    )
    router = _ScriptedRouter(responses=[response])
    problem = ProblemFrame(
        description="test",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(text="must do X", origin="user", type="hard", priority="critical", weight=1.0),
        ],
    )
    report = await audit_user_constraints(router, problem, "# Solution\n")
    assert report.overall_pass is True
    assert report.items[0].verdict == "pass"


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
