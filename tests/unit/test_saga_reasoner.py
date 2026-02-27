"""Tests for Reasoner ranking and constraint deduplication."""

from unittest.mock import patch

import pytest

from x_creative.core.types import ConstraintSpec, Hypothesis, HypothesisScores, ProblemFrame
from x_creative.saga.belief import BeliefState
from x_creative.saga.reasoner import Reasoner


class _DummyCompletion:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish_reason = "stop"
        self.model = "dummy/model"


class _DummyRouter:
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, task, messages, temperature=None, max_tokens=None, **kwargs):  # noqa: ANN001, ANN002, ARG002
        return _DummyCompletion(self._content)


class _CaptureRouter:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_messages = []

    async def complete(self, task, messages, temperature=None, max_tokens=None, **kwargs):  # noqa: ANN001, ANN002, ARG002
        self.last_messages = messages
        return _DummyCompletion(self._content)


def _hypothesis(
    hypothesis_id: str,
    *,
    score_base: float,
    final_score: float | None,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        description=f"Hypothesis {hypothesis_id}",
        source_domain="queueing_theory",
        source_structure="queue_dynamics",
        analogy_explanation="queue analogy",
        observable="queue_depth / trade_rate",
        scores=HypothesisScores(
            divergence=score_base,
            testability=score_base,
            rationale=score_base,
            robustness=score_base,
            feasibility=score_base,
        ),
        final_score=final_score,
    )


def test_reasoner_ranking_prefers_final_score() -> None:
    reasoner = Reasoner(router=_DummyRouter("[]"))
    low_final_high_composite = _hypothesis("h1", score_base=9.0, final_score=6.0)
    high_final_low_composite = _hypothesis("h2", score_base=5.0, final_score=7.0)

    ranked = sorted(
        [low_final_high_composite, high_final_low_composite],
        key=reasoner._hypothesis_rank_score,
        reverse=True,
    )

    assert reasoner._hypothesis_rank_score(low_final_high_composite) == 6.0
    assert ranked[0].id == "h2"


@pytest.mark.asyncio
async def test_risks_to_constraints_applies_semantic_dedupe() -> None:
    router = _DummyRouter(
        '["方案必须包含市场微观结构验证模块，使用 tick level 数据",'
        '"方案必须包含市场微观结构验证模块，使用tick-level数据",'
        '"所有因子必须经过独立的样本外验证期测试"]'
    )
    reasoner = Reasoner(router=router)

    constraints = await reasoner.risks_to_constraints(
        risks=[{"risk": "缺乏市场微观结构验证", "severity": "high"}],
        existing_constraints=["方案必须包含市场微观结构验证模块，使用tick-level数据"],
    )

    assert len(constraints) == 1
    assert constraints[0].text == "所有因子必须经过独立的样本外验证期测试"
    assert constraints[0].origin == "risk_refinement"


@pytest.mark.asyncio
async def test_solution_planning_injects_compiled_constraints() -> None:
    router = _CaptureRouter(
        '{'
        '"executive_summary":"summary",'
        '"key_insights":["k1"],'
        '"phases":[{"name":"p1","objective":"o","actions":["a1"],"evidence_refs":["E1"],"duration":"1w","success_metric":"m"}],'
        '"dependencies":[],'
        '"tools_and_resources":[]'
        '}'
    )
    reasoner = Reasoner(router=router)
    belief = BeliefState()
    belief.problem_analysis.core_challenge = "core challenge"
    belief.problem_analysis.sub_problems = ["sub-1"]
    belief.cross_validation.synthesis = "synthesis"

    problem = ProblemFrame(
        description="Design a robust implementation plan",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(
                text="must avoid lookahead bias",
                type="hard",
                priority="critical",
            ),
            ConstraintSpec(
                text="prefer out-of-sample validation",
                type="soft",
                priority="high",
            ),
        ],
    )

    await reasoner._step_solution_planning(
        belief=belief,
        problem=problem,
        verify_markdown="",
        selected=[],
        prior_risks=[{"risk": "microstructure validation missing", "severity": "high"}],
    )

    joined_prompt = "\n".join(str(message.get("content", "")) for message in router.last_messages)
    assert "HardCore constraints (must satisfy):" in joined_prompt
    assert "HardCore constraints (repeat):" in joined_prompt
    assert joined_prompt.count("must avoid lookahead bias") >= 2


@pytest.mark.asyncio
async def test_solution_planning_uses_configured_constraint_budget() -> None:
    router = _CaptureRouter(
        "{"
        '"executive_summary":"summary",'
        '"key_insights":["k1"],'
        '"phases":[{"name":"p1","objective":"o","actions":["a1"],"evidence_refs":["E1"],"duration":"1w","success_metric":"m"}],'
        '"dependencies":[],'
        '"tools_and_resources":[]'
        "}"
    )
    reasoner = Reasoner(router=router)
    reasoner._max_constraints = 9
    belief = BeliefState()
    belief.problem_analysis.core_challenge = "core challenge"

    problem = ProblemFrame(
        description="Plan under constraints",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(text=f"constraint-{i}", type="soft", priority="medium")
            for i in range(12)
        ],
    )

    with (
        patch(
            "x_creative.saga.constraint_checker.compile_constraint_activation"
        ) as mock_compile,
        patch(
            "x_creative.saga.constraint_checker.format_constraint_prompt_block",
            return_value="HardCore constraints (must satisfy):\n- constraint-0",
        ),
    ):
        mock_compile.return_value = object()
        await reasoner._step_solution_planning(
            belief=belief,
            problem=problem,
            verify_markdown="",
            selected=[],
        )

    assert mock_compile.call_count == 1
    assert mock_compile.call_args.kwargs["max_constraints"] == 9
