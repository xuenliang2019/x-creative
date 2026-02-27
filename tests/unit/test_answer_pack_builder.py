"""Tests for AnswerPackBuilder."""

import pytest

from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Hypothesis, ProblemFrame
from x_creative.session.types import Session


def _make_session():
    return Session(id="test-session-123", topic="How to design a viral open source tool")


def _make_frame():
    return ProblemFrame(
        description="How to design a viral open source tool?",
        target_domain="open_source_development",
        objective="Design a viral open source tool",
    )


def _make_plugin():
    return TargetDomainPlugin(
        id="open_source_development", name="开源软件开发选题", description="Open source development"
    )


def _make_hypothesis(idx: int):
    return Hypothesis(
        id=f"h_{idx}",
        description=f"Hypothesis {idx}",
        source_domain=f"domain_{idx}",
        source_structure=f"structure_{idx}",
        analogy_explanation=f"Analogy {idx}",
        observable=f"measure_{idx}",
    )


class TestAnswerPackBuilder:
    def test_build_basic(self):
        from x_creative.answer.pack_builder import AnswerPackBuilder

        pack = AnswerPackBuilder.build(
            session=_make_session(),
            problem_frame=_make_frame(),
            target_plugin=_make_plugin(),
            source_domains=[],
            verified_hypotheses=[],
            solve_result=None,
            question="How to design a viral open source tool?",
        )
        assert pack.question == "How to design a viral open source tool?"
        assert pack.session_id == "test-session-123"
        assert pack.needs_clarification is False
        assert "# How to design a viral open source tool?" in pack.answer_md
        assert pack.answer_json["version"] == "1.0"

    def test_build_with_hypotheses(self):
        from x_creative.answer.pack_builder import AnswerPackBuilder

        hypotheses = [_make_hypothesis(i) for i in range(3)]
        pack = AnswerPackBuilder.build(
            session=_make_session(),
            problem_frame=_make_frame(),
            target_plugin=_make_plugin(),
            source_domains=[],
            verified_hypotheses=hypotheses,
            solve_result={"solution_markdown": "Do X and Y."},
            question="How to design a viral open source tool?",
        )
        assert len(pack.answer_json["hypotheses"]) == 3
        assert "Hypothesis Ranking" in pack.answer_md

    def test_build_answer_json_structure(self):
        from x_creative.answer.pack_builder import AnswerPackBuilder

        pack = AnswerPackBuilder.build(
            session=_make_session(),
            problem_frame=_make_frame(),
            target_plugin=_make_plugin(),
            source_domains=[],
            verified_hypotheses=[],
            solve_result=None,
            question="test",
        )
        j = pack.answer_json
        assert "version" in j
        assert "question" in j
        assert "answer" in j
        assert "risks" in j
        assert "hypotheses" in j
        assert "metadata" in j
        assert j["metadata"]["target_domain"] == "open_source_development"
