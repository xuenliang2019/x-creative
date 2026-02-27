"""Tests for x_creative.answer.types — AnswerConfig, FrameBuildResult, AnswerPack."""

from __future__ import annotations

import pytest

from x_creative.answer.types import AnswerConfig, AnswerPack, FrameBuildResult
from x_creative.core.types import ProblemFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_frame() -> ProblemFrame:
    return ProblemFrame(description="Test problem")


# ---------------------------------------------------------------------------
# TestAnswerConfig
# ---------------------------------------------------------------------------

class TestAnswerConfig:
    """AnswerConfig dataclass tests."""

    def test_defaults(self):
        cfg = AnswerConfig()
        assert cfg.budget == 120
        assert cfg.mode == "deep_research"
        assert cfg.target_domain == "auto"
        assert cfg.num_per_domain == 3
        assert cfg.search_depth == 3
        assert cfg.search_breadth == 5
        assert cfg.verify_threshold == 5.0
        assert cfg.verify_top == 50
        assert cfg.max_ideas == 8
        assert cfg.auto_refine is True
        assert cfg.inner_max == 3
        assert cfg.outer_max == 2
        assert cfg.hkg_enabled is True
        assert cfg.saga_enabled is True
        assert cfg.saga_strategy == "ANOMALY_DRIVEN"

    def test_custom_values(self):
        cfg = AnswerConfig(
            budget=60,
            mode="quick",
            target_domain="open_source_development",
            num_per_domain=5,
            search_depth=1,
            search_breadth=10,
            verify_threshold=7.0,
            verify_top=20,
            max_ideas=4,
            auto_refine=False,
            inner_max=1,
            outer_max=1,
            hkg_enabled=False,
            saga_enabled=False,
            saga_strategy="FULL_SWEEP",
        )
        assert cfg.budget == 60
        assert cfg.mode == "quick"
        assert cfg.target_domain == "open_source_development"
        assert cfg.num_per_domain == 5
        assert cfg.search_depth == 1
        assert cfg.search_breadth == 10
        assert cfg.verify_threshold == 7.0
        assert cfg.verify_top == 20
        assert cfg.max_ideas == 4
        assert cfg.auto_refine is False
        assert cfg.inner_max == 1
        assert cfg.outer_max == 1
        assert cfg.hkg_enabled is False
        assert cfg.saga_enabled is False
        assert cfg.saga_strategy == "FULL_SWEEP"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            AnswerConfig(mode="invalid_mode")


# ---------------------------------------------------------------------------
# TestFrameBuildResult
# ---------------------------------------------------------------------------

class TestFrameBuildResult:
    """FrameBuildResult dataclass tests."""

    def test_success(self, sample_frame: ProblemFrame):
        result = FrameBuildResult(frame=sample_frame)
        assert result.frame is sample_frame
        assert result.needs_clarification is False
        assert result.clarification_question is None
        assert result.partial_frame is None
        assert result.confidence == 1.0

    def test_needs_clarification(self, sample_frame: ProblemFrame):
        result = FrameBuildResult(
            frame=None,
            needs_clarification=True,
            clarification_question="What project scope?",
            partial_frame=sample_frame,
            confidence=0.4,
        )
        assert result.frame is None
        assert result.needs_clarification is True
        assert result.clarification_question == "What project scope?"
        assert result.partial_frame is sample_frame
        assert result.confidence == 0.4


# ---------------------------------------------------------------------------
# TestAnswerPack
# ---------------------------------------------------------------------------

class TestAnswerPack:
    """AnswerPack dataclass tests."""

    def test_successful_pack(self):
        pack = AnswerPack(
            question="How to design a viral open source tool?",
            answer_md="# Viral Tool\nUse epidemic spreading models to design viral adoption.",
            answer_json={"ideas": [{"id": "h1"}]},
            session_id="sess-001",
        )
        assert pack.question == "How to design a viral open source tool?"
        assert pack.answer_md == "# Viral Tool\nUse epidemic spreading models to design viral adoption."
        assert pack.answer_json == {"ideas": [{"id": "h1"}]}
        assert pack.session_id == "sess-001"
        assert pack.needs_clarification is False
        assert pack.clarification_question is None
        assert pack.partial_frame is None

    def test_clarification_needed_factory(self, sample_frame: ProblemFrame):
        pack = AnswerPack.clarification_needed(
            partial_frame=sample_frame,
            question="Please specify the project scope.",
        )
        assert pack.needs_clarification is True
        assert pack.clarification_question == "Please specify the project scope."
        assert pack.partial_frame is sample_frame
        # The rest should be defaults
        assert pack.question == ""
        assert pack.answer_md == ""
        assert pack.answer_json == {}
        assert pack.session_id == ""

    def test_build_answer_json_structure(self):
        """Verify answer_json can hold the expected structure."""
        answer_json = {
            "session_id": "sess-002",
            "question": "How to design a viral open source tool?",
            "ideas": [
                {"id": "h1", "description": "Idea 1", "score": 7.5},
                {"id": "h2", "description": "Idea 2", "score": 6.8},
            ],
            "metadata": {"elapsed_s": 45.2, "mode": "deep_research"},
        }
        pack = AnswerPack(
            question="How to design a viral open source tool?",
            answer_md="# Results\n...",
            answer_json=answer_json,
            session_id="sess-002",
        )
        assert len(pack.answer_json["ideas"]) == 2
        assert pack.answer_json["ideas"][0]["score"] == 7.5
        assert pack.answer_json["metadata"]["mode"] == "deep_research"
