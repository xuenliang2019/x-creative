"""Integration test for the full answer pipeline (LLM mocked).

Unlike test_answer_engine.py which mocks every component at the engine level,
this test lets real components (SessionManager, ProblemFrameBuilder,
TargetDomainResolver, SourceDomainSelector, AnswerPackBuilder) run end-to-end,
only stubbing out LLM calls and the heavy creativity/solve engines.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from x_creative.answer.engine import AnswerEngine
from x_creative.answer.types import AnswerConfig


def _make_llm_response(content: str):
    """Create a mock LLM completion result with a .content attribute."""
    return type("CompletionResult", (), {"content": content})()


# --- Canned LLM responses ---

PROBLEM_FRAME_JSON = json.dumps({
    "objective": "Design a CLI tool that achieves viral adoption in the open source community",
    "constraints": [
        {"text": "Must be installable via a single command", "source": "inferred", "confidence": 0.9},
        {"text": "Must support cross-platform usage", "source": "inferred", "confidence": 0.8},
    ],
    "scope": {"in_scope": ["CLI tools", "developer experience"], "out_of_scope": ["GUI applications"]},
    "definitions": {"viral adoption": "organic growth driven by user recommendations"},
    "success_criteria": ["GitHub stars > 1000 in first month"],
    "domain_hint": {"domain_id": "open_source_development", "confidence": 0.9},
    "open_questions": [],
    "context": {},
})

# SourceDomainSelector relevance ranking response (scored list).
# We return a JSON array of {domain_id, relevance} — the selector
# expects this format from the LLM.  We include a few domain IDs
# that are actually in the open_source_development YAML so the test works
# regardless of exact domain count.
SOURCE_RELEVANCE_JSON = json.dumps([
    {"domain_id": "fluid_dynamics", "relevance": 0.9},
    {"domain_id": "epidemiology", "relevance": 0.85},
    {"domain_id": "ecology", "relevance": 0.8},
    {"domain_id": "thermodynamics", "relevance": 0.75},
    {"domain_id": "game_theory", "relevance": 0.7},
])


class TestAnswerIntegration:
    """Full pipeline integration test: question -> AnswerPack with real components."""

    @pytest.mark.asyncio
    async def test_end_to_end_with_mock_llm(self, tmp_path):
        """Full pipeline: question -> AnswerPack, only LLM transport mocked."""
        # Point SessionManager at tmp_path so it doesn't pollute local_data/
        os.environ["X_CREATIVE_DATA_DIR"] = str(tmp_path)

        problem_response = _make_llm_response(PROBLEM_FRAME_JSON)
        relevance_response = _make_llm_response(SOURCE_RELEVANCE_JSON)

        # We need to mock ModelRouter at every import site that creates one.
        # ProblemFrameBuilder, TargetDomainResolver, and SourceDomainSelector
        # all import ModelRouter from x_creative.llm.router and instantiate it.
        # The engine itself also creates a ModelRouter for CreativityEngine.
        #
        # Strategy: patch ModelRouter at each import location so that every
        # .complete() call returns the appropriate canned response.

        # Track call count to return the right response:
        #   Call 1 (ProblemFrameBuilder.build): problem frame JSON
        #   Call 2+ (SourceDomainSelector._rank_by_relevance): relevance JSON
        complete_calls = []

        async def mock_complete(**kwargs):
            complete_calls.append(kwargs)
            # First call is from ProblemFrameBuilder
            if len(complete_calls) == 1:
                return problem_response
            # Subsequent calls are from SourceDomainSelector ranking
            return relevance_response

        def make_mock_router():
            """Create a mock ModelRouter instance with async complete/close."""
            from unittest.mock import MagicMock
            router = MagicMock()
            router.complete = AsyncMock(side_effect=mock_complete)
            router.close = AsyncMock()
            return router

        # We patch at each import site.  TargetDomainResolver with
        # domain_hint confidence=0.9 >= 0.7 will do an exact match on
        # the built-in YAML and won't need an LLM call.
        with (
            patch(
                "x_creative.answer.problem_frame.ModelRouter",
                side_effect=lambda: make_mock_router(),
            ),
            patch(
                "x_creative.answer.target_resolver.ModelRouter",
                side_effect=lambda: make_mock_router(),
            ),
            patch(
                "x_creative.answer.source_selector.ModelRouter",
                side_effect=lambda: make_mock_router(),
            ),
            patch(
                "x_creative.answer.engine.CreativityEngine",
            ) as MockCE,
            patch(
                "x_creative.answer.engine.ModelRouter",
                side_effect=lambda: make_mock_router(),
            ),
            patch(
                "x_creative.answer.engine.TalkerReasonerSolver",
            ) as MockSolver,
        ):
            # CreativityEngine returns mock hypotheses with scores above threshold
            from x_creative.core.types import Hypothesis
            mock_hypotheses = [
                Hypothesis(
                    id="h_mock", description="Mock hypothesis",
                    source_domain="epidemiology", source_structure="compartment_flow",
                    analogy_explanation="Mock analogy", observable="mock_metric",
                    final_score=7.0,
                ),
            ]
            mock_ce_instance = MockCE.return_value
            mock_ce_instance.generate = AsyncMock(return_value=mock_hypotheses)
            mock_ce_instance.close = AsyncMock()

            # Solver (won't be called because auto_refine=False)
            mock_solver_instance = MockSolver.return_value
            mock_solver_instance.run = AsyncMock(return_value=None)
            mock_solver_instance.close = AsyncMock()

            # SessionManager uses data_dir; we override via constructor param
            # since the AnswerEngine creates it internally.
            with patch(
                "x_creative.answer.engine.SessionManager",
            ) as MockSM:
                from x_creative.session import SessionManager as RealSM

                real_sm = RealSM(data_dir=tmp_path)
                MockSM.return_value = real_sm

                config = AnswerConfig(saga_enabled=False, auto_refine=False)
                engine = AnswerEngine(config=config)
                pack = await engine.answer("How to design a viral open source tool?")

        # --- Assertions ---

        # Not a clarification response
        assert pack.needs_clarification is False

        # Session was created and persisted
        assert pack.session_id != ""
        assert "how-to-design-a-viral-open-source-tool" in pack.session_id

        # Markdown starts with the question as heading
        assert pack.answer_md.startswith("# How to design a viral open source tool?")

        # JSON structure
        assert pack.answer_json["version"] == "1.0"
        assert pack.answer_json["question"] == "How to design a viral open source tool?"

        # Target domain was resolved to open_source_development (via exact match)
        assert pack.answer_json["metadata"]["target_domain"] == "open_source_development"

        # Source domains were selected (real SourceDomainSelector ran)
        assert pack.answer_json["metadata"]["source_domains_count"] > 0

        # ProblemFrameBuilder's LLM was called at least once
        assert len(complete_calls) >= 1

    @pytest.mark.asyncio
    async def test_clarification_path_integration(self, tmp_path):
        """When confidence is below threshold, pipeline short-circuits to clarification."""
        os.environ["X_CREATIVE_DATA_DIR"] = str(tmp_path)

        low_confidence_json = json.dumps({
            "objective": "Unknown",
            "constraints": [],
            "scope": {"in_scope": [], "out_of_scope": []},
            "definitions": {},
            "success_criteria": [],
            "domain_hint": {"domain_id": "unknown", "confidence": 0.1},
            "open_questions": ["Which specific domain does this relate to?"],
            "context": {},
        })
        low_response = _make_llm_response(low_confidence_json)

        def make_mock_router():
            from unittest.mock import MagicMock
            router = MagicMock()
            router.complete = AsyncMock(return_value=low_response)
            router.close = AsyncMock()
            return router

        with (
            patch(
                "x_creative.answer.problem_frame.ModelRouter",
                side_effect=lambda: make_mock_router(),
            ),
            patch(
                "x_creative.answer.engine.SessionManager",
            ) as MockSM,
        ):
            from x_creative.session import SessionManager as RealSM

            real_sm = RealSM(data_dir=tmp_path)
            MockSM.return_value = real_sm

            config = AnswerConfig(saga_enabled=False, auto_refine=False)
            engine = AnswerEngine(config=config)
            pack = await engine.answer("Do something")

        assert pack.needs_clarification is True
        assert pack.clarification_question == "Which specific domain does this relate to?"
