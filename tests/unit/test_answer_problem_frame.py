"""Tests for ProblemFrameBuilder."""

import asyncio

from unittest.mock import AsyncMock, patch

import pytest


class TestProblemFrameBuilder:
    @pytest.mark.asyncio
    async def test_build_returns_frame(self):
        """Normal question returns a complete ProblemFrame."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        llm_response = """{
            "objective": "Design a CLI tool that achieves viral adoption",
            "constraints": [{"text": "Must be cross-platform", "source": "inferred", "confidence": 0.95}],
            "scope": {"in_scope": ["CLI tools", "developer experience"], "out_of_scope": ["GUI apps"]},
            "definitions": {"viral adoption": "organic growth driven by user recommendations"},
            "success_criteria": ["GitHub stars > 1000 in first month"],
            "domain_hint": {"domain_id": "open_source_development", "confidence": 0.85},
            "open_questions": [],
            "context": {"ecosystem": "open source"}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(
                return_value=type("R", (), {"content": llm_response})()
            )
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = await builder.build("How to design a viral open source tool?")

        assert result.needs_clarification is False
        assert result.frame is not None
        assert result.frame.objective == "Design a CLI tool that achieves viral adoption"
        assert result.frame.domain_hint["domain_id"] == "open_source_development"
        assert result.frame.domain_hint["confidence"] == 0.85
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_build_low_confidence_triggers_clarification(self):
        """Very low confidence domain hint triggers smart interrupt."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        llm_response = """{
            "objective": "Something vague",
            "constraints": [],
            "scope": {"in_scope": [], "out_of_scope": []},
            "definitions": {},
            "success_criteria": [],
            "domain_hint": {"domain_id": "unknown", "confidence": 0.15},
            "open_questions": ["Cannot determine domain"],
            "context": {}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(
                return_value=type("R", (), {"content": llm_response})()
            )
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = await builder.build("Do the thing")

        assert result.needs_clarification is True
        assert result.clarification_question is not None
        assert result.partial_frame is not None

    @pytest.mark.asyncio
    async def test_build_medium_confidence_continues(self):
        """Medium confidence (0.3-0.7) continues without interrupt."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        llm_response = """{
            "objective": "Improve developer tooling",
            "constraints": [],
            "scope": {"in_scope": ["developer tools"], "out_of_scope": []},
            "definitions": {},
            "success_criteria": [],
            "domain_hint": {"domain_id": "open_source_development", "confidence": 0.5},
            "open_questions": ["Exact tool category unclear"],
            "context": {}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(
                return_value=type("R", (), {"content": llm_response})()
            )
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = await builder.build("Improve my developer tooling")

        assert result.needs_clarification is False
        assert result.frame is not None
        assert len(result.frame.open_questions) > 0

    @pytest.mark.asyncio
    async def test_build_with_known_domains(self):
        """Builder is aware of available target domain IDs."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        builder = ProblemFrameBuilder()
        assert "open_source_development" in builder.available_domains

    def test_build_preserves_explicit_numbered_constraints_from_question(self):
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        question = (
            "寻找一个适用于中国大陆 A 股市场的板块轮动策略架构，要求：\n"
            "1. 能够适应不同市场 regime 的切换，\n"
            "2. 可用数据只有来自于 tushare 的个股日线、ETF日线、财报、券商研报、融资融券、可转债，以及来自于淘宝的个股逐笔成交数据，\n"
            "3. 满足涨停无法买入跌停无法卖出、T+1交割的交易规则。"
        )
        llm_response = """{
            "objective": "寻找板块轮动策略架构",
            "constraints": [{"text": "Data sources are restricted to Tushare (daily/financial/margin/reports) and Taobao tick data.", "source": "inferred", "confidence": 0.9}],
            "scope": {"in_scope": ["A股"], "out_of_scope": []},
            "definitions": {},
            "success_criteria": [],
            "domain_hint": {"domain_id": "general", "confidence": 0.8},
            "open_questions": [],
            "context": {}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(return_value=type("R", (), {"content": llm_response})())
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = asyncio.run(builder.build(question))

        assert result.frame is not None
        assert any("可转债" in text for text in result.frame.constraints)
