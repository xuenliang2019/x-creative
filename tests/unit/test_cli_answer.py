# tests/unit/test_cli_answer.py
"""Tests for CLI answer command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from x_creative.answer.types import AnswerPack


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIAnswer:
    def test_answer_command_exists(self, runner):
        """The answer command is registered."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["answer", "--help"])
        assert result.exit_code == 0
        assert "question" in result.output.lower() or "QUESTION" in result.output

    def test_answer_requires_question(self, runner):
        """The answer command requires -q option."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["answer"])
        assert result.exit_code != 0

    def test_answer_invokes_engine(self, runner):
        """The answer command calls AnswerEngine.answer()."""
        from x_creative.cli.main import app

        mock_pack = AnswerPack(
            question="test question",
            answer_md="# Test\n\nAnswer here.",
            answer_json={"version": "1.0"},
            session_id="test-session",
        )

        with patch("x_creative.cli.answer.AnswerEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.answer = AsyncMock(return_value=mock_pack)

            result = runner.invoke(app, ["answer", "-q", "test question"])

        assert result.exit_code == 0
        assert "Answer here" in result.output or "Test" in result.output

    def test_answer_passes_progress_callback(self, runner):
        """The answer command should pass a progress_callback for live progress UI."""
        from x_creative.cli.main import app

        mock_pack = AnswerPack(
            question="test question",
            answer_md="# Test\n\nAnswer here.",
            answer_json={"version": "1.0"},
            session_id="test-session",
        )

        with patch("x_creative.cli.answer.AnswerEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.answer = AsyncMock(return_value=mock_pack)

            result = runner.invoke(app, ["answer", "-q", "test question"])

        assert result.exit_code == 0
        assert mock_engine.answer.call_count == 1
        _, kwargs = mock_engine.answer.call_args
        assert "progress_callback" in kwargs
        assert callable(kwargs["progress_callback"])

    def test_answer_default_budget_is_60(self, runner):
        """Default --budget should be low enough for reasonable runtime."""
        from x_creative.cli.main import app

        mock_pack = AnswerPack(
            question="test question",
            answer_md="# Test\n\nAnswer here.",
            answer_json={"version": "1.0"},
            session_id="test-session",
        )

        with patch("x_creative.cli.answer.AnswerEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.answer = AsyncMock(return_value=mock_pack)

            result = runner.invoke(app, ["answer", "-q", "test question"])

        assert result.exit_code == 0
        _, kwargs = MockEngine.call_args
        config = kwargs.get("config")
        assert config is not None
        assert getattr(config, "budget") == 60

    def test_answer_exits_with_conflict_report(self, runner):
        """Conflicting constraints should exit non-zero and print a full report."""
        from x_creative.cli.main import app
        from x_creative.answer.constraint_preflight import UserConstraintConflictError

        report = {
            "message": "Found 1 conflicting user-constraint pairs",
            "constraints": [
                {"id": "C1", "text": "must use daily frequency data"},
                {"id": "C2", "text": "do not use daily frequency data"},
            ],
            "conflict_pairs": [
                {
                    "left_id": "C1",
                    "right_id": "C2",
                    "left_text": "must use daily frequency data",
                    "right_text": "do not use daily frequency data",
                }
            ],
        }

        with patch("x_creative.cli.answer.AnswerEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.answer = AsyncMock(side_effect=UserConstraintConflictError(report))

            result = runner.invoke(app, ["answer", "-q", "test question"])

        assert result.exit_code != 0
        assert "C1" in result.output
        assert "C2" in result.output

    def test_answer_exits_with_compliance_report(self, runner):
        """Constraint compliance failures should exit non-zero and print audit details."""
        from x_creative.cli.main import app
        from x_creative.saga.constraint_compliance import UserConstraintComplianceError

        report = {
            "message": "User constraint compliance failed after 2 revision rounds",
            "audit_report": {
                "overall_pass": False,
                "items": [
                    {
                        "id": "C1",
                        "text": "must do X",
                        "verdict": "fail",
                        "rationale": "missing",
                        "suggested_fix": "add X",
                    }
                ],
            },
        }

        with patch("x_creative.cli.answer.AnswerEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.answer = AsyncMock(side_effect=UserConstraintComplianceError(report))

            result = runner.invoke(app, ["answer", "-q", "test question"])

        assert result.exit_code != 0
        assert "C1" in result.output
