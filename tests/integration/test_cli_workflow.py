# tests/integration/test_cli_workflow.py
"""Integration tests for the CLI workflow."""

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from x_creative.core.types import Hypothesis


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Set up temporary data directory."""
    data_dir = tmp_path / "local_data"
    data_dir.mkdir()
    monkeypatch.setenv("X_CREATIVE_DATA_DIR", str(data_dir))
    return data_dir


class TestCLIWorkflow:
    """Test the complete CLI workflow."""

    def test_session_lifecycle(self, runner, temp_data_dir):
        """Test session creation, switching, and deletion."""
        from x_creative.cli.main import app

        # Create sessions
        result = runner.invoke(app, ["session", "new", "Session 1", "--id", "s1"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["session", "new", "Session 2", "--id", "s2"])
        assert result.exit_code == 0

        # List sessions
        result = runner.invoke(app, ["session", "list"])
        assert "s1" in result.stdout
        assert "s2" in result.stdout

        # Switch session
        result = runner.invoke(app, ["session", "switch", "s1"])
        assert result.exit_code == 0

        # Status shows s1
        result = runner.invoke(app, ["session", "status"])
        assert "s1" in result.stdout

        # Delete session
        result = runner.invoke(app, ["session", "delete", "s2", "--yes"])
        assert result.exit_code == 0

        # List should not show s2
        result = runner.invoke(app, ["session", "list"])
        assert "s2" not in result.stdout

    def test_problem_stage(self, runner, temp_data_dir):
        """Test problem definition stage."""
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test", "--id", "test"])

        # Use current CLI options: --context for domain-specific settings
        result = runner.invoke(app, [
            "run", "problem",
            "--description", "How to design a viral open source tool?",
            "--target-domain", "open_source_development",
            "--context", '{"language": "python", "license": "MIT"}',
        ])
        assert result.exit_code == 0

        # Verify files exist
        assert (temp_data_dir / "test" / "problem.json").exists()
        assert (temp_data_dir / "test" / "problem.md").exists()

        # Show problem
        result = runner.invoke(app, ["show", "problem"])
        assert result.exit_code == 0
        assert "How to design a viral open source tool?" in result.stdout

        # Status shows problem completed
        result = runner.invoke(app, ["session", "status"])
        assert "completed" in result.stdout.lower()

    def test_stage_dependency_check(self, runner, temp_data_dir):
        """Test that stages check dependencies."""
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test", "--id", "test"])

        # Try to run biso without problem
        result = runner.invoke(app, ["run", "biso"])
        assert result.exit_code != 0
        assert "problem" in result.stdout.lower()

        # Run problem first
        runner.invoke(app, [
            "run", "problem",
            "--description", "Test",
        ])

        # Mock the LLM call to avoid slow API requests
        mock_hypotheses = [
            Hypothesis(
                id="hyp_mock1",
                description="Mock hypothesis",
                source_domain="thermodynamics",
                source_structure="entropy",
                analogy_explanation="Mock explanation",
                observable="Mock observable",
                generation=0,
            )
        ]

        with patch(
            "x_creative.creativity.biso.BISOModule.generate_all_analogies",
            new_callable=AsyncMock,
            return_value=mock_hypotheses,
        ):
            # Now biso should run successfully with mocked LLM
            result = runner.invoke(app, ["run", "biso"])
            # Should not fail due to dependency check
            assert "Cannot run BISO: problem stage not completed" not in result.stdout
            # Should complete successfully with mock
            assert result.exit_code == 0
