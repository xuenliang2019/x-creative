"""Tests for CLI interface."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from x_creative.core.types import Hypothesis, HypothesisScores, ProblemFrame, SearchConfig


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIBasic:
    """Basic CLI tests."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test that help command works."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "x-creative" in result.stdout.lower() or "creativity" in result.stdout.lower()

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test version command."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout


class TestGenerateCommand:
    """Tests for the generate command."""

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.stdout.lower()

    def test_generate_basic(self, runner: CliRunner) -> None:
        """Test basic generate command."""
        from x_creative.cli.main import app

        with patch("x_creative.cli.main._generate_async") as mock_generate:
            mock_generate.return_value = [
                Hypothesis(
                    id="hyp_1",
                    description="Test hypothesis",
                    source_domain="test",
                    source_structure="test",
                    analogy_explanation="test explanation here",
                    observable="test_observable",
                    scores=HypothesisScores(
                        divergence=8.0,
                        testability=8.0,
                        rationale=8.0,
                        robustness=8.0,
                        feasibility=8.0,
                    ),
                )
            ]

            with patch("x_creative.cli.main.asyncio.run", return_value=mock_generate.return_value):
                result = runner.invoke(
                    app,
                    ["generate", "设计一个能实现病毒式传播的开源命令行工具", "--num-hypotheses", "5"],
                )

                # Should complete without error
                assert result.exit_code == 0

    def test_generate_with_options(self, runner: CliRunner) -> None:
        """Test generate with various options."""
        from x_creative.cli.main import app

        with patch("x_creative.cli.main.asyncio.run", return_value=[]):
            result = runner.invoke(
                app,
                [
                    "generate",
                    "test problem",
                    "--num-hypotheses", "10",
                    "--search-depth", "2",
                ],
            )

            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_generate_async_uses_saga_coordinator(self, monkeypatch) -> None:
        from x_creative.cli import main as cli_main

        expected = Hypothesis(
            id="h1",
            description="test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )

        class DummyCoordinator:
            called = False

            def __init__(self, engine):  # noqa: ANN001
                self.engine = engine

            async def run(self, problem, config, source_domains=None):  # noqa: ANN001
                DummyCoordinator.called = True
                assert source_domains is None
                return MagicMock(hypotheses=[expected])

        class DummyEngine:
            async def generate(self, problem, config):  # noqa: ANN001
                return []

            async def close(self) -> None:
                return None

        monkeypatch.setattr(cli_main, "SAGACoordinator", DummyCoordinator, raising=False)
        monkeypatch.setattr(cli_main, "CreativityEngine", DummyEngine)
        monkeypatch.setattr(cli_main, "load_target_domain", lambda _domain_id: None)

        result = await cli_main._generate_async(
            ProblemFrame(description="test"),
            SearchConfig(num_hypotheses=1, search_depth=1, search_breadth=1),
        )

        assert DummyCoordinator.called is True
        assert len(result) == 1
        assert result[0].id == "h1"

    @pytest.mark.asyncio
    async def test_generate_async_applies_stage_b_source_filter(self, monkeypatch) -> None:
        from x_creative.cli import main as cli_main

        expected = Hypothesis(
            id="h1",
            description="test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )
        source_domains = [MagicMock(id="queueing_theory"), MagicMock(id="ecology")]

        class DummyCoordinator:
            called = False

            def __init__(self, engine):  # noqa: ANN001
                self.engine = engine

            async def run(self, problem, config, source_domains=None):  # noqa: ANN001
                DummyCoordinator.called = True
                assert source_domains == source_domains_filter_result
                return MagicMock(hypotheses=[expected])

        class DummyEngine:
            async def close(self) -> None:
                return None

        class DummySelector:
            async def filter_by_mapping_feasibility(self, frame, target):  # noqa: ANN001
                assert frame.description == "test"
                assert target is plugin
                return source_domains

        plugin = MagicMock()
        source_domains_filter_result = source_domains

        monkeypatch.setattr(cli_main, "SAGACoordinator", DummyCoordinator, raising=False)
        monkeypatch.setattr(cli_main, "CreativityEngine", DummyEngine)
        monkeypatch.setattr(cli_main, "load_target_domain", lambda _domain_id: plugin)
        monkeypatch.setattr(cli_main, "SourceDomainSelector", DummySelector)

        result = await cli_main._generate_async(
            ProblemFrame(description="test"),
            SearchConfig(num_hypotheses=1, search_depth=1, search_breadth=1),
        )

        assert DummyCoordinator.called is True
        assert len(result) == 1
        assert result[0].id == "h1"


class TestDomainsCommand:
    """Tests for the domains command."""

    def test_domains_list(self, runner: CliRunner) -> None:
        """Test listing domains."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["domains", "list"])
        assert result.exit_code == 0
        assert "epidemiology" in result.stdout.lower()

    def test_domains_show(self, runner: CliRunner) -> None:
        """Test showing a specific domain."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["domains", "show", "流行病学与传播动力学"])
        assert result.exit_code == 0
        assert "流行病学" in result.stdout or "epidemiology" in result.stdout.lower()

    def test_domains_show_unknown(self, runner: CliRunner) -> None:
        """Test showing an unknown domain."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["domains", "show", "unknown_domain"])
        assert result.exit_code != 0 or "not found" in result.stdout.lower()


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_show(self, runner: CliRunner) -> None:
        """Test showing configuration."""
        from x_creative.cli.main import app

        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "openrouter" in result.stdout.lower()
        assert "yunwu" in result.stdout.lower()
