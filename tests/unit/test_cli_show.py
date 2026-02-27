"""Tests for show CLI commands."""

import json
import pytest
from typer.testing import CliRunner


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


@pytest.fixture
def session_with_data(temp_data_dir, runner):
    """Create a session with problem data."""
    from x_creative.cli.main import app

    runner.invoke(app, ["session", "new", "Test", "--id", "show-test"])

    # Create problem.json
    problem_data = {
        "description": "Test problem",
        "target_domain": "open_source_development",
        "constraints": [],
        "context": {},
    }
    session_dir = temp_data_dir / "show-test"
    with open(session_dir / "problem.json", "w") as f:
        json.dump(problem_data, f)

    return session_dir


class TestShowProblem:
    def test_show_problem(self, runner, session_with_data):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "problem"])
        assert result.exit_code == 0
        assert "Test problem" in result.stdout
        assert "Target Domain" in result.stdout

    def test_show_problem_no_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "problem"])
        assert result.exit_code != 0

    def test_show_problem_raw(self, runner, session_with_data):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "problem", "--raw"])
        assert result.exit_code == 0
        # Should output JSON
        assert '"description"' in result.stdout
        assert '"Test problem"' in result.stdout

    def test_show_problem_with_context(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test", "--id", "ctx-show"])

        # Create problem with context
        problem_data = {
            "description": "Problem with context",
            "target_domain": "research",
            "context": {"key1": "value1", "key2": 123},
            "constraints": ["constraint A", "constraint B"],
        }
        with open(temp_data_dir / "ctx-show" / "problem.json", "w") as f:
            json.dump(problem_data, f)

        result = runner.invoke(app, ["show", "problem"])
        assert result.exit_code == 0
        assert "research" in result.stdout
        assert "Context:" in result.stdout
        assert "constraint A" in result.stdout


class TestShowBiso:
    def test_show_biso_not_run(self, runner, session_with_data):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "biso"])
        # Should indicate no data
        assert result.exit_code == 0
        assert "not run yet" in result.stdout.lower() or "biso" in result.stdout.lower()

    def test_show_biso_with_data(self, runner, session_with_data):
        from x_creative.cli.main import app

        # Create biso.json
        biso_data = {
            "hypotheses": [
                {
                    "id": "h1",
                    "description": "Test hypothesis 1",
                    "source_domain": "physics",
                    "source_structure": "momentum",
                    "analogy_explanation": "Momentum in physics maps to price momentum",
                    "observable": "price_change / volume",
                    "generation": 0,
                }
            ]
        }
        with open(session_with_data / "biso.json", "w") as f:
            json.dump(biso_data, f)

        result = runner.invoke(app, ["show", "biso"])
        assert result.exit_code == 0
        assert "Test hypothesis 1" in result.stdout or "1 hypotheses" in result.stdout


class TestShowSearch:
    def test_show_search_not_run(self, runner, session_with_data):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "search"])
        assert result.exit_code == 0
        assert "not run yet" in result.stdout.lower() or "search" in result.stdout.lower()

    def test_show_search_with_data(self, runner, session_with_data):
        from x_creative.cli.main import app

        # Create search.json
        search_data = {
            "hypotheses": [
                {
                    "id": "h1",
                    "description": "Test hypothesis 1",
                    "source_domain": "physics",
                    "source_structure": "momentum",
                    "analogy_explanation": "Explanation 1",
                    "observable": "obs1",
                    "generation": 0,
                },
                {
                    "id": "h2",
                    "description": "Test hypothesis 2",
                    "source_domain": "physics",
                    "source_structure": "wave",
                    "analogy_explanation": "Explanation 2",
                    "observable": "obs2",
                    "generation": 1,
                },
            ]
        }
        with open(session_with_data / "search.json", "w") as f:
            json.dump(search_data, f)

        result = runner.invoke(app, ["show", "search"])
        assert result.exit_code == 0
        assert "2 hypotheses" in result.stdout


class TestShowVerify:
    def test_show_verify_not_run(self, runner, session_with_data):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "verify"])
        assert result.exit_code == 0
        assert "not run yet" in result.stdout.lower() or "verify" in result.stdout.lower()

    def test_show_verify_with_data(self, runner, session_with_data):
        from x_creative.cli.main import app

        # Create verify.json with scored hypotheses
        verify_data = {
            "hypotheses": [
                {
                    "id": "h1",
                    "description": "Verified hypothesis",
                    "source_domain": "physics",
                    "source_structure": "momentum",
                    "analogy_explanation": "Momentum analogy",
                    "observable": "price_momentum",
                    "generation": 0,
                    "scores": {
                        "divergence": 7.5,
                        "testability": 8.0,
                        "rationale": 7.0,
                        "robustness": 6.5,
                        "feasibility": 7.0,
                    },
                }
            ]
        }
        with open(session_with_data / "verify.json", "w") as f:
            json.dump(verify_data, f)

        result = runner.invoke(app, ["show", "verify"])
        assert result.exit_code == 0
        assert "Verified hypothesis" in result.stdout or "verified" in result.stdout.lower()


class TestShowReport:
    def test_show_report_not_found(self, runner, session_with_data):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "report", "problem"])
        assert result.exit_code == 0
        assert "not found" in result.stdout.lower()

    def test_show_report_exists(self, runner, session_with_data):
        from x_creative.cli.main import app

        # Create problem.md report
        report_content = "# Problem Report\n\nThis is a test report."
        with open(session_with_data / "problem.md", "w") as f:
            f.write(report_content)

        result = runner.invoke(app, ["show", "report", "problem"])
        assert result.exit_code == 0
        assert "Problem Report" in result.stdout or "test report" in result.stdout

    def test_show_report_output_to_file(self, runner, session_with_data, tmp_path):
        from x_creative.cli.main import app

        # Create problem.md report
        report_content = "# Problem Report\n\nThis is a test report."
        with open(session_with_data / "problem.md", "w") as f:
            f.write(report_content)

        output_file = tmp_path / "output.md"
        result = runner.invoke(app, ["show", "report", "problem", "--output", str(output_file)])
        assert result.exit_code == 0
        assert "saved to" in result.stdout.lower()
        assert output_file.exists()
        assert output_file.read_text() == report_content


class TestShowWithSessionOption:
    def test_show_problem_with_session_id(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        # Create two sessions
        runner.invoke(app, ["session", "new", "Session 1", "--id", "s1"])
        runner.invoke(app, ["session", "new", "Session 2", "--id", "s2"])

        # Add problem to s1
        problem_data = {
            "description": "Problem for session 1",
            "target_domain": "open_source_development",
            "constraints": [],
            "context": {},
        }
        with open(temp_data_dir / "s1" / "problem.json", "w") as f:
            json.dump(problem_data, f)

        # Current session is s2, but we want to show s1
        result = runner.invoke(app, ["show", "problem", "--session", "s1"])
        assert result.exit_code == 0
        assert "Problem for session 1" in result.stdout

    def test_show_problem_session_not_found(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["show", "problem", "--session", "nonexistent"])
        assert result.exit_code != 0
