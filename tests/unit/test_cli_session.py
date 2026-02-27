"""Tests for session CLI commands."""

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


class TestSessionNew:
    def test_create_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["session", "new", "Test Topic"])
        assert result.exit_code == 0
        assert "Created session" in result.stdout or "Test Topic" in result.stdout

    def test_create_session_with_id(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["session", "new", "Test", "--id", "custom-id"])
        assert result.exit_code == 0
        assert "custom-id" in result.stdout


class TestSessionList:
    def test_list_empty(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["session", "list"])
        assert result.exit_code == 0

    def test_list_sessions(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Session 1", "--id", "s1"])
        runner.invoke(app, ["session", "new", "Session 2", "--id", "s2"])

        result = runner.invoke(app, ["session", "list"])
        assert result.exit_code == 0
        assert "s1" in result.stdout
        assert "s2" in result.stdout


class TestSessionStatus:
    def test_status_no_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["session", "status"])
        assert result.exit_code == 0 or result.exit_code == 1
        # Should indicate no current session

    def test_status_with_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test", "--id", "test-status"])
        result = runner.invoke(app, ["session", "status"])
        assert result.exit_code == 0
        assert "test-status" in result.stdout


class TestSessionSwitch:
    def test_switch_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Session 1", "--id", "s1"])
        runner.invoke(app, ["session", "new", "Session 2", "--id", "s2"])

        result = runner.invoke(app, ["session", "switch", "s1"])
        assert result.exit_code == 0

        status_result = runner.invoke(app, ["session", "status"])
        assert "s1" in status_result.stdout


class TestSessionDelete:
    def test_delete_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "To Delete", "--id", "delete-me"])

        result = runner.invoke(app, ["session", "delete", "delete-me", "--yes"])
        assert result.exit_code == 0

        list_result = runner.invoke(app, ["session", "list"])
        assert "delete-me" not in list_result.stdout
