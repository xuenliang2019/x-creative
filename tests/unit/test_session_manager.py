"""Tests for SessionManager."""

import pytest
from pathlib import Path
from datetime import datetime


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    return tmp_path / "local_data"


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_create_session(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        session = manager.create_session("Test Topic")

        assert session.topic == "Test Topic"
        assert session.id is not None
        assert (temp_data_dir / session.id / "session.json").exists()

    def test_create_session_with_custom_id(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        session = manager.create_session("Test", session_id="my-custom-id")

        assert session.id == "my-custom-id"

    def test_load_session(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        created = manager.create_session("Test", session_id="test-session")

        loaded = manager.load_session("test-session")
        assert loaded.id == created.id
        assert loaded.topic == created.topic

    def test_load_nonexistent_session(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        result = manager.load_session("nonexistent")
        assert result is None

    def test_list_sessions(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        manager.create_session("Session 1", session_id="s1")
        manager.create_session("Session 2", session_id="s2")

        sessions = manager.list_sessions()
        assert len(sessions) == 2
        ids = [s.id for s in sessions]
        assert "s1" in ids
        assert "s2" in ids

    def test_delete_session(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        manager.create_session("Test", session_id="to-delete")
        assert (temp_data_dir / "to-delete").exists()

        manager.delete_session("to-delete")
        assert not (temp_data_dir / "to-delete").exists()

    def test_current_session(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        session = manager.create_session("Test", session_id="current-test")

        # Should be set as current
        current = manager.get_current_session()
        assert current is not None
        assert current.id == "current-test"

    def test_switch_session(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        manager.create_session("Session 1", session_id="s1")
        manager.create_session("Session 2", session_id="s2")

        manager.switch_session("s1")
        current = manager.get_current_session()
        assert current.id == "s1"

    def test_save_stage_data(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        session = manager.create_session("Test", session_id="data-test")

        test_data = {"description": "Test problem"}
        manager.save_stage_data(session.id, "problem", test_data)

        # Verify JSON file exists
        assert (temp_data_dir / "data-test" / "problem.json").exists()

    def test_load_stage_data(self, temp_data_dir):
        from x_creative.session.manager import SessionManager

        manager = SessionManager(data_dir=temp_data_dir)
        session = manager.create_session("Test", session_id="load-test")

        test_data = {"key": "value"}
        manager.save_stage_data(session.id, "problem", test_data)
        loaded = manager.load_stage_data(session.id, "problem")

        assert loaded == test_data
