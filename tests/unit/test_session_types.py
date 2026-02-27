"""Tests for session types."""

import pytest
from datetime import datetime


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_stage_status_values(self):
        from x_creative.session.types import StageStatus

        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"


class TestStageInfo:
    """Tests for StageInfo model."""

    def test_create_pending_stage(self):
        from x_creative.session.types import StageInfo, StageStatus

        stage = StageInfo(status=StageStatus.PENDING)
        assert stage.status == StageStatus.PENDING
        assert stage.started_at is None
        assert stage.completed_at is None
        assert stage.error is None

    def test_create_completed_stage(self):
        from x_creative.session.types import StageInfo, StageStatus

        now = datetime.now()
        stage = StageInfo(
            status=StageStatus.COMPLETED,
            started_at=now,
            completed_at=now,
        )
        assert stage.status == StageStatus.COMPLETED
        assert stage.started_at == now


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        from x_creative.session.types import Session

        session = Session(
            id="2026-02-02-test",
            topic="Test Session",
        )
        assert session.id == "2026-02-02-test"
        assert session.topic == "Test Session"
        assert session.current_stage == "problem"

    def test_session_has_all_stages(self):
        from x_creative.session.types import Session, StageStatus

        session = Session(id="test", topic="Test")
        assert "problem" in session.stages
        assert "biso" in session.stages
        assert "search" in session.stages
        assert "verify" in session.stages
        for stage_info in session.stages.values():
            assert stage_info.status == StageStatus.PENDING

    def test_session_get_stage_status(self):
        from x_creative.session.types import Session, StageStatus

        session = Session(id="test", topic="Test")
        assert session.get_stage_status("problem") == StageStatus.PENDING

    def test_session_is_stage_completed(self):
        from x_creative.session.types import Session, StageStatus

        session = Session(id="test", topic="Test")
        assert session.is_stage_completed("problem") is False

        session.stages["problem"].status = StageStatus.COMPLETED
        assert session.is_stage_completed("problem") is True

    def test_session_can_run_stage(self):
        from x_creative.session.types import Session, StageStatus

        session = Session(id="test", topic="Test")
        # problem has no dependencies
        assert session.can_run_stage("problem") is True
        # biso requires problem
        assert session.can_run_stage("biso") is False

        session.stages["problem"].status = StageStatus.COMPLETED
        assert session.can_run_stage("biso") is True
