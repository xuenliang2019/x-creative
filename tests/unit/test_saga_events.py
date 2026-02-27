"""Tests for SAGA event system."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from x_creative.saga.events import (
    DirectiveType,
    EventBus,
    EventType,
    FastAgentEvent,
    SlowAgentDirective,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self) -> None:
        """Test that all expected event types are defined."""
        expected = [
            "domain_loaded", "domain_added", "domain_extended",
            "biso_started", "biso_domain_completed", "biso_hypothesis_batch",
            "biso_completed",
            "search_started", "search_round_completed", "search_expanded",
            "search_completed",
            "verify_started", "verify_hypothesis_scored", "verify_batch_scored",
            "verify_completed",
            "pipeline_completed",
            "solve_round_completed",
            "hkg_path_found", "hkg_path_not_found", "hkg_expansion_created",
            "mapping_table_missing", "mapping_low_systematicity",
            "mapping_padding_suspected",
            "verify_position_inconsistent", "verify_escalated",
            "verify_abstained", "verify_injection_detected",
            "constraint_set_growth_alert", "constraint_conflict_detected",
            "constraint_reframe_triggered",
            "constraint_violation_found", "constraint_patch_applied",
            "blend_expand_completed",
            "transform_proposed", "transform_accepted_or_rejected",
            "transform_space_applied",
        ]
        actual = [e.value for e in EventType]
        for ev in expected:
            assert ev in actual, f"Missing event type: {ev}"

    def test_event_type_is_str_enum(self) -> None:
        """Test that EventType values are strings."""
        assert EventType.BISO_STARTED == "biso_started"
        assert isinstance(EventType.BISO_STARTED, str)


class TestDirectiveType:
    """Tests for DirectiveType enum."""

    def test_all_directive_types_exist(self) -> None:
        """Test that all expected directive types are defined."""
        expected = [
            "adjust_weights", "adjust_search_params", "inject_challenge",
            "pause_pipeline", "resume_pipeline", "flag_hypothesis",
            "rescore_batch", "add_constraint",
        ]
        actual = [d.value for d in DirectiveType]
        for dt in expected:
            assert dt in actual, f"Missing directive type: {dt}"


class TestFastAgentEvent:
    """Tests for FastAgentEvent model."""

    def test_creation(self) -> None:
        """Test creating a FastAgentEvent."""
        event = FastAgentEvent(
            event_type=EventType.BISO_COMPLETED,
            stage="biso",
            payload={"hypothesis_count": 42},
            metrics={"score_mean": 6.5},
        )
        assert event.event_type == EventType.BISO_COMPLETED
        assert event.stage == "biso"
        assert event.payload["hypothesis_count"] == 42
        assert event.metrics["score_mean"] == 6.5
        assert event.timestamp > 0

    def test_default_values(self) -> None:
        """Test FastAgentEvent default values."""
        event = FastAgentEvent(
            event_type=EventType.BISO_STARTED,
            stage="biso",
        )
        assert event.payload == {}
        assert event.metrics == {}
        assert event.timestamp > 0


class TestSlowAgentDirective:
    """Tests for SlowAgentDirective model."""

    def test_creation(self) -> None:
        """Test creating a SlowAgentDirective."""
        directive = SlowAgentDirective(
            directive_type=DirectiveType.ADJUST_WEIGHTS,
            reason="Score compression detected",
            confidence=0.85,
            payload={"weights": {"divergence": 0.30}},
            priority=2,
        )
        assert directive.directive_type == DirectiveType.ADJUST_WEIGHTS
        assert directive.confidence == 0.85
        assert directive.priority == 2

    def test_default_priority(self) -> None:
        """Test default priority is 5."""
        directive = SlowAgentDirective(
            directive_type=DirectiveType.FLAG_HYPOTHESIS,
            reason="test",
            confidence=0.5,
        )
        assert directive.priority == 5

    def test_confidence_validation(self) -> None:
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(Exception):
            SlowAgentDirective(
                directive_type=DirectiveType.FLAG_HYPOTHESIS,
                reason="test",
                confidence=1.5,
            )

    def test_priority_validation(self) -> None:
        """Test that priority must be between 1 and 10."""
        with pytest.raises(Exception):
            SlowAgentDirective(
                directive_type=DirectiveType.FLAG_HYPOTHESIS,
                reason="test",
                confidence=0.5,
                priority=0,
            )


class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_emit_and_subscribe_events(self) -> None:
        """Test emitting and receiving events."""
        bus = EventBus()
        event = FastAgentEvent(
            event_type=EventType.BISO_STARTED,
            stage="biso",
        )

        await bus.emit_event(event)
        bus.stop()

        received = []
        async for e in bus.subscribe_events():
            received.append(e)

        assert len(received) == 1
        assert received[0].event_type == EventType.BISO_STARTED

    @pytest.mark.asyncio
    async def test_emit_event_calls_progress_callback(self) -> None:
        """EventBus should forward emitted events into the optional progress callback."""
        progress_cb = AsyncMock()
        bus = EventBus(progress_callback=progress_cb)
        event = FastAgentEvent(
            event_type=EventType.SEARCH_STARTED,
            stage="search",
            payload={"total_rounds": 3},
            metrics={"hypothesis_count": 10.0},
        )

        await bus.emit_event(event)

        progress_cb.assert_awaited_once()
        args, kwargs = progress_cb.call_args
        assert not kwargs
        assert args[0] == EventType.SEARCH_STARTED.value
        forwarded = args[1]
        assert forwarded["pipeline_stage"] == "search"
        assert forwarded["total_rounds"] == 3
        assert forwarded["metrics"]["hypothesis_count"] == 10.0

    @pytest.mark.asyncio
    async def test_emit_and_poll_directives(self) -> None:
        """Test emitting and polling directives."""
        bus = EventBus()

        d1 = SlowAgentDirective(
            directive_type=DirectiveType.ADJUST_WEIGHTS,
            reason="test1",
            confidence=0.8,
            priority=3,
        )
        d2 = SlowAgentDirective(
            directive_type=DirectiveType.FLAG_HYPOTHESIS,
            reason="test2",
            confidence=0.6,
            priority=1,
        )

        await bus.emit_directive(d1)
        await bus.emit_directive(d2)

        directives = await bus.poll_directives()

        # Should be sorted by priority (1 first)
        assert len(directives) == 2
        assert directives[0].priority == 1
        assert directives[1].priority == 3

    @pytest.mark.asyncio
    async def test_poll_empty_directives(self) -> None:
        """Test polling when no directives are queued."""
        bus = EventBus()
        directives = await bus.poll_directives()
        assert directives == []

    @pytest.mark.asyncio
    async def test_stop_signal(self) -> None:
        """Test that stop() causes subscribe_events to exit."""
        bus = EventBus()

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.2)
            bus.stop()

        asyncio.create_task(stop_after_delay())

        received = []
        async for e in bus.subscribe_events():
            received.append(e)

        assert len(received) == 0
        assert bus.is_stopped

    @pytest.mark.asyncio
    async def test_multiple_events(self) -> None:
        """Test emitting and receiving multiple events."""
        bus = EventBus()

        for i in range(5):
            await bus.emit_event(FastAgentEvent(
                event_type=EventType.BISO_HYPOTHESIS_BATCH,
                stage="biso",
                payload={"batch": i},
            ))

        bus.stop()

        received = []
        async for e in bus.subscribe_events():
            received.append(e)

        assert len(received) == 5

    @pytest.mark.asyncio
    async def test_jsonl_persistence(self) -> None:
        """Test JSONL file persistence of events and directives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "saga"
            bus = EventBus(session_dir=session_dir)

            event = FastAgentEvent(
                event_type=EventType.BISO_COMPLETED,
                stage="biso",
                payload={"count": 10},
            )
            await bus.emit_event(event)

            directive = SlowAgentDirective(
                directive_type=DirectiveType.ADJUST_WEIGHTS,
                reason="test",
                confidence=0.9,
            )
            await bus.emit_directive(directive)

            # Check files exist and have content
            events_file = session_dir / "events.jsonl"
            directives_file = session_dir / "directives.jsonl"

            assert events_file.exists()
            assert directives_file.exists()

            events_lines = events_file.read_text().strip().split("\n")
            assert len(events_lines) == 1
            assert "biso_completed" in events_lines[0]

            directives_lines = directives_file.read_text().strip().split("\n")
            assert len(directives_lines) == 1
            assert "adjust_weights" in directives_lines[0]

    @pytest.mark.asyncio
    async def test_no_persistence_without_session_dir(self) -> None:
        """Test that no files are created when session_dir is None."""
        bus = EventBus()
        await bus.emit_event(FastAgentEvent(
            event_type=EventType.BISO_STARTED,
            stage="biso",
        ))
        # Should not raise
        assert bus._events_file is None
