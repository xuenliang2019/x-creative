"""Tests for split-screen progress renderer."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from x_creative.cli.progress import AnswerProgress
from x_creative.cli.split_screen import (
    HypothesisSummary,
    SplitScreenProgress,
    _DisplayState,
    _LogBufferHandler,
    _buffer_processor,
    _build_summary_renderable,
    _format_elapsed,
    _install_log_capture,
    _render_progress_bar,
    _sync_display_state,
    _uninstall_log_capture,
)


# ---------------------------------------------------------------------------
# _buffer_processor
# ---------------------------------------------------------------------------


class TestBufferProcessor:
    def test_captures_rendered_line_and_drops(self):
        buf: deque[str] = deque(maxlen=100)
        proc = _buffer_processor(buf)
        event_dict = {"event": "hello world", "log_level": "info", "timestamp": "12:00:00"}

        with pytest.raises(structlog.DropEvent):
            proc(None, "info", event_dict)

        assert len(buf) == 1
        assert "hello world" in buf[0]

    def test_multiple_calls_accumulate(self):
        buf: deque[str] = deque(maxlen=100)
        proc = _buffer_processor(buf)

        for i in range(5):
            try:
                proc(None, "info", {"event": f"msg-{i}", "log_level": "info"})
            except structlog.DropEvent:
                pass

        assert len(buf) == 5


# ---------------------------------------------------------------------------
# _LogBufferHandler
# ---------------------------------------------------------------------------


class TestLogBufferHandler:
    def test_handler_writes_to_buffer(self):
        buf: deque[str] = deque(maxlen=100)
        handler = _LogBufferHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        handler.emit(record)

        assert len(buf) == 1
        assert "test message" in buf[0]


# ---------------------------------------------------------------------------
# HypothesisSummary / _DisplayState
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_hypothesis_summary_fields(self):
        h = HypothesisSummary(
            id="h1",
            description_snippet="short desc",
            score=7.5,
            source_domain="bio",
            verify_status="passed",
        )
        assert h.id == "h1"
        assert h.score == 7.5

    def test_display_state_defaults(self):
        d = _DisplayState()
        assert d.stage_index == 0
        assert d.stage_total == 8
        assert d.stage_name == "init"
        assert d.hypothesis_count == 0
        assert d.top_hypotheses == []


# ---------------------------------------------------------------------------
# SplitScreenProgress — fallback strategy
# ---------------------------------------------------------------------------


class TestFallbackStrategy:
    def test_non_tty_disables_everything(self):
        """Non-TTY: both _enabled and _use_split should be False."""
        with patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            progress = SplitScreenProgress()

        assert not progress._enabled
        assert not progress._use_split

    def test_tty_narrow_falls_back_to_inner(self):
        """TTY but < 120 cols: _enabled=True, _use_split=False (fallback bar)."""
        with patch("sys.stderr") as mock_stderr, \
             patch("shutil.get_terminal_size", return_value=(80, 24)):
            mock_stderr.isatty.return_value = True
            progress = SplitScreenProgress()

        assert progress._enabled
        assert not progress._use_split

    def test_tty_wide_enables_split(self):
        """TTY >= 120 cols: full split-screen."""
        with patch("sys.stderr") as mock_stderr, \
             patch("shutil.get_terminal_size") as mock_size:
            mock_stderr.isatty.return_value = True
            mock_size.return_value = type("TermSize", (), {"columns": 160})()
            progress = SplitScreenProgress()

        assert progress._enabled
        assert progress._use_split

    def test_explicit_disabled(self):
        """enabled=False overrides TTY detection."""
        progress = SplitScreenProgress(enabled=False)
        assert not progress._enabled
        assert not progress._use_split


# ---------------------------------------------------------------------------
# SplitScreenProgress — callback forwarding (non-split mode)
# ---------------------------------------------------------------------------


class TestCallbackForwarding:
    def test_callback_noop_when_disabled(self):
        progress = SplitScreenProgress(enabled=False)
        # Should not raise
        progress.callback("answer_started", {"session_id": "s1"})

    def test_callback_forwards_to_inner_when_not_split(self):
        """When _use_split=False, callback should delegate to inner AnswerProgress."""
        with patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            progress = SplitScreenProgress()

        # Inner is disabled too, but calling callback should not error
        progress.callback("answer_started", {"session_id": "s1", "stage_total": 8})
        progress.callback("biso_completed", {"hypothesis_count": 10})


# ---------------------------------------------------------------------------
# SplitScreenProgress — run() method (non-split path)
# ---------------------------------------------------------------------------


class TestRunMethod:
    def test_run_nonsplit_returns_result(self):
        """run() should return the engine coroutine result in non-split mode."""
        progress = SplitScreenProgress(enabled=False)

        async def fake_engine(cb):
            cb("answer_started", {"stage_total": 8})
            return "the_result"

        result = progress.run(fake_engine)
        assert result == "the_result"

    def test_run_nonsplit_propagates_exception(self):
        """run() should propagate engine exceptions in non-split mode."""
        progress = SplitScreenProgress(enabled=False)

        async def failing_engine(cb):
            raise ValueError("engine error")

        with pytest.raises(ValueError, match="engine error"):
            progress.run(failing_engine)

    def test_run_nonsplit_callback_receives_events(self):
        """The progress callback should be called during run()."""
        progress = SplitScreenProgress(enabled=False)
        events_seen: list[str] = []

        async def fake_engine(cb):
            cb("answer_started", {"stage_total": 8})
            events_seen.append("answer_started")
            cb("biso_completed", {"hypothesis_count": 5})
            events_seen.append("biso_completed")
            return "done"

        progress.run(fake_engine)
        assert events_seen == ["answer_started", "biso_completed"]


# ---------------------------------------------------------------------------
# _sync_display_state
# ---------------------------------------------------------------------------


class TestDisplayStateSync:
    def test_sync_hypothesis_count(self):
        display = _DisplayState()
        inner = AnswerProgress(enabled=False)
        _sync_display_state(display, inner, {"hypothesis_count": 42}, None)
        assert display.hypothesis_count == 42

    def test_sync_top_hypotheses(self):
        payload = {
            "top_hypotheses": [
                {
                    "id": "h1",
                    "description": "A" * 100,
                    "score": 8.5,
                    "source_domain": "bio",
                    "verify_status": "passed",
                },
                {
                    "id": "h2",
                    "description": "B",
                    "score": 7.0,
                    "source_domain": "physics",
                    "verify_status": "",
                },
            ]
        }
        display = _DisplayState()
        inner = AnswerProgress(enabled=False)
        _sync_display_state(display, inner, payload, None)

        assert len(display.top_hypotheses) == 2
        # Description should be truncated to 60 chars
        assert len(display.top_hypotheses[0].description_snippet) <= 60
        assert display.top_hypotheses[1].source_domain == "physics"

    def test_sync_ignores_non_list_top_hypotheses(self):
        display = _DisplayState()
        inner = AnswerProgress(enabled=False)
        _sync_display_state(display, inner, {"top_hypotheses": "not a list"}, None)
        assert display.top_hypotheses == []

    def test_sync_elapsed_time(self):
        display = _DisplayState()
        inner = AnswerProgress(enabled=False)
        import time
        started = time.monotonic() - 10.0
        _sync_display_state(display, inner, {}, started)
        assert display.elapsed_seconds >= 10.0


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


class TestRenderingHelpers:
    def test_progress_bar_zero(self):
        bar = _render_progress_bar(0.0, width=10)
        assert "░" * 10 in bar
        assert "0%" in bar

    def test_progress_bar_full(self):
        bar = _render_progress_bar(1.0, width=10)
        assert "█" * 10 in bar
        assert "100%" in bar

    def test_progress_bar_half(self):
        bar = _render_progress_bar(0.5, width=10)
        assert "█" * 5 in bar
        assert "50%" in bar

    def test_progress_bar_clamps(self):
        bar_neg = _render_progress_bar(-0.5, width=10)
        bar_over = _render_progress_bar(1.5, width=10)
        assert "0%" in bar_neg
        assert "100%" in bar_over

    def test_format_elapsed_seconds(self):
        assert _format_elapsed(0) == "0s"
        assert _format_elapsed(45) == "45s"

    def test_format_elapsed_minutes(self):
        assert _format_elapsed(125) == "2m05s"


# ---------------------------------------------------------------------------
# Log capture install/uninstall
# ---------------------------------------------------------------------------


class TestLogCapture:
    def test_install_and_uninstall_restores_defaults(self):
        """Log capture installs and cleanly uninstalls."""
        buf: deque[str] = deque(maxlen=500)

        handler, fh = _install_log_capture(buf)
        assert handler in logging.getLogger().handlers
        assert fh is None

        _uninstall_log_capture(handler, fh)
        assert handler not in logging.getLogger().handlers

    def test_structlog_messages_captured_during_install(self):
        """structlog messages go into the buffer while capture is active."""
        buf: deque[str] = deque(maxlen=500)

        handler, fh = _install_log_capture(buf)
        try:
            log = structlog.get_logger()
            log.info("test_capture_message")
            assert any("test_capture_message" in line for line in buf)
        finally:
            _uninstall_log_capture(handler, fh)

    def test_context_manager_non_split(self):
        """Non-split context manager delegates to inner AnswerProgress."""
        progress = SplitScreenProgress(enabled=False)
        with progress as p:
            assert p is progress
        # Should not raise on exit


# ---------------------------------------------------------------------------
# _build_summary_renderable
# ---------------------------------------------------------------------------


class TestSummaryRenderable:
    def test_renders_without_hypotheses(self):
        d = _DisplayState()
        result = _build_summary_renderable(d)
        assert result is not None

    def test_renders_with_hypotheses(self):
        d = _DisplayState()
        d.stage_index = 5
        d.stage_name = "search"
        d.top_hypotheses = [
            HypothesisSummary(
                id="h1",
                description_snippet="test hypothesis",
                score=8.2,
                source_domain="bio",
                verify_status="passed",
            )
        ]
        result = _build_summary_renderable(d)
        assert result is not None


# ---------------------------------------------------------------------------
# Log file support
# ---------------------------------------------------------------------------


class TestLogFileSupport:
    def test_buffer_processor_writes_to_file(self, tmp_path: Path):
        """_buffer_processor writes rendered lines to file_handle when provided."""
        log_path = tmp_path / "test.log"
        buf: deque[str] = deque(maxlen=100)

        with open(log_path, "a", encoding="utf-8") as fh:
            proc = _buffer_processor(buf, file_handle=fh)
            event_dict = {"event": "file_test", "log_level": "info", "timestamp": "12:00:00"}
            with pytest.raises(structlog.DropEvent):
                proc(None, "info", event_dict)

        content = log_path.read_text(encoding="utf-8")
        assert "file_test" in content
        # File content must be free of ANSI escape codes
        assert "\x1b[" not in content

    def test_log_buffer_handler_writes_to_file(self, tmp_path: Path):
        """_LogBufferHandler writes formatted messages to file_handle when provided."""
        log_path = tmp_path / "handler.log"
        buf: deque[str] = deque(maxlen=100)

        with open(log_path, "a", encoding="utf-8") as fh:
            handler = _LogBufferHandler(buf, file_handle=fh)
            handler.setFormatter(logging.Formatter("%(message)s"))

            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="handler file test", args=(), exc_info=None,
            )
            handler.emit(record)

        assert len(buf) == 1
        content = log_path.read_text(encoding="utf-8")
        assert "handler file test" in content

    def test_install_log_capture_with_log_file(self, tmp_path: Path):
        """_install_log_capture opens file and returns handle when log_file is given."""
        log_path = tmp_path / "capture.log"
        buf: deque[str] = deque(maxlen=500)

        handler, fh = _install_log_capture(buf, log_file=log_path)
        try:
            assert fh is not None
            assert not fh.closed
            log = structlog.get_logger()
            log.info("capture_file_test")
            assert any("capture_file_test" in line for line in buf)
        finally:
            _uninstall_log_capture(handler, fh)

        assert fh.closed
        content = log_path.read_text(encoding="utf-8")
        assert "capture_file_test" in content

    def test_install_log_capture_creates_parent_dirs(self, tmp_path: Path):
        """_install_log_capture creates parent directories for log_file."""
        log_path = tmp_path / "sub" / "dir" / "nested.log"
        buf: deque[str] = deque(maxlen=500)

        handler, fh = _install_log_capture(buf, log_file=log_path)
        _uninstall_log_capture(handler, fh)
        assert log_path.parent.exists()

    def test_run_nonsplit_with_log_file(self, tmp_path: Path):
        """SplitScreenProgress.run() writes logs to file in non-split mode."""
        log_path = tmp_path / "run.log"
        progress = SplitScreenProgress(enabled=False, log_file=log_path)

        async def fake_engine(cb):
            log = structlog.get_logger()
            log.info("engine_log_line")
            cb("answer_started", {"stage_total": 8})
            return "result"

        result = progress.run(fake_engine)
        assert result == "result"

        content = log_path.read_text(encoding="utf-8")
        assert "engine_log_line" in content
        assert "\x1b[" not in content
