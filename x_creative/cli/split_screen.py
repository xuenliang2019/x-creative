"""Split-screen terminal renderer for `x-creative answer`.

Uses Textual for a flicker-free split-screen TUI:
- Left panel (60%): scrolling log output via RichLog
- Right panel (40%): real-time progress summary

Fallback strategy:
- Non-TTY (nohup/pipe/CI): fully disabled, no rendering
- TTY but width < 120 cols: falls back to original AnswerProgress single-line bar
- TTY and width >= 120 cols: full split-screen layout via Textual
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import IO, Any, Callable, Coroutine

import structlog
from rich.table import Table
from rich.text import Text

from x_creative.cli.progress import AnswerProgress

_MIN_SPLIT_WIDTH = 120
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ── Data types ──────────────────────────────────────────────────────────


@dataclass
class HypothesisSummary:
    """Lightweight summary of a hypothesis for display."""

    id: str
    description_snippet: str
    score: float
    source_domain: str
    verify_status: str


@dataclass
class _DisplayState:
    """Aggregated right-panel rendering data."""

    stage_index: int = 0
    stage_total: int = 8
    stage_name: str = "init"
    stage_progress: float = 0.0
    elapsed_seconds: float = 0.0
    hypothesis_count: int = 0
    top_hypotheses: list[HypothesisSummary] = field(default_factory=list)


# ── Log capture ─────────────────────────────────────────────────────────


class _LogBufferHandler(logging.Handler):
    """Standard logging.Handler that writes formatted records into a deque."""

    def __init__(self, buffer: deque[str], file_handle: IO[str] | None = None) -> None:
        super().__init__()
        self._buffer = buffer
        self._file_handle = file_handle

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._buffer.append(msg)
            if self._file_handle is not None:
                self._file_handle.write(_ANSI_RE.sub("", msg) + "\n")
                self._file_handle.flush()
        except Exception:
            self.handleError(record)


def _buffer_processor(
    buffer: deque[str],
    file_handle: IO[str] | None = None,
) -> structlog.types.Processor:
    """Return a structlog processor that captures the formatted line and drops it."""

    def processor(
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        renderer = structlog.dev.ConsoleRenderer()
        rendered = renderer(logger, method_name, event_dict)
        buffer.append(rendered)
        if file_handle is not None:
            file_handle.write(_ANSI_RE.sub("", rendered) + "\n")
            file_handle.flush()
        raise structlog.DropEvent

    return processor


def _install_log_capture(
    log_buffer: deque[str],
    log_file: Path | None = None,
) -> tuple[_LogBufferHandler, IO[str] | None]:
    """Intercept structlog + stdlib logging into the deque buffer.

    Returns:
        A tuple of (handler, file_handle). file_handle is None when log_file is None.
    """
    fh: IO[str] | None = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_file, "a", encoding="utf-8")  # noqa: SIM115

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            _buffer_processor(log_buffer, file_handle=fh),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    handler = _LogBufferHandler(log_buffer, file_handle=fh)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(handler)
    return handler, fh


def _uninstall_log_capture(
    handler: _LogBufferHandler | None,
    file_handle: IO[str] | None = None,
) -> None:
    """Restore default structlog + stdlib logging."""
    structlog.reset_defaults()
    if handler is not None:
        logging.getLogger().removeHandler(handler)
    if file_handle is not None:
        file_handle.close()


# ── Display helpers ─────────────────────────────────────────────────────


def _render_progress_bar(progress: float, width: int = 24) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(progress * width)
    empty = width - filled
    pct = int(progress * 100)
    return f"[{'█' * filled}{'░' * empty}] {pct:>3}%"


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _sync_display_state(
    display: _DisplayState,
    inner: AnswerProgress,
    payload: dict[str, Any],
    started_at: float | None,
) -> None:
    """Sync _DisplayState from AnswerProgress internal counters and event payload."""
    display.stage_index = inner._stage_index
    display.stage_total = inner._stage_total
    display.stage_name = inner._stage_name
    display.stage_progress = inner._stage_progress

    if started_at is not None:
        display.elapsed_seconds = time.monotonic() - started_at

    if isinstance(payload.get("hypothesis_count"), int):
        display.hypothesis_count = payload["hypothesis_count"]

    top_raw = payload.get("top_hypotheses")
    if isinstance(top_raw, list):
        display.top_hypotheses = [
            HypothesisSummary(
                id=str(item.get("id", "")),
                description_snippet=str(item.get("description", ""))[:60],
                score=float(item.get("score", 0)),
                source_domain=str(item.get("source_domain", "")),
                verify_status=str(item.get("verify_status", "")),
            )
            for item in top_raw[:8]
            if isinstance(item, dict)
        ]


def _build_summary_renderable(d: _DisplayState) -> Any:
    """Build a Rich renderable for the summary panel."""
    from rich.console import Group

    parts: list[Any] = []

    stage_line = f"Stage {d.stage_index}/{d.stage_total}: {d.stage_name}"
    parts.append(Text(stage_line, style="bold"))

    bar = _render_progress_bar(d.stage_progress, width=24)
    elapsed = _format_elapsed(d.elapsed_seconds)
    parts.append(Text(f"{bar}  {elapsed}"))
    parts.append(Text(""))

    parts.append(Text(f"Hypotheses: {d.hypothesis_count}"))
    parts.append(Text(""))

    if d.top_hypotheses:
        table = Table(title="Top Hypotheses", expand=True, show_lines=False)
        table.add_column("#", width=3, justify="right")
        table.add_column("Description", ratio=5, no_wrap=True, overflow="ellipsis")
        table.add_column("Score", width=6, justify="right")
        table.add_column("Domain", width=10, no_wrap=True, overflow="ellipsis")
        table.add_column("Status", width=8)

        for i, h in enumerate(d.top_hypotheses, 1):
            table.add_row(
                str(i),
                h.description_snippet,
                f"{h.score:.1f}",
                h.source_domain,
                h.verify_status or "-",
            )
        parts.append(table)

    return Group(*parts)


# ── Main class ──────────────────────────────────────────────────────────


class SplitScreenProgress:
    """Split-screen progress renderer.

    Primary interface: ``run(coro_factory)`` — handles both split and non-split
    modes internally.  Context manager + ``callback()`` remain available for the
    non-split fallback path.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        log_file: Path | None = None,
    ) -> None:
        is_tty = sys.stderr.isatty()
        self._enabled = is_tty if enabled is None else bool(enabled)
        self._log_file = log_file

        self._use_split = False
        if self._enabled and is_tty:
            try:
                import shutil

                cols = shutil.get_terminal_size().columns
                self._use_split = cols >= _MIN_SPLIT_WIDTH
            except Exception:
                pass

        self._inner = AnswerProgress(enabled=self._enabled and not self._use_split)

    # ── Primary interface ───────────────────────────────────────────

    def run(
        self,
        coro_factory: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Any:
        """Run an engine coroutine with progress display.

        Args:
            coro_factory: ``callback -> coroutine``.  Example::

                lambda cb: engine.answer(q, progress_callback=cb)

        Returns:
            The value returned by the engine coroutine.
        """
        if not self._use_split:
            if self._log_file is not None:
                log_buffer: deque[str] = deque(maxlen=500)
                handler, fh = _install_log_capture(log_buffer, self._log_file)
                try:
                    with self._inner:
                        return asyncio.run(coro_factory(self._inner.callback))
                finally:
                    _uninstall_log_capture(handler, fh)
            else:
                with self._inner:
                    return asyncio.run(coro_factory(self._inner.callback))

        return self._run_split(coro_factory)

    # ── Textual split-screen ────────────────────────────────────────

    def _run_split(
        self,
        coro_factory: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Any:
        """Launch a Textual app that runs the engine as a worker."""
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal
        from textual.widgets import RichLog, Static

        log_buffer: deque[str] = deque(maxlen=500)
        log_file = self._log_file
        display = _DisplayState()
        inner = AnswerProgress(enabled=False)  # state machine only
        started_at = time.monotonic()

        class _App(App):
            CSS = (
                "Horizontal { height: 100%; }\n"
                "#log-panel  { width: 3fr; border: round blue;  padding: 0 1; }\n"
                "#summary    { width: 2fr; border: round green; padding: 0 1; }\n"
            )

            def __init__(self) -> None:
                super().__init__()
                self.engine_result: Any = None
                self.engine_error: BaseException | None = None
                self._log_handler: _LogBufferHandler | None = None
                self._log_fh: IO[str] | None = None

            def compose(self) -> ComposeResult:
                with Horizontal():
                    yield RichLog(id="log-panel", highlight=False, auto_scroll=True)
                    yield Static(id="summary")

            def on_mount(self) -> None:
                self.query_one("#log-panel").border_title = "Log"
                self.query_one("#summary").border_title = "Summary"

                self._log_handler, self._log_fh = _install_log_capture(
                    log_buffer, log_file
                )
                self.set_interval(1 / 8, self._flush_logs)
                self.set_interval(1 / 4, self._refresh_summary)
                self.run_worker(self._engine_worker(), exclusive=True)

            async def _engine_worker(self) -> None:
                try:
                    self.engine_result = await coro_factory(self._on_progress)
                except Exception as exc:
                    self.engine_error = exc
                finally:
                    _uninstall_log_capture(self._log_handler, self._log_fh)
                    self._flush_logs()  # final flush
                    self.exit()

            def _on_progress(self, event: str, payload: dict[str, Any]) -> None:
                inner._ingest(event, payload or {})
                _sync_display_state(display, inner, payload or {}, started_at)

            def _flush_logs(self) -> None:
                try:
                    widget = self.query_one("#log-panel", RichLog)
                except Exception:
                    return
                while log_buffer:
                    line = log_buffer.popleft()
                    widget.write(Text.from_ansi(line))

            def _refresh_summary(self) -> None:
                try:
                    widget = self.query_one("#summary", Static)
                except Exception:
                    return
                widget.update(_build_summary_renderable(display))

        app = _App()
        app.run()

        if app.engine_error is not None:
            raise app.engine_error
        return app.engine_result

    # ── Context manager + callback (non-split fallback) ─────────────

    def __enter__(self) -> SplitScreenProgress:
        if not self._use_split:
            self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self._use_split:
            self._inner.__exit__(exc_type, exc, tb)

    def callback(self, event: str, payload: dict[str, Any]) -> None:
        """progress_callback(event, payload) compatible entry point (non-split only)."""
        if not self._enabled:
            return
        self._inner.callback(event, payload)
