"""CLI progress renderer for `x-creative answer`.

This consumes the engine-wide `progress_callback(event, payload)` stream and
renders a stage-based overall progress bar with dynamic counters + ETA.

Design goals:
- Keep logs intact (progress draws to stderr).
- Auto-disable when not in an interactive TTY (tests/CI/pipes).
- Be cheap: callback is synchronous and throttles renders.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


_STAGE_INDEX: dict[str, int] = {
    "problem": 1,
    "target": 2,
    "sources": 3,
    "biso": 4,
    "search": 5,
    "verify": 6,
    "solve": 7,
    "finalize": 8,
}


@dataclass
class _Counters:
    # BISO
    biso_total_domains: int | None = None
    biso_completed_domains: int = 0
    biso_generated_total: int = 0

    # SEARCH
    search_total_rounds: int | None = None
    search_round_index: int = 0
    search_pool_size: int | None = None
    search_new_count: int | None = None

    # VERIFY
    verify_score_total: int | None = None
    verify_score_done: int = 0
    verify_dual_total: int | None = None
    verify_dual_done: int = 0

    # SOLVE
    solve_outer_max: int | None = None
    solve_outer_round: int = 0
    solve_reasoning_steps: int | None = None


class AnswerProgress:
    """Render progress bar + ETA for the `answer` CLI command."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        refresh_hz: float = 10.0,
    ) -> None:
        self.enabled = sys.stderr.isatty() if enabled is None else bool(enabled)
        self._refresh_s = 1.0 / max(1.0, float(refresh_hz))

        self._console = Console(stderr=True)
        self._progress: Progress | None = None
        self._task_id: int | None = None

        self._started_at: float | None = None
        self._last_render_at: float = 0.0

        self._session_id: str | None = None
        self._stage_total: int = 8
        self._stage_index: int = 0
        self._stage_name: str = "init"
        self._stage_progress: float = 0.0
        self._last_completed_units: int = 0

        self._c = _Counters()

    def __enter__(self) -> "AnswerProgress":
        if not self.enabled:
            return self
        self._started_at = time.monotonic()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("Starting...", total=1000)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task_id = None

    # ---------------------------------------------------------------------
    # Public callback API
    # ---------------------------------------------------------------------

    def callback(self, event: str, payload: dict[str, Any]) -> None:
        """`progress_callback(event, payload)` compatible entry point."""
        self.on_event(event, payload)

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        """Consume a progress event and update internal counters/state."""
        if not self.enabled:
            return

        self._ingest(event, payload or {})

        if self._progress is None or self._task_id is None:
            return

        now = time.monotonic()
        stage_change = event in {
            "answer_started",
            "answer_stage_started",
            "answer_stage_completed",
            "biso_started",
            "biso_completed",
            "search_started",
            "search_completed",
            "verify_started",
            "verify_completed",
            "run_started",
            "run_completed",
            "answer_completed",
        }
        if not stage_change and (now - self._last_render_at) < self._refresh_s:
            return
        self._last_render_at = now

        desc = self._format_description()
        completed = self._compute_completed_units()
        self._progress.update(self._task_id, description=desc, completed=completed)

    # ---------------------------------------------------------------------
    # Internal state machine
    # ---------------------------------------------------------------------

    def _ingest(self, event: str, payload: dict[str, Any]) -> None:
        # Session/stage bookkeeping from AnswerEngine.
        if event == "answer_started":
            self._session_id = str(payload.get("session_id") or "") or None
            self._stage_total = int(payload.get("stage_total") or self._stage_total)
            return

        if event in {"answer_stage_started", "answer_stage_completed"}:
            stage = str(payload.get("stage") or "").strip()
            if stage:
                self._set_stage(stage)
            if event == "answer_stage_completed":
                # Treat pre-SAGA stages as instantly completed.
                self._stage_progress = 1.0
            else:
                self._stage_progress = 0.0
            return

        # Generic stage detection for SAGA events (EventBus forwards pipeline_stage).
        stage = str(payload.get("pipeline_stage") or "").strip()
        if stage:
            self._set_stage(stage)

        # BISO
        if event == "biso_started":
            self._set_stage("biso")
            total_domains = payload.get("total_domains")
            if isinstance(total_domains, int) and total_domains > 0:
                self._c.biso_total_domains = total_domains
            self._c.biso_completed_domains = 0
            self._c.biso_generated_total = 0
            self._stage_progress = 0.0
            return

        if event == "biso_domain_completed":
            completed = payload.get("completed")
            total = payload.get("total")
            generated = payload.get("generated")
            if isinstance(total, int) and total > 0:
                self._c.biso_total_domains = total
            if isinstance(completed, int) and completed >= 0:
                self._c.biso_completed_domains = completed
            if isinstance(generated, int) and generated >= 0:
                self._c.biso_generated_total += generated
            self._stage_progress = self._ratio(
                self._c.biso_completed_domains, self._c.biso_total_domains
            )
            return

        if event == "biso_completed":
            self._set_stage("biso")
            self._stage_progress = 1.0
            return

        # SEARCH
        if event == "search_started":
            self._set_stage("search")
            total_rounds = payload.get("total_rounds")
            if isinstance(total_rounds, int) and total_rounds >= 0:
                self._c.search_total_rounds = total_rounds
            self._c.search_round_index = 0
            self._c.search_pool_size = payload.get("initial_count") if isinstance(payload.get("initial_count"), int) else None
            self._c.search_new_count = None
            self._stage_progress = 0.0
            return

        if event == "search_round_completed":
            round_index = payload.get("round_index")
            total_rounds = payload.get("total_rounds")
            if isinstance(total_rounds, int) and total_rounds >= 0:
                self._c.search_total_rounds = total_rounds
            if isinstance(round_index, int) and round_index >= 0:
                self._c.search_round_index = round_index
            if isinstance(payload.get("hypothesis_count"), int):
                self._c.search_pool_size = int(payload["hypothesis_count"])
            if isinstance(payload.get("new_count"), int):
                self._c.search_new_count = int(payload["new_count"])
            self._stage_progress = self._ratio(
                self._c.search_round_index, self._c.search_total_rounds
            )
            return

        if event == "search_completed":
            self._set_stage("search")
            self._stage_progress = 1.0
            return

        # VERIFY
        if event == "verify_started":
            self._set_stage("verify")
            score_total = payload.get("score_total")
            if isinstance(score_total, int) and score_total >= 0:
                self._c.verify_score_total = score_total
            self._c.verify_score_done = 0
            self._c.verify_dual_total = None
            self._c.verify_dual_done = 0
            self._stage_progress = 0.0
            return

        if event == "verify_hypothesis_scored":
            phase = str(payload.get("phase") or "").strip().lower()
            completed = payload.get("completed")
            total = payload.get("total")
            if phase == "scoring":
                if isinstance(total, int) and total >= 0:
                    self._c.verify_score_total = total
                if isinstance(completed, int) and completed >= 0:
                    self._c.verify_score_done = completed
            elif phase == "dual_verify":
                if isinstance(total, int) and total >= 0:
                    self._c.verify_dual_total = total
                if isinstance(completed, int) and completed >= 0:
                    self._c.verify_dual_done = completed
            self._stage_progress = self._verify_stage_progress()
            return

        if event in {"verify_completed", "verify_batch_scored"}:
            self._set_stage("verify")
            # verify_completed semantics differ between SAGA and non-SAGA pipelines.
            # Treat it as stage completion (UI-wise).
            if event == "verify_completed":
                self._stage_progress = 1.0
            return

        # SOLVE
        if event == "run_started":
            self._set_stage("solve")
            self._c.solve_outer_round = 0
            self._c.solve_outer_max = None
            self._c.solve_reasoning_steps = None
            self._stage_progress = 0.0
            return

        if event == "refine_outer_round":
            self._set_stage("solve")
            rnd = payload.get("round")
            mx = payload.get("max_rounds")
            if isinstance(mx, int) and mx > 0:
                self._c.solve_outer_max = mx
            if isinstance(rnd, int) and rnd >= 0:
                self._c.solve_outer_round = rnd - 1  # "round 1" means progress just started
            self._stage_progress = self._ratio(
                self._c.solve_outer_round, self._c.solve_outer_max
            )
            return

        if event == "solve_round_completed":
            self._set_stage("solve")
            outer_round = payload.get("outer_round")
            outer_max = payload.get("outer_max")
            if isinstance(outer_max, int) and outer_max > 0:
                self._c.solve_outer_max = outer_max
            if isinstance(outer_round, int) and outer_round >= 0:
                self._c.solve_outer_round = outer_round
            if isinstance(payload.get("reasoning_steps"), int):
                self._c.solve_reasoning_steps = int(payload["reasoning_steps"])
            self._stage_progress = self._ratio(
                self._c.solve_outer_round, self._c.solve_outer_max
            )
            return

        if event == "run_completed":
            self._set_stage("solve")
            self._stage_progress = 1.0
            return

        # Finalize
        if event == "answer_completed":
            self._set_stage("finalize")
            self._stage_progress = 1.0
            return

    def _set_stage(self, stage: str) -> None:
        stage = stage.strip()
        if stage in _STAGE_INDEX:
            self._stage_name = stage
            self._stage_index = _STAGE_INDEX[stage]

    # ---------------------------------------------------------------------
    # Formatting / progress math
    # ---------------------------------------------------------------------

    def _compute_completed_units(self) -> int:
        if self._stage_total <= 0 or self._stage_index <= 0:
            return 0
        stage_progress = max(0.0, min(1.0, float(self._stage_progress)))
        overall = ((self._stage_index - 1) + stage_progress) / float(self._stage_total)
        units = int(round(max(0.0, min(1.0, overall)) * 1000))
        # Keep monotonic for stable ETA.
        if units < self._last_completed_units:
            units = self._last_completed_units
        self._last_completed_units = units
        return units

    def _format_description(self) -> str:
        prefix = ""
        if self._stage_index > 0:
            prefix = f"Stage {self._stage_index}/{self._stage_total} "

        stage = self._stage_name
        details = self._stage_details(stage)
        if details:
            msg = f"{prefix}{stage}: {details}"
        else:
            msg = f"{prefix}{stage}"

        # Keep session id short but useful.
        if self._session_id and stage == "problem":
            msg = f"{msg} (session {self._session_id})"
        return msg

    def _stage_details(self, stage: str) -> str:
        c = self._c
        if stage == "biso":
            total = c.biso_total_domains
            if total is None:
                return f"domains {c.biso_completed_domains}/?"
            return f"domains {c.biso_completed_domains}/{total} (gen {c.biso_generated_total})"
        if stage == "search":
            total = c.search_total_rounds
            if total is None or total <= 0:
                return "round ?/?"
            parts = [f"round {c.search_round_index}/{total}"]
            if c.search_pool_size is not None:
                parts.append(f"pool {c.search_pool_size}")
            if c.search_new_count is not None:
                parts.append(f"+{c.search_new_count}")
            return " | ".join(parts)
        if stage == "verify":
            parts: list[str] = []
            if c.verify_score_total is not None:
                parts.append(f"score {c.verify_score_done}/{c.verify_score_total}")
            else:
                parts.append(f"score {c.verify_score_done}/?")
            if c.verify_dual_total is not None:
                parts.append(f"dual {c.verify_dual_done}/{c.verify_dual_total}")
            return " | ".join(parts)
        if stage == "solve":
            parts: list[str] = []
            if c.solve_outer_max is not None:
                parts.append(f"outer {c.solve_outer_round}/{c.solve_outer_max}")
            if c.solve_reasoning_steps is not None:
                parts.append(f"steps {c.solve_reasoning_steps}")
            return " | ".join(parts)
        if stage == "problem":
            return "frame"
        if stage == "target":
            return "resolve domain"
        if stage == "sources":
            return "select domains"
        if stage == "finalize":
            return "pack output"
        return ""

    @staticmethod
    def _ratio(numer: int, denom: int | None) -> float:
        if denom is None or denom <= 0:
            return 0.0
        return max(0.0, min(1.0, float(numer) / float(denom)))

    def _verify_stage_progress(self) -> float:
        c = self._c
        score_total = c.verify_score_total or 0
        dual_total = c.verify_dual_total or 0
        total = score_total + dual_total
        if total <= 0:
            return 0.0
        done = max(0, c.verify_score_done) + max(0, c.verify_dual_done)
        return max(0.0, min(1.0, float(done) / float(total)))
