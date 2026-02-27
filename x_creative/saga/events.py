"""Event system for SAGA Fast/Slow Agent communication.

Provides the event bus for asynchronous communication:
- Fast Agent → Slow Agent: Events (observations from pipeline stages)
- Slow Agent → Fast Agent: Directives (intervention commands)

Events are non-blocking. Directives can be blocking (e.g. PAUSE_PIPELINE).
All events and directives are optionally persisted to JSONL for post-hoc analysis.
"""

import asyncio
import inspect
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger()

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class EventType(str, Enum):
    """Event types emitted by Fast Agent."""

    # xc-domain events
    DOMAIN_LOADED = "domain_loaded"
    DOMAIN_ADDED = "domain_added"
    DOMAIN_EXTENDED = "domain_extended"

    # BISO stage events
    BISO_STARTED = "biso_started"
    BISO_DOMAIN_COMPLETED = "biso_domain_completed"
    BISO_HYPOTHESIS_BATCH = "biso_hypothesis_batch"
    BISO_COMPLETED = "biso_completed"

    # SEARCH stage events
    SEARCH_STARTED = "search_started"
    SEARCH_ROUND_COMPLETED = "search_round_completed"
    SEARCH_EXPANDED = "search_expanded"
    SEARCH_COMPLETED = "search_completed"

    # VERIFY stage events
    VERIFY_STARTED = "verify_started"
    VERIFY_HYPOTHESIS_SCORED = "verify_hypothesis_scored"
    VERIFY_BATCH_SCORED = "verify_batch_scored"
    VERIFY_COMPLETED = "verify_completed"

    # Pipeline lifecycle
    PIPELINE_COMPLETED = "pipeline_completed"

    # SOLVE stage events
    SOLVE_ROUND_COMPLETED = "solve_round_completed"

    # HKG events
    HKG_PATH_FOUND = "hkg_path_found"
    HKG_PATH_NOT_FOUND = "hkg_path_not_found"
    HKG_EXPANSION_CREATED = "hkg_expansion_created"

    # Mapping quality events
    MAPPING_TABLE_MISSING = "mapping_table_missing"
    MAPPING_LOW_SYSTEMATICITY = "mapping_low_systematicity"
    MAPPING_PADDING_SUSPECTED = "mapping_padding_suspected"

    # VERIFY confidence events
    VERIFY_POSITION_INCONSISTENT = "verify_position_inconsistent"
    VERIFY_ESCALATED = "verify_escalated"
    VERIFY_ABSTAINED = "verify_abstained"
    VERIFY_INJECTION_DETECTED = "verify_injection_detected"

    # Constraint events
    CONSTRAINT_SET_GROWTH_ALERT = "constraint_set_growth_alert"
    CONSTRAINT_CONFLICT_DETECTED = "constraint_conflict_detected"
    CONSTRAINT_REFRAME_TRIGGERED = "constraint_reframe_triggered"
    CONSTRAINT_VIOLATION_FOUND = "constraint_violation_found"
    CONSTRAINT_PATCH_APPLIED = "constraint_patch_applied"

    # Blend events
    BLEND_EXPAND_COMPLETED = "blend_expand_completed"

    # Transform space events
    TRANSFORM_PROPOSED = "transform_proposed"
    TRANSFORM_ACCEPTED_OR_REJECTED = "transform_accepted_or_rejected"
    TRANSFORM_SPACE_APPLIED = "transform_space_applied"

    # CK events (for batch 3)
    CK_PHASE_TRANSITION = "ck_phase_transition"
    CK_K_EXPANSION_TRIGGERED = "ck_k_expansion_triggered"
    CK_KC_WRITEBACK = "ck_kc_writeback"


class DirectiveType(str, Enum):
    """Directive types issued by Slow Agent."""

    ADJUST_WEIGHTS = "adjust_weights"
    ADJUST_SEARCH_PARAMS = "adjust_search_params"
    INJECT_CHALLENGE = "inject_challenge"
    PAUSE_PIPELINE = "pause_pipeline"
    RESUME_PIPELINE = "resume_pipeline"
    FLAG_HYPOTHESIS = "flag_hypothesis"
    RESCORE_BATCH = "rescore_batch"
    ADD_CONSTRAINT = "add_constraint"


class FastAgentEvent(BaseModel):
    """An event emitted by Fast Agent during pipeline execution."""

    event_type: EventType = Field(..., description="Type of event")
    stage: str = Field(..., description="Current pipeline stage identifier")
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of event emission",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event data (hypotheses, scores, etc.)",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Statistical metrics (diversity, distribution features, etc.)",
    )


class SlowAgentDirective(BaseModel):
    """A directive issued by Slow Agent to Fast Agent."""

    directive_type: DirectiveType = Field(..., description="Type of directive")
    reason: str = Field(..., description="Human-readable trigger reason")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level of the directive"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Directive parameters",
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Priority (1=highest, 10=lowest)"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of directive emission",
    )


class EventBus:
    """Async event bus for Fast↔Slow Agent communication.

    Uses asyncio.Queue for both event and directive channels.
    Supports optional JSONL file persistence for post-hoc analysis.

    Args:
        session_dir: If provided, events and directives are persisted
            to JSONL files in this directory.
    """

    def __init__(
        self,
        session_dir: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._event_queue: asyncio.Queue[FastAgentEvent] = asyncio.Queue()
        self._directive_queue: asyncio.Queue[SlowAgentDirective] = asyncio.Queue()
        self._stopped = False
        self._session_dir = session_dir
        self._progress_callback: ProgressCallback | None = progress_callback

        if session_dir:
            session_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = session_dir / "events.jsonl"
            self._directives_file = session_dir / "directives.jsonl"
        else:
            self._events_file = None
            self._directives_file = None

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Attach an optional progress callback for streaming UI updates.

        The callback receives (event_type, payload) for each emitted event.
        """
        self._progress_callback = callback

    async def emit_event(self, event: FastAgentEvent) -> None:
        """Emit an event from Fast Agent to Slow Agent.

        Non-blocking: Fast Agent continues execution after emitting.

        Args:
            event: The event to emit.
        """
        await self._event_queue.put(event)
        self._persist_event(event)
        cb = self._progress_callback
        if cb is not None:
            try:
                forwarded = dict(event.payload or {})
                forwarded["pipeline_stage"] = event.stage
                forwarded["metrics"] = dict(event.metrics or {})
                forwarded["timestamp"] = float(event.timestamp)
                maybe_awaitable = cb(event.event_type.value, forwarded)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                logger.debug(
                    "Progress callback failed",
                    event_type=event.event_type.value,
                    stage=event.stage,
                    error=str(exc),
                )
        logger.debug(
            "Event emitted",
            event_type=event.event_type.value,
            stage=event.stage,
        )

    async def subscribe_events(self) -> AsyncIterator[FastAgentEvent]:
        """Subscribe to Fast Agent events (used by Slow Agent).

        Yields events until stop() is called and the queue is drained.

        Yields:
            FastAgentEvent instances as they arrive.
        """
        while True:
            if self._stopped and self._event_queue.empty():
                break
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.5
                )
                yield event
            except asyncio.TimeoutError:
                if self._stopped:
                    break
                continue

    async def emit_directive(self, directive: SlowAgentDirective) -> None:
        """Emit a directive from Slow Agent to Fast Agent.

        Args:
            directive: The directive to emit.
        """
        await self._directive_queue.put(directive)
        self._persist_directive(directive)
        logger.debug(
            "Directive emitted",
            directive_type=directive.directive_type.value,
            priority=directive.priority,
        )

    async def poll_directives(self) -> list[SlowAgentDirective]:
        """Poll all pending directives (used by Fast Agent at checkpoints).

        Returns all currently queued directives sorted by priority
        (lowest number = highest priority).

        Returns:
            List of pending directives, sorted by priority.
        """
        directives: list[SlowAgentDirective] = []
        while not self._directive_queue.empty():
            try:
                directive = self._directive_queue.get_nowait()
                directives.append(directive)
            except asyncio.QueueEmpty:
                break
        directives.sort(key=lambda d: d.priority)
        return directives

    def stop(self) -> None:
        """Signal the event bus to stop.

        After calling stop(), subscribe_events() will drain remaining
        events and then exit.
        """
        self._stopped = True
        logger.debug("EventBus stop signal received")

    @property
    def is_stopped(self) -> bool:
        """Whether the event bus has been stopped."""
        return self._stopped

    def _persist_event(self, event: FastAgentEvent) -> None:
        """Persist event to JSONL file if session_dir is configured."""
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception as e:
            logger.warning("Failed to persist event", error=str(e))

    def _persist_directive(self, directive: SlowAgentDirective) -> None:
        """Persist directive to JSONL file if session_dir is configured."""
        if self._directives_file is None:
            return
        try:
            with open(self._directives_file, "a", encoding="utf-8") as f:
                f.write(directive.model_dump_json() + "\n")
        except Exception as e:
            logger.warning("Failed to persist directive", error=str(e))
