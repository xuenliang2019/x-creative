"""Core data types for the Answer Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from x_creative.core.types import ProblemFrame

_VALID_MODES = {"quick", "deep_research", "exhaustive"}


@dataclass
class AnswerConfig:
    """Configuration for a single answer-engine run."""

    budget: int = 120
    mode: str = "deep_research"
    target_domain: str = "auto"
    num_per_domain: int = 3
    search_depth: int = 3
    search_breadth: int = 5
    verify_threshold: float = 5.0
    verify_top: int = 50
    max_ideas: int = 8
    auto_refine: bool = True
    inner_max: int = 3
    outer_max: int = 2
    hkg_enabled: bool = True
    saga_enabled: bool = True
    saga_strategy: str = "ANOMALY_DRIVEN"
    fresh: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {self.mode!r}"
            )


@dataclass
class FrameBuildResult:
    """Result of building a ProblemFrame from a user question."""

    frame: ProblemFrame | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    partial_frame: ProblemFrame | None = None
    confidence: float = 1.0


@dataclass
class AnswerPack:
    """Final output of the answer engine."""

    question: str = ""
    answer_md: str = ""
    answer_json: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    needs_clarification: bool = False
    clarification_question: str | None = None
    partial_frame: ProblemFrame | None = None

    @classmethod
    def clarification_needed(
        cls,
        partial_frame: ProblemFrame | None,
        question: str,
    ) -> AnswerPack:
        """Factory for a clarification-needed response."""
        return cls(
            needs_clarification=True,
            clarification_question=question,
            partial_frame=partial_frame,
        )
