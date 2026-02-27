"""User-constraint preflight for AnswerEngine.

This module enforces two invariants early (before expensive stages):
1) Detect irreconcilable contradictions among user constraints and fail fast.
2) Freeze user constraints into structured hard/critical ConstraintSpec so downstream
   components can preserve them end-to-end.
"""

from __future__ import annotations

import re
from typing import Any

from x_creative.core.types import ConstraintSpec, ProblemFrame
from x_creative.saga.constraint_checker import detect_conflicting_constraints


class UserConstraintConflictError(RuntimeError):
    """Raised when the user-provided constraint set contains contradictions."""

    def __init__(self, report: dict[str, Any]) -> None:
        message = str(report.get("message") or "User constraints contain contradictions")
        super().__init__(message)
        self.report = report


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text).strip())


def _canonical_user_constraints(problem: ProblemFrame) -> list[str]:
    """Return de-duped user constraint texts in stable order."""
    seen: set[str] = set()
    canonical: list[str] = []

    for raw in problem.constraints or []:
        text = _normalize_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        canonical.append(text)

    for spec in problem.structured_constraints or []:
        if getattr(spec, "origin", "user") != "user":
            continue
        text = _normalize_text(spec.text)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        canonical.append(text)

    return canonical


def preflight_user_constraints(problem: ProblemFrame) -> ProblemFrame:
    """Validate user constraints, raising on contradictions.

    On success, freezes user constraints into structured hard/critical specs and
    returns an updated ProblemFrame (without mutating the input).
    """
    user_constraints = _canonical_user_constraints(problem)
    constraints_index = {text: f"C{i}" for i, text in enumerate(user_constraints, start=1)}

    conflicts = detect_conflicting_constraints(user_constraints)
    if conflicts.has_conflicts:
        conflict_pairs = []
        for left, right in conflicts.conflict_pairs:
            left_id = constraints_index.get(left, "")
            right_id = constraints_index.get(right, "")
            conflict_pairs.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_text": left,
                    "right_text": right,
                }
            )
        report = {
            "message": f"Found {len(conflict_pairs)} conflicting user-constraint pairs",
            "constraints": [
                {"id": constraints_index[text], "text": text} for text in user_constraints
            ],
            "conflict_pairs": conflict_pairs,
        }
        raise UserConstraintConflictError(report)

    # Freeze: all user constraints become hard/critical and are preserved downstream.
    structured = [
        ConstraintSpec(
            text=text,
            origin="user",
            type="hard",
            priority="critical",
            weight=1.0,
        )
        for text in user_constraints
    ]
    return problem.model_copy(
        update={
            "constraints": list(user_constraints),
            "structured_constraints": structured,
        }
    )

