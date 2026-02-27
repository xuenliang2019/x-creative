"""Constraint budget, conflict detection, and repeated-risk detection.

Prevents constraint inflation in the adaptive risk refinement loop.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from x_creative.core.types import ConstraintSpec


class RepeatedRiskResult(BaseModel):
    """Result of repeated risk detection."""
    has_repeats: bool = Field(default=False)
    repeated_pairs: list[tuple[str, str]] = Field(default_factory=list)


class ConstraintConflictResult(BaseModel):
    """Result of constraint conflict detection."""
    has_conflicts: bool = Field(default=False)
    conflict_pairs: list[tuple[str, str]] = Field(default_factory=list)


class ConstraintActivation(BaseModel):
    """Compiled constraint activation for one solve round."""

    budgeted_constraints: list[ConstraintSpec] = Field(default_factory=list)
    hard_core: list[ConstraintSpec] = Field(default_factory=list)
    active_soft_set: list[ConstraintSpec] = Field(default_factory=list)


class ConstraintCoverageResult(BaseModel):
    """Coverage audit between active constraints and unresolved risks."""

    has_violations: bool = Field(default=False)
    uncovered_risks: list[str] = Field(default_factory=list)
    matched_pairs: list[tuple[str, str]] = Field(default_factory=list)


def apply_constraint_budget(
    constraints: list[ConstraintSpec],
    max_constraints: int = 15,
) -> list[ConstraintSpec]:
    """Apply budget to constraint set.

    Semantics:
    - All user-origin constraints are preserved (never truncated).
    - The remaining budget (if any) is used for non-user constraints using:
      1) critical/hard first
      2) then by weight (descending)

    Note: This means the returned list may exceed ``max_constraints`` when the
    user constraint set alone exceeds the budget.
    """
    user_constraints = [c for c in constraints if c.origin == "user"]
    non_user_constraints = [c for c in constraints if c.origin != "user"]

    budget_non_user = max(0, int(max_constraints) - len(user_constraints))
    if budget_non_user <= 0 or not non_user_constraints:
        return list(user_constraints)

    critical = [
        c
        for c in non_user_constraints
        if c.priority == "critical" or c.type == "hard"
    ]
    non_critical = [c for c in non_user_constraints if c not in critical]

    if len(critical) >= budget_non_user:
        return list(user_constraints) + critical[:budget_non_user]

    remaining_budget = budget_non_user - len(critical)
    non_critical_sorted = sorted(non_critical, key=lambda c: c.weight, reverse=True)
    return list(user_constraints) + critical + non_critical_sorted[:remaining_budget]


def _priority_rank(priority: str) -> int:
    order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return order.get(priority, 0)


def _text_similarity(left: str, right: str) -> float:
    """Hybrid lexical similarity for stable, zero-LLM comparisons."""
    left_terms = _constraint_terms(left)
    right_terms = _constraint_terms(right)
    if not left_terms or not right_terms:
        lexical = 0.0
    else:
        lexical = len(left_terms & right_terms) / len(left_terms | right_terms)
    seq_ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return 0.6 * lexical + 0.4 * seq_ratio


def compile_constraint_activation(
    constraints: list[ConstraintSpec],
    reference_texts: list[str] | None = None,
    max_constraints: int = 15,
    active_soft_min: int = 3,
    active_soft_max: int = 6,
) -> ConstraintActivation:
    """Compile constraints into HardCore + ActiveSoftSet for this round."""
    if active_soft_min < 0:
        active_soft_min = 0
    if active_soft_max < active_soft_min:
        active_soft_max = active_soft_min

    budgeted = apply_constraint_budget(constraints, max_constraints=max_constraints)
    hard_core = [
        c for c in budgeted if c.type == "hard" or c.priority == "critical"
    ]
    soft_pool = [c for c in budgeted if c not in hard_core]
    refs = [text for text in (reference_texts or []) if str(text).strip()]

    ranked_soft = sorted(
        soft_pool,
        key=lambda c: (
            max((_text_similarity(c.text, ref) for ref in refs), default=0.0),
            c.weight,
            _priority_rank(c.priority),
        ),
        reverse=True,
    )

    target_soft = min(active_soft_max, len(ranked_soft))
    if ranked_soft and target_soft < active_soft_min:
        target_soft = min(active_soft_min, len(ranked_soft))

    active_soft_set = ranked_soft[:target_soft]
    return ConstraintActivation(
        budgeted_constraints=budgeted,
        hard_core=hard_core,
        active_soft_set=active_soft_set,
    )


def format_constraint_prompt_block(compiled: ConstraintActivation) -> str:
    """Build prompt block with HardCore duplicated at head/tail."""
    if not compiled.budgeted_constraints:
        return ""

    head_lines = ["HardCore constraints (must satisfy):"]
    if compiled.hard_core:
        for spec in compiled.hard_core:
            head_lines.append(f"- {spec.text}")
    else:
        head_lines.append("- none")

    soft_lines = ["Active soft constraints (this round):"]
    if compiled.active_soft_set:
        for spec in compiled.active_soft_set:
            soft_lines.append(f"- {spec.text}")
    else:
        soft_lines.append("- none")

    tail_lines = ["HardCore constraints (repeat):"]
    if compiled.hard_core:
        for spec in compiled.hard_core:
            tail_lines.append(f"- {spec.text}")
    else:
        tail_lines.append("- none")

    return "\n".join(head_lines + [""] + soft_lines + [""] + tail_lines)


def audit_constraint_coverage(
    constraints: list[ConstraintSpec],
    risks: list[dict],
    similarity_threshold: float = 0.6,
) -> ConstraintCoverageResult:
    """Audit whether unresolved risks are covered by current constraints."""
    uncovered: list[str] = []
    matched_pairs: list[tuple[str, str]] = []
    constraint_texts = [c.text.strip() for c in constraints if c.text.strip()]

    for risk in risks:
        risk_text = str(risk.get("risk", "")).strip()
        if not risk_text:
            continue
        best_constraint = ""
        best_score = 0.0
        for constraint_text in constraint_texts:
            score = _text_similarity(risk_text, constraint_text)
            if score > best_score:
                best_score = score
                best_constraint = constraint_text
        if best_score >= similarity_threshold and best_constraint:
            matched_pairs.append((risk_text, best_constraint))
        else:
            uncovered.append(risk_text)

    return ConstraintCoverageResult(
        has_violations=len(uncovered) > 0,
        uncovered_risks=uncovered,
        matched_pairs=matched_pairs,
    )


def detect_repeated_risks(
    risk_history: list[list[dict]],
    similarity_threshold: float = 0.6,
) -> RepeatedRiskResult:
    """Detect if the same type of risk appears in consecutive rounds.

    Args:
        risk_history: List of risk lists from consecutive rounds.
        similarity_threshold: Minimum similarity ratio to consider a repeat.

    Returns:
        RepeatedRiskResult indicating if repeats were found.
    """
    if len(risk_history) < 2:
        return RepeatedRiskResult()

    prev_risks = risk_history[-2]
    curr_risks = risk_history[-1]

    repeated_pairs: list[tuple[str, str]] = []

    for prev in prev_risks:
        prev_text = prev.get("risk", "")
        for curr in curr_risks:
            curr_text = curr.get("risk", "")
            similarity = SequenceMatcher(None, prev_text, curr_text).ratio()
            if similarity >= similarity_threshold:
                repeated_pairs.append((prev_text, curr_text))

    return RepeatedRiskResult(
        has_repeats=len(repeated_pairs) > 0,
        repeated_pairs=repeated_pairs,
    )


_NEGATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdo not\b",
        r"\bdon't\b",
        r"\bnot\b",
        r"\bno\b",
        r"不使用",
        r"不要",
        r"禁止",
        r"不可",
    )
]

_STOPWORDS = {
    "must", "should", "use", "using", "with", "without", "data", "the", "a", "an",
    "and", "or", "to", "of", "for", "is", "be",
}


def _is_negative_constraint(text: str) -> bool:
    return any(pattern.search(text) for pattern in _NEGATION_PATTERNS)


def _constraint_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
    return {token for token in tokens if token not in _STOPWORDS}


def detect_conflicting_constraints(
    constraints: list[str],
    similarity_threshold: float = 0.6,
) -> ConstraintConflictResult:
    """Detect contradictory constraints by polarity + semantic term overlap."""
    conflict_pairs: list[tuple[str, str]] = []
    for idx, left in enumerate(constraints):
        left_text = left.strip()
        if not left_text:
            continue
        left_negative = _is_negative_constraint(left_text)
        left_terms = _constraint_terms(left_text)

        for right in constraints[idx + 1:]:
            right_text = right.strip()
            if not right_text:
                continue
            right_negative = _is_negative_constraint(right_text)
            if left_negative == right_negative:
                continue

            right_terms = _constraint_terms(right_text)
            if not left_terms or not right_terms:
                continue
            overlap = left_terms & right_terms
            min_size = min(len(left_terms), len(right_terms))
            if min_size == 0:
                continue
            if len(overlap) / min_size >= similarity_threshold:
                conflict_pairs.append((left_text, right_text))

    return ConstraintConflictResult(
        has_conflicts=len(conflict_pairs) > 0,
        conflict_pairs=conflict_pairs,
    )


def resolve_conflicting_constraints(
    constraints: list[ConstraintSpec],
    conflict_pairs: list[tuple[str, str]],
) -> list[ConstraintSpec]:
    """Resolve conflicts by dropping non-user constraints first.

    Policy:
    - If a conflict involves a user-origin constraint and a non-user constraint,
      drop the non-user constraint.
    - If both are non-user, drop the right side (deterministic, compatible with
      prior behavior).
    - If both are user constraints, do not auto-drop (caller should treat this
      as a fatal preflight failure upstream).
    """
    if not constraints or not conflict_pairs:
        return list(constraints)

    # Map exact text -> indices for stable drops without relying on object identity.
    text_to_indices: dict[str, list[int]] = {}
    for idx, spec in enumerate(constraints):
        text_to_indices.setdefault(spec.text.strip(), []).append(idx)

    drop_indices: set[int] = set()
    for left_text, right_text in conflict_pairs:
        left_candidates = [
            i for i in text_to_indices.get(left_text.strip(), []) if i not in drop_indices
        ]
        right_candidates = [
            i for i in text_to_indices.get(right_text.strip(), []) if i not in drop_indices
        ]
        if not left_candidates or not right_candidates:
            continue

        li = left_candidates[0]
        ri = right_candidates[0]
        left_origin = constraints[li].origin
        right_origin = constraints[ri].origin

        if left_origin == "user" and right_origin != "user":
            drop_indices.add(ri)
        elif right_origin == "user" and left_origin != "user":
            drop_indices.add(li)
        elif left_origin == "user" and right_origin == "user":
            # User-user conflicts should have been caught by preflight.
            continue
        else:
            drop_indices.add(ri)

    return [spec for idx, spec in enumerate(constraints) if idx not in drop_indices]
