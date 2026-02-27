"""Strict user-constraint compliance audit and error types."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from x_creative.core.types import ProblemFrame


class UserConstraintComplianceError(RuntimeError):
    """Raised when the final solution cannot satisfy all user constraints."""

    def __init__(self, report: dict[str, Any]) -> None:
        message = str(report.get("message") or "User constraint compliance failed")
        super().__init__(message)
        self.report = report


class ConstraintComplianceItem(BaseModel):
    id: str
    text: str
    verdict: Literal["pass", "fail", "unknown"]
    rationale: str = ""
    suggested_fix: str = ""


class ConstraintComplianceReport(BaseModel):
    overall_pass: bool = False
    items: list[ConstraintComplianceItem] = Field(default_factory=list)


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> dict[str, Any]:
    match = _JSON_OBJ_RE.search(text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _canonical_user_constraints(problem: ProblemFrame) -> list[tuple[str, str]]:
    """Return user constraint list as (C#, text) in stable order with dedupe."""
    texts: list[str] = []
    seen: set[str] = set()

    for spec in getattr(problem, "structured_constraints", []) or []:
        if getattr(spec, "origin", "user") != "user":
            continue
        text = str(getattr(spec, "text", "")).strip()
        if not text:
            continue
        key = " ".join(text.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        texts.append(" ".join(text.split()))

    if not texts:
        for raw in getattr(problem, "constraints", []) or []:
            text = str(raw).strip()
            if not text:
                continue
            key = " ".join(text.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(" ".join(text.split()))

    return [(f"C{i}", text) for i, text in enumerate(texts, start=1)]


async def audit_user_constraints(
    router: Any,
    problem: ProblemFrame,
    solution_markdown: str,
) -> ConstraintComplianceReport:
    """Audit whether solution satisfies all user constraints.

    Returns a structured report. The caller decides whether to revise or fail.
    """
    constraints = _canonical_user_constraints(problem)
    if not constraints:
        return ConstraintComplianceReport(overall_pass=True, items=[])

    constraints_text = "\n".join([f"- {cid}: {text}" for cid, text in constraints])
    prompt = (
        "You are a strict compliance auditor.\n"
        "Given a set of user hard constraints (C#) and a draft solution markdown, "
        "judge whether each constraint is satisfied.\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "overall_pass": true|false,\n'
        '  "items": [\n'
        '    {"id":"C1","text":"...","verdict":"pass|fail|unknown","rationale":"...","suggested_fix":"..."}\n'
        "  ]\n"
        "}\n\n"
        f"User hard constraints:\n{constraints_text}\n\n"
        f"Solution markdown:\n{solution_markdown}\n"
    )

    result = await router.complete(
        task="constraint_compliance_audit",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )

    data = _extract_json_object(str(getattr(result, "content", "")))
    try:
        report = ConstraintComplianceReport.model_validate(data)
    except ValidationError:
        # Best-effort fallback: treat as failure and include a synthetic unknown.
        items = [
            ConstraintComplianceItem(
                id=cid,
                text=text,
                verdict="unknown",
                rationale="audit_response_parse_failed",
                suggested_fix="Return valid JSON with per-constraint verdicts",
            )
            for cid, text in constraints
        ]
        report = ConstraintComplianceReport(overall_pass=False, items=items)

    # Ensure all constraints appear at least once (fill missing as unknown).
    present = {item.id for item in report.items}
    for cid, text in constraints:
        if cid not in present:
            report.items.append(
                ConstraintComplianceItem(
                    id=cid,
                    text=text,
                    verdict="unknown",
                    rationale="missing_in_audit_output",
                    suggested_fix="Explicitly assess this constraint",
                )
            )
            report.overall_pass = False

    return report

