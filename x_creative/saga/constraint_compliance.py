"""Strict user-constraint compliance audit and error types."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from x_creative.core.types import ProblemFrame
from x_creative.creativity.utils import extract_json_object

logger = logging.getLogger(__name__)

# LLMs may return verdict in non-canonical forms; normalise to Literal values.
_VERDICT_MAP: dict[str, Literal["pass", "fail", "unknown"]] = {
    "pass": "pass",
    "passed": "pass",
    "satisfied": "pass",
    "compliant": "pass",
    "yes": "pass",
    "true": "pass",
    "fail": "fail",
    "failed": "fail",
    "violated": "fail",
    "non-compliant": "fail",
    "no": "fail",
    "false": "fail",
    "unknown": "unknown",
    "partial": "unknown",
    "unclear": "unknown",
    "n/a": "unknown",
}


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


def _normalise_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a single audit item dict so it survives Pydantic validation.

    Handles:
    - verdict case-insensitivity and synonyms (e.g. "Pass", "satisfied")
    - field name aliases ("constraint_text" → "text")
    - missing ``text`` field (fill with id)
    """
    out = dict(raw)

    # Normalise verdict
    v = str(out.get("verdict", "unknown")).strip().lower()
    out["verdict"] = _VERDICT_MAP.get(v, "unknown")

    # Field aliases: some models use "constraint_text" or "constraint" instead of "text"
    if "text" not in out or not out["text"]:
        for alt in ("constraint_text", "constraint", "description"):
            if alt in out and out[alt]:
                out["text"] = out[alt]
                break
        else:
            out["text"] = str(out.get("id", ""))

    return out


def _extract_json_object(text: str) -> dict[str, Any]:
    return extract_json_object(text or "")


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
    )

    raw_content = str(getattr(result, "content", ""))
    data = _extract_json_object(raw_content)

    if not data:
        logger.warning(
            "Audit: extract_json_object returned empty. Raw content:\n%s",
            raw_content[:2000],
        )

    # Normalise each item before Pydantic validation
    if "items" in data and isinstance(data["items"], list):
        data["items"] = [_normalise_item(item) for item in data["items"] if isinstance(item, dict)]

    try:
        report = ConstraintComplianceReport.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "Audit: Pydantic validation failed after normalisation. "
            "data_keys=%s, error=%s",
            list(data.keys()) if data else "empty",
            str(exc)[:500],
        )
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

