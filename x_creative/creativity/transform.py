"""Transform Space operator (Boden's transformational creativity)."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from x_creative.core.transform_types import (
    SpaceTransformDiff,
    TransformAction,
    TransformStatus,
)
from x_creative.core.types import FailureMode, Hypothesis

if TYPE_CHECKING:
    from x_creative.core.concept_space import ConceptSpace

logger = structlog.get_logger()


async def transform_space(
    hypothesis: Hypothesis,
    concept_space: "ConceptSpace",
    router: Any,
) -> list[Hypothesis]:
    """Apply space transformations to generate new hypotheses.

    Uses the ConceptSpace's allowed_ops and mutable assumptions to
    generate transformations that alter the conceptual framework.
    """
    prompt = _build_transform_prompt(hypothesis, concept_space)

    try:
        response = await router.complete(
            task="transform_space",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content if hasattr(response, "content") else str(response)
        transforms_data = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("transform_space: failed to parse LLM response", error=str(e))
        return []

    if not isinstance(transforms_data, list):
        transforms_data = [transforms_data]

    results: list[Hypothesis] = []
    for raw in transforms_data:
        try:
            hyp = _parse_transform_result(hypothesis, concept_space, raw)
            results.append(hyp)
        except Exception as e:
            logger.warning("transform_space: failed to parse transform", error=str(e))
            continue

    return results


def _build_transform_prompt(h: Hypothesis, cs: "ConceptSpace") -> str:
    ops_text = "\n".join(
        f"  - {op.id}: {op.name} ({op.op_type} on {op.target_type}): {op.description}"
        for op in cs.allowed_ops
    )
    assumptions_text = "\n".join(
        f"  - {a.id}: {a.text} (mutable={a.mutable})"
        for a in cs.assumptions_mutable
    )
    constraints_text = "\n".join(
        f"  - {c.id}: {c.text}"
        for c in cs.hard_constraints
    )

    return f"""You are a Boden Transformational Creativity expert.

Given a hypothesis and a concept space, apply one or more allowed transformations
to create novel hypotheses that alter the conceptual framework itself.

Hypothesis: {h.description}
  Observable: {h.observable}
  Source domain: {h.source_domain}

Concept Space (v{cs.version}):
  Mutable assumptions:
{assumptions_text}
  Hard constraints:
{constraints_text}
  Allowed operations:
{ops_text}

For each transformation, return a JSON array of objects with:
- "target_id": ID of the element being transformed
- "op_id": ID of the allowed operation
- "op_type": operation type
- "before_state": original state
- "after_state": new state
- "rationale": why this transformation
- "description": new hypothesis description
- "observable": new observable
- "failure_modes": [{{"scenario": str, "why_breaks": str, "detectable_signal": str}}]

Respond ONLY with valid JSON array."""


def _parse_transform_result(
    h: Hypothesis, cs: "ConceptSpace", raw: dict[str, Any]
) -> Hypothesis:
    action = TransformAction(
        op_id=raw.get("op_id", "unknown"),
        op_type=raw.get("op_type", "unknown"),
        target_id=raw.get("target_id", "unknown"),
        before_state=raw.get("before_state", ""),
        after_state=raw.get("after_state", ""),
        rationale=raw.get("rationale", ""),
    )

    failure_modes = [
        FailureMode(
            scenario=fm["scenario"],
            why_breaks=fm["why_breaks"],
            detectable_signal=fm.get("detectable_signal", ""),
        )
        for fm in raw.get("failure_modes", [])
    ]

    diff = SpaceTransformDiff(
        concept_space_version=cs.version,
        actions=[action],
        new_failure_modes=failure_modes,
        new_detectable_signals=[
            fm.detectable_signal for fm in failure_modes if fm.detectable_signal
        ],
        new_observables=[raw.get("observable", "")] if raw.get("observable") else [],
        transform_status=TransformStatus.PROPOSED,
    )

    return Hypothesis(
        id=f"transform_{uuid.uuid4().hex[:8]}",
        description=raw.get("description", "Transformed hypothesis"),
        source_domain=h.source_domain,
        source_structure=f"transform({h.source_structure})",
        analogy_explanation=f"Space transformation of {h.id}",
        observable=raw.get("observable", h.observable),
        space_transform_diff=diff,
        failure_modes=failure_modes,
        expansion_type="transform_space",
        parent_id=h.id,
    )
