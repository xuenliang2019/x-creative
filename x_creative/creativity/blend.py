"""Conceptual Blending operator (Fauconnier-Turner four-space model)."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from x_creative.core.blend_types import (
    BlendNetwork,
    BlendSpaceMapping,
    EmergentStructure,
)
from x_creative.core.types import Hypothesis

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


async def blend_expand(
    hypothesis_a: Hypothesis,
    hypothesis_b: Hypothesis,
    router: Any,
) -> list[Hypothesis]:
    """Create conceptual blends from two hypotheses.

    Calls LLM to produce a four-space blend network:
    input1 (hypothesis_a) + input2 (hypothesis_b) -> generic_space -> blend.
    """
    prompt = _build_blend_prompt(hypothesis_a, hypothesis_b)

    try:
        response = await router.complete(
            task="blend_expansion",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content if hasattr(response, "content") else str(response)
        blends_data = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("blend_expand: failed to parse LLM response", error=str(e))
        return []

    if not isinstance(blends_data, list):
        blends_data = [blends_data]

    results: list[Hypothesis] = []
    for blend_raw in blends_data:
        try:
            hyp = _parse_blend_result(hypothesis_a, hypothesis_b, blend_raw)
            results.append(hyp)
        except Exception as e:
            logger.warning("blend_expand: failed to parse blend", error=str(e))
            continue

    return results


def _build_blend_prompt(ha: Hypothesis, hb: Hypothesis) -> str:
    return f"""You are a Conceptual Blending expert (Fauconnier-Turner).

Given two input hypotheses, create a conceptual blend that produces emergent structures
not present in either input.

Input Space 1: {ha.description}
  Source domain: {ha.source_domain}
  Observable: {ha.observable}

Input Space 2: {hb.description}
  Source domain: {hb.source_domain}
  Observable: {hb.observable}

Return a JSON array of blend objects, each with:
- "generic_space": shared abstract structure
- "blend_description": description of the blend
- "cross_space_mappings": [{{"source_element": str, "blend_element": str, "projection_type": "identity"|"compression"|"elaboration"|"completion"}}]
- "emergent_structures": [{{"description": str, "emergence_type": "composition"|"completion"|"elaboration", "observable_link": str|null}}]
- "description": new hypothesis description
- "observable": new observable variable/formula

Respond ONLY with valid JSON array."""


def _parse_blend_result(
    ha: Hypothesis, hb: Hypothesis, raw: dict[str, Any]
) -> Hypothesis:
    observable = str(raw.get("observable", "")).strip()
    if not observable:
        raise ValueError("blend output missing observable")

    mappings = [
        BlendSpaceMapping(
            source_element=m["source_element"],
            blend_element=m["blend_element"],
            projection_type=m.get("projection_type", "identity"),
        )
        for m in raw.get("cross_space_mappings", [])
    ]

    emergent = [
        EmergentStructure(
            description=e["description"],
            emergence_type=e.get("emergence_type", "composition"),
            observable_link=e.get("observable_link"),
            testable_prediction=e.get("testable_prediction"),
        )
        for e in raw.get("emergent_structures", [])
    ]

    blend_network = BlendNetwork(
        input1_summary=ha.description,
        input2_summary=hb.description,
        generic_space=raw.get("generic_space", ""),
        blend_description=raw.get("blend_description", ""),
        cross_space_mappings=mappings,
        emergent_structures=emergent,
    )

    return Hypothesis(
        id=f"blend_{uuid.uuid4().hex[:8]}",
        description=raw.get("description", "Blended hypothesis"),
        source_domain=f"{ha.source_domain}+{hb.source_domain}",
        source_structure=f"blend({ha.source_structure}, {hb.source_structure})",
        analogy_explanation=f"Conceptual blend of {ha.id} and {hb.id}",
        observable=observable,
        blend_network=blend_network,
        expansion_type="blend",
        parent_id=ha.id,
    )
