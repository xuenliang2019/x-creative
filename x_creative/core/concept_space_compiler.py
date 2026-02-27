"""ConceptSpace Compiler: builds ConceptSpace from multiple sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from x_creative.core.concept_space import (
    AllowedTransformOp,
    ConceptSpace,
    ConceptSpaceAssumption,
    ConceptSpaceConstraint,
    ConceptSpacePrimitive,
    ConceptSpaceRelation,
)
from x_creative.core.transform_types import TransformAction


class ConceptSpaceCompiler:
    """Compiles ConceptSpace from YAML, TargetDomainPlugin, or LLM fallback."""

    def compile_from_yaml(self, yaml_path: Path, domain_id: str) -> ConceptSpace:
        """Load ConceptSpace from a YAML file's concept_space section."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cs_data = data.get("concept_space")
        if cs_data is None:
            return ConceptSpace(
                version="0.0.0", domain_id=domain_id, provenance="llm_inferred",
                primitives=[], relations=[], hard_constraints=[],
                soft_preferences=[], assumptions_fixed=[], assumptions_mutable=[],
                allowed_ops=[], evaluation_criteria=[],
            )

        return ConceptSpace(
            version=cs_data.get("version", "0.0.0"),
            domain_id=domain_id,
            provenance="yaml",
            primitives=[ConceptSpacePrimitive(**p) for p in cs_data.get("primitives", [])],
            relations=[ConceptSpaceRelation(**r) for r in cs_data.get("relations", [])],
            hard_constraints=[ConceptSpaceConstraint(**c) for c in cs_data.get("hard_constraints", [])],
            soft_preferences=[ConceptSpaceConstraint(**c) for c in cs_data.get("soft_preferences", [])],
            assumptions_fixed=[ConceptSpaceAssumption(**a) for a in cs_data.get("assumptions_fixed", [])],
            assumptions_mutable=[ConceptSpaceAssumption(**a) for a in cs_data.get("assumptions_mutable", [])],
            allowed_ops=[AllowedTransformOp(**o) for o in cs_data.get("allowed_ops", [])],
            evaluation_criteria=cs_data.get("evaluation_criteria", []),
        )

    def diff(self, old: ConceptSpace, new: ConceptSpace) -> list[TransformAction]:
        """Compute differences between two ConceptSpace versions."""
        actions: list[TransformAction] = []
        old_constraints = {c.id: c for c in old.hard_constraints}
        new_constraints = {c.id: c for c in new.hard_constraints}

        for cid, new_c in new_constraints.items():
            old_c = old_constraints.get(cid)
            if old_c is None:
                actions.append(TransformAction(
                    op_id="auto_diff", op_type="add_constraint",
                    target_id=cid, before_state="(not present)",
                    after_state=new_c.text, rationale="Added in new version",
                ))
            elif old_c.text != new_c.text:
                actions.append(TransformAction(
                    op_id="auto_diff", op_type="modify_constraint",
                    target_id=cid, before_state=old_c.text,
                    after_state=new_c.text, rationale="Modified in new version",
                ))

        for cid in old_constraints:
            if cid not in new_constraints:
                actions.append(TransformAction(
                    op_id="auto_diff", op_type="drop_constraint",
                    target_id=cid, before_state=old_constraints[cid].text,
                    after_state="(removed)", rationale="Removed in new version",
                ))

        return actions

    def validate(self, cs: ConceptSpace) -> list[str]:
        """Validate ConceptSpace consistency."""
        errors: list[str] = []
        all_ids: list[str] = []
        for p in cs.primitives:
            all_ids.append(p.id)
        for r in cs.relations:
            all_ids.append(r.id)
        for c in cs.hard_constraints:
            all_ids.append(c.id)
        for c in cs.soft_preferences:
            all_ids.append(c.id)
        for a in cs.assumptions_fixed:
            all_ids.append(a.id)
        for a in cs.assumptions_mutable:
            all_ids.append(a.id)

        seen: set[str] = set()
        for aid in all_ids:
            if aid in seen:
                errors.append(f"Duplicate ID: {aid}")
            seen.add(aid)

        prim_ids = {p.id for p in cs.primitives}
        for r in cs.relations:
            for c in r.connects:
                if c not in prim_ids:
                    errors.append(f"Relation '{r.id}' connects unknown primitive '{c}'")

        return errors
