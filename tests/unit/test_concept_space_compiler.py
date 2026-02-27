# tests/unit/test_concept_space_compiler.py
"""Tests for ConceptSpaceCompiler."""

import pytest
from pathlib import Path

from x_creative.core.concept_space_compiler import ConceptSpaceCompiler
from x_creative.core.concept_space import ConceptSpace


class TestConceptSpaceCompiler:
    def test_compile_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
concept_space:
  version: "1.0.0"
  primitives:
    - id: price
      name: 价格
      description: 市场价格
      type: variable
  relations:
    - id: momentum
      name: 动量
      description: 价格趋势
      connects: [price]
  hard_constraints:
    - id: no_lookahead
      text: 不使用未来信息
      constraint_type: hard
      rationale: 基本要求
  soft_preferences: []
  assumptions_fixed: []
  assumptions_mutable:
    - id: linear
      text: 线性模型假设
      mutable: true
      rationale: 常见近似
  allowed_ops:
    - id: negate
      name: 否定假设
      description: 取反
      op_type: negate_assumption
      target_type: assumption
  evaluation_criteria:
    - User Growth Rate
"""
        yaml_path = tmp_path / "test_domain.yaml"
        yaml_path.write_text(yaml_content)

        compiler = ConceptSpaceCompiler()
        cs = compiler.compile_from_yaml(yaml_path, domain_id="test")
        assert cs.version == "1.0.0"
        assert cs.domain_id == "test"
        assert cs.provenance == "yaml"
        assert len(cs.primitives) == 1
        assert len(cs.hard_constraints) == 1
        assert len(cs.assumptions_mutable) == 1
        assert len(cs.allowed_ops) == 1

    def test_compile_missing_concept_space_section(self, tmp_path: Path) -> None:
        yaml_content = """
id: test
name: Test Domain
"""
        yaml_path = tmp_path / "test_domain.yaml"
        yaml_path.write_text(yaml_content)

        compiler = ConceptSpaceCompiler()
        cs = compiler.compile_from_yaml(yaml_path, domain_id="test")
        assert cs.provenance == "llm_inferred"
        assert len(cs.primitives) == 0

    def test_diff_detects_changes(self) -> None:
        compiler = ConceptSpaceCompiler()
        from x_creative.core.concept_space import (
            ConceptSpaceConstraint,
            ConceptSpacePrimitive,
        )

        old = ConceptSpace(
            version="1.0.0", domain_id="test", provenance="yaml",
            primitives=[ConceptSpacePrimitive(id="price", name="价格", description="d", type="variable")],
            relations=[], hard_constraints=[
                ConceptSpaceConstraint(id="c1", text="old", constraint_type="hard", rationale="r")
            ],
            soft_preferences=[], assumptions_fixed=[], assumptions_mutable=[],
            allowed_ops=[], evaluation_criteria=[],
        )
        new = ConceptSpace(
            version="1.1.0", domain_id="test", provenance="yaml",
            primitives=[ConceptSpacePrimitive(id="price", name="价格", description="d", type="variable")],
            relations=[], hard_constraints=[
                ConceptSpaceConstraint(id="c1", text="new", constraint_type="hard", rationale="r")
            ],
            soft_preferences=[], assumptions_fixed=[], assumptions_mutable=[],
            allowed_ops=[], evaluation_criteria=[],
        )
        diffs = compiler.diff(old, new)
        assert len(diffs) > 0

    def test_validate_valid(self) -> None:
        compiler = ConceptSpaceCompiler()
        cs = ConceptSpace(
            version="1.0.0", domain_id="test", provenance="yaml",
            primitives=[], relations=[], hard_constraints=[],
            soft_preferences=[], assumptions_fixed=[], assumptions_mutable=[],
            allowed_ops=[], evaluation_criteria=[],
        )
        errors = compiler.validate(cs)
        assert isinstance(errors, list)
