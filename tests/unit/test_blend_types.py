"""Tests for Conceptual Blending data types."""

from x_creative.core.blend_types import (
    BlendNetwork,
    BlendSpaceMapping,
    EmergentStructure,
)


class TestBlendSpaceMapping:
    def test_create_mapping(self) -> None:
        m = BlendSpaceMapping(
            source_element="entropy",
            blend_element="information_disorder",
            projection_type="compression",
            conflict_resolution="select input1 framing",
        )
        assert m.projection_type == "compression"
        assert m.conflict_resolution is not None

    def test_minimal_mapping(self) -> None:
        m = BlendSpaceMapping(
            source_element="a",
            blend_element="b",
            projection_type="identity",
        )
        assert m.conflict_resolution is None


class TestEmergentStructure:
    def test_create_emergent(self) -> None:
        e = EmergentStructure(
            description="Order flow entropy predicts regime shift",
            emergence_type="completion",
            observable_link="entropy_delta > threshold",
            testable_prediction="Entropy spike 2h before reversal",
        )
        assert e.emergence_type == "completion"

    def test_minimal_emergent(self) -> None:
        e = EmergentStructure(
            description="New structure",
            emergence_type="composition",
        )
        assert e.observable_link is None


class TestBlendNetwork:
    def test_create_full(self) -> None:
        bn = BlendNetwork(
            input1_summary="Thermodynamic entropy increase",
            input2_summary="Order book information dynamics",
            generic_space="System disorder measurement",
            blend_description="Market entropy as order flow disorder",
            cross_space_mappings=[
                BlendSpaceMapping(
                    source_element="temperature",
                    blend_element="volatility",
                    projection_type="compression",
                ),
            ],
            emergent_structures=[
                EmergentStructure(
                    description="Entropy-volatility feedback loop",
                    emergence_type="elaboration",
                ),
            ],
            blend_consistency_score=7.5,
        )
        assert len(bn.cross_space_mappings) == 1
        assert len(bn.emergent_structures) == 1
        assert bn.blend_consistency_score == 7.5

    def test_empty_blend(self) -> None:
        bn = BlendNetwork(
            input1_summary="A",
            input2_summary="B",
            generic_space="C",
            blend_description="D",
            cross_space_mappings=[],
            emergent_structures=[],
        )
        assert bn.blend_consistency_score is None
