"""Tests for BDExtractor — extracts BehaviorDescriptor from Hypothesis."""

from x_creative.core.types import Hypothesis, MappingItem, FailureMode
from x_creative.creativity.bd_extractor import BDExtractor
from x_creative.creativity.qd_types import BDSchema, GridConfig
from x_creative.hkg.types import HKGEvidence


def _make_schema() -> BDSchema:
    return BDSchema(
        version="1.0.0",
        grid_dimensions=[
            GridConfig(
                name="mechanism_family",
                dim_type="categorical",
                labels=["thermodynamic", "information", "game_theory", "biological", "other"],
            ),
            GridConfig(
                name="data_granularity",
                dim_type="categorical",
                labels=["tick", "minute", "daily", "weekly"],
            ),
        ],
        raw_dimensions=["causal_chain_length", "constraint_count"],
    )


def _make_hypothesis(
    source_domain: str = "thermodynamics",
    observable: str = "entropy = -sum(p_i * log(p_i))",
    mapping_count: int = 3,
    failure_count: int = 1,
    hkg_evidence: HKGEvidence | None = None,
) -> Hypothesis:
    return Hypothesis(
        id="hyp_test",
        description="test hypothesis",
        source_domain=source_domain,
        source_structure="entropy_increase",
        analogy_explanation="test",
        observable=observable,
        mapping_table=[
            MappingItem(
                source_concept=f"src_{i}",
                target_concept=f"tgt_{i}",
                source_relation=f"rel_{i}",
                target_relation=f"tgt_rel_{i}",
                mapping_type="relation",
                systematicity_group_id="g1",
            )
            for i in range(mapping_count)
        ],
        failure_modes=[
            FailureMode(
                scenario=f"scenario_{i}",
                why_breaks=f"reason_{i}",
                detectable_signal=f"signal_{i}",
            )
            for i in range(failure_count)
        ],
        hkg_evidence=hkg_evidence,
    )


class TestBDExtractor:
    def test_extract_rule_based(self) -> None:
        schema = _make_schema()
        extractor = BDExtractor(schema=schema)
        h = _make_hypothesis(source_domain="thermodynamics")
        bd = extractor.extract(h)

        assert bd.version == "1.0.0"
        assert bd.extraction_method == "rule"
        assert "mechanism_family" in bd.grid_dims
        assert "data_granularity" in bd.grid_dims
        assert bd.grid_dims["mechanism_family"] == 0  # thermodynamic = index 0
        assert bd.bd_version == "1.0.0"

    def test_unknown_domain_maps_to_other(self) -> None:
        schema = _make_schema()
        extractor = BDExtractor(schema=schema)
        h = _make_hypothesis(source_domain="quantum_chromodynamics")
        bd = extractor.extract(h)

        assert bd.grid_dims["mechanism_family"] == 4  # "other"

    def test_raw_dimensions(self) -> None:
        schema = _make_schema()
        extractor = BDExtractor(schema=schema)
        h = _make_hypothesis(mapping_count=5, failure_count=3)
        bd = extractor.extract(h)

        assert bd.raw["causal_chain_length"] == 5.0
        assert bd.raw["constraint_count"] == 3.0

    def test_data_granularity_from_observable(self) -> None:
        schema = _make_schema()
        extractor = BDExtractor(schema=schema)

        h_tick = _make_hypothesis(observable="tick_volume_imbalance")
        bd_tick = extractor.extract(h_tick)
        assert bd_tick.grid_dims["data_granularity"] == 0  # tick

        h_daily = _make_hypothesis(observable="daily_return_momentum")
        bd_daily = extractor.extract(h_daily)
        assert bd_daily.grid_dims["data_granularity"] == 2  # daily

    def test_mixed_hint_uses_constrained_enum_with_replay_cache(self) -> None:
        schema = _make_schema()
        extractor = BDExtractor(schema=schema, allow_mixed_hints=True)

        h = _make_hypothesis(
            source_domain="unknown_domain",
            hkg_evidence=HKGEvidence(
                coverage={
                    "bd_hint": {
                        "mechanism_family": "information",
                        "confidence": 0.82,
                        "evidence_path": ["start_match:n1", "edge:e2"],
                    }
                }
            ),
        )
        bd1 = extractor.extract(h)
        bd2 = extractor.extract(h)

        assert bd1.extraction_method == "hybrid"
        assert bd1.grid_dims["mechanism_family"] == 1  # information
        assert bd1.confidence == 0.82
        assert bd1.evidence_path == ["start_match:n1", "edge:e2"]
        assert bd1.grid_dims == bd2.grid_dims
