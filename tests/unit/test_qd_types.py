"""Tests for QD / MAP-Elites data types."""

from x_creative.creativity.qd_types import BDSchema, BehaviorDescriptor, GridConfig


class TestGridConfig:
    def test_categorical(self) -> None:
        gc = GridConfig(
            name="mechanism_family",
            dim_type="categorical",
            labels=["thermodynamic", "information", "game_theory", "biological", "other"],
        )
        assert gc.dim_type == "categorical"
        assert len(gc.labels) == 5

    def test_continuous(self) -> None:
        gc = GridConfig(
            name="novelty",
            dim_type="continuous",
            num_bins=5,
        )
        assert gc.labels is None
        assert gc.num_bins == 5

    def test_custom_bins(self) -> None:
        gc = GridConfig(
            name="chain_length",
            dim_type="continuous",
            bin_edges=[0.0, 2.0, 4.0, 7.0, 10.0],
        )
        assert gc.bin_edges is not None
        assert len(gc.bin_edges) == 4


class TestBDSchema:
    def test_create(self) -> None:
        schema = BDSchema(
            version="1.0.0",
            grid_dimensions=[
                GridConfig(
                    name="mechanism_family",
                    dim_type="categorical",
                    labels=["thermo", "info", "game", "bio", "other"],
                ),
                GridConfig(
                    name="data_granularity",
                    dim_type="categorical",
                    labels=["tick", "minute", "daily", "weekly"],
                ),
            ],
            raw_dimensions=["causal_chain_length", "observable_type", "constraint_count"],
        )
        assert schema.version == "1.0.0"
        assert len(schema.grid_dimensions) == 2
        assert len(schema.raw_dimensions) == 3


class TestBehaviorDescriptor:
    def test_create_full(self) -> None:
        bd = BehaviorDescriptor(
            grid_dims={"mechanism_family": 0, "data_granularity": 2},
            raw={"causal_chain_length": 3.0, "observable_type": "formula", "constraint_count": 2.0},
            version="1.0.0",
            extraction_method="hybrid",
            confidence=0.85,
        )
        assert bd.grid_dims["mechanism_family"] == 0
        assert bd.raw["observable_type"] == "formula"
        assert bd.extraction_method == "hybrid"

    def test_grid_coord_as_tuple(self) -> None:
        bd = BehaviorDescriptor(
            grid_dims={"a": 1, "b": 2},
            raw={},
            version="1.0.0",
            extraction_method="rule",
        )
        coord = tuple(bd.grid_dims[k] for k in sorted(bd.grid_dims))
        assert coord == (1, 2)
