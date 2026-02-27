"""Tests for MOME (Multi-Objective MAP-Elites) archive."""

import pytest

from x_creative.core.types import Hypothesis, HypothesisScores
from x_creative.creativity.mome import MOMEArchive, MOMECell
from x_creative.creativity.bd_extractor import BDExtractor
from x_creative.creativity.qd_types import BDSchema, GridConfig


def _hyp(hid: str, divergence: float, feasibility_avg: float) -> Hypothesis:
    """Create a Hypothesis with specific novelty/feasibility axes."""
    return Hypothesis(
        id=hid,
        description=f"hyp {hid}",
        source_domain="test",
        source_structure="test",
        analogy_explanation="test",
        observable="test",
        scores=HypothesisScores(
            divergence=divergence,
            testability=feasibility_avg,
            rationale=feasibility_avg,
            robustness=feasibility_avg,
            feasibility=feasibility_avg,
        ),
    )


class TestMOMECell:
    def test_add_to_empty_cell(self) -> None:
        cell = MOMECell(capacity=5)
        h = _hyp("a", 7.0, 6.0)
        assert cell.try_add(h) is True
        assert len(cell) == 1

    def test_dominated_hypothesis_rejected(self) -> None:
        cell = MOMECell(capacity=5)
        h1 = _hyp("a", 8.0, 8.0)
        h2 = _hyp("b", 5.0, 5.0)
        cell.try_add(h1)
        assert cell.try_add(h2) is False
        assert len(cell) == 1

    def test_dominating_hypothesis_replaces(self) -> None:
        cell = MOMECell(capacity=5)
        h1 = _hyp("a", 5.0, 5.0)
        h2 = _hyp("b", 8.0, 8.0)
        cell.try_add(h1)
        assert cell.try_add(h2) is True
        assert len(cell) == 1
        ids = [h.id for h in cell.hypotheses]
        assert "b" in ids
        assert "a" not in ids

    def test_non_dominated_both_kept(self) -> None:
        cell = MOMECell(capacity=5)
        h1 = _hyp("a", 9.0, 3.0)
        h2 = _hyp("b", 3.0, 9.0)
        cell.try_add(h1)
        cell.try_add(h2)
        assert len(cell) == 2

    def test_capacity_eviction(self) -> None:
        cell = MOMECell(capacity=2)
        h1 = _hyp("a", 9.0, 1.0)
        h2 = _hyp("b", 1.0, 9.0)
        h3 = _hyp("c", 5.0, 5.0)
        cell.try_add(h1)
        cell.try_add(h2)
        cell.try_add(h3)
        assert len(cell) == 2

    def test_select(self) -> None:
        cell = MOMECell(capacity=5)
        h1 = _hyp("a", 9.0, 3.0)
        h2 = _hyp("b", 3.0, 9.0)
        h3 = _hyp("c", 6.0, 6.0)
        cell.try_add(h1)
        cell.try_add(h2)
        cell.try_add(h3)
        selected = cell.select(2)
        assert len(selected) == 2


def _schema() -> BDSchema:
    return BDSchema(
        version="1.0.0",
        grid_dimensions=[
            GridConfig(
                name="mechanism_family",
                dim_type="categorical",
                labels=["thermodynamic", "information", "other"],
            ),
            GridConfig(
                name="data_granularity",
                dim_type="categorical",
                labels=["tick", "daily"],
            ),
        ],
        raw_dimensions=["causal_chain_length"],
    )


class TestMOMEArchive:
    def test_add_and_select(self) -> None:
        schema = _schema()
        archive = MOMEArchive(bd_schema=schema, cell_capacity=5)
        h1 = _hyp("a", 8.0, 6.0)
        h1 = h1.model_copy(update={"source_domain": "thermodynamics", "observable": "daily_entropy"})
        h2 = _hyp("b", 3.0, 9.0)
        h2 = h2.model_copy(update={"source_domain": "information_theory", "observable": "tick_signal"})

        archive.add(h1)
        archive.add(h2)
        assert archive.cell_count >= 1

        selected = archive.select(2)
        assert len(selected) == 2

    def test_coverage(self) -> None:
        schema = _schema()
        archive = MOMEArchive(bd_schema=schema, cell_capacity=5)
        assert archive.total_cells == 6

        h = _hyp("a", 7.0, 7.0)
        h = h.model_copy(update={"source_domain": "thermodynamics", "observable": "daily_x"})
        archive.add(h)
        assert archive.filled_cells >= 1
        assert archive.coverage_ratio > 0.0

    def test_select_prefers_sparse_cells(self) -> None:
        schema = _schema()
        archive = MOMEArchive(bd_schema=schema, cell_capacity=5)

        for i in range(3):
            h = _hyp(f"thermo_{i}", 5.0 + i, 5.0 + i)
            h = h.model_copy(update={"source_domain": "thermodynamics", "observable": "daily_x"})
            archive.add(h)

        h_info = _hyp("info_0", 6.0, 6.0)
        h_info = h_info.model_copy(update={"source_domain": "information_theory", "observable": "daily_y"})
        archive.add(h_info)

        selected = archive.select(2)
        ids = [h.id for h in selected]
        assert "info_0" in ids

    def test_empty_archive_returns_empty(self) -> None:
        schema = _schema()
        archive = MOMEArchive(bd_schema=schema, cell_capacity=5)
        assert archive.select(5) == []


from x_creative.creativity.mome import novelty_distance
from x_creative.creativity.qd_types import BehaviorDescriptor


class TestNoveltyDistance:
    def test_identical_descriptors(self) -> None:
        bd = BehaviorDescriptor(
            grid_dims={"a": 1}, raw={"x": 5.0}, version="1", extraction_method="rule"
        )
        dist = novelty_distance(bd, [bd], k=1)
        assert dist == 0.0

    def test_different_descriptors(self) -> None:
        bd1 = BehaviorDescriptor(
            grid_dims={"a": 0}, raw={"x": 0.0}, version="1", extraction_method="rule"
        )
        bd2 = BehaviorDescriptor(
            grid_dims={"a": 1}, raw={"x": 10.0}, version="1", extraction_method="rule"
        )
        dist = novelty_distance(bd1, [bd2], k=1)
        assert dist > 0.0

    def test_empty_archive(self) -> None:
        bd = BehaviorDescriptor(
            grid_dims={"a": 0}, raw={"x": 5.0}, version="1", extraction_method="rule"
        )
        dist = novelty_distance(bd, [], k=5)
        assert dist == float("inf")

    def test_k_neighbors(self) -> None:
        bd = BehaviorDescriptor(
            grid_dims={"a": 0}, raw={"x": 5.0}, version="1", extraction_method="rule"
        )
        neighbors = [
            BehaviorDescriptor(
                grid_dims={"a": 0}, raw={"x": float(i)}, version="1", extraction_method="rule"
            )
            for i in range(10)
        ]
        dist = novelty_distance(bd, neighbors, k=3)
        assert dist > 0.0
