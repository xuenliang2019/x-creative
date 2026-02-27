"""Tests for MOME integration in SearchModule."""

import pytest

from x_creative.core.types import Hypothesis, HypothesisScores
from x_creative.creativity.mome import MOMEArchive
from x_creative.creativity.qd_types import BDSchema, GridConfig
from x_creative.creativity.search import SearchModule


def _schema() -> BDSchema:
    return BDSchema(
        version="1.0.0",
        grid_dimensions=[
            GridConfig(name="mechanism_family", dim_type="categorical", labels=["a", "b", "other"]),
            GridConfig(name="data_granularity", dim_type="categorical", labels=["tick", "daily"]),
        ],
        raw_dimensions=[],
    )


def _scored_hyp(hid: str, div: float, feas: float, domain: str = "a_domain") -> Hypothesis:
    return Hypothesis(
        id=hid,
        description=f"hyp {hid}",
        source_domain=domain,
        source_structure="s",
        analogy_explanation="a",
        observable="daily_test",
        scores=HypothesisScores(
            divergence=div,
            testability=feas,
            rationale=feas,
            robustness=feas,
            feasibility=feas,
        ),
    )


class TestSearchMOMEIntegration:
    def test_select_with_mome(self) -> None:
        archive = MOMEArchive(bd_schema=_schema(), cell_capacity=5)
        search = SearchModule(mome_archive=archive)

        hyps = [
            _scored_hyp("h1", 8.0, 3.0, "a_domain"),
            _scored_hyp("h2", 3.0, 8.0, "b_domain"),
            _scored_hyp("h3", 5.0, 5.0, "other_domain"),
        ]
        selected = search._select_for_expansion(hyps, 2)
        assert len(selected) == 2

    def test_mome_takes_priority_over_pareto(self) -> None:
        from x_creative.creativity.pareto import ParetoArchive

        archive = MOMEArchive(bd_schema=_schema(), cell_capacity=5)
        pareto = ParetoArchive()
        search = SearchModule(mome_archive=archive, pareto_archive=pareto)

        hyps = [_scored_hyp("h1", 7.0, 7.0)]
        selected = search._select_for_expansion(hyps, 1)
        assert len(selected) == 1
