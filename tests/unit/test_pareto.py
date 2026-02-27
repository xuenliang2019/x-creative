"""Tests for NSGA-II Pareto selection algorithms and ParetoArchive."""

from __future__ import annotations

import math

import pytest

from x_creative.core.types import Hypothesis, HypothesisScores
from x_creative.creativity.pareto import (
    ParetoArchive,
    ParetoPoint,
    RankedPoint,
    crowding_distance,
    dominates,
    dynamic_weight_novelty,
    non_dominated_sort,
    novelty_bin_index,
    pareto_rank_and_crowd,
    pareto_select,
)


# ---------------------------------------------------------------------------
# dominates()
# ---------------------------------------------------------------------------

class TestDominates:
    def test_strict_domination(self) -> None:
        a = ParetoPoint(0, novelty=8.0, feasibility=7.0)
        b = ParetoPoint(1, novelty=5.0, feasibility=4.0)
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_equal_on_one_axis_better_on_other(self) -> None:
        a = ParetoPoint(0, novelty=8.0, feasibility=7.0)
        b = ParetoPoint(1, novelty=8.0, feasibility=5.0)
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_identical_points_do_not_dominate(self) -> None:
        a = ParetoPoint(0, novelty=5.0, feasibility=5.0)
        b = ParetoPoint(1, novelty=5.0, feasibility=5.0)
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_neither_dominates_tradeoff(self) -> None:
        a = ParetoPoint(0, novelty=9.0, feasibility=3.0)
        b = ParetoPoint(1, novelty=3.0, feasibility=9.0)
        assert dominates(a, b) is False
        assert dominates(b, a) is False


# ---------------------------------------------------------------------------
# non_dominated_sort()
# ---------------------------------------------------------------------------

class TestNonDominatedSort:
    def test_empty(self) -> None:
        assert non_dominated_sort([]) == []

    def test_single_point(self) -> None:
        pts = [ParetoPoint(0, 5.0, 5.0)]
        fronts = non_dominated_sort(pts)
        assert len(fronts) == 1
        assert len(fronts[0]) == 1

    def test_all_non_dominated(self) -> None:
        """Trade-off points should all be on the first front."""
        pts = [
            ParetoPoint(0, novelty=9.0, feasibility=1.0),
            ParetoPoint(1, novelty=1.0, feasibility=9.0),
            ParetoPoint(2, novelty=5.0, feasibility=5.0),
        ]
        fronts = non_dominated_sort(pts)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_two_fronts(self) -> None:
        """Dominated point should be on front 1, rest on front 0."""
        pts = [
            ParetoPoint(0, novelty=9.0, feasibility=8.0),
            ParetoPoint(1, novelty=8.0, feasibility=9.0),
            ParetoPoint(2, novelty=3.0, feasibility=3.0),  # dominated by both
        ]
        fronts = non_dominated_sort(pts)
        assert len(fronts) == 2
        front0_indices = {p.index for p in fronts[0]}
        assert front0_indices == {0, 1}
        assert fronts[1][0].index == 2

    def test_three_fronts_chain(self) -> None:
        """Chain: A dominates B dominates C → three fronts."""
        pts = [
            ParetoPoint(0, novelty=9.0, feasibility=9.0),
            ParetoPoint(1, novelty=5.0, feasibility=5.0),
            ParetoPoint(2, novelty=2.0, feasibility=2.0),
        ]
        fronts = non_dominated_sort(pts)
        assert len(fronts) == 3
        assert fronts[0][0].index == 0
        assert fronts[1][0].index == 1
        assert fronts[2][0].index == 2


# ---------------------------------------------------------------------------
# crowding_distance()
# ---------------------------------------------------------------------------

class TestCrowdingDistance:
    def test_two_points_get_inf(self) -> None:
        front = [
            ParetoPoint(0, 1.0, 9.0),
            ParetoPoint(1, 9.0, 1.0),
        ]
        dists = crowding_distance(front)
        assert all(d == math.inf for d in dists)

    def test_single_point_gets_inf(self) -> None:
        front = [ParetoPoint(0, 5.0, 5.0)]
        dists = crowding_distance(front)
        assert dists == [math.inf]

    def test_boundary_points_get_inf(self) -> None:
        front = [
            ParetoPoint(0, 1.0, 9.0),
            ParetoPoint(1, 5.0, 5.0),
            ParetoPoint(2, 9.0, 1.0),
        ]
        dists = crowding_distance(front)
        # Boundary points (min/max on each axis) should be inf
        assert dists[0] == math.inf  # min novelty, max feasibility
        assert dists[2] == math.inf  # max novelty, min feasibility
        # Middle point should be finite
        assert dists[1] < math.inf
        assert dists[1] > 0

    def test_middle_point_distance(self) -> None:
        """Three equally spaced points: middle should get distance 2.0."""
        front = [
            ParetoPoint(0, 0.0, 10.0),
            ParetoPoint(1, 5.0, 5.0),
            ParetoPoint(2, 10.0, 0.0),
        ]
        dists = crowding_distance(front)
        # Middle on each axis: (next - prev) / span = (10-0)/10 + (10-0)/10 = 2.0
        assert dists[1] == pytest.approx(2.0)

    def test_all_same_novelty_zero_span(self) -> None:
        """When novelty is identical, span=0 → no contribution from that axis."""
        front = [
            ParetoPoint(0, 5.0, 1.0),
            ParetoPoint(1, 5.0, 5.0),
            ParetoPoint(2, 5.0, 9.0),
        ]
        dists = crowding_distance(front)
        # Boundary on feasibility axis → inf for index 0 and 2
        assert dists[0] == math.inf
        assert dists[2] == math.inf
        # Middle point: only feasibility contributes
        assert dists[1] < math.inf
        assert dists[1] > 0


# ---------------------------------------------------------------------------
# pareto_rank_and_crowd()
# ---------------------------------------------------------------------------

class TestParetoRankAndCrowd:
    def test_produces_ranked_points(self) -> None:
        pts = [
            ParetoPoint(0, 9.0, 8.0),
            ParetoPoint(1, 3.0, 3.0),
        ]
        ranked = pareto_rank_and_crowd(pts)
        assert len(ranked) == 2
        ranks = {rp.point.index: rp.rank for rp in ranked}
        assert ranks[0] == 0
        assert ranks[1] == 1

    def test_empty_input(self) -> None:
        assert pareto_rank_and_crowd([]) == []


# ---------------------------------------------------------------------------
# pareto_select()
# ---------------------------------------------------------------------------

class TestParetoSelect:
    def test_selects_lower_rank_first(self) -> None:
        ranked = [
            RankedPoint(ParetoPoint(0, 3.0, 3.0), rank=1, crowding_distance=5.0),
            RankedPoint(ParetoPoint(1, 9.0, 8.0), rank=0, crowding_distance=1.0),
        ]
        selected = pareto_select(ranked, 1)
        assert len(selected) == 1
        assert selected[0].point.index == 1  # rank 0

    def test_within_rank_selects_higher_crowding(self) -> None:
        ranked = [
            RankedPoint(ParetoPoint(0, 5.0, 5.0), rank=0, crowding_distance=1.0),
            RankedPoint(ParetoPoint(1, 8.0, 2.0), rank=0, crowding_distance=math.inf),
        ]
        selected = pareto_select(ranked, 1)
        assert selected[0].point.index == 1  # higher crowding

    def test_count_exceeds_pool(self) -> None:
        ranked = [
            RankedPoint(ParetoPoint(0, 5.0, 5.0), rank=0, crowding_distance=1.0),
        ]
        selected = pareto_select(ranked, 10)
        assert len(selected) == 1

    def test_empty(self) -> None:
        assert pareto_select([], 5) == []


# ---------------------------------------------------------------------------
# dynamic_weight_novelty()
# ---------------------------------------------------------------------------

class TestDynamicWeightNovelty:
    def test_high_novelty_gets_low_wn(self) -> None:
        """High novelty (10) → push toward feasibility → wN near wn_min."""
        wn, wf = dynamic_weight_novelty(10.0)
        assert wn == pytest.approx(0.15)  # wn_min
        assert wf == pytest.approx(0.85)

    def test_low_novelty_gets_high_wn(self) -> None:
        """Low novelty (0) → push toward novelty → wN near wn_max."""
        wn, wf = dynamic_weight_novelty(0.0)
        assert wn == pytest.approx(0.55)  # wn_max
        assert wf == pytest.approx(0.45)

    def test_weights_sum_to_one(self) -> None:
        for n in [0.0, 2.5, 5.0, 7.5, 10.0]:
            wn, wf = dynamic_weight_novelty(n)
            assert wn + wf == pytest.approx(1.0)

    def test_monotonically_decreasing_wn(self) -> None:
        """wN should decrease as novelty increases."""
        prev_wn = 1.0
        for n in range(11):
            wn, _ = dynamic_weight_novelty(float(n))
            assert wn <= prev_wn
            prev_wn = wn

    def test_custom_params(self) -> None:
        wn, wf = dynamic_weight_novelty(5.0, wn_min=0.1, wn_max=0.9, gamma=1.0)
        # n=0.5, wN = 0.1 + 0.8 * 0.5^1 = 0.1 + 0.4 = 0.5
        assert wn == pytest.approx(0.5)
        assert wf == pytest.approx(0.5)

    def test_clamps_novelty_above_10(self) -> None:
        wn_10, _ = dynamic_weight_novelty(10.0)
        wn_15, _ = dynamic_weight_novelty(15.0)
        assert wn_15 == pytest.approx(wn_10)

    def test_clamps_novelty_below_0(self) -> None:
        wn_0, _ = dynamic_weight_novelty(0.0)
        wn_neg, _ = dynamic_weight_novelty(-5.0)
        assert wn_neg == pytest.approx(wn_0)


# ---------------------------------------------------------------------------
# novelty_bin_index()
# ---------------------------------------------------------------------------

class TestNoveltyBinIndex:
    def test_bins_five(self) -> None:
        assert novelty_bin_index(0.0, 5) == 0
        assert novelty_bin_index(1.9, 5) == 0
        assert novelty_bin_index(2.0, 5) == 1
        assert novelty_bin_index(9.9, 5) == 4
        assert novelty_bin_index(10.0, 5) == 4  # clamped to last bin

    def test_clamps_above_10(self) -> None:
        assert novelty_bin_index(15.0, 5) == 4

    def test_clamps_below_0(self) -> None:
        assert novelty_bin_index(-3.0, 5) == 0

    def test_single_bin(self) -> None:
        assert novelty_bin_index(5.0, 1) == 0

    def test_zero_bins(self) -> None:
        assert novelty_bin_index(5.0, 0) == 0


# ---------------------------------------------------------------------------
# ParetoArchive
# ---------------------------------------------------------------------------

def _make_hyp(
    id: str,
    divergence: float = 5.0,
    testability: float = 5.0,
    rationale: float = 5.0,
    robustness: float = 5.0,
    feasibility: float = 5.0,
    scored: bool = True,
) -> Hypothesis:
    """Helper to create a scored or unscored Hypothesis."""
    scores = (
        HypothesisScores(
            divergence=divergence,
            testability=testability,
            rationale=rationale,
            robustness=robustness,
            feasibility=feasibility,
        )
        if scored
        else None
    )
    return Hypothesis(
        id=id,
        description=f"Hypothesis {id}",
        source_domain="d",
        source_structure="s",
        analogy_explanation="a",
        observable="o",
        scores=scores,
    )


class TestParetoArchive:
    def test_pareto_optimal_preferred_over_dominated(self) -> None:
        """Non-dominated point should be selected before dominated one."""
        archive = ParetoArchive()

        # h1: high novelty, high feasibility → dominates h2
        h1 = _make_hyp("h1", divergence=9.0, testability=9.0, rationale=9.0, robustness=9.0, feasibility=9.0)
        # h2: low on everything → dominated
        h2 = _make_hyp("h2", divergence=3.0, testability=3.0, rationale=3.0, robustness=3.0, feasibility=3.0)

        selected = archive.select([h2, h1], count=1)
        assert len(selected) == 1
        assert selected[0].id == "h1"

    def test_fallback_when_no_scored(self) -> None:
        """When no hypotheses have scores, fallback to composite sort."""
        archive = ParetoArchive()

        h1 = _make_hyp("h1", scored=False)
        h2 = _make_hyp("h2", scored=False)

        selected = archive.select([h1, h2], count=2)
        assert len(selected) == 2

    def test_mixed_scored_and_unscored(self) -> None:
        """Scored hypotheses selected via Pareto, unscored fill remaining."""
        archive = ParetoArchive()

        h_scored = _make_hyp("h_scored", divergence=8.0, testability=8.0, rationale=8.0, robustness=8.0, feasibility=8.0)
        h_unscored = _make_hyp("h_unscored", scored=False)

        selected = archive.select([h_unscored, h_scored], count=2)
        assert len(selected) == 2
        # Scored should come first
        assert selected[0].id == "h_scored"

    def test_empty_input(self) -> None:
        archive = ParetoArchive()
        assert archive.select([], count=5) == []

    def test_count_exceeds_pool(self) -> None:
        archive = ParetoArchive()
        h1 = _make_hyp("h1")
        selected = archive.select([h1], count=10)
        assert len(selected) == 1

    def test_diversity_via_crowding(self) -> None:
        """Trade-off points should both be selected over a duplicate."""
        archive = ParetoArchive()

        # Two trade-off points (neither dominates the other)
        h_novel = _make_hyp("h_novel", divergence=9.0, testability=3.0, rationale=3.0, robustness=3.0, feasibility=3.0)
        h_feasible = _make_hyp("h_feasible", divergence=3.0, testability=9.0, rationale=9.0, robustness=9.0, feasibility=9.0)
        # A middle point dominated by neither but less diverse
        h_mid = _make_hyp("h_mid", divergence=5.0, testability=5.0, rationale=5.0, robustness=5.0, feasibility=5.0)

        selected = archive.select([h_mid, h_novel, h_feasible], count=2)
        selected_ids = {h.id for h in selected}
        # The two boundary/extreme trade-off points should be preferred
        assert "h_novel" in selected_ids
        assert "h_feasible" in selected_ids
