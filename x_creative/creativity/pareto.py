"""NSGA-II Pareto selection with novelty-conditioned dynamic weights.

Provides non-dominated sorting, crowding distance, and a ParetoArchive
that replaces single-axis composite ranking in SEARCH selection.

Feature-gated by ``pareto_selection_enabled`` (default False).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

from x_creative.core.types import Hypothesis

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParetoPoint:
    """A point in 2-D objective space (novelty, feasibility)."""

    index: int        # back-reference to source list
    novelty: float
    feasibility: float


@dataclass
class RankedPoint:
    """A ParetoPoint annotated with Pareto rank and crowding distance."""

    point: ParetoPoint
    rank: int = 0                    # Pareto front index (0 = optimal)
    crowding_distance: float = 0.0   # diversity measure


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Return True if *a* Pareto-dominates *b* (maximization on both axes).

    *a* dominates *b* iff a >= b on both axes AND strictly > on at least one.
    """
    return (
        a.novelty >= b.novelty
        and a.feasibility >= b.feasibility
        and (a.novelty > b.novelty or a.feasibility > b.feasibility)
    )


def non_dominated_sort(points: list[ParetoPoint]) -> list[list[ParetoPoint]]:
    """Fast non-dominated sorting (O(M·N²) for M objectives, N points).

    Returns a list of fronts, where front[0] is the Pareto-optimal set.
    """
    n = len(points)
    if n == 0:
        return []

    # For each point: set of points it dominates, and count of points dominating it
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count: list[int] = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(points[i], points[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(points[j], points[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    # Build fronts
    fronts: list[list[ParetoPoint]] = []
    current_front_indices = [i for i in range(n) if domination_count[i] == 0]

    while current_front_indices:
        front = [points[i] for i in current_front_indices]
        fronts.append(front)

        next_front_indices: list[int] = []
        for i in current_front_indices:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front_indices.append(j)
        current_front_indices = next_front_indices

    return fronts


def crowding_distance(front: list[ParetoPoint]) -> list[float]:
    """Compute crowding distance for each point in a single front.

    Returns a list of distances aligned with *front*.
    Boundary points receive ``inf``.
    """
    n = len(front)
    if n <= 2:
        return [math.inf] * n

    distances = [0.0] * n

    for attr in ("novelty", "feasibility"):
        sorted_indices = sorted(range(n), key=lambda i: getattr(front[i], attr))
        obj_min = getattr(front[sorted_indices[0]], attr)
        obj_max = getattr(front[sorted_indices[-1]], attr)
        span = obj_max - obj_min

        # Boundary points
        distances[sorted_indices[0]] = math.inf
        distances[sorted_indices[-1]] = math.inf

        if span > 0:
            for k in range(1, n - 1):
                prev_val = getattr(front[sorted_indices[k - 1]], attr)
                next_val = getattr(front[sorted_indices[k + 1]], attr)
                distances[sorted_indices[k]] += (next_val - prev_val) / span

    return distances


def pareto_rank_and_crowd(points: list[ParetoPoint]) -> list[RankedPoint]:
    """Run non-dominated sort + crowding distance, return ranked points."""
    fronts = non_dominated_sort(points)
    ranked: list[RankedPoint] = []

    for rank, front in enumerate(fronts):
        distances = crowding_distance(front)
        for pt, dist in zip(front, distances):
            ranked.append(RankedPoint(point=pt, rank=rank, crowding_distance=dist))

    return ranked


def pareto_select(ranked: list[RankedPoint], count: int) -> list[RankedPoint]:
    """Select top *count* points by (rank ascending, crowding descending)."""
    # Sort: lower rank is better; higher crowding distance is better
    ordered = sorted(
        ranked,
        key=lambda rp: (rp.rank, -rp.crowding_distance),
    )
    return ordered[:count]


def dynamic_weight_novelty(
    novelty: float,
    wn_min: float = 0.15,
    wn_max: float = 0.55,
    gamma: float = 2.0,
) -> tuple[float, float]:
    """Novelty-conditioned dynamic weight.

    High-novelty ideas get lower novelty weight (push toward feasibility).
    Low-novelty ideas get higher novelty weight (push toward more novelty).

    Args:
        novelty: Novelty value (0-10 scale).
        wn_min: Minimum novelty weight.
        wn_max: Maximum novelty weight.
        gamma: Curvature parameter (>1 = more concave).

    Returns:
        (wN, wF) where wN + wF = 1.
    """
    # Normalize novelty to [0, 1]
    n = max(0.0, min(1.0, novelty / 10.0))
    wn = wn_min + (wn_max - wn_min) * (1.0 - n) ** gamma
    wf = 1.0 - wn
    return (wn, wf)


def novelty_bin_index(novelty: float, num_bins: int = 5) -> int:
    """Map a novelty value (0-10) to an equal-width bin index.

    Returns an integer in [0, num_bins - 1].
    """
    if num_bins <= 0:
        return 0
    # Normalize to [0, 1), then scale to bins
    n = max(0.0, min(10.0, novelty))
    idx = int(n / 10.0 * num_bins)
    return min(idx, num_bins - 1)


# ---------------------------------------------------------------------------
# ParetoArchive
# ---------------------------------------------------------------------------

class ParetoArchive:
    """NSGA-II selection with novelty-conditioned dynamic weights.

    Replaces single-axis composite ranking in SEARCH selection when
    ``pareto_selection_enabled`` is True.
    """

    def __init__(
        self,
        wn_min: float = 0.15,
        wn_max: float = 0.55,
        gamma: float = 2.0,
        num_bins: int = 5,
    ) -> None:
        self.wn_min = wn_min
        self.wn_max = wn_max
        self.gamma = gamma
        self.num_bins = num_bins

    def select(
        self,
        hypotheses: list[Hypothesis],
        count: int,
    ) -> list[Hypothesis]:
        """Select hypotheses using NSGA-II Pareto ranking.

        1. Partition into scored (have both axes) vs unscored.
        2. If no scored → fallback to composite_score sort.
        3. For each scored: apply dynamic_weight_novelty → create ParetoPoint.
        4. Run pareto_rank_and_crowd → pareto_select.
        5. Map back to Hypothesis objects.
        6. Fill remaining slots from unscored if needed.
        """
        if not hypotheses:
            return []

        scored: list[tuple[int, Hypothesis]] = []
        unscored: list[tuple[int, Hypothesis]] = []

        for i, h in enumerate(hypotheses):
            n = h.novelty_axis()
            f = h.feasibility_axis()
            if n is not None and f is not None:
                scored.append((i, h))
            else:
                unscored.append((i, h))

        if not scored:
            return self._fallback_select(hypotheses, count)

        # Build ParetoPoints with dynamic-weight adjusted axes
        points: list[ParetoPoint] = []
        scored_hyps: list[Hypothesis] = []
        for idx, (orig_idx, h) in enumerate(scored):
            nov = h.novelty_axis()
            feas = h.feasibility_axis()
            assert nov is not None and feas is not None  # guaranteed by partition

            wn, wf = dynamic_weight_novelty(nov, self.wn_min, self.wn_max, self.gamma)
            # Weighted objectives for Pareto ranking
            points.append(ParetoPoint(
                index=idx,
                novelty=wn * nov,
                feasibility=wf * feas,
            ))
            scored_hyps.append(h)

        # NSGA-II selection
        ranked = pareto_rank_and_crowd(points)
        selected_ranked = pareto_select(ranked, min(count, len(ranked)))

        result: list[Hypothesis] = []
        for rp in selected_ranked:
            result.append(scored_hyps[rp.point.index])

        # Fill remaining slots from unscored (by composite score)
        remaining = count - len(result)
        if remaining > 0 and unscored:
            unscored_hyps = [h for _, h in unscored]
            fallback = self._fallback_select(unscored_hyps, remaining)
            result.extend(fallback)

        return result

    @staticmethod
    def _fallback_select(
        hypotheses: list[Hypothesis],
        count: int,
    ) -> list[Hypothesis]:
        """Fallback: same tuple-based sort as SearchModule._select_for_expansion."""

        def _key(h: Hypothesis) -> tuple[int, float]:
            if h.scores is not None:
                return (2, h.composite_score())
            if h.quick_score is not None:
                return (1, float(h.quick_score))
            return (0, 0.0)

        ranked = sorted(hypotheses, key=_key, reverse=True)
        return ranked[:count]
