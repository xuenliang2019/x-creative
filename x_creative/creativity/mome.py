"""MOME (Multi-Objective MAP-Elites) archive.

Each cell maintains a capacity-limited Pareto front using the existing
NSGA-II primitives from x_creative.creativity.pareto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from x_creative.creativity.bd_extractor import BDExtractor
from x_creative.creativity.pareto import (
    ParetoPoint,
    crowding_distance,
    dominates,
    pareto_rank_and_crowd,
    pareto_select,
)
from x_creative.creativity.qd_types import BDSchema, BehaviorDescriptor

if TYPE_CHECKING:
    from x_creative.core.types import Hypothesis

logger = structlog.get_logger()


def _hyp_to_point(h: "Hypothesis", index: int) -> ParetoPoint | None:
    """Convert Hypothesis to ParetoPoint using novelty/feasibility axes."""
    n = h.novelty_axis()
    f = h.feasibility_axis()
    if n is None or f is None:
        return None
    return ParetoPoint(index=index, novelty=n, feasibility=f)


class MOMECell:
    """A single cell in the MAP-Elites grid, maintaining a Pareto front."""

    def __init__(self, capacity: int = 10) -> None:
        self._capacity = capacity
        self._hypotheses: list[Hypothesis] = []

    def __len__(self) -> int:
        return len(self._hypotheses)

    @property
    def hypotheses(self) -> list["Hypothesis"]:
        return list(self._hypotheses)

    def try_add(self, hypothesis: "Hypothesis") -> bool:
        """Try to add a hypothesis to this cell's Pareto front. Returns True if added."""
        new_pt = _hyp_to_point(hypothesis, 0)
        if new_pt is None:
            if len(self._hypotheses) < self._capacity:
                self._hypotheses.append(hypothesis)
                return True
            return False

        existing_pts = [_hyp_to_point(h, i) for i, h in enumerate(self._hypotheses)]
        for pt in existing_pts:
            if pt is not None and dominates(pt, new_pt):
                return False

        surviving: list[Hypothesis] = []
        for h, pt in zip(self._hypotheses, existing_pts):
            if pt is None or not dominates(new_pt, pt):
                surviving.append(h)
        surviving.append(hypothesis)

        if len(surviving) > self._capacity:
            pts = []
            for i, h in enumerate(surviving):
                p = _hyp_to_point(h, i)
                if p is not None:
                    pts.append(p)
            if pts:
                ranked = pareto_rank_and_crowd(pts)
                selected = pareto_select(ranked, self._capacity)
                keep_indices = {rp.point.index for rp in selected}
                surviving = [h for i, h in enumerate(surviving) if i in keep_indices]

        self._hypotheses = surviving
        return True

    def select(self, count: int) -> list["Hypothesis"]:
        """Select from cell's Pareto front by rank + crowding distance."""
        if not self._hypotheses:
            return []
        pts = []
        for i, h in enumerate(self._hypotheses):
            p = _hyp_to_point(h, i)
            if p is not None:
                pts.append(p)
        if not pts:
            return self._hypotheses[:count]
        ranked = pareto_rank_and_crowd(pts)
        selected = pareto_select(ranked, min(count, len(ranked)))
        indices = [rp.point.index for rp in selected]
        return [self._hypotheses[i] for i in indices]


class MOMEArchive:
    """Multi-Objective MAP-Elites archive with per-cell Pareto fronts."""

    def __init__(self, bd_schema: BDSchema, cell_capacity: int = 10) -> None:
        self._schema = bd_schema
        self._cell_capacity = cell_capacity
        self._extractor = BDExtractor(schema=bd_schema)
        self._grid: dict[tuple[int, ...], MOMECell] = {}
        self._total_cells = self._compute_total_cells()
        if len(self._schema.grid_dimensions) > 3:
            logger.warning(
                "mome_high_dimensional_grid",
                grid_dims=len(self._schema.grid_dimensions),
                recommendation="Keep MAP-Elites grid <= 3 dimensions; move extra signals to novelty_distance raw dims.",
            )

    def _compute_total_cells(self) -> int:
        total = 1
        for dim in self._schema.grid_dimensions:
            if dim.dim_type == "categorical" and dim.labels:
                total *= len(dim.labels)
            else:
                total *= dim.num_bins
        return total

    @property
    def total_cells(self) -> int:
        return self._total_cells

    @property
    def filled_cells(self) -> int:
        return len(self._grid)

    @property
    def cell_count(self) -> int:
        return len(self._grid)

    @property
    def coverage_ratio(self) -> float:
        if self._total_cells == 0:
            return 0.0
        return self.filled_cells / self._total_cells

    def add(self, hypothesis: "Hypothesis") -> bool:
        """Add a hypothesis to the archive, placing it in the appropriate cell."""
        bd = self._extractor.extract(hypothesis)
        hypothesis = hypothesis.model_copy(update={"behavior_descriptor": bd})
        coord = self._grid_coord(bd)
        if coord not in self._grid:
            self._grid[coord] = MOMECell(capacity=self._cell_capacity)
        return self._grid[coord].try_add(hypothesis)

    def select(self, count: int) -> list["Hypothesis"]:
        """Select hypotheses preferring sparse (less populated) cells first."""
        if not self._grid:
            return []
        sorted_cells = sorted(self._grid.values(), key=lambda c: len(c))
        result: list[Hypothesis] = []
        remaining = count
        cell_idx = 0
        while remaining > 0 and sorted_cells:
            cell = sorted_cells[cell_idx % len(sorted_cells)]
            selected_from_cell = cell.select(1)
            for h in selected_from_cell:
                if h.id not in {r.id for r in result}:
                    result.append(h)
                    remaining -= 1
                    if remaining <= 0:
                        break
            cell_idx += 1
            if cell_idx >= len(sorted_cells) * 2:
                break
        if remaining > 0:
            for cell in sorted_cells:
                for h in cell.hypotheses:
                    if h.id not in {r.id for r in result}:
                        result.append(h)
                        remaining -= 1
                        if remaining <= 0:
                            break
                if remaining <= 0:
                    break
        return result[:count]

    def _grid_coord(self, bd: "BehaviorDescriptor") -> tuple[int, ...]:
        """Convert a BehaviorDescriptor into a grid coordinate tuple."""
        dims = sorted(self._schema.grid_dimensions, key=lambda d: d.name)
        return tuple(bd.grid_dims.get(d.name, 0) for d in dims)


def novelty_distance(
    bd: BehaviorDescriptor,
    archive: list[BehaviorDescriptor],
    k: int = 5,
) -> float:
    """Compute kNN average distance from bd to archive."""
    if not archive:
        return float("inf")
    distances: list[float] = []
    for other in archive:
        d = _descriptor_distance(bd, other)
        distances.append(d)
    distances.sort()
    k_actual = min(k, len(distances))
    if k_actual == 0:
        return float("inf")
    return sum(distances[:k_actual]) / k_actual


def _descriptor_distance(a: BehaviorDescriptor, b: BehaviorDescriptor) -> float:
    """Distance between two BehaviorDescriptors."""
    dist = 0.0
    # Grid dims (categorical): hamming
    all_keys = set(a.grid_dims) | set(b.grid_dims)
    for key in all_keys:
        va = a.grid_dims.get(key, -1)
        vb = b.grid_dims.get(key, -1)
        if va != vb:
            dist += 1.0
    # Raw dims: absolute difference
    all_raw = set(a.raw) | set(b.raw)
    for key in all_raw:
        va = a.raw.get(key, 0.0)
        vb = b.raw.get(key, 0.0)
        if isinstance(va, str) or isinstance(vb, str):
            dist += 0.0 if va == vb else 1.0
        else:
            dist += abs(float(va) - float(vb))
    return dist
