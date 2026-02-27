"""Structural grounding score for the VERIFY stage.

Evaluates how well a hypothesis is grounded in structural evidence
from the hypergraph by scoring hyperpath quality across four metrics.
"""

from __future__ import annotations

from x_creative.hkg.types import HKGEvidence, HyperpathEvidence


def _average_intersection_size(path: HyperpathEvidence) -> float:
    """Average shared-node count across consecutive hyperedges."""
    edges = path.hyperedges
    if len(edges) <= 1:
        return 0.0
    intersections: list[int] = []
    for idx in range(len(edges) - 1):
        left_nodes = set(edges[idx].nodes)
        right_nodes = set(edges[idx + 1].nodes)
        intersections.append(len(left_nodes & right_nodes))
    return sum(intersections) / len(intersections) if intersections else 0.0


def _hub_ratio(path: HyperpathEvidence) -> float:
    """Max edge-participation ratio of any node in the path."""
    edges = path.hyperedges
    if len(edges) <= 1:
        return 0.0

    node_edge_count: dict[str, int] = {}
    for edge in edges:
        for node_id in set(edge.nodes):
            node_edge_count[node_id] = node_edge_count.get(node_id, 0) + 1

    if not node_edge_count:
        return 0.0

    max_participation = max(node_edge_count.values())
    return max_participation / len(edges)


def _score_single_path(
    path: HyperpathEvidence,
    max_path_len: int,
    expected_is: int,
) -> float:
    """Compute the structural grounding score for a single hyperpath.

    Returns a value in [0, 10].
    """
    edges = path.hyperedges
    num_edges = len(edges)

    # --- 1. path_length_score (weight 0.30) ---
    if num_edges <= 2:
        path_length_score = 10.0
    elif num_edges <= max_path_len and max_path_len > 2:
        # Linear decay from 10.0 (at length 2) to 1.0 (at max_path_len)
        path_length_score = 10.0 - (num_edges - 2) * (9.0 / (max_path_len - 2))
    else:
        path_length_score = 1.0

    # --- 2. intersection_quality (weight 0.25) ---
    if num_edges <= 1:
        intersection_quality = 10.0
    else:
        avg_intersection_size = _average_intersection_size(path)
        target_is = max(1, expected_is)
        intersection_quality = max(
            0.0,
            min((avg_intersection_size / target_is) * 10.0, 10.0),
        )

    # --- 3. provenance_coverage (weight 0.25) ---
    if num_edges > 0:
        avg_prov_per_edge = sum(len(e.provenance_refs) for e in edges) / num_edges
    else:
        avg_prov_per_edge = 0.0
    provenance_coverage = min(avg_prov_per_edge * 5.0, 10.0)

    # --- 4. hub_penalty (weight 0.20) ---
    if num_edges <= 1:
        hub_penalty = 10.0
    else:
        ratio = _hub_ratio(path)
        # Keep full score when ratio <= 0.5; linearly penalize above that.
        hub_penalty = max(
            0.0,
            min(10.0, 10.0 * (1.0 - max(0.0, (ratio - 0.5) / 0.5))),
        )

    # --- Weighted sum ---
    score = (
        0.30 * path_length_score
        + 0.25 * intersection_quality
        + 0.25 * provenance_coverage
        + 0.20 * hub_penalty
    )

    return score


def _match_confidence_metrics(coverage: dict[str, object] | None) -> tuple[float, float] | None:
    """Return (min_confidence, avg_confidence) from match coverage chain."""
    if not isinstance(coverage, dict):
        return None

    confidences: list[float] = []
    for key in ("start_match", "end_match"):
        value = coverage.get(key)
        if not isinstance(value, dict):
            continue
        confidence = value.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(max(0.0, min(1.0, float(confidence))))

    if not confidences:
        return None

    min_conf = min(confidences)
    avg_conf = sum(confidences) / len(confidences)
    return min_conf, avg_conf


def structural_grounding_score(
    evidence: HKGEvidence | None,
    max_path_len: int = 6,
) -> float | None:
    """Score how well a hypothesis is grounded in hypergraph structure.

    Parameters
    ----------
    evidence:
        Aggregated HKG evidence. *None* or empty hyperpaths -> *None*.
    max_path_len:
        Maximum path length used for the path-length score decay.

    Returns
    -------
    float | None
        A score in [0, 10], or *None* when there is no evidence to score.
    """
    if evidence is None or not evidence.hyperpaths:
        return None

    expected_is = evidence.hkg_params.IS if evidence.hkg_params is not None else 1
    scores = [
        _score_single_path(
            p,
            max_path_len=max_path_len,
            expected_is=expected_is,
        )
        for p in evidence.hyperpaths
    ]
    base_score = sum(scores) / len(scores)

    confidence_metrics = _match_confidence_metrics(evidence.coverage)
    if confidence_metrics is None:
        return base_score

    min_conf, avg_conf = confidence_metrics
    # Gate high structural scores when term->node matching confidence is weak.
    confidence_factor = 0.35 + 0.65 * (0.7 * min_conf + 0.3 * avg_conf)
    return base_score * confidence_factor
