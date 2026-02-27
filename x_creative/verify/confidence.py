"""Judge confidence estimation for VERIFY stage.

Computes confidence based on:
- Score variance across k samples (lower variance = higher confidence)
- Position consistency (consistent = higher confidence)
"""

from __future__ import annotations

import math
import statistics


def compute_judge_confidence(
    scores_per_dim: dict[str, list[float]],
    position_consistent: bool = True,
    position_bias_confidence_factor: float = 0.7,
) -> float:
    """Compute judge confidence from multi-sample scores.

    Args:
        scores_per_dim: Dict mapping dimension name to list of scores across k samples.
        position_consistent: Whether scores are consistent across position swaps.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not scores_per_dim:
        return 0.0

    # Compute average standard deviation across dimensions
    stds = []
    for dim_scores in scores_per_dim.values():
        if len(dim_scores) >= 2:
            stds.append(statistics.stdev(dim_scores))
        else:
            stds.append(0.0)

    avg_std = statistics.mean(stds) if stds else 0.0

    # Convert std to confidence: std=0 -> 1.0, std=3 -> ~0.2
    # Using exponential decay: conf = exp(-0.5 * std)
    variance_confidence = math.exp(-0.5 * avg_std)

    # Position consistency penalty (configurable)
    penalty = max(0.0, min(1.0, float(position_bias_confidence_factor)))
    position_factor = 1.0 if position_consistent else penalty

    confidence = variance_confidence * position_factor

    return round(max(0.0, min(1.0, confidence)), 3)
