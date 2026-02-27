"""Tests for JudgeConfidenceEstimator."""

import pytest
import statistics


class TestJudgeConfidence:
    def test_high_confidence_from_consistent_scores(self) -> None:
        from x_creative.verify.confidence import compute_judge_confidence

        scores_per_dim = {
            "analogy_validity": [8.0, 8.2, 7.9],
            "internal_consistency": [9.0, 8.8, 9.1],
            "causal_rigor": [7.5, 7.3, 7.6],
        }

        confidence = compute_judge_confidence(
            scores_per_dim=scores_per_dim,
            position_consistent=True,
        )

        assert confidence > 0.7

    def test_low_confidence_from_inconsistent_scores(self) -> None:
        from x_creative.verify.confidence import compute_judge_confidence

        scores_per_dim = {
            "analogy_validity": [3.0, 9.0, 5.0],  # High variance
            "internal_consistency": [2.0, 8.0, 6.0],
            "causal_rigor": [4.0, 7.0, 2.0],
        }

        confidence = compute_judge_confidence(
            scores_per_dim=scores_per_dim,
            position_consistent=False,
        )

        assert confidence < 0.5

    def test_position_inconsistency_lowers_confidence(self) -> None:
        from x_creative.verify.confidence import compute_judge_confidence

        scores = {
            "analogy_validity": [8.0, 8.0, 8.0],
            "internal_consistency": [8.0, 8.0, 8.0],
            "causal_rigor": [8.0, 8.0, 8.0],
        }

        conf_consistent = compute_judge_confidence(scores, position_consistent=True)
        conf_inconsistent = compute_judge_confidence(scores, position_consistent=False)

        assert conf_inconsistent < conf_consistent

    def test_position_bias_factor_is_configurable(self) -> None:
        from x_creative.verify.confidence import compute_judge_confidence

        scores = {
            "analogy_validity": [8.0, 8.0, 8.0],
            "internal_consistency": [8.0, 8.0, 8.0],
            "causal_rigor": [8.0, 8.0, 8.0],
        }

        low_penalty = compute_judge_confidence(
            scores,
            position_consistent=False,
            position_bias_confidence_factor=0.9,
        )
        high_penalty = compute_judge_confidence(
            scores,
            position_consistent=False,
            position_bias_confidence_factor=0.5,
        )
        assert low_penalty > high_penalty
