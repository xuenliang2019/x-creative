"""Baseline SAGA anomaly detectors used by unit tests."""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from math import sqrt

from x_creative.saga.events import EventType, FastAgentEvent
from x_creative.saga.slow_agent import BaseDetector
from x_creative.saga.state import CognitionAlert, SharedCognitionState


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort numeric conversion for detector inputs."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_alert(
    alert_type: str,
    severity: str,
    description: str,
    evidence: dict | None = None,
) -> CognitionAlert:
    return CognitionAlert(
        alert_type=alert_type,
        severity=severity,
        description=description,
        evidence=evidence or {},
    )


class ScoreCompressionDetector(BaseDetector):
    """Detect suspiciously compressed score distributions."""

    def __init__(self, std_threshold: float = 0.8, min_hypotheses: int = 5) -> None:
        self._std_threshold = std_threshold
        self._min_hypotheses = min_hypotheses

    async def detect(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,  # noqa: ARG002
    ) -> list[CognitionAlert]:
        if event.event_type != EventType.VERIFY_BATCH_SCORED:
            return []

        score_std = _safe_float(event.metrics.get("score_std"), 0.0)
        hypothesis_count = int(
            _safe_float(
                event.metrics.get("hypothesis_count", event.metrics.get("scored_count")),
                0.0,
            )
        )
        if hypothesis_count < self._min_hypotheses:
            return []
        if score_std >= self._std_threshold:
            return []

        return [
            _make_alert(
                alert_type="score_compression",
                severity="critical",
                description="Score distribution is overly compressed",
                evidence={
                    "score_std": score_std,
                    "threshold": self._std_threshold,
                    "hypothesis_count": hypothesis_count,
                },
            )
        ]


class StructureCollapseDetector(BaseDetector):
    """Detect collapse to too few unique structures in search output."""

    def __init__(self, min_hypotheses: int = 5, min_structures: int = 2) -> None:
        self._min_hypotheses = min_hypotheses
        self._min_structures = min_structures

    async def detect(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,  # noqa: ARG002
    ) -> list[CognitionAlert]:
        if event.event_type != EventType.SEARCH_COMPLETED:
            return []

        hypothesis_count = int(_safe_float(event.metrics.get("hypothesis_count"), 0.0))
        unique_structure_count = int(
            _safe_float(event.metrics.get("unique_structure_count"), 0.0)
        )
        if hypothesis_count < self._min_hypotheses:
            return []
        if unique_structure_count >= self._min_structures:
            return []

        return [
            _make_alert(
                alert_type="structure_collapse",
                severity="critical",
                description="Search output collapsed to too few unique structures",
                evidence={
                    "hypothesis_count": hypothesis_count,
                    "unique_structure_count": unique_structure_count,
                },
            )
        ]


def _pearson_corr(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [x - mean_a for x in values_a]
    centered_b = [x - mean_b for x in values_b]
    numerator = sum(a * b for a, b in zip(centered_a, centered_b, strict=False))
    denom_a = sqrt(sum(a * a for a in centered_a))
    denom_b = sqrt(sum(b * b for b in centered_b))
    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return numerator / (denom_a * denom_b)


class DimensionCollinearityDetector(BaseDetector):
    """Detect excessive correlation across scoring dimensions."""

    SCORE_FIELDS = ("divergence", "testability", "rationale", "robustness", "feasibility")

    def __init__(self, corr_threshold: float = 0.7, min_hypotheses: int = 5) -> None:
        self._corr_threshold = corr_threshold
        self._min_hypotheses = min_hypotheses

    async def detect(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,
    ) -> list[CognitionAlert]:
        if event.event_type != EventType.VERIFY_COMPLETED:
            return []

        pool = state.hypotheses_pool
        if len(pool) < self._min_hypotheses:
            return []

        rows: list[dict] = []
        for item in pool:
            data = item if isinstance(item, dict) else {}
            scores = data.get("scores")
            if isinstance(scores, dict):
                rows.append(scores)

        if len(rows) < self._min_hypotheses:
            return []

        for lhs, rhs in combinations(self.SCORE_FIELDS, 2):
            a_vals = [_safe_float(row.get(lhs), 0.0) for row in rows]
            b_vals = [_safe_float(row.get(rhs), 0.0) for row in rows]
            corr = _pearson_corr(a_vals, b_vals)
            if abs(corr) >= self._corr_threshold:
                return [
                    _make_alert(
                        alert_type="dimension_collinearity",
                        severity="warning",
                        description="Scoring dimensions are highly collinear",
                        evidence={
                            "pair": f"{lhs}:{rhs}",
                            "correlation": round(corr, 4),
                            "threshold": self._corr_threshold,
                        },
                    )
                ]
        return []


class SourceDomainBiasDetector(BaseDetector):
    """Detect source-domain score imbalance."""

    def __init__(
        self,
        f_threshold: float = 1.2,
        min_domains: int = 2,
        min_samples: int = 6,
    ) -> None:
        self._f_threshold = f_threshold
        self._min_domains = min_domains
        self._min_samples = min_samples

    @staticmethod
    def _anova_f_statistic(groups: list[list[float]]) -> float:
        """Compute one-way ANOVA F statistic for multiple score groups."""
        valid_groups = [group for group in groups if len(group) >= 2]
        if len(valid_groups) < 2:
            return 0.0

        total_samples = sum(len(group) for group in valid_groups)
        if total_samples <= len(valid_groups):
            return 0.0

        group_means = [sum(group) / len(group) for group in valid_groups]
        grand_mean = sum(sum(group) for group in valid_groups) / total_samples

        ss_between = sum(
            len(group) * (group_mean - grand_mean) ** 2
            for group, group_mean in zip(valid_groups, group_means, strict=False)
        )
        ss_within = sum(
            sum((value - group_mean) ** 2 for value in group)
            for group, group_mean in zip(valid_groups, group_means, strict=False)
        )

        df_between = len(valid_groups) - 1
        df_within = total_samples - len(valid_groups)
        if df_between <= 0 or df_within <= 0:
            return 0.0

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        if ms_within == 0.0:
            return float("inf") if ms_between > 0.0 else 0.0
        return ms_between / ms_within

    async def detect(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,
    ) -> list[CognitionAlert]:
        if event.event_type != EventType.VERIFY_BATCH_SCORED:
            return []

        buckets: dict[str, list[float]] = {}
        for item in state.hypotheses_pool:
            data = item if isinstance(item, dict) else {}
            domain = str(data.get("source_domain", "")).strip()
            score = data.get("final_score")
            if not domain or score is None:
                continue
            buckets.setdefault(domain, []).append(_safe_float(score, 0.0))

        domain_count = len(buckets)
        sample_count = sum(len(scores) for scores in buckets.values())
        if domain_count < self._min_domains or sample_count < self._min_samples:
            return []

        groups = [scores for scores in buckets.values() if scores]
        if len(groups) < self._min_domains:
            return []
        anova_f = self._anova_f_statistic(groups)
        if anova_f < self._f_threshold:
            return []

        return [
            _make_alert(
                alert_type="source_domain_bias",
                severity="warning",
                description="Potential source-domain bias detected (ANOVA)",
                evidence={
                    "domain_count": domain_count,
                    "sample_count": sample_count,
                    "anova_f": round(anova_f, 4),
                    "threshold": self._f_threshold,
                    "method": "one_way_anova_f_statistic",
                },
            )
        ]


class ShallowRewriteDetector(BaseDetector):
    """Detect near-duplicate rewrites with score inflation/deflation."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        score_gap_threshold: float = 2.0,
        min_hypotheses: int = 2,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._score_gap_threshold = score_gap_threshold
        self._min_hypotheses = min_hypotheses

    async def detect(
        self,
        event: FastAgentEvent,
        state: SharedCognitionState,
    ) -> list[CognitionAlert]:
        if event.event_type != EventType.VERIFY_COMPLETED:
            return []

        pool = [item for item in state.hypotheses_pool if isinstance(item, dict)]
        if len(pool) < self._min_hypotheses:
            return []

        for left, right in combinations(pool, 2):
            left_desc = str(left.get("description", ""))
            right_desc = str(right.get("description", ""))
            similarity = SequenceMatcher(None, left_desc, right_desc).ratio()
            if similarity < self._similarity_threshold:
                continue

            left_score = _safe_float(left.get("final_score"), 0.0)
            right_score = _safe_float(right.get("final_score"), 0.0)
            score_gap = abs(left_score - right_score)
            if score_gap < self._score_gap_threshold:
                continue

            return [
                _make_alert(
                    alert_type="shallow_rewrite",
                    severity="warning",
                    description="Near-duplicate rewrite with large score gap",
                    evidence={
                        "left_id": left.get("id"),
                        "right_id": right.get("id"),
                        "similarity": round(similarity, 4),
                        "score_gap": round(score_gap, 4),
                    },
                )
            ]
        return []
