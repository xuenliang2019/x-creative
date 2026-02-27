"""BDExtractor: extracts BehaviorDescriptor from Hypothesis.

Default path is rule-based and deterministic.
Mixed extraction supports constrained enum hints (e.g., LLM-produced labels)
with cache-backed replay stability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from x_creative.creativity.qd_types import BDSchema, BehaviorDescriptor, GridConfig

if TYPE_CHECKING:
    from x_creative.core.types import Hypothesis

# Mapping from source_domain keywords to mechanism family labels
_MECHANISM_FAMILY_MAP: dict[str, str] = {
    "thermodynamic": "thermodynamic",
    "entropy": "thermodynamic",
    "heat": "thermodynamic",
    "information": "information",
    "signal": "information",
    "communication": "information",
    "game": "game_theory",
    "nash": "game_theory",
    "auction": "game_theory",
    "biological": "biological",
    "ecology": "biological",
    "evolution": "biological",
    "predator": "biological",
}

# Keywords for data granularity detection in observable text
_GRANULARITY_KEYWORDS: dict[str, list[str]] = {
    "tick": ["tick", "order_book", "orderbook", "microstructure", "bid_ask"],
    "minute": ["minute", "intraday", "5min", "15min", "1min"],
    "daily": ["daily", "close", "open", "ohlc", "return"],
    "weekly": ["weekly", "month", "quarter"],
}


class BDExtractor:
    """Extracts BehaviorDescriptor from Hypothesis with replay-stable logic."""

    def __init__(self, schema: BDSchema, allow_mixed_hints: bool = True) -> None:
        self._schema = schema
        self._allow_mixed_hints = allow_mixed_hints
        self._mixed_cache: dict[tuple[str, str], tuple[int, float, list[str]]] = {}

    def extract(self, hypothesis: Hypothesis) -> BehaviorDescriptor:
        """Extract a BehaviorDescriptor from a hypothesis."""
        grid_dims: dict[str, int] = {}
        extraction_method = "rule"
        extraction_confidence = 1.0
        evidence_path: list[str] = []

        for dim_config in self._schema.grid_dimensions:
            if dim_config.name == "mechanism_family":
                (
                    mechanism_index,
                    mechanism_method,
                    mechanism_confidence,
                    mechanism_evidence,
                ) = self._extract_mechanism_family(hypothesis, dim_config)
                grid_dims[dim_config.name] = mechanism_index
                if mechanism_method == "hybrid":
                    extraction_method = "hybrid"
                    extraction_confidence = min(extraction_confidence, mechanism_confidence)
                    evidence_path.extend(mechanism_evidence)
            elif dim_config.name == "data_granularity":
                grid_dims[dim_config.name] = self._extract_data_granularity(
                    hypothesis, dim_config
                )
            else:
                grid_dims[dim_config.name] = (
                    len(dim_config.labels) - 1 if dim_config.labels else 0
                )

        raw: dict[str, float | str] = {}
        for raw_dim in self._schema.raw_dimensions:
            if raw_dim == "causal_chain_length":
                raw[raw_dim] = float(
                    len([m for m in hypothesis.mapping_table if m.mapping_type == "relation"])
                    or len(hypothesis.mapping_table)
                )
            elif raw_dim == "constraint_count":
                raw[raw_dim] = float(len(hypothesis.failure_modes))
            else:
                raw[raw_dim] = 0.0

        return BehaviorDescriptor(
            grid_dims=grid_dims,
            raw=raw,
            version=self._schema.version,
            bd_version=self._schema.version,
            extraction_method=extraction_method,
            confidence=max(0.0, min(1.0, extraction_confidence)),
            evidence_path=evidence_path,
        )

    def _extract_mechanism_family(
        self,
        hypothesis: Hypothesis,
        config: GridConfig,
    ) -> tuple[int, str, float, list[str]]:
        rule_index = self._extract_mechanism_family_rule(hypothesis, config)
        if not self._allow_mixed_hints or config.labels is None:
            return rule_index, "rule", 1.0, [f"source_domain:{hypothesis.source_domain}"]

        hint = self._extract_mixed_hint(hypothesis, config.labels)
        if hint is None:
            return rule_index, "rule", 1.0, [f"source_domain:{hypothesis.source_domain}"]

        label, confidence, hint_evidence = hint
        cache_key = (hypothesis.id, self._schema.version)
        cached = self._mixed_cache.get(cache_key)
        if cached is None:
            mixed_index = config.labels.index(label)
            cached = (mixed_index, confidence, hint_evidence)
            self._mixed_cache[cache_key] = cached

        mixed_index, mixed_confidence, mixed_evidence = cached
        return mixed_index, "hybrid", mixed_confidence, mixed_evidence

    @staticmethod
    def _extract_mixed_hint(
        hypothesis: Hypothesis,
        labels: list[str],
    ) -> tuple[str, float, list[str]] | None:
        """Read constrained enum hint from hypothesis evidence coverage.

        Expected shape:
        hypothesis.hkg_evidence.coverage["bd_hint"] = {
            "mechanism_family": <label in labels>,
            "confidence": <0..1>,
            "evidence_path": ["...", ...]
        }
        """
        if hypothesis.hkg_evidence is None:
            return None
        coverage = hypothesis.hkg_evidence.coverage
        if not isinstance(coverage, dict):
            return None

        raw_hint = coverage.get("bd_hint")
        if not isinstance(raw_hint, dict):
            return None

        label = str(raw_hint.get("mechanism_family", "")).strip()
        if not label or label not in labels:
            return None

        confidence = raw_hint.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        evidence = raw_hint.get("evidence_path", [])
        if not isinstance(evidence, list):
            evidence = []
        evidence_path = [str(item) for item in evidence if str(item).strip()]

        return label, confidence, evidence_path

    @staticmethod
    def _extract_mechanism_family_rule(hypothesis: Hypothesis, config: GridConfig) -> int:
        if config.labels is None:
            return 0
        domain_lower = hypothesis.source_domain.lower()
        for keyword, family in _MECHANISM_FAMILY_MAP.items():
            if keyword in domain_lower:
                try:
                    return config.labels.index(family)
                except ValueError:
                    continue
        return len(config.labels) - 1

    @staticmethod
    def _extract_data_granularity(hypothesis: Hypothesis, config: GridConfig) -> int:
        if config.labels is None:
            return 0
        obs_lower = hypothesis.observable.lower()
        for label in config.labels:
            keywords = _GRANULARITY_KEYWORDS.get(label, [])
            for kw in keywords:
                if kw in obs_lower:
                    return config.labels.index(label)
        try:
            return config.labels.index("daily")
        except ValueError:
            return 0
