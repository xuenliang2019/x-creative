"""Tests for MappingScorer (rule + LLM hybrid)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from x_creative.core.types import FailureMode, Hypothesis, MappingItem


def _make_hypothesis(
    mapping_table: list[MappingItem] | None = None,
    failure_modes: list[FailureMode] | None = None,
    observable: str = "signal_0 + signal_1 + signal_2 + signal_3",
) -> Hypothesis:
    return Hypothesis(
        id="hyp_test",
        description="Test hypothesis about entropy",
        source_domain="thermodynamics",
        source_structure="entropy_increase",
        analogy_explanation="Entropy maps to market disorder",
        observable=observable,
        mapping_table=mapping_table or [],
        failure_modes=failure_modes
        or [
            FailureMode(
                scenario="regime shift",
                why_breaks="signal_4 no longer tracks entropy",
                detectable_signal="signal_4 regime drift",
            ),
            FailureMode(
                scenario="microstructure shock",
                why_breaks="signal_5 breaks under latency spikes",
                detectable_signal="signal_5 instability",
            ),
        ],
    )


def _make_good_mapping_table() -> list[MappingItem]:
    """Create a mapping table with effective rows, usage links, and systematic groups."""
    items = []
    for i in range(6):
        items.append(
            MappingItem(
                source_concept=f"source_{i}",
                target_concept=f"target_{i}",
                source_relation=f"relation_s_{i}",
                target_relation=f"relation_t_{i}",
                mapping_type="relation" if i < 3 else "process",
                systematicity_group_id="group_a" if i < 4 else "group_b",
                observable_link=f"signal_{i}",
            )
        )
    return items


class TestMappingScorerRules:
    """Test the anti-padding rule layer of MappingScorer."""

    def test_empty_mapping_table_scores_low(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        scorer = MappingScorer()
        h = _make_hypothesis(mapping_table=[])
        result = scorer.rule_score(h)

        assert result.score <= 4.0
        assert not result.rules_passed

    def test_effective_rows_gate_blocks_padding(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        scorer = MappingScorer()
        table = [
            MappingItem(
                source_concept=f"s{i}",
                target_concept=f"t{i}",
                source_relation="same_source_relation",
                target_relation="same_target_relation",
                mapping_type="relation",
                systematicity_group_id="g1",
                observable_link="signal_dup",
            )
            for i in range(8)
        ]
        h = _make_hypothesis(mapping_table=table)
        result = scorer.rule_score(h)

        assert not result.rules_passed
        assert result.effective_rows < 6
        assert any("effective_rows" in v for v in result.violations)

    def test_duplicate_ratio_gate(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        scorer = MappingScorer()
        base = MappingItem(
            source_concept="heat",
            target_concept="volatility",
            source_relation="heat transfer",
            target_relation="risk transfer",
            mapping_type="relation",
            systematicity_group_id="g1",
            observable_link="signal_dup",
        )
        table = [base.model_copy() for _ in range(6)]
        h = _make_hypothesis(mapping_table=table)
        result = scorer.rule_score(h)

        assert not result.rules_passed
        assert result.duplicate_ratio > 0.3
        assert any("duplicate_ratio" in v for v in result.violations)

    def test_row_usage_coverage_gate(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        scorer = MappingScorer()
        table = _make_good_mapping_table()
        h = _make_hypothesis(
            mapping_table=table,
            observable="unrelated_metric = x / y",
            failure_modes=[
                FailureMode(
                    scenario="none",
                    why_breaks="no mapping links mentioned",
                    detectable_signal="none",
                )
            ],
        )
        result = scorer.rule_score(h)

        assert not result.rules_passed
        assert result.row_usage_coverage < 0.7
        assert any("row_usage_coverage" in v for v in result.violations)

    def test_good_mapping_table_passes_rules(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        scorer = MappingScorer()
        table = _make_good_mapping_table()
        h = _make_hypothesis(mapping_table=table)
        result = scorer.rule_score(h)

        assert result.rules_passed
        assert result.score > 4.0
        assert result.effective_rows >= 6
        assert result.duplicate_ratio <= 0.3
        assert result.row_usage_coverage >= 0.7


class TestMappingScorerRuleResult:
    """Test RuleScoreResult structure."""

    def test_result_has_violations(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        scorer = MappingScorer()
        h = _make_hypothesis(mapping_table=[])
        result = scorer.rule_score(h)

        assert len(result.violations) > 0
        assert any("effective_rows" in v.lower() for v in result.violations)


class TestMappingUnpackAudit:
    """Test unpack audit behavior and score capping."""

    @pytest.mark.asyncio
    async def test_unpack_audit_failure_caps_score_and_emits_event(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        events: list[tuple[str, dict[str, object]]] = []

        def _on_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        mock_router = AsyncMock()
        mock_router.complete.side_effect = [
            AsyncMock(content='{"systematicity": 9, "depth": 8, "reasoning": "strong"}'),
            AsyncMock(
                content=(
                    '{"rows": ['
                    '{"row_index": 0, "supported": false, "reason": "not grounded"}, '
                    '{"row_index": 1, "supported": false, "reason": "not grounded"}, '
                    '{"row_index": 2, "supported": false, "reason": "not grounded"}, '
                    '{"row_index": 3, "supported": false, "reason": "not grounded"}, '
                    '{"row_index": 4, "supported": false, "reason": "not grounded"}, '
                    '{"row_index": 5, "supported": false, "reason": "not grounded"}'
                    '], "overall_reasoning": "possible padding"}'
                )
            ),
        ]

        scorer = MappingScorer(router=mock_router, event_callback=_on_event, random_seed=7)
        h = _make_hypothesis(mapping_table=_make_good_mapping_table())

        result = await scorer.score(h)

        assert result.llm_score == 8.5
        assert result.score == 6.5  # capped by unpack audit failure
        assert result.score_cap_applied is True
        assert "mapping_padding_suspected" in result.events
        assert result.unpack_audit is not None
        assert result.unpack_audit.passed is False
        assert any(event_type == "mapping_padding_suspected" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_unpack_audit_pass_keeps_llm_score(self) -> None:
        from x_creative.verify.mapping_scorer import MappingScorer

        mock_router = AsyncMock()
        mock_router.complete.side_effect = [
            AsyncMock(content='{"systematicity": 8, "depth": 8, "reasoning": "strong"}'),
            AsyncMock(
                content=(
                    '{"rows": ['
                    '{"row_index": 0, "supported": true, "reason": "used"}, '
                    '{"row_index": 1, "supported": true, "reason": "used"}, '
                    '{"row_index": 2, "supported": true, "reason": "used"}, '
                    '{"row_index": 3, "supported": true, "reason": "used"}, '
                    '{"row_index": 4, "supported": true, "reason": "used"}, '
                    '{"row_index": 5, "supported": true, "reason": "used"}'
                    '], "overall_reasoning": "grounded"}'
                )
            ),
        ]

        scorer = MappingScorer(router=mock_router, random_seed=7)
        h = _make_hypothesis(mapping_table=_make_good_mapping_table())

        result = await scorer.score(h)

        assert result.llm_score == 8.0
        assert result.score == 8.0
        assert result.score_cap_applied is False
        assert result.unpack_audit is not None
        assert result.unpack_audit.passed is True
