"""Tests for blend_expand and transform_space integration in SearchModule."""

from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.core.concept_space import (
    AllowedTransformOp,
    ConceptSpace,
    ConceptSpaceAssumption,
    ConceptSpaceConstraint,
)
from x_creative.core.types import Hypothesis, SearchConfig
from x_creative.core.transform_types import SpaceTransformDiff, TransformAction, TransformStatus


def _hypothesis(hypothesis_id: str, source_domain: str = "domain_a") -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        description=f"description {hypothesis_id}",
        source_domain=source_domain,
        source_structure="structure",
        analogy_explanation="analogy",
        observable=f"observable_{hypothesis_id}",
    )


def _concept_space() -> ConceptSpace:
    return ConceptSpace(
        version="1.0.0",
        domain_id="open_source_development",
        provenance="yaml",
        hard_constraints=[
            ConceptSpaceConstraint(
                id="c1",
                text="no lookahead",
                constraint_type="hard",
                rationale="test",
            )
        ],
        assumptions_mutable=[
            ConceptSpaceAssumption(
                id="a1",
                text="stationarity",
                mutable=True,
                rationale="test",
            )
        ],
        allowed_ops=[
            AllowedTransformOp(
                id="op1",
                name="negate assumption",
                description="negate mutable assumption",
                op_type="negate_assumption",
                target_type="assumption",
            )
        ],
    )


class TestHeavyQuotaManager:
    def test_from_config_interactive_disables_all(self) -> None:
        from x_creative.creativity.search import HeavyQuotaManager

        config = SearchConfig(runtime_profile="interactive")
        quota = HeavyQuotaManager.from_config(
            config=config, heavy_ops_enabled=False, top_n_size=5, hyperpath_expand_topN=5,
        )
        assert quota.blend == 0
        assert quota.transform == 0
        assert quota.hyperbridge == 0

    def test_from_config_research_populates_budgets(self) -> None:
        from x_creative.creativity.search import HeavyQuotaManager

        config = SearchConfig(
            runtime_profile="research",
            blend_expand_budget_per_round=4,
            transform_space_budget_per_round=3,
        )
        quota = HeavyQuotaManager.from_config(
            config=config, heavy_ops_enabled=True, top_n_size=6, hyperpath_expand_topN=5,
        )
        assert quota.blend == 4
        assert quota.transform == 3
        # hyperbridge = min(6, 5) // 2 = 2
        assert quota.hyperbridge == 2

    def test_allow_and_consume(self) -> None:
        from x_creative.creativity.search import HeavyQuotaManager

        quota = HeavyQuotaManager(blend=2, transform=0, hyperbridge=1)
        assert quota.allow("blend") is True
        assert quota.allow("transform") is False
        assert quota.allow("hyperbridge") is True
        quota.consume("blend")
        assert quota.blend == 1
        quota.consume("blend")
        assert quota.allow("blend") is False

    def test_consume_raises_on_exhaustion(self) -> None:
        from x_creative.creativity.search import HeavyQuotaManager

        quota = HeavyQuotaManager(blend=0)
        with pytest.raises(ValueError, match="Quota exhausted"):
            quota.consume("blend")

    def test_record_degradation(self) -> None:
        from x_creative.creativity.search import HeavyQuotaManager

        quota = HeavyQuotaManager()
        quota.record_degradation("blend")
        quota.record_degradation("blend")
        quota.record_degradation("transform")
        assert quota.degraded_ops == {"blend": 2, "transform": 1}

    def test_hyperbridge_quota_derivation(self) -> None:
        """§2.2: hyperbridge = min(hkg_top_n, hyperpath_expand_topN) // 2"""
        from x_creative.creativity.search import HeavyQuotaManager

        config = SearchConfig()
        quota = HeavyQuotaManager.from_config(
            config=config, heavy_ops_enabled=True, top_n_size=3, hyperpath_expand_topN=5,
        )
        assert quota.hyperbridge == min(3, 5) // 2  # == 1


class TestSearchConfigExtensions:
    def test_expansion_type_includes_all_search_operators(self) -> None:
        from x_creative.creativity.search import ExpansionType

        values = set(get_args(ExpansionType))
        assert "refine" in values
        assert "variant" in values
        assert "combine" in values
        assert "oppose" in values
        assert "extreme" in values
        assert "hyperpath_expand" in values
        assert "hyperbridge" in values
        assert "blend_expand" in values
        assert "transform_space" in values

    def test_blend_defaults(self) -> None:
        config = SearchConfig()
        assert config.enable_extreme is True
        assert config.enable_blending is False
        assert config.enable_transform_space is False
        assert config.max_blend_pairs == 3
        assert config.max_transform_hypotheses == 2
        assert config.runtime_profile == "research"
        assert config.blend_expand_budget_per_round == 3
        assert config.transform_space_budget_per_round == 2
        assert config.hyperpath_expand_topN == 5

    def test_blend_enabled(self) -> None:
        config = SearchConfig(enable_blending=True, max_blend_pairs=5)
        assert config.enable_blending is True
        assert config.max_blend_pairs == 5

    def test_transform_enabled(self) -> None:
        config = SearchConfig(enable_transform_space=True, max_transform_hypotheses=4)
        assert config.enable_transform_space is True
        assert config.max_transform_hypotheses == 4

    def test_max_blend_pairs_bounds(self) -> None:
        with pytest.raises(Exception):
            SearchConfig(max_blend_pairs=0)
        with pytest.raises(Exception):
            SearchConfig(max_blend_pairs=11)

    def test_max_transform_hypotheses_bounds(self) -> None:
        with pytest.raises(Exception):
            SearchConfig(max_transform_hypotheses=0)
        with pytest.raises(Exception):
            SearchConfig(max_transform_hypotheses=6)


class TestSelectBlendPairs:
    def test_prefers_cross_domain(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        hyps = [
            Hypothesis(
                id="a",
                description="d",
                source_domain="thermo",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
            ),
            Hypothesis(
                id="b",
                description="d",
                source_domain="info",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
            ),
            Hypothesis(
                id="c",
                description="d",
                source_domain="thermo",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
            ),
        ]
        pairs = search._select_blend_pairs(hyps, 2)
        assert len(pairs) >= 1
        # At least one cross-domain pair
        assert any(ha.source_domain != hb.source_domain for ha, hb in pairs)

    def test_limits_pairs(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        hyps = [
            Hypothesis(
                id=f"h{i}",
                description="d",
                source_domain=f"dom{i}",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
            )
            for i in range(10)
        ]
        pairs = search._select_blend_pairs(hyps, 3)
        assert len(pairs) <= 3

    def test_fallback_same_domain(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        hyps = [
            Hypothesis(
                id=f"h{i}",
                description="d",
                source_domain="same",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
            )
            for i in range(4)
        ]
        pairs = search._select_blend_pairs(hyps, 2)
        # Should fall back to same-domain pairs
        assert len(pairs) == 2

    def test_empty_with_single_hypothesis(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        hyps = [
            Hypothesis(
                id="solo",
                description="d",
                source_domain="dom",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
            ),
        ]
        pairs = search._select_blend_pairs(hyps, 3)
        assert pairs == []


class TestSearchOperatorScheduling:
    @pytest.mark.asyncio
    async def test_search_iteration_includes_extreme_when_enabled(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        base = _hypothesis("h1")
        config = SearchConfig(
            search_depth=1,
            search_breadth=1,
            enable_combination=False,
            enable_opposition=False,
            enable_extreme=True,
        )

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])) as mock_expand:
            await search.search_iteration([base], config)

        expansion_types = mock_expand.await_args.kwargs["expansion_types"]
        assert "extreme" in expansion_types

    @pytest.mark.asyncio
    async def test_search_iteration_skips_extreme_when_disabled(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        base = _hypothesis("h1")
        config = SearchConfig(
            search_depth=1,
            search_breadth=1,
            enable_combination=False,
            enable_opposition=False,
            enable_extreme=False,
        )

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])) as mock_expand:
            await search.search_iteration([base], config)

        expansion_types = mock_expand.await_args.kwargs["expansion_types"]
        assert "extreme" not in expansion_types

    @pytest.mark.asyncio
    async def test_interactive_profile_disables_heavy_operators(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule(concept_space=_concept_space(), enable_hyperbridge=True)
        h1 = _hypothesis("h1", "thermo")
        h2 = _hypothesis("h2", "ecology")
        config = SearchConfig(
            search_depth=1,
            search_breadth=2,
            enable_combination=False,
            enable_opposition=False,
            enable_blending=True,
            enable_transform_space=True,
            runtime_profile="interactive",
            blend_expand_budget_per_round=3,
            transform_space_budget_per_round=2,
        )

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])), \
             patch("x_creative.creativity.blend.blend_expand", AsyncMock(return_value=[_hypothesis("blend_1")] )) as mock_blend, \
             patch("x_creative.creativity.transform.transform_space", AsyncMock(return_value=[_hypothesis("transform_1")])) as mock_transform:
            await search.search_iteration([h1, h2], config)

        assert mock_blend.await_count == 0
        assert mock_transform.await_count == 0


class TestSearchOperatorEvents:
    @pytest.mark.asyncio
    async def test_blend_expand_emits_event_callback(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        h1 = _hypothesis("h1", "thermo")
        h2 = _hypothesis("h2", "ecology")
        blended = _hypothesis("blend_1", "thermo+ecology")
        config = SearchConfig(
            search_depth=1,
            search_breadth=2,
            enable_combination=False,
            enable_opposition=False,
            enable_blending=True,
            max_blend_pairs=1,
            enable_transform_space=False,
        )

        events: list[tuple[str, dict[str, object]]] = []

        async def on_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])), \
             patch("x_creative.creativity.blend.blend_expand", AsyncMock(return_value=[blended])):
            await search.search_iteration([h1, h2], config, on_hkg_event=on_event)

        assert any(event_type == "blend_expand_completed" for event_type, _ in events)
        blend_payload = [payload for event_type, payload in events if event_type == "blend_expand_completed"][0]
        assert blend_payload["new_count"] == 1

    @pytest.mark.asyncio
    async def test_transform_space_emits_event_callback(self) -> None:
        from x_creative.creativity.search import SearchModule

        search = SearchModule(concept_space=_concept_space())
        base = _hypothesis("h1", "open_source_development")
        transformed = _hypothesis("transform_1", "open_source_development").model_copy(
            update={
                "space_transform_diff": SpaceTransformDiff(
                    concept_space_version="1.0.0",
                    actions=[
                        TransformAction(
                            op_id="op1",
                            op_type="negate_assumption",
                            target_id="a1",
                            before_state="stationarity=true",
                            after_state="stationarity=false",
                            rationale="test transform event chain",
                        )
                    ],
                    new_failure_modes=[],
                    transform_status=TransformStatus.PROPOSED,
                )
            }
        )
        config = SearchConfig(
            search_depth=1,
            search_breadth=1,
            enable_combination=False,
            enable_opposition=False,
            enable_blending=False,
            enable_transform_space=True,
            max_transform_hypotheses=1,
        )

        events: list[tuple[str, dict[str, object]]] = []

        async def on_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        # Mock C→K LLM verification to accept (§10.5.1)
        ck_accept_result = MagicMock()
        ck_accept_result.content = '{"verdict": "accept", "confidence": 0.9, "reasoning": "ok", "issues": []}'
        ck_accept_response = AsyncMock(return_value=ck_accept_result)

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])), \
             patch("x_creative.creativity.transform.transform_space", AsyncMock(return_value=[transformed])), \
             patch.object(search._router, "complete", ck_accept_response):
            await search.search_iteration([base], config, on_hkg_event=on_event)

        assert any(event_type == "transform_proposed" for event_type, _ in events)
        assert any(
            event_type == "transform_accepted_or_rejected"
            for event_type, _ in events
        )
        assert any(event_type == "transform_space_applied" for event_type, _ in events)

        gate_payload = [
            payload
            for event_type, payload in events
            if event_type == "transform_accepted_or_rejected"
        ][0]
        assert gate_payload["transform_status"] == "ACCEPTED"
        assert "ck_verification_passed" in gate_payload["validation_notes"]

        transform_payload = [payload for event_type, payload in events if event_type == "transform_space_applied"][0]
        assert transform_payload["new_count"] == 1
        assert transform_payload["concept_space_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_transform_gate_rejects_unvalidated_high_risk_proposal(self) -> None:
        """Op type mismatch is caught in Phase 1 (before C→K LLM call)."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule(concept_space=_concept_space())
        base = _hypothesis("h1", "open_source_development")
        proposed = _hypothesis("transform_unsafe", "open_source_development").model_copy(
            update={
                "space_transform_diff": SpaceTransformDiff(
                    concept_space_version="1.0.0",
                    actions=[
                        TransformAction(
                            op_id="op1",
                            op_type="change_representation",
                            target_id="a1",
                            before_state="x",
                            after_state="y",
                            rationale="try alternative representation",
                        )
                    ],
                    new_failure_modes=[],
                    transform_status=TransformStatus.PROPOSED,
                )
            }
        )
        config = SearchConfig(
            search_depth=1,
            search_breadth=1,
            enable_combination=False,
            enable_opposition=False,
            enable_blending=False,
            enable_transform_space=True,
            runtime_profile="research",
            transform_space_budget_per_round=1,
            max_transform_hypotheses=1,
        )

        events: list[tuple[str, dict[str, object]]] = []

        async def on_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])), \
             patch("x_creative.creativity.transform.transform_space", AsyncMock(return_value=[proposed])):
            out = await search.search_iteration([base], config, on_hkg_event=on_event)

        assert all(h.id != "transform_unsafe" for h in out)
        gate_payload = [
            payload
            for event_type, payload in events
            if event_type == "transform_accepted_or_rejected"
        ][0]
        assert gate_payload["transform_status"] == "REJECTED"
        assert gate_payload["rejection_reason"] == "op_type_mismatch:op1:change_representation!=negate_assumption"
        payload = [p for e, p in events if e == "transform_space_applied"][0]
        assert payload["proposed_count"] == 1
        assert payload["rejected_count"] == 1
        assert payload["new_count"] == 0

    @pytest.mark.asyncio
    async def test_transform_gate_rejects_on_ck_verification_failure(self) -> None:
        """C→K LLM Logic Verifier rejects the transform (§10.5.1)."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule(concept_space=_concept_space())
        base = _hypothesis("h1", "open_source_development")
        proposed = _hypothesis("transform_ck_fail", "open_source_development").model_copy(
            update={
                "space_transform_diff": SpaceTransformDiff(
                    concept_space_version="1.0.0",
                    actions=[
                        TransformAction(
                            op_id="op1",
                            op_type="negate_assumption",
                            target_id="a1",
                            before_state="stationarity=true",
                            after_state="stationarity=false",
                            rationale="negate stationarity",
                        )
                    ],
                    new_failure_modes=[],
                    transform_status=TransformStatus.PROPOSED,
                )
            }
        )
        config = SearchConfig(
            search_depth=1,
            search_breadth=1,
            enable_combination=False,
            enable_opposition=False,
            enable_blending=False,
            enable_transform_space=True,
            runtime_profile="research",
            transform_space_budget_per_round=1,
            max_transform_hypotheses=1,
        )

        events: list[tuple[str, dict[str, object]]] = []

        async def on_event(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        # Mock C→K LLM verification to reject
        ck_reject_result = MagicMock()
        ck_reject_result.content = '{"verdict": "reject", "confidence": 0.8, "reasoning": "incoherent", "issues": ["no testable consequence"]}'
        ck_reject_response = AsyncMock(return_value=ck_reject_result)

        with patch.object(search, "expand_hypothesis", AsyncMock(return_value=[])), \
             patch("x_creative.creativity.transform.transform_space", AsyncMock(return_value=[proposed])), \
             patch.object(search._router, "complete", ck_reject_response):
            out = await search.search_iteration([base], config, on_hkg_event=on_event)

        assert all(h.id != "transform_ck_fail" for h in out)
        gate_payload = [
            payload
            for event_type, payload in events
            if event_type == "transform_accepted_or_rejected"
        ][0]
        assert gate_payload["transform_status"] == "REJECTED"
        assert gate_payload["rejection_reason"] == "ck_logic_verification_failed"
        assert "ck_verification_rejected" in gate_payload["validation_notes"]
