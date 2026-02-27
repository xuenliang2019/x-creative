"""Regression test: HKG disabled should not change existing behavior."""

from x_creative.config.settings import Settings
from x_creative.core.types import Hypothesis, LogicVerdict, NoveltyVerdict, VerifiedHypothesis
from x_creative.creativity.search import SearchModule


class TestHKGDisabledRegression:
    """Verify that with hkg_enabled=False, behavior is unchanged."""

    def test_search_module_no_hkg_by_default(self) -> None:
        sm = SearchModule()
        assert sm._hkg_store is None
        assert sm._hkg_matcher is None

    def test_hypothesis_without_hkg_unchanged(self) -> None:
        h = Hypothesis(
            id="h1",
            description="test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )
        # hkg_evidence should be None by default
        assert h.hkg_evidence is None
        # Existing serialization should work
        data = h.model_dump()
        h2 = Hypothesis(**data)
        assert h2.id == "h1"
        assert h2.hkg_evidence is None

    def test_verified_hypothesis_without_structural_score(self) -> None:
        vh = VerifiedHypothesis(
            id="h1",
            description="test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            logic_verdict=LogicVerdict(
                passed=True,
                analogy_validity=8,
                internal_consistency=8,
                causal_rigor=8,
                reasoning="ok",
            ),
            novelty_verdict=NoveltyVerdict(score=7, searched=False, novelty_analysis="ok"),
            final_score=7.5,
        )
        assert vh.structural_grounding_score is None

    def test_settings_hkg_disabled_by_default(self) -> None:
        s = Settings()
        assert s.hkg_enabled is False
        assert s.hkg_store_path is None
        assert s.hkg_enable_hyperbridge is False
        assert s.hkg_enable_structural_scoring is False
