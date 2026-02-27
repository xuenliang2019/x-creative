"""Tests for HKG integration into existing types."""
import pytest


class TestHypothesisHKGField:
    def test_hypothesis_without_hkg(self) -> None:
        from x_creative.core.types import Hypothesis
        h = Hypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
        )
        assert h.hkg_evidence is None

    def test_hypothesis_with_hkg(self) -> None:
        from x_creative.core.types import Hypothesis
        from x_creative.hkg.types import HKGEvidence, HyperpathEvidence, HyperedgeSummary
        ev = HKGEvidence(hyperpaths=[
            HyperpathEvidence(
                start_node_id="n1", end_node_id="n2",
                path_rank=1, path_length=1,
                hyperedges=[HyperedgeSummary(edge_id="e1", nodes=["n1", "n2"], relation="r")],
                intermediate_nodes=[],
            )
        ])
        h = Hypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
            hkg_evidence=ev,
        )
        assert h.hkg_evidence is not None
        assert len(h.hkg_evidence.hyperpaths) == 1

    def test_hypothesis_serialization_roundtrip(self) -> None:
        from x_creative.core.types import Hypothesis
        from x_creative.hkg.types import HKGEvidence
        h = Hypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
            hkg_evidence=HKGEvidence(),
        )
        data = h.model_dump()
        h2 = Hypothesis(**data)
        assert h2.hkg_evidence is not None


class TestVerifiedHypothesisStructuralScore:
    def test_verified_without_structural(self) -> None:
        from x_creative.core.types import VerifiedHypothesis, LogicVerdict, NoveltyVerdict
        vh = VerifiedHypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
            logic_verdict=LogicVerdict(
                passed=True, analogy_validity=8, internal_consistency=8,
                causal_rigor=8, reasoning="ok",
            ),
            novelty_verdict=NoveltyVerdict(score=7, searched=False, novelty_analysis="ok"),
            final_score=7.5,
        )
        assert vh.structural_grounding_score is None

    def test_verified_with_structural(self) -> None:
        from x_creative.core.types import VerifiedHypothesis, LogicVerdict, NoveltyVerdict
        vh = VerifiedHypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
            logic_verdict=LogicVerdict(
                passed=True, analogy_validity=8, internal_consistency=8,
                causal_rigor=8, reasoning="ok",
            ),
            novelty_verdict=NoveltyVerdict(score=7, searched=False, novelty_analysis="ok"),
            final_score=7.5,
            structural_grounding_score=6.5,
        )
        assert vh.structural_grounding_score == 6.5

    def test_from_hypothesis_with_structural_score(self) -> None:
        from x_creative.core.types import Hypothesis, VerifiedHypothesis, LogicVerdict, NoveltyVerdict
        h = Hypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
        )
        vh = VerifiedHypothesis.from_hypothesis(
            h,
            logic_verdict=LogicVerdict(
                passed=True, analogy_validity=8, internal_consistency=8,
                causal_rigor=8, reasoning="ok",
            ),
            novelty_verdict=NoveltyVerdict(score=7, searched=False, novelty_analysis="ok"),
            final_score=7.5,
            structural_grounding_score=6.0,
        )
        assert vh.structural_grounding_score == 6.0


class TestHKGEventTypes:
    def test_hkg_event_types_exist(self) -> None:
        from x_creative.saga.events import EventType
        assert hasattr(EventType, "HKG_PATH_FOUND")
        assert hasattr(EventType, "HKG_PATH_NOT_FOUND")
        assert hasattr(EventType, "HKG_EXPANSION_CREATED")


class TestHKGSettings:
    def test_default_hkg_disabled(self) -> None:
        from x_creative.config.settings import Settings
        s = Settings()
        assert s.hkg_enabled is False
        assert s.hkg_store_path is None

    def test_hkg_settings_fields_exist(self) -> None:
        from x_creative.config.settings import Settings
        s = Settings()
        assert s.hkg_enable_hyperbridge is False
        assert s.hkg_enable_structural_scoring is False
        assert s.hkg_structural_score_weight == 0.10
        assert s.hkg_K == 3
        assert s.hkg_IS == 1
        assert s.hkg_max_len == 6
        assert s.hkg_matcher == "auto"
        assert s.hkg_embedding_provider == "openrouter"
        assert s.hkg_embedding_model == "openai/text-embedding-3-small"
        assert s.hkg_embedding_index_path is None
        assert s.hkg_top_n_hypotheses == 5

    def test_hkg_expansion_route_exists(self) -> None:
        from x_creative.config.settings import Settings
        s = Settings()
        config = s.get_model_config("hkg_expansion")
        assert config is not None
