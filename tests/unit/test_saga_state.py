"""Tests for SAGA shared cognition state models."""

from x_creative.saga.state import (
    CognitionAlert,
    EvaluationAdjustments,
    GenerationMetrics,
    SharedCognitionState,
)


class TestGenerationMetrics:
    """Tests for GenerationMetrics model."""

    def test_default_creation(self) -> None:
        """Test creating GenerationMetrics with defaults."""
        metrics = GenerationMetrics()
        assert metrics.source_domain_distribution == {}
        assert metrics.unique_structure_count == 0
        assert metrics.expansion_type_distribution == {}
        assert metrics.score_mean == 0.0
        assert metrics.score_std == 0.0
        assert metrics.score_min == 0.0
        assert metrics.score_max == 0.0
        assert metrics.dimension_correlations == {}
        assert metrics.generation_depth_vs_quality == []
        assert metrics.deduplication_ratio == 0.0

    def test_creation_with_values(self) -> None:
        """Test creating GenerationMetrics with actual values."""
        metrics = GenerationMetrics(
            source_domain_distribution={"ecology": 5, "thermodynamics": 3},
            unique_structure_count=8,
            score_mean=6.5,
            score_std=1.2,
            score_min=3.0,
            score_max=9.0,
        )
        assert metrics.source_domain_distribution["ecology"] == 5
        assert metrics.unique_structure_count == 8
        assert metrics.score_mean == 6.5


class TestCognitionAlert:
    """Tests for CognitionAlert model."""

    def test_creation(self) -> None:
        """Test creating a CognitionAlert."""
        alert = CognitionAlert(
            alert_type="reward_hacking",
            severity="critical",
            description="Score compression detected",
            evidence={"std": 0.3},
            suggested_action="adjust_weights",
            timestamp=1000.0,
        )
        assert alert.alert_type == "reward_hacking"
        assert alert.severity == "critical"
        assert alert.evidence == {"std": 0.3}

    def test_default_values(self) -> None:
        """Test CognitionAlert default values."""
        alert = CognitionAlert(
            alert_type="diversity_decay",
            severity="warning",
            description="Diversity declining",
        )
        assert alert.evidence == {}
        assert alert.suggested_action == ""
        assert alert.timestamp == 0.0


class TestEvaluationAdjustments:
    """Tests for EvaluationAdjustments model."""

    def test_default_creation(self) -> None:
        """Test creating EvaluationAdjustments with defaults."""
        adj = EvaluationAdjustments()
        assert adj.weight_overrides is None
        assert adj.prune_threshold_override is None
        assert adj.extra_dimensions is None
        assert adj.hard_gates is None
        assert adj.adjustment_reasons == []

    def test_with_overrides(self) -> None:
        """Test creating EvaluationAdjustments with overrides."""
        adj = EvaluationAdjustments(
            weight_overrides={"divergence": 0.30, "robustness": 0.25},
            prune_threshold_override=6.0,
            hard_gates=["no_lookahead_bias"],
            adjustment_reasons=["Domain requires higher divergence"],
        )
        assert adj.weight_overrides["divergence"] == 0.30
        assert adj.prune_threshold_override == 6.0
        assert len(adj.hard_gates) == 1
        assert len(adj.adjustment_reasons) == 1


class TestSharedCognitionState:
    """Tests for SharedCognitionState model."""

    def test_default_creation(self) -> None:
        """Test creating SharedCognitionState with defaults."""
        state = SharedCognitionState()
        assert state.current_stage == "idle"
        assert state.hypotheses_pool == []
        assert state.generation_metrics is None
        assert state.active_alerts == []
        assert isinstance(state.evaluation_adjustments, EvaluationAdjustments)
        assert state.intervention_count == 0
        assert state.target_domain_id is None
        assert state.historical_hypothesis_hashes == set()
        assert state.adversarial_challenges == []

    def test_with_target_domain(self) -> None:
        """Test creating state with target domain."""
        state = SharedCognitionState(target_domain_id="open_source_development")
        assert state.target_domain_id == "open_source_development"

    def test_mutable_defaults_independent(self) -> None:
        """Test that mutable defaults are independent between instances."""
        state1 = SharedCognitionState()
        state2 = SharedCognitionState()
        state1.active_alerts.append(
            CognitionAlert(
                alert_type="test",
                severity="info",
                description="test",
            )
        )
        assert len(state1.active_alerts) == 1
        assert len(state2.active_alerts) == 0

    def test_intervention_count_increment(self) -> None:
        """Test incrementing intervention count."""
        state = SharedCognitionState()
        state.intervention_count += 1
        assert state.intervention_count == 1

    def test_historical_hashes(self) -> None:
        """Test historical hypothesis hashes."""
        state = SharedCognitionState()
        state.historical_hypothesis_hashes.add("hash_abc")
        state.historical_hypothesis_hashes.add("hash_def")
        assert len(state.historical_hypothesis_hashes) == 2
        assert "hash_abc" in state.historical_hypothesis_hashes
