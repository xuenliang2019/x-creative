"""Tests for configuration management."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from x_creative.config.settings import Settings


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self) -> None:
        """Test creating settings with defaults."""
        from x_creative.config.settings import Settings

        # Avoid reading project .env; this test asserts code defaults.
        settings = Settings(_env_file=None)

        assert settings.default_provider == "openrouter"
        assert settings.default_num_hypotheses == 50
        assert settings.default_search_depth == 3
        assert settings.biso_max_concurrency > 0

    def test_model_routing_defaults(self) -> None:
        """Test default model routing configuration."""
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)

        creativity_config = settings.task_routing.creativity
        # Model should be in provider/model format (e.g., anthropic/claude-sonnet-4)
        assert "/" in creativity_config.model
        assert len(creativity_config.model.split("/")) == 2
        assert creativity_config.temperature == 0.9

    def test_yunwu_defaults(self) -> None:
        """Test Yunwu provider default configuration."""
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)
        assert settings.yunwu.base_url == "https://yunwu.ai/v1"

    def test_yunwu_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test reading Yunwu API key from environment variable."""
        from x_creative.config.settings import Settings

        monkeypatch.setenv("YUNWU_API_KEY", "yunwu-key")
        settings = Settings(_env_file=None)
        assert settings.yunwu.api_key.get_secret_value() == "yunwu-key"

    def test_get_model_config(self) -> None:
        """Test getting model config for specific tasks."""
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)

        creativity_config = settings.get_model_config("creativity")
        assert creativity_config.temperature == 0.9

        scoring_config = settings.get_model_config("hypothesis_scoring")
        assert scoring_config.temperature == 0.3

    def test_get_model_config_unknown_task(self) -> None:
        """Test getting model config for unknown task returns creativity default."""
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)

        unknown_config = settings.get_model_config("unknown_task")
        # Should return creativity config as fallback
        assert unknown_config.model == settings.task_routing.creativity.model

    def test_score_weights_sum_to_one(self) -> None:
        """Test that default score weights sum to 1.0."""
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)

        total = (
            settings.score_weight_divergence
            + settings.score_weight_testability
            + settings.score_weight_rationale
            + settings.score_weight_robustness
            + settings.score_weight_feasibility
        )
        assert abs(total - 1.0) < 0.001

    def test_final_score_weights_sum_to_one(self) -> None:
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)
        total = settings.final_score_logic_weight + settings.final_score_novelty_weight
        assert abs(total - 1.0) < 0.001

    def test_theory_alignment_defaults(self) -> None:
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)
        assert settings.constraint_similarity_threshold == 0.6
        assert settings.multi_sample_evaluations == 3
        assert settings.position_bias_confidence_factor == 0.7
        assert settings.score_compression_threshold == 0.8
        assert settings.dimension_colinearity_threshold == 0.7

    def test_biso_pool_default_has_models(self) -> None:
        """biso_pool defaults to a non-empty model list for diversity."""
        settings = Settings(_env_file=None)
        assert len(settings.biso_pool) >= 2
        assert all("/" in m for m in settings.biso_pool)

    def test_biso_pool_from_init(self) -> None:
        """biso_pool can be set via constructor."""
        settings = Settings(
            _env_file=None,
            biso_pool=["google/gemini-2.5-pro", "anthropic/claude-sonnet-4"],
        )
        assert len(settings.biso_pool) == 2
        assert "google/gemini-2.5-pro" in settings.biso_pool

    def test_biso_dedup_enabled_default(self) -> None:
        """biso_dedup_enabled defaults to True."""
        settings = Settings(_env_file=None)
        assert settings.biso_dedup_enabled is True

    def test_pareto_settings_defaults(self) -> None:
        """Test QD-Pareto selection defaults."""
        from x_creative.config.settings import Settings

        settings = Settings(_env_file=None)

        assert settings.pareto_selection_enabled is False
        assert settings.pareto_novelty_bins == 5
        assert settings.pareto_wn_min == 0.15
        assert settings.pareto_wn_max == 0.55
        assert settings.pareto_gamma == 2.0


class TestDomainLoader:
    """Tests for domain loading functionality."""

    def test_load_domains_from_target_domain(self) -> None:
        """Test loading domains via from_target_domain."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        library = DomainLibrary.from_target_domain("open_source_development")

        # Should have loaded 7 domains from open_source_development
        assert len(library) == 7

        # Check specific domain exists
        ids = library.list_ids()
        assert "流行病学与传播动力学" in ids
        assert "复杂网络科学与级联扩散" in ids

    def test_domain_has_structures(self) -> None:
        """Test that loaded domains have structures."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        library = DomainLibrary.from_target_domain("open_source_development")

        epi = library.get("流行病学与传播动力学")
        assert epi is not None
        assert len(epi.structures) > 0
        assert any(s.id == "compartment_flow_model" for s in epi.structures)

    def test_domain_has_target_mappings(self) -> None:
        """Test that loaded domains have target mappings."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        library = DomainLibrary.from_target_domain("open_source_development")

        epi = library.get("流行病学与传播动力学")
        assert epi is not None
        assert len(epi.target_mappings) > 0

    def test_get_domain_by_id(self) -> None:
        """Test getting a domain by its ID."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        library = DomainLibrary.from_target_domain("open_source_development")

        epi = library.get("流行病学与传播动力学")
        assert epi is not None
        assert epi.name == "流行病学与传播动力学"

        nonexistent = library.get("nonexistent")
        assert nonexistent is None

    def test_domain_library_list_all(self) -> None:
        """Test listing all domain IDs."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        library = DomainLibrary.from_target_domain("open_source_development")

        ids = library.list_ids()
        assert len(ids) == 7
        assert "流行病学与传播动力学" in ids
        assert "复杂网络科学与级联扩散" in ids


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        from x_creative.config.settings import get_settings

        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that settings can be configured from environment."""
        from x_creative.config.settings import Settings

        monkeypatch.setenv("X_CREATIVE_DEFAULT_NUM_HYPOTHESES", "100")

        settings = Settings()
        assert settings.default_num_hypotheses == 100


class TestVerifierConfig:
    """Tests for verifier configuration."""

    def test_default_logic_verifier(self) -> None:
        from x_creative.config.settings import get_settings

        # Clear cache for fresh settings
        get_settings.cache_clear()

        settings = get_settings()

        # Model should be in provider/model format (can be overridden by .env)
        assert "/" in settings.verifiers.logic.model
        assert len(settings.verifiers.logic.model.split("/")) == 2
        assert settings.verifiers.logic.temperature == 0.2

    def test_default_novelty_verifier(self) -> None:
        from x_creative.config.settings import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        # Model should be in provider/model format (can be overridden by .env)
        assert "/" in settings.verifiers.novelty.model
        assert len(settings.verifiers.novelty.model.split("/")) == 2
        assert settings.verifiers.novelty.temperature == 0.3


class TestSearchConfig:
    """Tests for search configuration."""

    def test_default_search_config(self) -> None:
        from x_creative.config.settings import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.search.provider == "brave"
        assert settings.search.search_threshold == 6.0
        assert len(settings.search.rounds) == 3

    def test_search_rounds_weights(self) -> None:
        from x_creative.config.settings import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        weights = [r.weight for r in settings.search.rounds]
        assert abs(sum(weights) - 1.0) < 0.01  # Sum to 1.0

    def test_search_round_weight_validation(self) -> None:
        """Test that search round weight must be between 0 and 1."""
        from pydantic import ValidationError
        from x_creative.config.settings import SearchRoundConfig

        # Valid weights
        valid = SearchRoundConfig(name="test", weight=0.5)
        assert valid.weight == 0.5

        # Invalid weight > 1.0
        with pytest.raises(ValidationError):
            SearchRoundConfig(name="test", weight=1.5)

        # Invalid weight < 0.0
        with pytest.raises(ValidationError):
            SearchRoundConfig(name="test", weight=-0.1)


class TestMOMESettings:
    """Tests for MOME (Multi-Objective MAP-Elites) settings."""

    def test_mome_defaults(self) -> None:
        from x_creative.config.settings import Settings

        s = Settings()
        assert s.mome_enabled is False
        assert s.mome_cell_capacity == 10

    def test_mome_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_CREATIVE_MOME_ENABLED", "true")
        monkeypatch.setenv("X_CREATIVE_MOME_CELL_CAPACITY", "20")
        monkeypatch.setenv("X_CREATIVE_PARETO_SELECTION_ENABLED", "true")
        from x_creative.config.settings import Settings

        s = Settings()
        assert s.mome_enabled is True
        assert s.mome_cell_capacity == 20


class TestBlendTransformSettings:
    def test_blend_defaults(self) -> None:
        s = Settings()
        assert s.runtime_profile == "interactive"
        assert s.enable_extreme is True
        assert s.enable_blending is False
        assert s.enable_transform_space is False
        assert s.max_blend_pairs == 3
        assert s.max_transform_hypotheses == 2
        assert s.blend_expand_budget_per_round == 3
        assert s.transform_space_budget_per_round == 2
        assert s.hyperpath_expand_topN == 5

    def test_blend_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_CREATIVE_ENABLE_BLENDING", "true")
        monkeypatch.setenv("X_CREATIVE_MAX_BLEND_PAIRS", "5")
        s = Settings()
        assert s.enable_blending is True
        assert s.max_blend_pairs == 5

    def test_transform_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_CREATIVE_ENABLE_TRANSFORM_SPACE", "true")
        monkeypatch.setenv("X_CREATIVE_MAX_TRANSFORM_HYPOTHESES", "4")
        s = Settings()
        assert s.enable_transform_space is True
        assert s.max_transform_hypotheses == 4

    def test_extreme_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_CREATIVE_ENABLE_EXTREME", "false")
        s = Settings()
        assert s.enable_extreme is False

    def test_runtime_profile_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_CREATIVE_RUNTIME_PROFILE", "research")
        s = Settings()
        assert s.runtime_profile == "research"


class TestMOMEPrerequisiteValidator:
    def test_mome_enabled_requires_pareto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOME requires pareto_selection_enabled=True."""
        monkeypatch.setenv("X_CREATIVE_MOME_ENABLED", "true")
        monkeypatch.setenv("X_CREATIVE_PARETO_SELECTION_ENABLED", "false")
        with pytest.raises(ValidationError, match="pareto_selection_enabled"):
            Settings()

    def test_mome_enabled_with_pareto_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOME with pareto enabled should work."""
        monkeypatch.setenv("X_CREATIVE_MOME_ENABLED", "true")
        monkeypatch.setenv("X_CREATIVE_PARETO_SELECTION_ENABLED", "true")
        s = Settings()
        assert s.mome_enabled is True
        assert s.pareto_selection_enabled is True


class TestCognitiveBudgetAllocation:
    def test_default_allocation_sums_to_100(self) -> None:
        """§3.5: default cognitive budget percentages must sum to 100."""
        from x_creative.config.settings import CognitiveBudgetAllocation

        alloc = CognitiveBudgetAllocation()
        total = (
            alloc.emergency_reserve
            + alloc.domain_audit
            + alloc.biso_monitor
            + alloc.search_monitor
            + alloc.verify_monitor
            + alloc.adversarial
            + alloc.global_review
        )
        assert abs(total - 100.0) < 0.01

    def test_default_values_match_theory(self) -> None:
        """§3.5: verify each default matches theory spec."""
        from x_creative.config.settings import CognitiveBudgetAllocation

        alloc = CognitiveBudgetAllocation()
        assert alloc.emergency_reserve == 10.0
        assert alloc.domain_audit == 9.0
        assert alloc.biso_monitor == 13.5
        assert alloc.search_monitor == 13.5
        assert alloc.verify_monitor == 18.0
        assert alloc.adversarial == 22.5
        assert alloc.global_review == 13.5

    def test_invalid_sum_rejected(self) -> None:
        """Allocation that doesn't sum to 100 must fail."""
        from x_creative.config.settings import CognitiveBudgetAllocation

        with pytest.raises(ValidationError, match="sum to 100"):
            CognitiveBudgetAllocation(emergency_reserve=50.0)

    def test_settings_includes_allocation(self) -> None:
        """Settings.saga_cognitive_budget_allocation exists with defaults."""
        s = Settings()
        alloc = s.saga_cognitive_budget_allocation
        assert alloc.adversarial == 22.5
