"""Settings management with Pydantic Settings.

Configuration priority (highest to lowest):
1. Environment variables
2. .env file in current directory
3. User config file (~/.config/x-creative/config.yaml)
4. Default values
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# User config directory
USER_CONFIG_DIR = Path.home() / ".config" / "x-creative"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"


def _load_user_config() -> dict[str, Any]:
    """Load user configuration from ~/.config/x-creative/config.yaml."""
    if USER_CONFIG_FILE.exists():
        with open(USER_CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class ModelConfig(BaseSettings):
    """Configuration for a specific model."""

    model: str
    fallback: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096


class TaskRoutingConfig(BaseSettings):
    """Task-to-model routing configuration."""

    creativity: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-pro",
            fallback=["anthropic/claude-sonnet-4"],
            temperature=0.9,
        )
    )
    analogical_mapping: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.7,
        )
    )
    structured_search: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.5,
        )
    )
    hypothesis_scoring: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash-lite",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.3,
        )
    )
    code_generation: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="deepseek/deepseek-chat-v3-0324",
            fallback=["google/gemini-2.5-flash"],
            temperature=0.2,
        )
    )
    code_review: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.3,
        )
    )
    knowledge_extraction: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="deepseek/deepseek-chat-v3-0324",
            fallback=["google/gemini-2.5-flash-lite"],
            temperature=0.2,
        )
    )
    logic_verification: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.2,
            max_tokens=2048,
        )
    )
    novelty_verification: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash-lite",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.3,
            max_tokens=2048,
        )
    )

    # SAGA task routes
    saga_adversarial: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="deepseek/deepseek-chat-v3-0324",
            fallback=["google/gemini-2.5-flash"],
            temperature=0.4,
            max_tokens=4096,
        )
    )
    saga_checkpoint: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash-lite",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.2,
            max_tokens=2048,
        )
    )
    saga_deep_audit: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.3,
            max_tokens=4096,
        )
    )

    # HKG task route
    hkg_expansion: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.4,
            max_tokens=4096,
        )
    )

    # Blend/Transform task routes
    blend_expansion: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-pro",
            fallback=["deepseek/deepseek-r1"],
            temperature=0.8,
            max_tokens=4096,
        )
    )
    transform_space: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.6,
            max_tokens=4096,
        )
    )

    # Talker-Reasoner task routes
    reasoner_step: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="deepseek/deepseek-r1",
            fallback=["google/gemini-2.5-pro"],
            temperature=0.3,
            max_tokens=4096,
        )
    )
    talker_output: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-pro",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.2,
            max_tokens=8192,
        )
    )

    # Constraint compliance task routes (used by `answer` deep research)
    constraint_compliance_audit: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-flash",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.3,
            max_tokens=4096,
        )
    )
    constraint_compliance_revision: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="google/gemini-2.5-pro",
            fallback=["deepseek/deepseek-chat-v3-0324"],
            temperature=0.3,
            max_tokens=4096,
        )
    )


class OpenRouterConfig(BaseSettings):
    """OpenRouter-specific configuration."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="OPENROUTER_API_KEY",
    )
    default_fallback_behavior: str = "chain"  # chain | parallel | none

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,  # Allow both field name and alias
    )


class YunwuConfig(BaseSettings):
    """Yunwu-specific configuration."""

    base_url: str = "https://yunwu.ai/v1"
    api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="YUNWU_API_KEY",
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


class VerifierModelConfig(BaseSettings):
    """Configuration for a verifier model."""

    provider: str = "openrouter"
    model: str
    temperature: float = 0.3
    timeout: int = 60


class VerifiersConfig(BaseSettings):
    """Configuration for logic and novelty verifiers."""

    logic: VerifierModelConfig = Field(
        default_factory=lambda: VerifierModelConfig(
            model="google/gemini-2.5-flash",
            temperature=0.2,
            timeout=60,
        )
    )
    novelty: VerifierModelConfig = Field(
        default_factory=lambda: VerifierModelConfig(
            model="google/gemini-2.5-flash-lite",
            temperature=0.3,
            timeout=45,
        )
    )


class SearchRoundConfig(BaseSettings):
    """Configuration for a search round."""

    name: str
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight for this search round")
    max_results: int = 10


class SearchConfig(BaseSettings):
    """Configuration for novelty search."""

    provider: str = "brave"
    search_threshold: float = 6.0  # LLM score threshold to trigger search
    rounds: list[SearchRoundConfig] = Field(
        default_factory=lambda: [
            SearchRoundConfig(name="concept", weight=0.3, max_results=10),
            SearchRoundConfig(name="implementation", weight=0.5, max_results=15),
            SearchRoundConfig(name="cross_domain", weight=0.2, max_results=8),
        ]
    )


class BraveSearchConfig(BaseSettings):
    """Brave Search API configuration."""

    api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="BRAVE_SEARCH_API_KEY",
    )
    base_url: str = "https://api.search.brave.com/res/v1"

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,  # Allow both field name and alias
    )


class CognitiveBudgetAllocation(BaseSettings):
    """Cognitive budget allocation percentages for SAGA Slow Agent (§3.5).

    Controls how the total cognitive budget is distributed across
    monitoring categories.  Percentages must sum to 100.

    Configurable via env vars with prefix ``X_CREATIVE_COGNITIVE_BUDGET_``,
    e.g. ``X_CREATIVE_COGNITIVE_BUDGET_ADVERSARIAL=30``.
    """

    # Theory §3.5: emergency_reserve is 10% of total; remaining 90% is
    # split domain_audit:10%, biso:15%, search:15%, verify:20%,
    # adversarial:25%, global_review:15%.  Values below are expressed as
    # percentages of the **total** budget (not the remainder).
    emergency_reserve: float = Field(default=10.0, ge=0.0, le=100.0)
    domain_audit: float = Field(default=9.0, ge=0.0, le=100.0)
    biso_monitor: float = Field(default=13.5, ge=0.0, le=100.0)
    search_monitor: float = Field(default=13.5, ge=0.0, le=100.0)
    verify_monitor: float = Field(default=18.0, ge=0.0, le=100.0)
    adversarial: float = Field(default=22.5, ge=0.0, le=100.0)
    global_review: float = Field(default=13.5, ge=0.0, le=100.0)

    model_config = SettingsConfigDict(
        env_prefix="X_CREATIVE_COGNITIVE_BUDGET_",
        # Do NOT read .env here — parent Settings already reads it and
        # passes nested values down via env_nested_delimiter.
        extra="ignore",
    )

    @model_validator(mode="after")
    def _validate_sum(self) -> "CognitiveBudgetAllocation":
        total = (
            self.emergency_reserve
            + self.domain_audit
            + self.biso_monitor
            + self.search_monitor
            + self.verify_monitor
            + self.adversarial
            + self.global_review
        )
        if abs(total - 100.0) > 0.1:
            raise ValueError(
                f"Cognitive budget allocation must sum to 100%, got {total:.1f}%"
            )
        return self


class Settings(BaseSettings):
    """Main application settings.

    Configuration is loaded from multiple sources (highest priority first):
    1. Environment variables (X_CREATIVE_* prefix)
    2. .env file in current directory
    3. User config file (~/.config/x-creative/config.yaml)
    4. Default values

    Example .env file:
        OPENROUTER_API_KEY=sk-or-v1-your-key
        YUNWU_API_KEY=your-yunwu-key
        X_CREATIVE_DEFAULT_NUM_HYPOTHESES=100

    Example config.yaml:
        openrouter:
          api_key: sk-or-v1-your-key
        yunwu:
          api_key: your-yunwu-key
        default_num_hypotheses: 100
    """

    model_config = SettingsConfigDict(
        env_prefix="X_CREATIVE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # Provider selection
    default_provider: str = "openrouter"

    # Task routing
    task_routing: TaskRoutingConfig = Field(default_factory=TaskRoutingConfig)

    # OpenRouter config
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    yunwu: YunwuConfig = Field(default_factory=YunwuConfig)

    # Verifiers config
    verifiers: VerifiersConfig = Field(default_factory=VerifiersConfig)

    # Search config
    search: SearchConfig = Field(default_factory=SearchConfig)

    # Brave Search config
    brave_search: BraveSearchConfig = Field(default_factory=BraveSearchConfig)

    # Paths
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "x-creative"
    )

    # Creativity Engine defaults
    default_num_hypotheses: int = 50
    default_search_depth: int = 3
    default_search_breadth: int = 5
    biso_max_concurrency: int = 8

    # BISO LLM pool for diversity
    biso_pool: list[str] = Field(
        default_factory=lambda: [
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-4",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat-v3-0324",
        ]
    )

    # BISO semantic deduplication
    biso_dedup_enabled: bool = True

    # SAGA settings
    saga_enable_by_default: bool = False
    saga_default_budget: float = 100.0
    saga_cognitive_budget_allocation: CognitiveBudgetAllocation = Field(
        default_factory=CognitiveBudgetAllocation,
    )

    # Scoring weights (must sum to 1.0)
    score_weight_divergence: float = 0.21
    score_weight_testability: float = 0.26
    score_weight_rationale: float = 0.21
    score_weight_robustness: float = 0.17
    score_weight_feasibility: float = 0.15
    final_score_logic_weight: float = 0.4
    final_score_novelty_weight: float = 0.6

    # VERIFY confidence controls
    multi_sample_evaluations: int = Field(default=3, ge=1, le=9)
    position_bias_confidence_factor: float = Field(default=0.7, ge=0.0, le=1.0)

    # HKG settings
    hkg_enabled: bool = False
    hkg_store_path: Path | None = None
    hkg_enable_hyperbridge: bool = False
    hkg_enable_structural_scoring: bool = False
    hkg_structural_score_weight: float = 0.10
    hkg_K: int = 3
    hkg_IS: int = 1
    hkg_max_len: int = 6
    hkg_matcher: str = "auto"
    hkg_embedding_provider: str = "openrouter"
    hkg_embedding_model: str = "openai/text-embedding-3-small"
    hkg_embedding_index_path: Path | None = None
    hkg_top_n_hypotheses: int = 5

    # Mapping quality gate
    mapping_quality_gate_threshold: float = 6.0
    mapping_quality_gate_enabled: bool = True

    # Constraint budget
    max_constraints: int = 15
    constraint_similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # QD-Pareto selection
    pareto_selection_enabled: bool = False
    pareto_novelty_bins: int = 5
    pareto_wn_min: float = 0.15
    pareto_wn_max: float = 0.55
    pareto_gamma: float = 2.0

    # MOME (Multi-Objective MAP-Elites) settings
    mome_enabled: bool = False
    mome_cell_capacity: int = 10

    # Creativity search operator settings
    runtime_profile: str = "interactive"
    enable_extreme: bool = True
    enable_blending: bool = False
    enable_transform_space: bool = False
    max_blend_pairs: int = 3
    max_transform_hypotheses: int = 2
    blend_expand_budget_per_round: int = 3
    transform_space_budget_per_round: int = 2
    hyperpath_expand_topN: int = 5

    # C-K Coordinator
    ck_enabled: bool = False
    ck_min_phase_duration_s: float = 10.0
    ck_max_k_expansion_per_session: int = 5
    ck_coverage_plateau_threshold: int = 2
    ck_evidence_gap_threshold: float = 5.0

    # Slow-Agent detector thresholds
    score_compression_threshold: float = Field(default=0.8, ge=0.0, le=10.0)
    dimension_colinearity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "dimension_colinearity_threshold",
            "dimension_collinearity_threshold",
        ),
    )

    @model_validator(mode="after")
    def _mome_requires_pareto(self) -> "Settings":
        if self.mome_enabled and not self.pareto_selection_enabled:
            raise ValueError(
                "mome_enabled=True requires pareto_selection_enabled=True"
            )
        return self

    @model_validator(mode="after")
    def _validate_score_weights_sum(self) -> "Settings":
        total = (
            self.score_weight_divergence
            + self.score_weight_testability
            + self.score_weight_rationale
            + self.score_weight_robustness
            + self.score_weight_feasibility
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"score_weight_* must sum to 1.0, got {total:.4f}"
            )
        return self

    @model_validator(mode="after")
    def _validate_final_score_weights(self) -> "Settings":
        total = self.final_score_logic_weight + self.final_score_novelty_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "final_score_logic_weight + final_score_novelty_weight must equal 1.0"
            )
        return self

    def get_model_config(self, task: str) -> ModelConfig:
        """Get model configuration for a specific task."""
        routing: dict[str, Any] = self.task_routing.model_dump()
        if task in routing:
            return ModelConfig(**routing[task])
        # Default to creativity config
        return self.task_routing.creativity


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Loads configuration from:
    1. Default values
    2. User config file (~/.config/x-creative/config.yaml)
    3. .env file
    4. Environment variables (highest priority)
    """
    # Load user config as base
    user_config = _load_user_config()

    # Handle nested openrouter config
    openrouter_config = None
    if "openrouter" in user_config:
        openrouter_data = user_config.pop("openrouter")
        openrouter_config = OpenRouterConfig(**openrouter_data)

    # Handle nested brave_search config
    brave_search_config = None
    if "brave_search" in user_config:
        brave_search_data = user_config.pop("brave_search")
        brave_search_config = BraveSearchConfig(**brave_search_data)

    # Handle nested yunwu config
    yunwu_config = None
    if "yunwu" in user_config:
        yunwu_data = user_config.pop("yunwu")
        yunwu_config = YunwuConfig(**yunwu_data)

    # Create settings with user config as defaults
    # Environment variables and .env will override these
    settings = Settings(**user_config)

    # If we loaded openrouter config from user file and env var is not set,
    # use the user config value
    if (
        openrouter_config
        and openrouter_config.api_key.get_secret_value()
        and not settings.openrouter.api_key.get_secret_value()
    ):
        settings.openrouter = openrouter_config

    # If we loaded brave_search config from user file and env var is not set,
    # use the user config value
    if (
        brave_search_config
        and brave_search_config.api_key.get_secret_value()
        and not settings.brave_search.api_key.get_secret_value()
    ):
        settings.brave_search = brave_search_config

    # If we loaded yunwu config from user file and env var is not set,
    # use the user config value
    if (
        yunwu_config
        and yunwu_config.api_key.get_secret_value()
        and not settings.yunwu.api_key.get_secret_value()
    ):
        settings.yunwu = yunwu_config

    return settings


def init_user_config() -> Path:
    """Initialize user config directory and return the config file path.

    Creates ~/.config/x-creative/config.yaml with a template if it doesn't exist.
    """
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not USER_CONFIG_FILE.exists():
        template = """# X-Creative Configuration
# This file is loaded automatically. Environment variables take priority.

# OpenRouter API configuration
openrouter:
  api_key: ""  # Your OpenRouter API key (or set OPENROUTER_API_KEY env var)
  base_url: "https://openrouter.ai/api/v1"

# Yunwu API configuration
# yunwu:
#   api_key: ""  # Your Yunwu API key (or set YUNWU_API_KEY env var)
#   base_url: "https://yunwu.ai/v1"

# Brave Search API configuration (optional, for novelty verification)
# brave_search:
#   api_key: ""  # Your Brave Search API key (or set BRAVE_SEARCH_API_KEY env var)

# Default generation settings
# default_num_hypotheses: 50
# default_search_depth: 3
# default_search_breadth: 5

# BISO LLM pool for diversity
# Each domain randomly selects a model from this pool.
# Set to empty list to disable and use only the creativity task's primary model.
# biso_pool:
#   - "google/gemini-2.5-pro"
#   - "anthropic/claude-sonnet-4"
#   - "deepseek/deepseek-r1"
#   - "deepseek/deepseek-chat-v3-0324"

# Score weights (must sum to 1.0)
# score_weight_divergence: 0.21
# score_weight_testability: 0.26
# score_weight_rationale: 0.21
# score_weight_robustness: 0.17
# score_weight_feasibility: 0.15
"""
        USER_CONFIG_FILE.write_text(template, encoding="utf-8")

    return USER_CONFIG_FILE
