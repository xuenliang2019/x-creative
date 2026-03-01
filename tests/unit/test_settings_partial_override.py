"""Tests for partial env-var overrides of nested config (TaskRoutingConfig, VerifiersConfig)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from x_creative.config.settings import (
    ModelConfig,
    Settings,
    TaskRoutingConfig,
    VerifierModelConfig,
    VerifiersConfig,
)


class TestTaskRoutingPartialOverride:
    """TaskRoutingConfig merges partial dicts with defaults."""

    def test_partial_max_tokens_preserves_model(self) -> None:
        """Setting only max_tokens keeps the default model name."""
        config = TaskRoutingConfig(**{
            "novelty_verification": {"max_tokens": 8192},
        })
        nv = config.novelty_verification
        assert nv.max_tokens == 8192
        assert nv.model == "google/gemini-2.5-flash-lite"
        assert nv.temperature == 0.3

    def test_partial_model_preserves_max_tokens(self) -> None:
        """Setting only model keeps the default max_tokens."""
        config = TaskRoutingConfig(**{
            "creativity": {"model": "custom/model"},
        })
        cr = config.creativity
        assert cr.model == "custom/model"
        assert cr.temperature == 0.9

    def test_full_override_still_works(self) -> None:
        """Providing all fields still works as before."""
        config = TaskRoutingConfig(**{
            "novelty_verification": {
                "model": "custom/model",
                "fallback": [],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
        })
        nv = config.novelty_verification
        assert nv.model == "custom/model"
        assert nv.max_tokens == 1024
        assert nv.temperature == 0.1
        assert nv.fallback == []

    def test_unmentioned_tasks_keep_defaults(self) -> None:
        """Tasks not in the override dict keep their full defaults."""
        config = TaskRoutingConfig(**{
            "novelty_verification": {"max_tokens": 8192},
        })
        # creativity should be untouched
        assert config.creativity.model == "google/gemini-2.5-pro"
        assert config.creativity.temperature == 0.9

    def test_multiple_tasks_partial_override(self) -> None:
        """Multiple tasks can be partially overridden at once."""
        config = TaskRoutingConfig(**{
            "novelty_verification": {"max_tokens": 8192},
            "logic_verification": {"max_tokens": 16384},
            "reasoner_step": {"temperature": 0.1},
        })
        assert config.novelty_verification.max_tokens == 8192
        assert config.logic_verification.max_tokens == 16384
        assert config.reasoner_step.temperature == 0.1
        # model should be preserved for all
        assert config.novelty_verification.model == "google/gemini-2.5-flash-lite"
        assert config.logic_verification.model == "google/gemini-2.5-flash"
        assert config.reasoner_step.model == "deepseek/deepseek-r1"

    def test_modelconfig_instance_passthrough(self) -> None:
        """Passing a ModelConfig instance (not a dict) still works."""
        mc = ModelConfig(model="custom/m", max_tokens=1024)
        config = TaskRoutingConfig(**{
            "novelty_verification": mc,
        })
        assert config.novelty_verification.model == "custom/m"


class TestVerifiersPartialOverride:
    """VerifiersConfig merges partial dicts with defaults."""

    def test_partial_timeout_preserves_model(self) -> None:
        config = VerifiersConfig(**{
            "logic": {"timeout": 120},
        })
        assert config.logic.timeout == 120
        assert config.logic.model == "google/gemini-2.5-flash"
        assert config.logic.temperature == 0.2

    def test_partial_model_preserves_timeout(self) -> None:
        config = VerifiersConfig(**{
            "novelty": {"model": "custom/v"},
        })
        assert config.novelty.model == "custom/v"
        assert config.novelty.timeout == 45


class TestSettingsEnvVarOverride:
    """End-to-end: env vars flow through Settings to nested config."""

    def test_env_var_overrides_task_max_tokens(self) -> None:
        env = {
            "X_CREATIVE_TASK_ROUTING__NOVELTY_VERIFICATION__MAX_TOKENS": "8192",
        }
        with patch.dict(os.environ, env, clear=False):
            s = Settings()
        assert s.task_routing.novelty_verification.max_tokens == 8192
        # model preserved
        assert s.task_routing.novelty_verification.model != ""

    def test_env_var_overrides_verifier_timeout(self) -> None:
        env = {
            "X_CREATIVE_VERIFIERS__LOGIC__TIMEOUT": "120",
        }
        with patch.dict(os.environ, env, clear=False):
            s = Settings()
        assert s.verifiers.logic.timeout == 120
        assert s.verifiers.logic.model != ""
