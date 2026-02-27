"""Tests for LLM client and model routing."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMClient:
    """Tests for LLMClient."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        mock = MagicMock()
        mock.chat = MagicMock()
        mock.chat.completions = MagicMock()
        return mock

    def test_client_creation(self) -> None:
        """Test creating an LLM client."""
        from x_creative.llm.client import LLMClient

        client = LLMClient(api_key="test-key")
        assert client is not None

    def test_client_uses_openrouter_base_url(self) -> None:
        """Test that client uses OpenRouter base URL by default."""
        from x_creative.llm.client import LLMClient

        client = LLMClient(api_key="test-key")
        assert "openrouter" in client.base_url

    def test_client_custom_base_url(self) -> None:
        """Test client with custom base URL."""
        from x_creative.llm.client import LLMClient

        client = LLMClient(api_key="test-key", base_url="https://api.openai.com/v1")
        assert client.base_url == "https://api.openai.com/v1"

    def test_client_uses_yunwu_when_default_provider_is_yunwu(self) -> None:
        """Test default provider routing to Yunwu config."""
        from x_creative.llm.client import LLMClient

        mock_settings = MagicMock()
        mock_settings.default_provider = "yunwu"
        mock_settings.openrouter.api_key.get_secret_value.return_value = "openrouter-key"
        mock_settings.openrouter.base_url = "https://openrouter.ai/api/v1"
        mock_settings.yunwu.api_key.get_secret_value.return_value = "yunwu-key"
        mock_settings.yunwu.base_url = "https://yunwu.ai/v1"

        with patch("x_creative.llm.client.get_settings", return_value=mock_settings):
            client = LLMClient()
            assert client.base_url == "https://yunwu.ai/v1"

    def test_client_uses_provider_override(self) -> None:
        """Test explicit provider override beats default provider."""
        from x_creative.llm.client import LLMClient

        mock_settings = MagicMock()
        mock_settings.default_provider = "openrouter"
        mock_settings.openrouter.api_key.get_secret_value.return_value = "openrouter-key"
        mock_settings.openrouter.base_url = "https://openrouter.ai/api/v1"
        mock_settings.yunwu.api_key.get_secret_value.return_value = "yunwu-key"
        mock_settings.yunwu.base_url = "https://yunwu.ai/v1"

        with patch("x_creative.llm.client.get_settings", return_value=mock_settings):
            client = LLMClient(provider="yunwu")
            assert client.base_url == "https://yunwu.ai/v1"

    def test_client_raises_for_unknown_provider(self) -> None:
        """Test unknown provider is rejected early."""
        from x_creative.llm.client import LLMClient

        mock_settings = MagicMock()
        mock_settings.default_provider = "openrouter"
        mock_settings.openrouter.api_key.get_secret_value.return_value = "openrouter-key"
        mock_settings.openrouter.base_url = "https://openrouter.ai/api/v1"
        mock_settings.yunwu.api_key.get_secret_value.return_value = "yunwu-key"
        mock_settings.yunwu.base_url = "https://yunwu.ai/v1"

        with (
            patch("x_creative.llm.client.get_settings", return_value=mock_settings),
            pytest.raises(ValueError, match="Unsupported provider"),
        ):
            LLMClient(provider="unknown")

    @pytest.mark.asyncio
    async def test_complete_basic(self) -> None:
        """Test basic completion call."""
        from x_creative.llm.client import LLMClient

        with patch("x_creative.llm.client.AsyncOpenAI") as mock_openai:
            # Setup mock response
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Test response"
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20

            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_instance

            client = LLMClient(api_key="test-key")
            result = await client.complete(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result.content == "Test response"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_complete_with_temperature(self) -> None:
        """Test completion with temperature setting."""
        from x_creative.llm.client import LLMClient

        with patch("x_creative.llm.client.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 10

            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_instance

            client = LLMClient(api_key="test-key")
            await client.complete(
                model="openai/gpt-5.2",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.9,
            )

            # Verify temperature was passed
            call_kwargs = mock_instance.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_yunwu_model_alias_fallback_for_prefixed_model(self) -> None:
        """Yunwu should retry with unprefixed model alias."""
        from x_creative.llm.client import LLMClient

        with patch("x_creative.llm.client.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "ok"
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 1
            mock_response.usage.completion_tokens = 1

            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                side_effect=[Exception("No available channels for model openai/gpt-5.2"), mock_response]
            )
            mock_openai.return_value = mock_instance

            client = LLMClient(api_key="test-key", provider="yunwu")
            result = await client.complete(
                model="openai/gpt-5.2",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
            )

            assert result.model == "gpt-5.2"
            assert mock_instance.chat.completions.create.call_count == 2
            first_call = mock_instance.chat.completions.create.call_args_list[0].kwargs
            second_call = mock_instance.chat.completions.create.call_args_list[1].kwargs
            assert first_call["model"] == "openai/gpt-5.2"
            assert second_call["model"] == "gpt-5.2"

    @pytest.mark.asyncio
    async def test_openrouter_does_not_use_alias_fallback(self) -> None:
        """OpenRouter should keep original prefixed model without alias retry."""
        from x_creative.llm.client import LLMClient

        with patch("x_creative.llm.client.AsyncOpenAI") as mock_openai:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                side_effect=Exception("temporary failure")
            )
            mock_openai.return_value = mock_instance

            client = LLMClient(api_key="test-key", provider="openrouter")
            with pytest.raises(Exception, match="temporary failure"):
                await client.complete(
                    model="openai/gpt-5.2",
                    messages=[{"role": "user", "content": "hi"}],
                )

            assert mock_instance.chat.completions.create.call_count == 1


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_router_creation(self) -> None:
        """Test creating a model router."""
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()
        assert router is not None

    def test_get_model_for_task(self) -> None:
        """Test getting model for a specific task."""
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        creativity_model = router.get_model("creativity")
        # Model should be in provider/model format (can be overridden by .env)
        assert "/" in creativity_model
        assert len(creativity_model.split("/")) == 2

        scoring_model = router.get_model("hypothesis_scoring")
        assert scoring_model is not None

    def test_get_model_config_for_task(self) -> None:
        """Test getting full config for a task."""
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        config = router.get_config("creativity")
        assert config.model is not None
        assert config.temperature is not None
        assert config.fallback is not None

    def test_get_fallback_models(self) -> None:
        """Test getting fallback models for a task."""
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        fallbacks = router.get_fallbacks("creativity")
        # Fallbacks should be a list (possibly empty if overridden by env)
        assert isinstance(fallbacks, list)

    @pytest.mark.asyncio
    async def test_router_complete_with_fallback(self) -> None:
        """Test that router tries fallback on failure."""
        from x_creative.config.settings import ModelConfig
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        # Mock the config to ensure we have fallbacks for this test
        mock_config = ModelConfig(
            model="test/primary-model",
            fallback=["test/fallback-model"],
            temperature=0.9,
        )

        with (
            patch.object(router, "get_config", return_value=mock_config),
            patch.object(router, "_client") as mock_client,
        ):
            # First call fails, second succeeds
            mock_client.complete = AsyncMock(
                side_effect=[
                    Exception("Primary failed"),
                    MagicMock(content="Fallback response", prompt_tokens=5, completion_tokens=10),
                ]
            )

            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": "Hello"}],
            )

            # Should have tried twice (primary + 1 fallback)
            assert mock_client.complete.call_count == 2
            assert result.content == "Fallback response"

    @pytest.mark.asyncio
    async def test_router_complete_all_fallbacks_fail(self) -> None:
        """Test that router raises when all models fail."""
        from x_creative.llm.router import ModelRouter
        from x_creative.llm.exceptions import AllModelsFailedError

        router = ModelRouter()

        with patch.object(router, "_client") as mock_client:
            mock_client.complete = AsyncMock(side_effect=Exception("All failed"))

            with pytest.raises(AllModelsFailedError):
                await router.complete(
                    task="creativity",
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_router_retries_with_reduced_max_tokens_on_context_limit(self) -> None:
        """Router should retry once with reduced max_tokens when provider rejects context length."""
        from x_creative.config.settings import ModelConfig
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        mock_config = ModelConfig(
            model="test/primary-model",
            fallback=[],
            temperature=0.3,
            max_tokens=32768,
        )

        error = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 32768 tokens. "
            "However, you requested about 33758 tokens (990 of text input, 32768 in the output). "
            "Please reduce the length of either one.\", 'code': 400}}"
        )

        with (
            patch.object(router, "get_config", return_value=mock_config),
            patch.object(router, "_client") as mock_client,
        ):
            mock_client.complete = AsyncMock(
                side_effect=[
                    error,
                    MagicMock(content="Retry response", prompt_tokens=5, completion_tokens=10),
                ]
            )

            result = await router.complete(
                task="hypothesis_scoring",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result.content == "Retry response"
            assert mock_client.complete.call_count == 2

            # First attempt uses original config max_tokens.
            first_kwargs = mock_client.complete.call_args_list[0].kwargs
            assert first_kwargs["max_tokens"] == 32768

            # Retry should clamp max_tokens below max context - prompt tokens.
            retry_kwargs = mock_client.complete.call_args_list[1].kwargs
            assert retry_kwargs["max_tokens"] == 31522


    @pytest.mark.asyncio
    async def test_router_complete_with_model_override(self) -> None:
        """model_override replaces the primary model but keeps fallbacks."""
        from x_creative.config.settings import ModelConfig
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        mock_config = ModelConfig(
            model="test/primary-model",
            fallback=["test/fallback-model"],
            temperature=0.9,
        )

        with (
            patch.object(router, "get_config", return_value=mock_config),
            patch.object(router, "_client") as mock_client,
        ):
            mock_client.complete = AsyncMock(
                return_value=MagicMock(
                    content="Response",
                    model="override/model",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    finish_reason="stop",
                )
            )
            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": "test"}],
                model_override="override/model",
            )
            assert result.content == "Response"
            # Verify the override model was used, not the primary model
            call_args = mock_client.complete.call_args
            assert call_args.kwargs["model"] == "override/model"

    @pytest.mark.asyncio
    async def test_router_complete_override_falls_back_on_failure(self) -> None:
        """When override model fails, router falls back to task fallback chain."""
        from x_creative.config.settings import ModelConfig
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        mock_config = ModelConfig(
            model="test/primary-model",
            fallback=["test/fallback-model"],
            temperature=0.9,
        )

        with (
            patch.object(router, "get_config", return_value=mock_config),
            patch.object(router, "_client") as mock_client,
        ):
            mock_client.complete = AsyncMock(
                side_effect=[
                    Exception("Override model failed"),
                    MagicMock(
                        content="Fallback response",
                        model="test/fallback-model",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        finish_reason="stop",
                    ),
                ]
            )
            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": "test"}],
                model_override="override/model",
            )
            assert result.content == "Fallback response"
            assert mock_client.complete.call_count == 2


class TestCompletionResult:
    """Tests for CompletionResult."""

    def test_completion_result_creation(self) -> None:
        """Test creating a completion result."""
        from x_creative.llm.client import CompletionResult

        result = CompletionResult(
            content="Test content",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
        )

        assert result.content == "Test content"
        assert result.total_tokens == 30

    def test_completion_result_optional_fields(self) -> None:
        """Test completion result with optional fields."""
        from x_creative.llm.client import CompletionResult

        result = CompletionResult(
            content="Test",
            model="gpt-4",
            prompt_tokens=5,
            completion_tokens=10,
            finish_reason="stop",
        )

        assert result.finish_reason == "stop"
