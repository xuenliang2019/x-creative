"""Tests for ModelRouter per-model max_tokens cap enforcement."""

from unittest.mock import AsyncMock, patch

import pytest

from x_creative.llm.client import CompletionResult
from x_creative.llm.router import ModelRouter


def _make_result(model: str = "test/model") -> CompletionResult:
    return CompletionResult(
        content="ok",
        model=model,
        prompt_tokens=10,
        completion_tokens=20,
    )


@pytest.fixture()
def mock_settings():
    """Return a mock Settings with model_max_tokens and task routing."""
    with patch("x_creative.llm.router.get_settings") as mock_gs:
        settings = mock_gs.return_value
        settings.model_max_tokens = {}

        # Minimal ModelConfig-like object for task routing
        from x_creative.config.settings import ModelConfig

        settings.get_model_config.return_value = ModelConfig(
            model="primary/model",
            fallback=["fallback/model"],
            temperature=0.7,
            max_tokens=4096,
        )
        yield settings


class TestModelMaxTokensCapsRequest:
    """When a per-model cap is configured, the router should enforce it."""

    @pytest.mark.asyncio
    async def test_model_max_tokens_caps_request(self, mock_settings) -> None:
        """Router passes min(requested, cap) to client when cap < requested."""
        mock_settings.model_max_tokens = {"primary/model": 2048}

        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4096,
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_model_max_tokens_no_cap_passthrough(self, mock_settings) -> None:
        """When no cap is configured for a model, full max_tokens passes through."""
        mock_settings.model_max_tokens = {"other/model": 2048}

        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4096,
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_model_max_tokens_cap_higher_than_request(self, mock_settings) -> None:
        """When cap > requested, the requested value is used (min semantics)."""
        mock_settings.model_max_tokens = {"primary/model": 8192}

        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4096,
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_model_max_tokens_different_per_fallback(self, mock_settings) -> None:
        """Primary and fallback models can have different caps."""
        mock_settings.model_max_tokens = {
            "primary/model": 2048,
            "fallback/model": 1024,
        }

        call_log: list[dict] = []

        async def fake_complete(**kwargs):
            call_log.append(kwargs)
            if kwargs["model"] == "primary/model":
                raise RuntimeError("primary failed")
            return _make_result(kwargs["model"])

        client = AsyncMock()
        client.complete = AsyncMock(side_effect=fake_complete)

        router = ModelRouter(client=client)

        result = await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4096,
        )

        assert result.model == "fallback/model"
        # Primary was called with its cap
        assert call_log[0]["max_tokens"] == 2048
        # Fallback was called with its cap
        assert call_log[1]["max_tokens"] == 1024
