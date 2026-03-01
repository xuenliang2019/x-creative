"""Tests for JSON mode (response_format) feature."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.config.settings import ModelConfig, TaskRoutingConfig
from x_creative.llm.client import CompletionResult
from x_creative.llm.router import ModelRouter


def _make_result(model: str = "test/model") -> CompletionResult:
    return CompletionResult(
        content='{"ok": true}',
        model=model,
        prompt_tokens=10,
        completion_tokens=20,
    )


@pytest.fixture()
def mock_settings():
    """Return a mock Settings with task routing that has json_mode."""
    with patch("x_creative.llm.router.get_settings") as mock_gs:
        settings = mock_gs.return_value
        settings.model_max_tokens = {}

        settings.get_model_config.return_value = ModelConfig(
            model="primary/model",
            fallback=["fallback/model"],
            temperature=0.7,
            max_tokens=4096,
            json_mode=True,
        )
        yield settings


@pytest.fixture()
def mock_settings_no_json():
    """Return a mock Settings with json_mode=False."""
    with patch("x_creative.llm.router.get_settings") as mock_gs:
        settings = mock_gs.return_value
        settings.model_max_tokens = {}

        settings.get_model_config.return_value = ModelConfig(
            model="primary/model",
            fallback=["fallback/model"],
            temperature=0.7,
            max_tokens=4096,
            json_mode=False,
        )
        yield settings


class TestRouterJsonModeInjection:
    """Router should inject response_format when json_mode is effective."""

    @pytest.mark.asyncio
    async def test_router_injects_response_format_when_json_mode_true(
        self, mock_settings
    ) -> None:
        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_router_no_response_format_when_json_mode_false(
        self, mock_settings_no_json
    ) -> None:
        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="code_generation",
            messages=[{"role": "user", "content": "test"}],
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert "response_format" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_router_caller_override_json_mode_false(
        self, mock_settings
    ) -> None:
        """Task config has json_mode=True but caller passes json_mode=False."""
        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
            json_mode=False,
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert "response_format" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_router_caller_override_json_mode_true(
        self, mock_settings_no_json
    ) -> None:
        """Task config has json_mode=False but caller passes json_mode=True."""
        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        await router.complete(
            task="code_generation",
            messages=[{"role": "user", "content": "test"}],
            json_mode=True,
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_router_does_not_overwrite_explicit_response_format(
        self, mock_settings
    ) -> None:
        """If caller already provides response_format, don't overwrite it."""
        client = AsyncMock()
        client.complete = AsyncMock(return_value=_make_result("primary/model"))

        router = ModelRouter(client=client)

        custom_format = {"type": "json_schema", "json_schema": {"name": "test"}}
        await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": "test"}],
            response_format=custom_format,
        )

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["response_format"] == custom_format


class TestCheckJsonMode:
    """Tests for check_json_mode() in checker.py."""

    def test_check_json_mode_pass(self) -> None:
        from x_creative.config.checker import check_json_mode

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"ok":true}'), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        models = ["google/gemini-2.5-flash", "deepseek/deepseek-chat-v3-0324"]
        result = asyncio.run(
            check_json_mode(models=models, _openai_client=mock_client)
        )
        assert result.all_passed is True
        assert result.stage == "JSON Mode Support"
        assert len(result.items) == 2

    def test_check_json_mode_fail(self) -> None:
        from x_creative.config.checker import CheckStatus, check_json_mode

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "response_format not supported"
        )

        models = ["bad/model"]
        result = asyncio.run(
            check_json_mode(models=models, _openai_client=mock_client)
        )
        assert result.all_passed is False
        assert result.items[0].status == CheckStatus.FAIL
        assert "not supported" in result.items[0].message

    def test_check_json_mode_deduplicates(self) -> None:
        from x_creative.config.checker import check_json_mode

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"ok":true}'), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        models = ["google/gemini-2.5-flash", "google/gemini-2.5-flash"]
        result = asyncio.run(
            check_json_mode(models=models, _openai_client=mock_client)
        )
        assert len(result.items) == 1

    def test_check_json_mode_transient_error_is_warn(self) -> None:
        """Transient errors (429, 500, timeout) should be WARN, not FAIL."""
        from x_creative.config.checker import CheckStatus, check_json_mode

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Error code: 429 - {'error': {'message': 'Provider returned error', "
            "'code': 429, 'metadata': {'raw': 'rate-limited upstream'}}}"
        )

        models = ["deepseek/deepseek-v3.2-speciale"]
        result = asyncio.run(
            check_json_mode(models=models, _openai_client=mock_client)
        )
        # Should NOT fail — transient error
        assert result.all_passed is True
        assert result.items[0].status == CheckStatus.WARN
        assert "transient error" in result.items[0].message

    def test_check_json_mode_unsupported_error_is_fail(self) -> None:
        """Errors explicitly about response_format not supported should be FAIL."""
        from x_creative.config.checker import CheckStatus, check_json_mode

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "response_format is not supported by this model"
        )

        models = ["bad/model"]
        result = asyncio.run(
            check_json_mode(models=models, _openai_client=mock_client)
        )
        assert result.all_passed is False
        assert result.items[0].status == CheckStatus.FAIL

    def test_check_json_mode_sends_response_format(self) -> None:
        from x_creative.config.checker import check_json_mode

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"ok":true}'), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        asyncio.run(
            check_json_mode(models=["test/model"], _openai_client=mock_client)
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}


class TestJsonModeDefaults:
    """Verify that each task has the correct json_mode default value."""

    def test_json_mode_default_values(self) -> None:
        routing = TaskRoutingConfig()

        # Tasks that should have json_mode=True
        json_true_tasks = [
            "creativity",
            "analogical_mapping",
            "structured_search",
            "hypothesis_scoring",
            "knowledge_extraction",
            "logic_verification",
            "novelty_verification",
            "saga_adversarial",
            "saga_checkpoint",
            "saga_deep_audit",
            "hkg_expansion",
            "blend_expansion",
            "transform_space",
            "reasoner_step",
            "constraint_compliance_audit",
        ]
        for task in json_true_tasks:
            config = getattr(routing, task)
            assert config.json_mode is True, f"{task} should have json_mode=True"

        # Tasks that should have json_mode=False
        json_false_tasks = [
            "code_generation",
            "code_review",
            "talker_output",
            "constraint_compliance_revision",
        ]
        for task in json_false_tasks:
            config = getattr(routing, task)
            assert config.json_mode is False, f"{task} should have json_mode=False"
