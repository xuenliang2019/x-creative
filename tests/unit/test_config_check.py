"""Tests for config checker."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from x_creative.config.checker import CheckItem, CheckResult, CheckStatus


class TestCheckTypes:
    def test_check_item_pass(self) -> None:
        item = CheckItem(label="test", status=CheckStatus.PASS)
        assert item.status == CheckStatus.PASS
        assert item.message is None

    def test_check_item_fail_with_message(self) -> None:
        item = CheckItem(label="test", status=CheckStatus.FAIL, message="broken")
        assert item.message == "broken"

    def test_check_item_warn(self) -> None:
        item = CheckItem(label="test", status=CheckStatus.WARN, message="optional")
        assert item.status == CheckStatus.WARN

    def test_check_result_all_passed(self) -> None:
        result = CheckResult(
            stage="test",
            items=[
                CheckItem(label="a", status=CheckStatus.PASS),
                CheckItem(label="b", status=CheckStatus.PASS),
            ],
        )
        assert result.all_passed is True

    def test_check_result_has_failure(self) -> None:
        result = CheckResult(
            stage="test",
            items=[
                CheckItem(label="a", status=CheckStatus.PASS),
                CheckItem(label="b", status=CheckStatus.FAIL, message="err"),
            ],
        )
        assert result.all_passed is False

    def test_check_result_warn_counts_as_passed(self) -> None:
        result = CheckResult(
            stage="test",
            items=[
                CheckItem(label="a", status=CheckStatus.WARN, message="ok"),
            ],
        )
        assert result.all_passed is True


class TestStaticCheck:
    def test_env_file_missing(self, tmp_path: Path) -> None:
        from x_creative.config.checker import check_static

        result = check_static(env_path=tmp_path / ".env")
        assert result.all_passed is False
        assert any("not found" in (i.message or "") for i in result.items)

    def test_env_file_present_with_required_keys(self, tmp_path: Path) -> None:
        from x_creative.config.checker import check_static

        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-test\n"
            "X_CREATIVE_DEFAULT_PROVIDER=openrouter\n"
        )
        result = check_static(env_path=env_file)
        assert result.all_passed is True

    def test_missing_api_key_for_provider(self, tmp_path: Path) -> None:
        from x_creative.config.checker import check_static

        env_file = tmp_path / ".env"
        env_file.write_text("X_CREATIVE_DEFAULT_PROVIDER=openrouter\n")
        result = check_static(env_path=env_file)
        assert result.all_passed is False

    def test_brave_key_optional_warn(self, tmp_path: Path) -> None:
        from x_creative.config.checker import check_static

        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-test\n"
            "X_CREATIVE_DEFAULT_PROVIDER=openrouter\n"
        )
        result = check_static(env_path=env_file)
        warn_items = [i for i in result.items if i.status == CheckStatus.WARN]
        assert len(warn_items) >= 1  # BRAVE_SEARCH_API_KEY not set warning

    def test_score_weights_invalid_sum(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from x_creative.config.checker import check_static

        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-test\n"
            "X_CREATIVE_DEFAULT_PROVIDER=openrouter\n"
            "X_CREATIVE_SCORE_WEIGHT_DIVERGENCE=0.5\n"
            "X_CREATIVE_SCORE_WEIGHT_TESTABILITY=0.5\n"
            "X_CREATIVE_SCORE_WEIGHT_RATIONALE=0.5\n"
            "X_CREATIVE_SCORE_WEIGHT_ROBUSTNESS=0.5\n"
            "X_CREATIVE_SCORE_WEIGHT_FEASIBILITY=0.5\n"
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("X_CREATIVE_DEFAULT_PROVIDER", "openrouter")
        monkeypatch.setenv("X_CREATIVE_SCORE_WEIGHT_DIVERGENCE", "0.5")
        monkeypatch.setenv("X_CREATIVE_SCORE_WEIGHT_TESTABILITY", "0.5")
        monkeypatch.setenv("X_CREATIVE_SCORE_WEIGHT_RATIONALE", "0.5")
        monkeypatch.setenv("X_CREATIVE_SCORE_WEIGHT_ROBUSTNESS", "0.5")
        monkeypatch.setenv("X_CREATIVE_SCORE_WEIGHT_FEASIBILITY", "0.5")
        result = check_static(env_path=env_file)
        weight_items = [i for i in result.items if "weight" in i.label.lower()]
        assert any(i.status == CheckStatus.FAIL for i in weight_items)


class TestConnectivityCheck:
    def test_openrouter_success(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from x_creative.config.checker import check_connectivity

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        result = asyncio.run(check_connectivity(
            provider="openrouter",
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            brave_api_key=None,
            _openai_client=mock_client,
        ))
        assert result.all_passed is True

    def test_openrouter_auth_failure(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock

        from x_creative.config.checker import check_connectivity

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        result = asyncio.run(check_connectivity(
            provider="openrouter",
            api_key="bad-key",
            base_url="https://openrouter.ai/api/v1",
            brave_api_key=None,
            _openai_client=mock_client,
        ))
        assert result.all_passed is False

    def test_brave_search_success(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from x_creative.config.checker import check_connectivity

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200

        result = asyncio.run(check_connectivity(
            provider="openrouter",
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            brave_api_key="brave-key",
            _openai_client=mock_client,
            _brave_response=mock_httpx_response,
        ))
        assert result.all_passed is True
        assert len(result.items) == 2  # LLM + Brave


class TestModelCheck:
    def test_all_models_available(self) -> None:
        from x_creative.config.checker import check_models

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        models = ["openai/gpt-5.2", "google/gemini-3-flash-preview"]
        result = asyncio.run(check_models(models=models, _openai_client=mock_client))
        assert result.all_passed is True
        assert len(result.items) == 2

    def test_one_model_fails(self) -> None:
        from x_creative.config.checker import check_models

        mock_client = AsyncMock()

        async def side_effect(**kwargs):
            if kwargs.get("model") == "bad/model":
                raise Exception("404 model not found")
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="hi"), finish_reason="stop")]
            resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
            return resp

        mock_client.chat.completions.create.side_effect = side_effect

        models = ["openai/gpt-5.2", "bad/model"]
        result = asyncio.run(check_models(models=models, _openai_client=mock_client))
        assert result.all_passed is False
        fail_items = [i for i in result.items if i.status == CheckStatus.FAIL]
        assert len(fail_items) == 1
        assert "bad/model" in fail_items[0].label

    def test_deduplicates_models(self) -> None:
        from x_creative.config.checker import check_models

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        mock_client.chat.completions.create.return_value = mock_response

        # Same model twice should only test once
        models = ["openai/gpt-5.2", "openai/gpt-5.2"]
        result = asyncio.run(check_models(models=models, _openai_client=mock_client))
        assert len(result.items) == 1


# ---- CLI tests ----

from typer.testing import CliRunner
from x_creative.cli.main import app

runner = CliRunner()


class TestConfigCheckCLI:
    def test_config_check_quick_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--quick only does static check, no API calls."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-test-key\n"
            "X_CREATIVE_DEFAULT_PROVIDER=openrouter\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "check", "--quick"])
        assert result.exit_code == 0
        assert "Static Check" in result.output

    def test_config_check_quick_fails_no_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "check", "--quick"])
        assert result.exit_code == 1
        assert "not found" in result.output
