"""Configuration checker for validating .env and API connectivity."""

from __future__ import annotations

import enum
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field


class CheckStatus(enum.Enum):
    """Status of a single check item."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class CheckItem(BaseModel):
    """A single check result line."""

    label: str
    status: CheckStatus
    message: str | None = None
    elapsed_ms: float | None = None


class CheckResult(BaseModel):
    """Result of a check stage."""

    stage: str
    items: list[CheckItem] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True if no items have FAIL status (WARN is acceptable)."""
        return all(item.status != CheckStatus.FAIL for item in self.items)


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a dict of key=value pairs."""
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def check_static(env_path: Path | None = None) -> CheckResult:
    """Stage 1: Static validation of .env and Settings."""
    from x_creative.config.settings import Settings

    if env_path is None:
        env_path = Path(".env")

    items: list[CheckItem] = []

    # 1. .env file exists
    if not env_path.exists():
        items.append(
            CheckItem(
                label=".env file",
                status=CheckStatus.FAIL,
                message=f".env not found at {env_path.resolve()}",
            )
        )
        return CheckResult(stage="Static Check", items=items)

    items.append(CheckItem(label=".env file", status=CheckStatus.PASS, message="found"))

    # 2. Parse env file
    env_vars = _parse_env_file(env_path)

    # 3. Provider
    provider = env_vars.get("X_CREATIVE_DEFAULT_PROVIDER", "openrouter")
    items.append(
        CheckItem(
            label="DEFAULT_PROVIDER",
            status=CheckStatus.PASS,
            message=f"= {provider}",
        )
    )

    # 4. Required API key for provider
    if provider == "openrouter":
        key = env_vars.get("OPENROUTER_API_KEY", "")
        if key and key != "sk-or-v1-your-key-here":
            items.append(
                CheckItem(
                    label="OPENROUTER_API_KEY",
                    status=CheckStatus.PASS,
                    message="present",
                )
            )
        else:
            items.append(
                CheckItem(
                    label="OPENROUTER_API_KEY",
                    status=CheckStatus.FAIL,
                    message="missing or placeholder",
                )
            )
    elif provider == "yunwu":
        key = env_vars.get("YUNWU_API_KEY", "")
        if key and key != "your-yunwu-api-key-here":
            items.append(
                CheckItem(
                    label="YUNWU_API_KEY",
                    status=CheckStatus.PASS,
                    message="present",
                )
            )
        else:
            items.append(
                CheckItem(
                    label="YUNWU_API_KEY",
                    status=CheckStatus.FAIL,
                    message="missing or placeholder",
                )
            )

    # 5. Optional: Brave Search
    brave_key = env_vars.get("BRAVE_SEARCH_API_KEY", "")
    if brave_key and brave_key != "your-brave-api-key-here":
        items.append(
            CheckItem(
                label="BRAVE_SEARCH_API_KEY",
                status=CheckStatus.PASS,
                message="present (optional)",
            )
        )
    else:
        items.append(
            CheckItem(
                label="BRAVE_SEARCH_API_KEY",
                status=CheckStatus.WARN,
                message="not set (novelty verification will skip web search)",
            )
        )

    # 6. Try loading Settings
    settings = None
    try:
        settings = Settings(_env_file=str(env_path))
        items.append(
            CheckItem(label="Settings load", status=CheckStatus.PASS, message="OK")
        )
    except Exception as e:
        items.append(
            CheckItem(label="Settings load", status=CheckStatus.FAIL, message=str(e))
        )

    # 7. Model name format (provider/model-name)
    if settings is not None:
        model_format_ok = True
        routing = settings.task_routing
        for task_name in type(routing).model_fields:
            config = getattr(routing, task_name)
            if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$", config.model):
                items.append(
                    CheckItem(
                        label=f"Model format ({task_name})",
                        status=CheckStatus.FAIL,
                        message=f"'{config.model}' doesn't match provider/model-name",
                    )
                )
                model_format_ok = False
        if model_format_ok:
            items.append(
                CheckItem(
                    label="Model name format",
                    status=CheckStatus.PASS,
                    message="all valid",
                )
            )

    # 8. Score weights sum
    if settings is not None:
        total = (
            settings.score_weight_divergence
            + settings.score_weight_testability
            + settings.score_weight_rationale
            + settings.score_weight_robustness
            + settings.score_weight_feasibility
        )
    else:
        # Even if Settings fails to load (e.g., because the weights don't sum to 1.0),
        # report the weight-sum issue to the user instead of short-circuiting.
        weight_keys = {
            "X_CREATIVE_SCORE_WEIGHT_DIVERGENCE": Settings.model_fields[
                "score_weight_divergence"
            ].default,
            "X_CREATIVE_SCORE_WEIGHT_TESTABILITY": Settings.model_fields[
                "score_weight_testability"
            ].default,
            "X_CREATIVE_SCORE_WEIGHT_RATIONALE": Settings.model_fields[
                "score_weight_rationale"
            ].default,
            "X_CREATIVE_SCORE_WEIGHT_ROBUSTNESS": Settings.model_fields[
                "score_weight_robustness"
            ].default,
            "X_CREATIVE_SCORE_WEIGHT_FEASIBILITY": Settings.model_fields[
                "score_weight_feasibility"
            ].default,
        }

        weight_values: list[float] = []
        parse_errors: list[str] = []
        for key, default in weight_keys.items():
            raw = os.getenv(key) or env_vars.get(key)
            if raw is None or raw == "":
                weight_values.append(float(default))
                continue
            try:
                weight_values.append(float(raw))
            except ValueError:
                parse_errors.append(f"{key}={raw!r}")

        if parse_errors:
            items.append(
                CheckItem(
                    label="Score weights parse",
                    status=CheckStatus.FAIL,
                    message="Invalid float values: " + ", ".join(parse_errors),
                )
            )
            total = float("nan")
        else:
            total = sum(weight_values)

    if abs(total - 1.0) < 0.01:
        items.append(
            CheckItem(
                label="Score weights sum",
                status=CheckStatus.PASS,
                message=f"= {total:.2f}",
            )
        )
    else:
        items.append(
            CheckItem(
                label="Score weights sum",
                status=CheckStatus.FAIL,
                message=f"= {total:.2f} (should be ~1.0)",
            )
        )

    return CheckResult(stage="Static Check", items=items)


async def check_connectivity(
    provider: str,
    api_key: str,
    base_url: str,
    brave_api_key: str | None = None,
    *,
    _openai_client: object | None = None,
    _brave_response: object | None = None,
) -> CheckResult:
    """Stage 2: Verify API key connectivity.

    Sends a minimal request to the LLM provider and optionally to Brave Search.
    _openai_client and _brave_response are for testing injection.
    """
    import httpx
    from openai import AsyncOpenAI

    items: list[CheckItem] = []

    # 1. LLM provider connectivity
    client = _openai_client or AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        t0 = time.monotonic()
        await client.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
        )
        elapsed = (time.monotonic() - t0) * 1000
        items.append(
            CheckItem(
                label=f"{provider.capitalize()} API key",
                status=CheckStatus.PASS,
                message=f"valid ({elapsed:.0f}ms)",
                elapsed_ms=elapsed,
            )
        )
    except Exception as e:
        items.append(
            CheckItem(
                label=f"{provider.capitalize()} API key",
                status=CheckStatus.FAIL,
                message=str(e),
            )
        )
    finally:
        if _openai_client is None:
            await client.close()

    # 2. Brave Search connectivity (optional)
    if brave_api_key:
        try:
            t0 = time.monotonic()
            if _brave_response is not None:
                resp = _brave_response
            else:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": "test", "count": 1},
                        headers={
                            "X-Subscription-Token": brave_api_key,
                            "Accept": "application/json",
                        },
                        timeout=10,
                    )
            elapsed = (time.monotonic() - t0) * 1000
            status_code = getattr(resp, "status_code", 200)
            if status_code == 200:
                items.append(
                    CheckItem(
                        label="Brave Search API key",
                        status=CheckStatus.PASS,
                        message=f"valid ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                )
            else:
                items.append(
                    CheckItem(
                        label="Brave Search API key",
                        status=CheckStatus.FAIL,
                        message=f"HTTP {status_code}",
                    )
                )
        except Exception as e:
            items.append(
                CheckItem(
                    label="Brave Search API key",
                    status=CheckStatus.FAIL,
                    message=str(e),
                )
            )

    return CheckResult(stage="Connectivity Check", items=items)


import asyncio as _asyncio


async def _check_single_model(
    model: str,
    client: object,
) -> CheckItem:
    """Check a single model by sending a minimal completion."""
    try:
        t0 = time.monotonic()
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
        )
        elapsed = (time.monotonic() - t0) * 1000
        return CheckItem(
            label=model,
            status=CheckStatus.PASS,
            message=f"OK ({elapsed:.0f}ms)",
            elapsed_ms=elapsed,
        )
    except Exception as e:
        return CheckItem(
            label=model,
            status=CheckStatus.FAIL,
            message=str(e),
        )


async def check_models(
    models: list[str],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    _openai_client: object | None = None,
) -> CheckResult:
    """Stage 3: Verify each unique model is available.

    Sends concurrent minimal completion requests.
    _openai_client is for testing injection.
    """
    from x_creative.config.settings import get_settings
    from openai import AsyncOpenAI

    unique_models = list(dict.fromkeys(models))  # preserve order, deduplicate

    if _openai_client is not None:
        client = _openai_client
        own_client = False
    else:
        settings = get_settings()
        provider = settings.default_provider
        if provider == "openrouter":
            _api_key = api_key or settings.openrouter.api_key.get_secret_value()
            _base_url = base_url or settings.openrouter.base_url
        else:
            _api_key = api_key or settings.yunwu.api_key.get_secret_value()
            _base_url = base_url or settings.yunwu.base_url
        client = AsyncOpenAI(api_key=_api_key, base_url=_base_url)
        own_client = True

    try:
        tasks = [_check_single_model(m, client) for m in unique_models]
        items = await _asyncio.gather(*tasks)
    finally:
        if own_client:
            await client.close()

    return CheckResult(stage="Model Availability", items=list(items))
