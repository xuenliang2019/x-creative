"""LLM client for interacting with OpenRouter and other providers."""

from typing import Any

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from x_creative.config.settings import get_settings

logger = structlog.get_logger()


class CompletionResult(BaseModel):
    """Result from an LLM completion call."""

    content: str = Field(..., description="The generated content")
    model: str = Field(..., description="Model that generated the response")
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens")
    completion_tokens: int = Field(default=0, description="Number of completion tokens")
    finish_reason: str | None = Field(default=None, description="Why generation stopped")

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Client for making LLM API calls via configured providers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: API key for the provider. If None, uses settings.
            base_url: Base URL for the API. If None, uses provider default.
            provider: Provider identifier. If None, uses settings.default_provider.
        """
        settings = get_settings()
        selected_provider = (provider or settings.default_provider).lower()

        if selected_provider == "openrouter":
            default_api_key = settings.openrouter.api_key.get_secret_value()
            default_base_url = settings.openrouter.base_url
        elif selected_provider == "yunwu":
            default_api_key = settings.yunwu.api_key.get_secret_value()
            default_base_url = settings.yunwu.base_url
        else:
            raise ValueError(f"Unsupported provider: {selected_provider}")

        self._provider = selected_provider
        self._api_key = api_key if api_key is not None else default_api_key
        self._base_url = base_url if base_url is not None else default_base_url

        self._client: AsyncOpenAI | None = None

    @property
    def provider(self) -> str:
        """Get the selected provider."""
        return self._provider

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._base_url

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def _build_model_candidates(self, model: str) -> list[str]:
        """Build provider-specific model candidates for compatibility fallback."""
        if self._provider != "yunwu" or "/" not in model:
            return [model]

        alias = model.split("/", 1)[1]
        if alias == model:
            return [model]
        return [model, alias]

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Make a completion request.

        Args:
            model: Model identifier (e.g., "google/gemini-2.5-flash").
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments passed to the API.

        Returns:
            CompletionResult with the generated content.
        """
        client = self._get_client()

        model_candidates = self._build_model_candidates(model)
        last_error: Exception | None = None

        for idx, candidate in enumerate(model_candidates):
            # Build request parameters
            params: dict[str, Any] = {
                "model": candidate,
                "messages": messages,
            }

            if temperature is not None:
                params["temperature"] = temperature

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            params.update(kwargs)

            logger.debug("Making completion request", model=candidate, temperature=temperature)

            try:
                response = await client.chat.completions.create(**params)
            except Exception as e:
                last_error = e
                if idx < len(model_candidates) - 1:
                    logger.warning(
                        "Primary model format failed, trying compatibility alias",
                        provider=self._provider,
                        original_model=model,
                        alias_model=model_candidates[idx + 1],
                        error=str(e),
                    )
                    continue
                raise

            # Extract result
            if not response.choices:
                raise ValueError(
                    f"Model {candidate} returned empty choices "
                    f"(finish_reason may indicate content filtering)"
                )
            choice = response.choices[0]
            usage = response.usage

            return CompletionResult(
                content=choice.message.content or "",
                model=candidate,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason,
            )

        if last_error:
            raise last_error
        raise RuntimeError("No model candidates available")

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
