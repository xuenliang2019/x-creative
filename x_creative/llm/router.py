"""Model routing for task-specific model selection with fallback support."""

import re
from typing import Any

import structlog

from x_creative.config.settings import ModelConfig, get_settings
from x_creative.llm.client import CompletionResult, LLMClient
from x_creative.llm.exceptions import AllModelsFailedError

logger = structlog.get_logger()

# OpenRouter-style context-length error (sometimes bubbled up by compatible providers).
_CTX_MAX_RE = re.compile(r"maximum context length is (\d+) tokens", re.IGNORECASE)
_CTX_REQ_RE = re.compile(
    r"requested about (\d+) tokens\s*\((\d+) of text input,\s*(\d+) in the output\)",
    re.IGNORECASE,
)


def _extract_context_window(error: Exception) -> tuple[int | None, int | None, int | None]:
    """Best-effort parse of context length error.

    Returns:
        (max_context_tokens, input_tokens, requested_output_tokens)
    """
    text = str(error)
    max_ctx = None
    input_tokens = None
    output_tokens = None

    m_max = _CTX_MAX_RE.search(text)
    if m_max:
        try:
            max_ctx = int(m_max.group(1))
        except Exception:
            max_ctx = None

    m_req = _CTX_REQ_RE.search(text)
    if m_req:
        try:
            input_tokens = int(m_req.group(2))
        except Exception:
            input_tokens = None
        try:
            output_tokens = int(m_req.group(3))
        except Exception:
            output_tokens = None

    return max_ctx, input_tokens, output_tokens


def _is_context_length_error(error: Exception) -> bool:
    """Heuristic: detect 'context length exceeded' style provider errors."""
    text = str(error).lower()
    return "maximum context length" in text or "context length" in text


class ModelRouter:
    """Routes tasks to appropriate models with fallback support."""

    def __init__(
        self,
        client: LLMClient | None = None,
    ) -> None:
        """Initialize the model router.

        Args:
            client: LLM client to use. If None, creates one from settings.
        """
        self._client = client or LLMClient()
        self._settings = get_settings()
        self._model_max_tokens: dict[str, int] = dict(self._settings.model_max_tokens)

    def get_model(self, task: str) -> str:
        """Get the primary model for a task.

        Args:
            task: Task identifier (e.g., "creativity", "hypothesis_scoring").

        Returns:
            Model identifier string.
        """
        config = self.get_config(task)
        return config.model

    def get_config(self, task: str) -> ModelConfig:
        """Get the full configuration for a task.

        Args:
            task: Task identifier.

        Returns:
            ModelConfig with model, fallbacks, temperature, etc.
        """
        return self._settings.get_model_config(task)

    def get_fallbacks(self, task: str) -> list[str]:
        """Get fallback models for a task.

        Args:
            task: Task identifier.

        Returns:
            List of fallback model identifiers.
        """
        config = self.get_config(task)
        return config.fallback

    async def complete(
        self,
        task: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Make a completion with automatic fallback on failure.

        Args:
            task: Task identifier for model selection.
            messages: List of message dicts.
            temperature: Override temperature (uses task default if None).
            max_tokens: Override max tokens (uses task default if None).
            model_override: If set, use this model instead of the task's
                primary model. Fallback chain from task config is preserved.
            **kwargs: Additional arguments.

        Returns:
            CompletionResult from the first successful model.

        Raises:
            AllModelsFailedError: If all models (primary + fallbacks) fail.
        """
        config = self.get_config(task)

        # model_override replaces the primary model but keeps fallback chain
        if model_override:
            models_to_try = [model_override] + config.fallback
        else:
            models_to_try = [config.model] + config.fallback

        # Use provided values or fall back to config
        actual_temp = temperature if temperature is not None else config.temperature
        actual_max_tokens = max_tokens if max_tokens is not None else config.max_tokens

        errors: list[tuple[str, Exception]] = []

        for model in models_to_try:
            # Per-model output token cap
            model_cap = self._model_max_tokens.get(model)
            effective_max = (
                min(actual_max_tokens, model_cap)
                if model_cap and actual_max_tokens
                else actual_max_tokens
            )

            try:
                logger.debug(
                    "Attempting completion",
                    task=task,
                    model=model,
                    temperature=actual_temp,
                )

                result = await self._client.complete(
                    model=model,
                    messages=messages,
                    temperature=actual_temp,
                    max_tokens=effective_max,
                    **kwargs,
                )

                logger.info(
                    "Completion successful",
                    task=task,
                    model=model,
                    tokens=result.total_tokens,
                )

                return result

            except Exception as e:
                # Some providers reject requests when prompt+max_tokens exceed the
                # model context window. When the error provides the context length,
                # retry once with a reduced max_tokens before falling back.
                if (
                    effective_max is not None
                    and effective_max > 0
                    and _is_context_length_error(e)
                ):
                    max_ctx, input_tokens, _ = _extract_context_window(e)
                    safety_margin = 256
                    reduced_max = None
                    if max_ctx is not None:
                        budget = max_ctx - safety_margin
                        if input_tokens is not None:
                            budget = max(0, budget - input_tokens)
                        reduced_max = max(16, budget)

                    if reduced_max is not None and reduced_max < effective_max:
                        try:
                            logger.warning(
                                "Context length exceeded; retrying with reduced max_tokens",
                                task=task,
                                model=model,
                                max_tokens=effective_max,
                                retry_max_tokens=reduced_max,
                            )
                            result = await self._client.complete(
                                model=model,
                                messages=messages,
                                temperature=actual_temp,
                                max_tokens=reduced_max,
                                **kwargs,
                            )
                            logger.info(
                                "Completion successful",
                                task=task,
                                model=model,
                                tokens=result.total_tokens,
                            )
                            return result
                        except Exception:
                            # Fall through to normal fallback chain.
                            pass

                logger.warning(
                    "Model failed, trying fallback",
                    task=task,
                    model=model,
                    error=str(e),
                )
                errors.append((model, e))
                continue

        # All models failed
        raise AllModelsFailedError(task=task, errors=errors)

    async def complete_with_retry(
        self,
        task: str,
        messages: list[dict[str, Any]],
        max_retries: int = 3,
        **kwargs: Any,
    ) -> CompletionResult:
        """Make a completion with retry logic.

        Args:
            task: Task identifier.
            messages: List of message dicts.
            max_retries: Maximum retries per model.
            **kwargs: Additional arguments.

        Returns:
            CompletionResult from successful completion.

        Raises:
            AllModelsFailedError: If all attempts fail.
        """
        from tenacity import retry, stop_after_attempt, wait_exponential

        config = self.get_config(task)
        models_to_try = [config.model] + config.fallback
        errors: list[tuple[str, Exception]] = []

        for model in models_to_try:

            @retry(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
            )
            async def _complete_with_model() -> CompletionResult:
                return await self._client.complete(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    **kwargs,
                )

            try:
                return await _complete_with_model()
            except Exception as e:
                errors.append((model, e))
                continue

        raise AllModelsFailedError(task=task, errors=errors)

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()
