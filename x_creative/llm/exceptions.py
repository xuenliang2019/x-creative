"""Exceptions for LLM module."""


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class AllModelsFailedError(LLMError):
    """Raised when all models (primary and fallbacks) fail."""

    def __init__(self, task: str, errors: list[tuple[str, Exception]]) -> None:
        self.task = task
        self.errors = errors
        models_tried = [f"{model}: {error}" for model, error in errors]
        super().__init__(
            f"All models failed for task '{task}'. "
            f"Tried: {'; '.join(models_tried)}"
        )


class ModelNotFoundError(LLMError):
    """Raised when a requested model is not found."""

    pass
