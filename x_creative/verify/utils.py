"""Shared utilities for verification modules."""

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float value or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def strip_markdown_code_block(content: str) -> str:
    """Strip markdown code block markers from content.

    Args:
        content: Raw content that may contain markdown code blocks.

    Returns:
        Content with code block markers removed.
    """
    clean_content = content.strip()
    if clean_content.startswith("```"):
        lines = clean_content.split("\n")
        # Remove first line if it's a code block marker
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's a code block marker
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean_content = "\n".join(lines)
    return clean_content


# Default similarity score for LLM-identified similar works
# (web search will refine this value)
DEFAULT_LLM_SIMILARITY = 0.5
