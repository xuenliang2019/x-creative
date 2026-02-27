"""Shared utilities for creativity modules."""

import json
import re
from typing import Any


def safe_json_loads(json_str: str) -> Any:
    """Parse JSON string with fallback for common escape issues.

    LLMs sometimes generate invalid escape sequences in JSON.
    This function tries normal parsing first, then attempts to fix
    common issues if that fails.

    Args:
        json_str: JSON string to parse.

    Returns:
        Parsed JSON data.

    Raises:
        json.JSONDecodeError: If parsing fails even after fixes.
    """
    # First, try normal parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fix invalid escape sequences by escaping lone backslashes
    # This handles cases like \n in formulas that should be \\n
    # Match backslash not followed by valid JSON escape chars: " \ / b f n r t u
    fixed_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass

    # Try removing control characters that might have slipped in
    fixed_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed_str)

    return json.loads(fixed_str)
