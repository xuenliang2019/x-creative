"""Shared utilities for creativity modules."""

import json
import re
from typing import Any

import structlog

logger = structlog.get_logger()


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


def recover_json_array(text: str) -> list[dict[str, Any]]:
    """Extract complete JSON objects from a possibly truncated JSON array.

    When LLMs hit token limits, JSON arrays get cut mid-object. This function
    uses a stateful scanner (not regex) to correctly handle nested braces,
    escaped quotes, and strings containing ``{``/``}``.

    Args:
        text: Raw string that may contain a truncated JSON array.

    Returns:
        List of successfully parsed dicts. Returns ``[]`` when no complete
        objects can be recovered.
    """
    # Find the opening bracket of the array
    arr_start = text.find("[")
    if arr_start < 0:
        return []

    recovered: list[dict[str, Any]] = []
    pos = arr_start + 1
    length = len(text)

    while pos < length:
        # Skip whitespace and commas between objects
        while pos < length and text[pos] in " \t\r\n,":
            pos += 1

        if pos >= length or text[pos] == "]":
            break

        if text[pos] != "{":
            # Unexpected character — skip it
            pos += 1
            continue

        # --- Stateful scan for a complete { ... } block ---
        obj_start = pos
        depth = 0
        in_string = False
        i = pos

        while i < length:
            ch = text[i]

            if in_string:
                if ch == "\\":
                    i += 2  # skip escaped character
                    continue
                if ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        # Complete object found
                        obj_str = text[obj_start : i + 1]
                        try:
                            obj = safe_json_loads(obj_str)
                            if isinstance(obj, dict):
                                recovered.append(obj)
                        except (json.JSONDecodeError, ValueError):
                            pass
                        pos = i + 1
                        break

            i += 1
        else:
            # Ran out of text before closing brace — truncated object; stop
            break

    if recovered:
        logger.debug(
            "recover_json_array recovered objects from truncated JSON",
            recovered_count=len(recovered),
        )

    return recovered
