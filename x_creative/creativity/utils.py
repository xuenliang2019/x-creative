"""Shared utilities for creativity modules."""

import json
import re
from typing import Any

import structlog

logger = structlog.get_logger()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from *text*.

    Unlike a greedy regex ``\\{[\\s\\S]*\\}``, this uses depth / string
    tracking so that braces inside strings or in prose *after* the JSON
    block are ignored.  Falls back to :func:`safe_json_loads` for escape
    repair.

    Returns ``{}`` when no valid object can be found.
    """
    start = text.find("{")
    if start < 0:
        return {}

    depth = 0
    in_string = False
    length = len(text)
    i = start

    while i < length:
        ch = text[i]

        if in_string:
            if ch == "\\":
                i += 2
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
                    snippet = text[start : i + 1]
                    try:
                        obj = safe_json_loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    return {}

        i += 1

    return {}


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


def recover_truncated_json_object(text: str) -> dict[str, Any]:
    """Recover a truncated top-level JSON object.

    When LLMs hit ``max_tokens``, the output may be a valid JSON object prefix
    that is cut off mid-string or mid-array.  This function scans from the
    outermost ``{`` with depth / string tracking to locate the last position
    where a complete key-value pair ended, then closes any open arrays ``]``
    and the root object ``}``.

    Args:
        text: Raw string that may contain a truncated JSON object.

    Returns:
        Parsed ``dict`` on success, or ``{}`` when nothing can be recovered.
    """
    obj_start = text.find("{")
    if obj_start < 0:
        return {}

    length = len(text)
    i = obj_start
    depth = 0
    in_string = False
    open_arrays = 0  # track unclosed '[' outside strings
    last_good = -1  # position after last complete top-level value
    last_good_open_arrays = 0  # open_arrays snapshot at last_good

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
                    # The object is actually complete — just parse it
                    snippet = text[obj_start : i + 1]
                    try:
                        obj = safe_json_loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    return {}
                if depth == 1:
                    # Completed a nested object at top level
                    last_good = i + 1
                    last_good_open_arrays = open_arrays
            elif ch == "[":
                if depth == 1:
                    open_arrays += 1
            elif ch == "]":
                if depth == 1:
                    open_arrays = max(0, open_arrays - 1)
                    last_good = i + 1
                    last_good_open_arrays = open_arrays
            elif ch == "," and depth == 1 and not in_string:
                # Comma between top-level entries — the value before it was
                # complete (unless we're inside an array/nested obj, but
                # depth==1 means we're at the root object level).
                last_good = i
                last_good_open_arrays = open_arrays

        i += 1

    # We ran out of text — the object is truncated.
    # Greedy close: if we ended at depth 1 outside a string, the text may
    # finish right after a complete scalar value (string, number, bool, null).
    # Try closing open brackets and parsing before falling back to last_good.
    if not in_string and depth == 1:
        snippet = text[obj_start:i].rstrip().rstrip(",")
        suffix = "]" * open_arrays + "}"
        candidate = snippet + suffix
        try:
            obj = safe_json_loads(candidate)
            if isinstance(obj, dict) and obj:
                logger.warning(
                    "recover_truncated_json_object: recovered truncated JSON object",
                    recovered_keys=list(obj.keys()),
                )
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    if last_good <= obj_start:
        return {}

    # Take everything up to last_good, strip trailing comma, close brackets.
    # Use the open_arrays snapshot from when last_good was set, not the
    # current count — arrays opened after last_good aren't in the slice.
    truncated = text[obj_start : last_good].rstrip().rstrip(",")
    suffix = "]" * last_good_open_arrays + "}"
    candidate = truncated + suffix

    try:
        obj = safe_json_loads(candidate)
        if isinstance(obj, dict):
            logger.warning(
                "recover_truncated_json_object: recovered truncated JSON object",
                recovered_keys=list(obj.keys()),
            )
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    return {}


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
