"""Lightweight prompt injection scanner for VERIFY stage.

Scans hypothesis content for patterns that could manipulate LLM judges.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


# Patterns that indicate potential injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"override\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"(you\s+must|always)\s+score\s+(this\s+)?\d+", re.IGNORECASE),
    re.compile(r"as\s+a\s+judge.*score\s+(this|it)\s+\d+", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(your|the)\s+(previous|above)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)\s+(above|previous)", re.IGNORECASE),
    re.compile(r"(score|rate|give)\s+(me\s+)?a?\s*(perfect|maximum|10\s*/\s*10|10\.0)", re.IGNORECASE),
]


class InjectionScanResult(BaseModel):
    """Result of injection scan."""

    flagged: bool = Field(..., description="Whether injection was detected")
    matched_patterns: list[str] = Field(
        default_factory=list, description="Matched pattern descriptions"
    )


def scan_for_injection(text: str) -> InjectionScanResult:
    """Scan text for potential prompt injection patterns.

    Args:
        text: Content to scan.

    Returns:
        InjectionScanResult with flagged status and matched patterns.
    """
    matched = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(pattern.pattern)

    return InjectionScanResult(
        flagged=len(matched) > 0,
        matched_patterns=matched,
    )
