"""Minimal ruamel.yaml compatibility layer backed by PyYAML."""

from __future__ import annotations

from typing import Any, TextIO

import yaml as _pyyaml


class YAML:
    """Subset of ruamel.yaml.YAML API used by this project."""

    def __init__(self) -> None:
        self.preserve_quotes = True
        self.default_flow_style = False
        self.width = 120

    def load(self, stream: str | TextIO) -> Any:
        """Load YAML from text or file-like object."""
        if hasattr(stream, "read"):
            content = stream.read()
        else:
            content = stream
        return _pyyaml.safe_load(content)

    def dump(self, data: Any, stream: TextIO | None = None) -> str | None:
        """Dump YAML to file-like object, or return YAML string."""
        dumped = _pyyaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=self.default_flow_style,
            width=self.width,
        )
        if stream is None:
            return dumped
        stream.write(dumped)
        return None
