"""Domain Manager package with lazy TUI imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from x_creative.domain_manager.app import DomainManagerApp

__all__ = ["DomainManagerApp"]


def __getattr__(name: str):
    """Lazy-load submodule and app class for test-friendly imports."""
    if name == "app":
        return importlib.import_module("x_creative.domain_manager.app")
    if name == "DomainManagerApp":
        module = importlib.import_module("x_creative.domain_manager.app")
        return module.DomainManagerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
