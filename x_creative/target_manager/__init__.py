"""Target Manager package.

Keep TUI dependency (`textual`) optional at import time so service-only
unit tests can run without installing TUI extras.
"""

from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from x_creative.target_manager.app import TargetManagerApp

__all__ = ["TargetManagerApp"]


def __getattr__(name: str):
    """Lazy-load TUI app to avoid importing textual during service tests."""
    if name == "app":
        return importlib.import_module("x_creative.target_manager.app")
    if name == "TargetManagerApp":
        module = importlib.import_module("x_creative.target_manager.app")

        return module.TargetManagerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
