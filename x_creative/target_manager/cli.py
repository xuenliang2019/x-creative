"""CLI entry point for xc-target command."""

from x_creative.target_manager.app import TargetManagerApp


def main() -> None:
    """Entry point for xc-target command.

    Usage:
        xc-target           # launch target domain manager TUI
    """
    app = TargetManagerApp()
    app.run()
