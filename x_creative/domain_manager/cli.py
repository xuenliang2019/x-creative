"""CLI entry point for xc-domain command."""

import sys

from x_creative.domain_manager.app import DomainManagerApp


def main() -> None:
    """Entry point for xc-domain command.

    Usage:
        xc-domain                         # defaults to open_source_development
        xc-domain open_source_development
    """
    target_domain_id = sys.argv[1] if len(sys.argv) > 1 else "open_source_development"
    app = DomainManagerApp(target_domain_id=target_domain_id)
    app.run()


if __name__ == "__main__":
    main()
