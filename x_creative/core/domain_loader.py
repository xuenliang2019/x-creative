"""Domain library loading and management."""

from pathlib import Path
from typing import Iterator

import structlog
import yaml

from x_creative.core.types import Domain, DomainStructure, TargetMapping

logger = structlog.get_logger()


def load_domains(path: Path) -> list[Domain]:
    """Load domains from a YAML file.

    Args:
        path: Path to the YAML file containing domain definitions.

    Returns:
        List of Domain objects.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    domains: list[Domain] = []
    for domain_data in data.get("domains", []):
        # Parse structures
        structures = [
            DomainStructure(**s) for s in domain_data.get("structures", [])
        ]

        # Parse mappings
        raw_mappings = domain_data.get("target_mappings", [])
        mappings = [TargetMapping(**m) for m in raw_mappings]

        # Create domain
        domain = Domain(
            id=domain_data["id"],
            name=domain_data["name"],
            name_en=domain_data.get("name_en"),
            description=domain_data["description"],
            structures=structures,
            target_mappings=mappings,
        )
        domains.append(domain)

    return domains


class DomainLibrary:
    """Container for domain structures with lookup functionality."""

    def __init__(self, domains: list[Domain]) -> None:
        """Initialize with a list of domains.

        Args:
            domains: List of Domain objects.
        """
        self._domains = {d.id: d for d in domains}

    @classmethod
    def from_target_domain(cls, domain_id: str) -> "DomainLibrary":
        """Create a DomainLibrary from a target domain's embedded source domains.

        Args:
            domain_id: The target domain ID (e.g., 'open_source_development').

        Returns:
            DomainLibrary containing source domains for this target domain.

        Raises:
            ValueError: If the target domain has no embedded source domains.
        """
        from x_creative.core.plugin import load_target_domain

        plugin = load_target_domain(domain_id)
        if plugin and plugin.source_domains:
            library = plugin.get_domain_library()
            logger.info(
                "Loaded source domains from target domain",
                target_domain=domain_id,
                domain_count=len(library),
            )
            return library

        raise ValueError(
            f"Target domain '{domain_id}' not found or has no embedded source domains. "
            f"Ensure the target domain YAML file exists with a source_domains section."
        )

    @classmethod
    def from_path(cls, path: Path) -> "DomainLibrary":
        """Create a DomainLibrary from a specific path.

        Args:
            path: Path to the YAML file.
        """
        domains = load_domains(path)
        return cls(domains)

    def get(self, domain_id: str) -> Domain | None:
        """Get a domain by its ID.

        Args:
            domain_id: The domain ID to look up.

        Returns:
            The Domain if found, None otherwise.
        """
        return self._domains.get(domain_id)

    def list_ids(self) -> list[str]:
        """List all domain IDs.

        Returns:
            List of domain IDs.
        """
        return list(self._domains.keys())

    def __iter__(self) -> Iterator[Domain]:
        """Iterate over all domains."""
        return iter(self._domains.values())

    def __len__(self) -> int:
        """Return the number of domains."""
        return len(self._domains)
