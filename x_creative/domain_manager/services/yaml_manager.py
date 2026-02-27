"""YAML file management for target domain source domains.

Manages source domains within target-domain YAML files (e.g. open_source_development.yaml).
Supports reading, adding, and modifying source domains and their structures.
"""

from pathlib import Path
from typing import Any

import structlog
from ruamel.yaml import YAML

from x_creative.core.types import Domain, DomainStructure, TargetMapping

logger = structlog.get_logger()

# Built-in target domains directory
_TARGET_DOMAINS_DIR = Path(__file__).parent.parent.parent / "config" / "target_domains"


class YAMLManager:
    """Manages source domains within a target-domain YAML file.

    Operates on target_domains/{id}.yaml files, reading and writing
    the source_domains section while preserving plugin config.
    """

    def __init__(self, target_domain_id: str, *, path: Path | None = None) -> None:
        """Initialize for a specific target domain.

        Args:
            target_domain_id: Target domain plugin ID (e.g. "open_source_development").
            path: Explicit path override. If not provided, resolves automatically.
        """
        self._target_domain_id = target_domain_id
        self._path = path or self._resolve_path(target_domain_id)
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.default_flow_style = False
        self._yaml.width = 120
        self._data: dict[str, Any] | None = None
        self._domains: list[Domain] = []

    @staticmethod
    def _resolve_path(target_domain_id: str) -> Path:
        """Resolve path for a target domain YAML file.

        Args:
            target_domain_id: Target domain plugin ID.

        Returns:
            Path to the target domain YAML file.

        Raises:
            FileNotFoundError: If no YAML file found for the given ID.
        """
        path = _TARGET_DOMAINS_DIR / f"{target_domain_id}.yaml"
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Target domain YAML not found: {path}"
        )

    @property
    def target_domain_id(self) -> str:
        """Get the target domain ID."""
        return self._target_domain_id

    @property
    def path(self) -> Path:
        """Get the YAML file path."""
        return self._path

    def _ensure_loaded(self) -> None:
        """Ensure data is loaded from file."""
        if self._data is None:
            self.load_domains()

    def load_domains(self) -> list[Domain]:
        """Load source domains from the target domain YAML file."""
        with open(self._path, encoding="utf-8") as f:
            self._data = self._yaml.load(f)

        self._domains = []
        for domain_data in self._data.get("source_domains", []):
            structures = [
                DomainStructure(**s) for s in domain_data.get("structures", [])
            ]
            raw_mappings = domain_data.get("target_mappings", [])
            mappings = [TargetMapping(**m) for m in raw_mappings]
            domain = Domain(
                id=domain_data["id"],
                name=domain_data["name"],
                name_en=domain_data.get("name_en"),
                description=domain_data["description"],
                structures=structures,
                target_mappings=mappings,
            )
            self._domains.append(domain)

        logger.debug(
            "Loaded source domains from target domain",
            target_domain=self._target_domain_id,
            count=len(self._domains),
        )
        return self._domains

    def add_source_domain(self, domain: Domain) -> None:
        """Add a new source domain to the target domain.

        Args:
            domain: Domain to add.
        """
        self._ensure_loaded()

        domain_dict = {
            "id": domain.id,
            "name": domain.name,
            "name_en": domain.name_en,
            "description": domain.description,
            "structures": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "key_variables": list(s.key_variables),
                    "dynamics": s.dynamics,
                }
                for s in domain.structures
            ],
            "target_mappings": [
                {
                    "structure": m.structure,
                    "target": m.target,
                    "observable": m.observable,
                }
                for m in domain.target_mappings
            ],
        }

        if "source_domains" not in self._data:
            self._data["source_domains"] = []
        self._data["source_domains"].append(domain_dict)
        self._domains.append(domain)

        logger.info(
            "Added source domain",
            target_domain=self._target_domain_id,
            source_domain=domain.id,
        )

    def add_structure(
        self,
        domain_id: str,
        structure: DomainStructure,
        mapping: TargetMapping,
    ) -> None:
        """Add a structure and mapping to an existing source domain.

        Args:
            domain_id: Source domain ID to add the structure to.
            structure: New structure to add.
            mapping: Corresponding target mapping.
        """
        self._ensure_loaded()

        for domain_data in self._data.get("source_domains", []):
            if domain_data["id"] == domain_id:
                domain_data.setdefault("structures", []).append({
                    "id": structure.id,
                    "name": structure.name,
                    "description": structure.description,
                    "key_variables": list(structure.key_variables),
                    "dynamics": structure.dynamics,
                })
                domain_data.setdefault("target_mappings", []).append({
                    "structure": mapping.structure,
                    "target": mapping.target,
                    "observable": mapping.observable,
                })
                break

    def save(self) -> None:
        """Save changes to YAML file with backup."""
        if self._data is None:
            return

        # Create backup
        backup_path = self._path.with_suffix(".yaml.bak")
        if self._path.exists():
            backup_path.write_bytes(self._path.read_bytes())

        # Write updated content
        with open(self._path, "w", encoding="utf-8") as f:
            self._yaml.dump(self._data, f)

        logger.info(
            "Saved target domain YAML",
            target_domain=self._target_domain_id,
            path=str(self._path),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the target domain.

        Returns:
            Dict with target_domain_id, domain_count, structure_count,
            mapping_count, and target_domain_name.
        """
        self._ensure_loaded()

        structure_count = sum(len(d.structures) for d in self._domains)
        mapping_count = sum(len(d.target_mappings) for d in self._domains)
        return {
            "target_domain_id": self._target_domain_id,
            "target_domain_name": self._data.get("name", self._target_domain_id),
            "domain_count": len(self._domains),
            "structure_count": structure_count,
            "mapping_count": mapping_count,
        }

    def get_target_domain_info(self) -> dict[str, Any]:
        """Get target domain plugin configuration (constraints, etc.).

        Returns:
            Dict with id, name, description, and plugin config fields.
        """
        self._ensure_loaded()

        return {
            "id": self._data.get("id", self._target_domain_id),
            "name": self._data.get("name", ""),
            "description": self._data.get("description", ""),
            "constraints": self._data.get("constraints", []),
            "evaluation_criteria": self._data.get("evaluation_criteria", []),
        }

    def get_domain(self, domain_id: str) -> Domain | None:
        """Get a source domain by ID.

        Args:
            domain_id: Source domain ID to look up.

        Returns:
            Domain if found, None otherwise.
        """
        self._ensure_loaded()

        for domain in self._domains:
            if domain.id == domain_id:
                return domain
        return None

    @property
    def domains(self) -> list[Domain]:
        """Get loaded source domains."""
        self._ensure_loaded()
        return self._domains

    @classmethod
    def list_target_domains(cls) -> list[dict[str, Any]]:
        """List all available target domains with basic stats.

        Returns:
            List of dicts with id, name, path, domain_count.
        """
        result = []
        yaml = YAML()

        for yaml_file in sorted(_TARGET_DOMAINS_DIR.glob("*.yaml")):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.load(f)
                if not isinstance(data, dict) or "id" not in data:
                    continue
                source_domains = data.get("source_domains", [])
                result.append({
                    "id": data["id"],
                    "name": data.get("name", data["id"]),
                    "path": str(yaml_file),
                    "domain_count": len(source_domains),
                    "structure_count": sum(
                        len(d.get("structures", []))
                        for d in source_domains
                    ),
                })
            except Exception as e:
                logger.warning(
                    "Failed to load target domain",
                    path=str(yaml_file),
                    error=str(e),
                )

        return result
