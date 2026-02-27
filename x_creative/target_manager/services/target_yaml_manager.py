"""YAML file management for target domain configuration.

Manages target domain YAML files (create, read, update).
Supports listing all target domains, loading full plugin data,
and updating individual sections.
"""

import re
from pathlib import Path
from typing import Any

import structlog
from ruamel.yaml import YAML

from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Domain

logger = structlog.get_logger()

# Built-in target domains directory
_TARGET_DOMAINS_DIR = Path(__file__).parent.parent.parent / "config" / "target_domains"


class TargetYAMLManager:
    """Manages target domain YAML files.

    Supports creating new target domains, loading existing ones,
    and updating individual sections (constraints, terminology, etc.).
    """

    def __init__(self, domains_dir: Path | None = None) -> None:
        """Initialize the target YAML manager.

        Args:
            domains_dir: Directory containing target domain YAML files.
                         Defaults to built-in target_domains directory.
        """
        self._domains_dir = domains_dir or _TARGET_DOMAINS_DIR
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.default_flow_style = False
        self._yaml.width = 120

    @property
    def domains_dir(self) -> Path:
        """Get the target domains directory."""
        return self._domains_dir

    def list_target_domains(self) -> list[dict[str, Any]]:
        """List all available target domains with basic stats.

        Returns:
            List of dicts with id, name, description, path,
            domain_count, constraint_count.
        """
        result = []
        yaml = YAML()

        if not self._domains_dir.exists():
            return result

        for yaml_file in sorted(self._domains_dir.glob("*.yaml")):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.load(f)
                if not isinstance(data, dict) or "id" not in data:
                    continue
                source_domains = data.get("source_domains", [])
                result.append({
                    "id": data["id"],
                    "name": data.get("name", data["id"]),
                    "description": data.get("description", ""),
                    "path": str(yaml_file),
                    "domain_count": len(source_domains),
                    "constraint_count": len(data.get("constraints", [])),
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

    def load_target_domain(self, target_id: str) -> TargetDomainPlugin:
        """Load a target domain as a TargetDomainPlugin.

        Args:
            target_id: Target domain ID.

        Returns:
            TargetDomainPlugin instance.

        Raises:
            FileNotFoundError: If no YAML file found for the given ID.
        """
        path = self._resolve_path(target_id)
        with open(path, encoding="utf-8") as f:
            data = self._yaml.load(f)
        return TargetDomainPlugin(**data)

    def get_stats(self, target_id: str) -> dict[str, Any]:
        """Get statistics about a target domain.

        Args:
            target_id: Target domain ID.

        Returns:
            Dict with target_id, name, domain_count, structure_count,
            constraint_count, evaluation_criteria_count, etc.
        """
        path = self._resolve_path(target_id)
        with open(path, encoding="utf-8") as f:
            data = self._yaml.load(f)

        source_domains = data.get("source_domains", [])
        return {
            "target_id": data.get("id", target_id),
            "name": data.get("name", target_id),
            "description": data.get("description", ""),
            "domain_count": len(source_domains),
            "structure_count": sum(
                len(d.get("structures", []))
                for d in source_domains
            ),
            "constraint_count": len(data.get("constraints", [])),
            "evaluation_criteria_count": len(data.get("evaluation_criteria", [])),
            "anti_pattern_count": len(data.get("anti_patterns", [])),
            "terminology_count": len(data.get("terminology", {})),
            "stale_idea_count": len(data.get("stale_ideas", [])),
        }

    def create_target_domain(
        self,
        target_id: str,
        name: str,
        description: str,
        constraints: list[dict[str, Any]] | None = None,
        evaluation_criteria: list[str] | None = None,
        anti_patterns: list[str] | None = None,
        terminology: dict[str, str] | None = None,
        stale_ideas: list[str] | None = None,
        source_domains: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Create a new target domain YAML file.

        Args:
            target_id: Unique ID (snake_case).
            name: Display name.
            description: Brief description.
            constraints: Optional list of constraint dicts.
            evaluation_criteria: Optional list of criteria strings.
            anti_patterns: Optional list of anti-pattern strings.
            terminology: Optional dict of term -> definition.
            stale_ideas: Optional list of stale idea strings.
            source_domains: Optional list of source domain dicts.

        Returns:
            Path to the created YAML file.

        Raises:
            FileExistsError: If a target domain with this ID already exists.
        """
        self._domains_dir.mkdir(parents=True, exist_ok=True)
        path = self._domains_dir / f"{target_id}.yaml"

        if path.exists():
            raise FileExistsError(
                f"Target domain already exists: {path}"
            )

        data: dict[str, Any] = {
            "id": target_id,
            "name": name,
            "description": description,
        }

        if constraints is not None:
            data["constraints"] = constraints
        if evaluation_criteria is not None:
            data["evaluation_criteria"] = evaluation_criteria
        if anti_patterns is not None:
            data["anti_patterns"] = anti_patterns
        if terminology is not None:
            data["terminology"] = terminology
        if stale_ideas is not None:
            data["stale_ideas"] = stale_ideas
        if source_domains is not None:
            data["source_domains"] = source_domains

        with open(path, "w", encoding="utf-8") as f:
            self._yaml.dump(data, f)

        logger.info(
            "Created target domain",
            target_id=target_id,
            path=str(path),
        )
        return path

    def update_section(
        self,
        target_id: str,
        section: str,
        data: Any,
    ) -> None:
        """Update a specific section of a target domain.

        Creates a backup before writing.

        Args:
            target_id: Target domain ID.
            section: Section key (e.g. "constraints", "terminology").
            data: New data for the section.

        Raises:
            FileNotFoundError: If no YAML file found for the given ID.
            ValueError: If section is not a valid target domain field.
        """
        valid_sections = {
            "constraints", "evaluation_criteria", "anti_patterns",
            "terminology", "stale_ideas", "source_domains",
            "name", "description",
        }
        if section not in valid_sections:
            raise ValueError(
                f"Invalid section '{section}'. Valid sections: {sorted(valid_sections)}"
            )

        path = self._resolve_path(target_id)
        with open(path, encoding="utf-8") as f:
            file_data = self._yaml.load(f)

        # Create backup
        self._backup(path)

        file_data[section] = data

        with open(path, "w", encoding="utf-8") as f:
            self._yaml.dump(file_data, f)

        logger.info(
            "Updated target domain section",
            target_id=target_id,
            section=section,
        )

    def add_source_domains(
        self,
        target_id: str,
        domains: list[Domain],
    ) -> None:
        """Add source domains to a target domain.

        Creates a backup before writing.

        Args:
            target_id: Target domain ID.
            domains: List of Domain objects to add.

        Raises:
            FileNotFoundError: If no YAML file found for the given ID.
        """
        path = self._resolve_path(target_id)
        with open(path, encoding="utf-8") as f:
            file_data = self._yaml.load(f)

        # Create backup
        self._backup(path)

        if "source_domains" not in file_data:
            file_data["source_domains"] = []

        for domain in domains:
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
            file_data["source_domains"].append(domain_dict)

        with open(path, "w", encoding="utf-8") as f:
            self._yaml.dump(file_data, f)

        logger.info(
            "Added source domains to target domain",
            target_id=target_id,
            count=len(domains),
        )

    def validate_target_id(self, target_id: str) -> tuple[bool, str]:
        """Validate a target domain ID.

        Args:
            target_id: ID to validate.

        Returns:
            Tuple of (is_valid, message).
        """
        if not target_id:
            return False, "ID 不能为空"

        if not re.match(r"^[a-z][a-z0-9_]*$", target_id):
            return False, "ID 必须是小写字母开头，只包含小写字母、数字和下划线"

        if len(target_id) > 50:
            return False, "ID 长度不能超过 50 个字符"

        # Check if already exists
        path = self._domains_dir / f"{target_id}.yaml"
        if path.exists():
            return False, f"ID '{target_id}' 已存在"

        return True, "ID 有效"

    def _resolve_path(self, target_id: str) -> Path:
        """Resolve path for a target domain YAML file.

        Args:
            target_id: Target domain ID.

        Returns:
            Path to the YAML file.

        Raises:
            FileNotFoundError: If no YAML file found.
        """
        path = self._domains_dir / f"{target_id}.yaml"
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Target domain YAML not found: {path}"
        )

    def _backup(self, path: Path) -> None:
        """Create a backup of a YAML file.

        Args:
            path: Path to the file to backup.
        """
        if path.exists():
            backup_path = path.with_suffix(".yaml.bak")
            backup_path.write_bytes(path.read_bytes())
