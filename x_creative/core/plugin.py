"""Target domain plugin for multi-domain creativity support."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from x_creative.core.domain_loader import DomainLibrary


class DomainConstraint(BaseModel):
    """A constraint specific to a target domain."""

    name: str = Field(..., description="Unique identifier for the constraint")
    description: str = Field(..., description="Human-readable description")
    severity: Literal["critical", "important", "advisory"] = Field(
        ..., description="How important this constraint is"
    )
    check_prompt: str | None = Field(
        default=None,
        description="Optional prompt fragment for LLM to check this constraint",
    )


class ValidatorConfig(BaseModel):
    """Configuration for a domain-specific validator."""

    name: str = Field(..., description="Validator name")
    prompt_template: str = Field(..., description="Prompt template for validation")
    model: str | None = Field(default=None, description="Specific model to use")
    threshold: float = Field(default=5.0, ge=0.0, le=10.0, description="Passing threshold (0-10)")


class TargetDomainPlugin(BaseModel):
    """Target domain plugin - progressive configuration.

    Defines the characteristics, constraints, and evaluation criteria
    for a specific application domain (e.g., open source development,
    drug discovery, materials science).

    Optionally embeds source domains with target-specific mappings,
    enabling self-contained target domain files.
    """

    # === Required (minimal configuration) ===
    id: str = Field(..., description="Unique identifier, e.g. 'open_source_development'")
    name: str = Field(..., description="Display name, e.g. '开源软件开发选题'")
    description: str = Field(..., description="Brief description of the domain")

    # === Optional (progressive enrichment) ===
    constraints: list[DomainConstraint] = Field(
        default_factory=list,
        description="Domain-specific constraints",
    )
    evaluation_criteria: list[str] = Field(
        default_factory=list,
        description="How to evaluate results in this domain",
    )
    anti_patterns: list[str] = Field(
        default_factory=list,
        description="Known bad practices to avoid",
    )
    terminology: dict[str, str] = Field(
        default_factory=dict,
        description="Domain-specific terms and definitions",
    )
    stale_ideas: list[str] = Field(
        default_factory=list,
        description="Known stale/overused ideas for novelty checking",
    )
    custom_validators: list[ValidatorConfig] = Field(
        default_factory=list,
        description="Domain-specific validation configurations",
    )

    # === Source Domains (embedded, for self-contained target domain files) ===
    source_domains: list[dict] = Field(
        default_factory=list,
        description="Source domains embedded in this target domain with target-specific mappings",
    )

    def get_constraint(self, name: str) -> DomainConstraint | None:
        """Get a constraint by its name."""
        for c in self.constraints:
            if c.name == name:
                return c
        return None

    def get_critical_constraints(self) -> list[DomainConstraint]:
        """Get all critical constraints."""
        return [c for c in self.constraints if c.severity == "critical"]

    def get_domain_library(self) -> DomainLibrary:
        """Create a DomainLibrary from embedded source domains.

        Returns:
            DomainLibrary containing the source domains defined in this plugin.
            Returns an empty DomainLibrary if no source domains are embedded.
        """
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.types import Domain

        domains = [Domain(**d) for d in self.source_domains]
        return DomainLibrary(domains)


# Target domains directory
TARGET_DOMAINS_DIR = Path(__file__).parent.parent / "config" / "target_domains"
USER_DOMAINS_DIR = Path.home() / ".config" / "x-creative" / "domains"


def _load_yaml_plugin(path: Path) -> TargetDomainPlugin | None:
    """Load a target domain plugin from YAML file."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TargetDomainPlugin(**data)
    except Exception:
        return None


@lru_cache(maxsize=32)
def load_target_domain(domain_id: str) -> TargetDomainPlugin | None:
    """Load a target domain plugin by ID.

    Searches in order:
    1. User custom domains (~/.config/x-creative/domains/)
    2. Built-in domains (x_creative/config/target_domains/)

    Args:
        domain_id: The domain identifier (e.g., 'open_source_development').

    Returns:
        TargetDomainPlugin or None if not found.
    """
    # Check user domains first
    user_path = USER_DOMAINS_DIR / f"{domain_id}.yaml"
    if user_path.exists():
        return _load_yaml_plugin(user_path)

    # Check built-in domains
    builtin_path = TARGET_DOMAINS_DIR / f"{domain_id}.yaml"
    return _load_yaml_plugin(builtin_path)


def list_target_domains() -> list[str]:
    """List all available target domain IDs.

    Returns:
        List of domain IDs from both user and built-in directories.
    """
    domains = set()

    # Collect from built-in
    if TARGET_DOMAINS_DIR.exists():
        for f in TARGET_DOMAINS_DIR.glob("*.yaml"):
            domains.add(f.stem)

    # Collect from user (can override built-in)
    if USER_DOMAINS_DIR.exists():
        for f in USER_DOMAINS_DIR.glob("*.yaml"):
            domains.add(f.stem)

    return sorted(domains)
