"""Core abstractions and data types for X-Creative."""

from x_creative.core.types import (
    Domain,
    DomainStructure,
    Hypothesis,
    HypothesisScores,
    LogicVerdict,
    NoveltyVerdict,
    ProblemFrame,
    SearchConfig,
    SimilarWork,
    TargetMapping,
    VerifiedHypothesis,
)
from x_creative.core.domain_loader import DomainLibrary, load_domains
from x_creative.core.plugin import (
    DomainConstraint,
    TargetDomainPlugin,
    ValidatorConfig,
    list_target_domains,
    load_target_domain,
)

__all__ = [
    # Types
    "Domain",
    "DomainLibrary",
    "DomainStructure",
    "TargetMapping",
    "Hypothesis",
    "HypothesisScores",
    "LogicVerdict",
    "NoveltyVerdict",
    "ProblemFrame",
    "SearchConfig",
    "SimilarWork",
    "VerifiedHypothesis",
    "load_domains",
    # Plugin system
    "DomainConstraint",
    "TargetDomainPlugin",
    "ValidatorConfig",
    "list_target_domains",
    "load_target_domain",
]
