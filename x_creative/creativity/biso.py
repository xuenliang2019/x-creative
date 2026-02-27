"""BISO (Bisociation) module for generating analogies from distant domains."""

import asyncio
import json
import random
import uuid
from collections.abc import Awaitable, Callable
import inspect
from typing import Any

import structlog

from x_creative.core.domain_loader import DomainLibrary
from x_creative.core.types import Domain, FailureMode, Hypothesis, MappingItem, ProblemFrame
from x_creative.config.settings import get_settings
from x_creative.creativity.prompts import BISO_ANALOGY_PROMPT
from x_creative.creativity.utils import safe_json_loads
from x_creative.llm.router import ModelRouter

logger = structlog.get_logger()


class BISOModule:
    """Module for bisociation-based hypothesis generation.

    Generates creative analogies by mapping structures from distant
    domains to target domain concepts.
    """

    def __init__(
        self,
        router: ModelRouter | None = None,
        domain_library: DomainLibrary | None = None,
    ) -> None:
        """Initialize the BISO module.

        Args:
            router: Model router for LLM calls. Creates one if not provided.
            domain_library: Explicit domain library override. When provided,
                bypasses target-domain-based loading entirely.
        """
        self._router = router or ModelRouter()
        self._explicit_domains = domain_library
        self._domain_cache: dict[str, DomainLibrary] = {}
        settings = get_settings()
        try:
            self._default_max_concurrency = max(
                1, int(settings.biso_max_concurrency)
            )
        except Exception:
            self._default_max_concurrency = 8
        self._biso_pool: list[str] = list(settings.biso_pool)

    def _get_domains(self, target_domain: str | None = None) -> DomainLibrary:
        """Get domain library, preferring target-domain-specific source domains.

        Resolution order:
        1. Explicit domain_library passed to constructor (always wins)
        2. Embedded source_domains from target domain plugin YAML

        Args:
            target_domain: Target domain plugin ID (e.g. "open_source_development").

        Returns:
            DomainLibrary with source domains for the given target domain.
        """
        if self._explicit_domains is not None:
            return self._explicit_domains

        if not target_domain:
            raise ValueError(
                "target_domain is required. Set problem.target_domain or pass "
                "an explicit domain_library to BISOModule()."
            )

        if target_domain in self._domain_cache:
            return self._domain_cache[target_domain]

        library = DomainLibrary.from_target_domain(target_domain)
        self._domain_cache[target_domain] = library
        return library

    def _build_domain_context(self, problem: ProblemFrame) -> str:
        """Build domain context text from problem frame."""
        lines: list[str] = []

        if problem.context:
            for key, value in problem.context.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else "No specific context provided"

    async def generate_analogies(
        self,
        domain: Domain,
        problem: ProblemFrame,
        num_analogies: int = 5,
    ) -> list[Hypothesis]:
        """Generate analogies from a single domain.

        Args:
            domain: Source domain to generate analogies from.
            problem: Problem frame with constraints.
            num_analogies: Target number of analogies to generate.

        Returns:
            List of Hypothesis objects generated from this domain.
        """
        # Format structures for prompt
        structures_text = "\n".join(
            f"- **{s.id}** ({s.name}): {s.description}\n"
            f"  Key variables: {', '.join(s.key_variables)}\n"
            f"  Dynamics: {s.dynamics}"
            for s in domain.structures
        )

        # Format existing mappings for reference
        mappings_text = "\n".join(
            f"- {m.structure} -> {m.target}: {m.observable}"
            for m in domain.target_mappings
        )

        # Format constraints
        constraints_text = (
            "\n".join(f"- {c}" for c in problem.constraints) if problem.constraints else "None"
        )

        # Build domain context
        domain_context = self._build_domain_context(problem)

        prompt = BISO_ANALOGY_PROMPT.format(
            domain_name=domain.name,
            domain_name_en=domain.name_en or domain.id,
            domain_description=domain.description,
            structures_text=structures_text,
            mappings_text=mappings_text or "None provided",
            problem_description=problem.description,
            target_domain=problem.target_domain,
            domain_context=domain_context,
            constraints_text=constraints_text,
            num_analogies=num_analogies,
        )

        logger.debug("Generating analogies", domain=domain.id, num_analogies=num_analogies)

        # Build optional model override from BISO pool
        pool_kwargs: dict[str, Any] = {}
        if self._biso_pool:
            selected_model = random.choice(self._biso_pool)
            pool_kwargs["model_override"] = selected_model
            logger.debug("BISO pool selected model", model=selected_model, domain=domain.id)

        result = await self._router.complete(
            task="creativity",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32768,
            **pool_kwargs,
        )

        # Parse response
        hypotheses = self._parse_analogies(result.content, domain)

        logger.info(
            "Generated analogies",
            domain=domain.id,
            count=len(hypotheses),
        )

        return hypotheses

    def _parse_analogies(self, content: str, domain: Domain) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects.

        Args:
            content: Raw LLM response content.
            domain: Source domain for these analogies.

        Returns:
            List of parsed Hypothesis objects.
        """
        hypotheses: list[Hypothesis] = []
        rejected_missing_structural_evidence = 0

        try:
            # Try to extract JSON from the response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                analogies = safe_json_loads(json_str)

                for analogy in analogies:
                    if not isinstance(analogy, dict):
                        continue
                    observable = str(analogy.get("observable", "")).strip()
                    if not observable:
                        continue

                    # Parse mapping_table (graceful: skip invalid items)
                    mapping_table = []
                    for m in analogy.get("mapping_table", []):
                        if isinstance(m, dict):
                            try:
                                mapping_table.append(MappingItem(**m))
                            except Exception:
                                pass

                    # Parse failure_modes (graceful: skip invalid items)
                    failure_modes = []
                    for fm in analogy.get("failure_modes", []):
                        if isinstance(fm, dict):
                            try:
                                failure_modes.append(FailureMode(**fm))
                            except Exception:
                                pass

                    # Enforce structural evidence contract: BISO output must
                    # include both mapping rows and failure modes.
                    if not mapping_table or not failure_modes:
                        rejected_missing_structural_evidence += 1
                        continue

                    hyp = Hypothesis(
                        id=f"hyp_{uuid.uuid4().hex[:8]}",
                        description=analogy.get("analogy", ""),
                        source_domain=domain.id,
                        source_structure=analogy.get("structure_id", ""),
                        analogy_explanation=analogy.get("explanation", ""),
                        observable=observable,
                        generation=0,
                        mapping_table=mapping_table,
                        failure_modes=failure_modes,
                    )
                    hypotheses.append(hyp)

                if rejected_missing_structural_evidence:
                    logger.debug(
                        "Rejected BISO candidates missing structural evidence",
                        domain=domain.id,
                        rejected_count=rejected_missing_structural_evidence,
                    )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Failed to parse analogies",
                error=str(e),
                content_preview=content[:200],
            )

        return hypotheses

    async def generate_all_analogies(
        self,
        problem: ProblemFrame,
        num_per_domain: int = 3,
        max_domains: int | None = None,
        max_concurrency: int | None = None,
        source_domains: list[Domain] | None = None,
        on_domain_complete: Callable[[str, int, int, int], Awaitable[None] | None] | None = None,
    ) -> list[Hypothesis]:
        """Generate analogies from all available domains concurrently.

        Args:
            problem: Problem frame with constraints.
            num_per_domain: Number of analogies to generate per domain.
            max_domains: Maximum number of domains to process (for testing).
            max_concurrency: Maximum concurrent API calls. None uses configured default.
            on_domain_complete: Optional callback invoked after each domain completes:
                (domain_id, completed_count, total_domains, generated_hypotheses_in_domain).

        Returns:
            List of all generated Hypothesis objects.
        """
        library = DomainLibrary(source_domains) if source_domains is not None else self._get_domains(problem.target_domain)
        domain_ids = library.list_ids()

        if max_domains is not None:
            domain_ids = domain_ids[:max_domains]

        # Get valid domains
        domains = [
            library.get(domain_id)
            for domain_id in domain_ids
            if library.get(domain_id) is not None
        ]

        effective_concurrency = (
            max_concurrency
            if isinstance(max_concurrency, int) and max_concurrency > 0
            else self._default_max_concurrency
        )

        logger.info(
            "Generating analogies from all domains",
            num_domains=len(domains),
            num_per_domain=num_per_domain,
            max_concurrency=effective_concurrency,
        )

        total_domains = len(domains)
        completed_domains = 0
        completed_lock = asyncio.Lock()

        # Always apply semaphore control to avoid unbounded fan-out.
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def _report_domain_complete(domain_id: str, generated: int) -> None:
            nonlocal completed_domains
            if on_domain_complete is None:
                return
            async with completed_lock:
                completed_domains += 1
                current = completed_domains
            try:
                maybe_awaitable = on_domain_complete(
                    domain_id,
                    current,
                    total_domains,
                    generated,
                )
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                logger.debug(
                    "on_domain_complete callback failed",
                    domain=domain_id,
                    error=str(exc),
                )

        async def generate_with_limit(domain: Domain) -> list[Hypothesis]:
            """Generate analogies for a domain with optional concurrency limit."""
            async def _generate() -> list[Hypothesis]:
                try:
                    return await self.generate_analogies(
                        domain=domain,
                        problem=problem,
                        num_analogies=num_per_domain,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to generate analogies for domain",
                        domain=domain.id,
                        error=str(e),
                    )
                    return []

            async with semaphore:
                result = await _generate()
                await _report_domain_complete(domain.id, len(result))
                return result

        # Run all domains concurrently
        results = await asyncio.gather(
            *[generate_with_limit(domain) for domain in domains],
            return_exceptions=False,
        )

        # Flatten results
        all_hypotheses: list[Hypothesis] = []
        for hypotheses in results:
            all_hypotheses.extend(hypotheses)

        logger.info(
            "Completed analogy generation",
            total_hypotheses=len(all_hypotheses),
        )

        return all_hypotheses
