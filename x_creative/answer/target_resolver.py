"""TargetDomainResolver - three-level waterfall domain resolution."""

from __future__ import annotations

import json
import logging
from typing import Any

from x_creative.answer.prompts import EPHEMERAL_DOMAIN_PROMPT, FRESH_DOMAIN_PROMPT, TARGET_DOMAIN_RESOLVE_PROMPT
from x_creative.core.plugin import (
    DomainConstraint,
    TargetDomainPlugin,
    list_target_domains,
    load_target_domain,
)
from x_creative.core.types import ProblemFrame
from x_creative.creativity.utils import (
    recover_json_array,
    recover_truncated_json_object,
    safe_json_loads,
)
from x_creative.llm.router import ModelRouter

logger = logging.getLogger(__name__)

EXACT_MATCH_THRESHOLD = 0.7
SEMANTIC_MATCH_THRESHOLD = 0.5


class TargetDomainResolver:
    """Resolves a target domain using a three-level waterfall strategy.

    Level 0: Explicit override -- load by domain_id directly.
    Level 1: domain_hint confidence >= 0.7 -- exact match against built-in YAML.
    Level 2: LLM semantic match confidence >= 0.5 -- use that domain.
    Level 3: LLM generates ephemeral TargetDomainPlugin.
    """

    def __init__(self, router: ModelRouter | None = None) -> None:
        self._router = router

    async def resolve(
        self, frame: ProblemFrame, target_override: str = "auto", *, fresh: bool = False,
    ) -> TargetDomainPlugin:
        """Resolve the best target domain for the given problem frame.

        Args:
            frame: The problem frame describing the user's question.
            target_override: If not "auto", load this domain directly.
            fresh: If True and target_override is "auto", skip pre-defined
                YAML domains and generate both target and source domains
                from scratch via LLM.

        Returns:
            A TargetDomainPlugin instance.

        Raises:
            ValueError: If an explicit override domain is not found.
        """
        # Level 0: Explicit override
        if target_override != "auto":
            plugin = load_target_domain(target_override)
            if plugin is None:
                raise ValueError(
                    f"Target domain '{target_override}' not found. "
                    f"Available: {list_target_domains()}"
                )
            return plugin

        # Fresh mode: skip Level 1 & 2, generate everything from scratch
        if fresh:
            logger.info("Fresh mode: generating domain and source domains from scratch")
            return await self._generate_fresh(frame)

        # Level 1: Exact match on domain_hint
        if frame.domain_hint:
            domain_id = frame.domain_hint.get("domain_id", "")
            confidence = frame.domain_hint.get("confidence", 0.0)
            if confidence >= EXACT_MATCH_THRESHOLD:
                plugin = load_target_domain(domain_id)
                if plugin is not None:
                    return plugin

        # Level 2: Semantic match via LLM
        plugin = await self._semantic_match(frame)
        if plugin is not None:
            return plugin

        # Level 3: Generate ephemeral domain via LLM
        return await self._generate_ephemeral(frame)

    async def _semantic_match(self, frame: ProblemFrame) -> TargetDomainPlugin | None:
        """Use LLM to find the best matching domain from available options.

        Returns None if no domain matches with sufficient confidence,
        or if an error occurs.
        """
        available = list_target_domains()
        if not available:
            return None

        router = self._router or ModelRouter()
        owns_router = self._router is None
        try:
            domain_descs: list[str] = []
            for did in available:
                p = load_target_domain(did)
                if p:
                    domain_descs.append(f"- {did}: {p.description}")

            terms = ""
            if frame.definitions:
                terms = ", ".join(frame.definitions.keys())

            prompt = TARGET_DOMAIN_RESOLVE_PROMPT.format(
                objective=frame.objective or frame.description,
                description=frame.description,
                terms=terms,
                domain_descriptions="\n".join(domain_descs),
            )

            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            text = result.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]

            parsed: dict[str, Any] = json.loads(text)
            confidence = parsed.get("confidence", 0.0)
            domain_id = parsed.get("domain_id", "")

            if confidence >= SEMANTIC_MATCH_THRESHOLD:
                plugin = load_target_domain(domain_id)
                if plugin is not None:
                    return plugin

            return None
        except Exception:
            logger.warning("Semantic match failed", exc_info=True)
            return None
        finally:
            if owns_router:
                await router.close()

    async def _generate_fresh(self, frame: ProblemFrame) -> TargetDomainPlugin:
        """Use LLM to generate a domain with embedded source domains from scratch.

        Falls back to a minimal general-purpose plugin on failure.
        """
        router = self._router or ModelRouter()
        owns_router = self._router is None
        try:
            short_id = frame.description[:20].replace(" ", "_").lower()
            prompt = FRESH_DOMAIN_PROMPT.format(
                description=frame.description,
                objective=frame.objective or frame.description,
                short_id=short_id,
            )

            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            text = result.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]

            try:
                parsed: dict[str, Any] = safe_json_loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "Fresh domain: JSON parse failed, attempting recovery. Raw text:\n%s",
                    text,
                )
                parsed = recover_truncated_json_object(text)
                # If source_domains array was truncated, recover individual objects
                if "source_domains" not in parsed or not parsed["source_domains"]:
                    sd_marker = '"source_domains"'
                    sd_idx = text.find(sd_marker)
                    if sd_idx >= 0:
                        sd_fragment = text[sd_idx + len(sd_marker):]
                        recovered_sd = recover_json_array(sd_fragment)
                        if recovered_sd:
                            parsed["source_domains"] = recovered_sd

            if not isinstance(parsed, dict) or not parsed:
                logger.warning("Fresh domain: recovery yielded empty result, falling back to general")
                return TargetDomainPlugin(
                    id="general",
                    name="General",
                    description="General-purpose domain",
                )

            raw_constraints = parsed.get("constraints", [])
            constraints: list[DomainConstraint] = []
            for rc in raw_constraints:
                if isinstance(rc, dict) and "name" in rc:
                    constraints.append(
                        DomainConstraint(
                            name=rc["name"],
                            description=rc.get("description", ""),
                            severity=rc.get("severity", "advisory"),
                        )
                    )

            plugin = TargetDomainPlugin(
                id=parsed.get("id", "fresh"),
                name=parsed.get("name", "Fresh Domain"),
                description=parsed.get("description", frame.description),
                constraints=constraints,
                evaluation_criteria=parsed.get("evaluation_criteria", []),
                anti_patterns=parsed.get("anti_patterns", []),
                terminology=parsed.get("terminology", {}),
                stale_ideas=parsed.get("stale_ideas", []),
                source_domains=parsed.get("source_domains", []),
            )
            logger.info(
                "Fresh domain generated: %s with %d source domains",
                plugin.id, len(plugin.source_domains),
            )
            return plugin
        except Exception:
            logger.warning("Fresh domain generation failed", exc_info=True)
            return TargetDomainPlugin(
                id="general",
                name="General",
                description="General-purpose domain",
            )
        finally:
            if owns_router:
                await router.close()

    async def _generate_ephemeral(self, frame: ProblemFrame) -> TargetDomainPlugin:
        """Use LLM to generate a full ephemeral domain plugin.

        Falls back to a minimal general-purpose plugin on failure.
        """
        router = self._router or ModelRouter()
        owns_router = self._router is None
        try:
            short_id = frame.description[:20].replace(" ", "_").lower()
            prompt = EPHEMERAL_DOMAIN_PROMPT.format(
                description=frame.description,
                objective=frame.objective or frame.description,
                short_id=short_id,
            )

            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )

            text = result.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]

            try:
                parsed: dict[str, Any] = safe_json_loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "Ephemeral domain: JSON parse failed, attempting recovery. Raw text:\n%s",
                    text,
                )
                parsed = recover_truncated_json_object(text)
                if "source_domains" not in parsed or not parsed["source_domains"]:
                    sd_marker = '"source_domains"'
                    sd_idx = text.find(sd_marker)
                    if sd_idx >= 0:
                        sd_fragment = text[sd_idx + len(sd_marker):]
                        recovered_sd = recover_json_array(sd_fragment)
                        if recovered_sd:
                            parsed["source_domains"] = recovered_sd

            if not isinstance(parsed, dict) or not parsed:
                logger.warning("Ephemeral domain: recovery yielded empty result, falling back to general")
                return TargetDomainPlugin(
                    id="general",
                    name="General",
                    description="General-purpose domain",
                )

            raw_constraints = parsed.get("constraints", [])
            constraints: list[DomainConstraint] = []
            for rc in raw_constraints:
                if isinstance(rc, dict) and "name" in rc:
                    constraints.append(
                        DomainConstraint(
                            name=rc["name"],
                            description=rc.get("description", ""),
                            severity=rc.get("severity", "advisory"),
                        )
                    )

            return TargetDomainPlugin(
                id=parsed.get("id", "ephemeral"),
                name=parsed.get("name", "Ephemeral Domain"),
                description=parsed.get("description", frame.description),
                constraints=constraints,
                evaluation_criteria=parsed.get("evaluation_criteria", []),
                anti_patterns=parsed.get("anti_patterns", []),
                terminology=parsed.get("terminology", {}),
                stale_ideas=parsed.get("stale_ideas", []),
                source_domains=parsed.get("source_domains", []),
            )
        except Exception:
            logger.warning("Ephemeral generation failed", exc_info=True)
            return TargetDomainPlugin(
                id="general",
                name="General",
                description="General-purpose domain",
            )
        finally:
            if owns_router:
                await router.close()
