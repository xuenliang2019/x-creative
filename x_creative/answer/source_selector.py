"""SourceDomainSelector - intelligent source domain selection with diversity."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from x_creative.answer.prompts import SOURCE_RELEVANCE_PROMPT
from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Domain, ProblemFrame
from x_creative.llm.router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class ScoredDomain:
    domain: Domain
    relevance: float


class SourceDomainSelector:
    def __init__(
        self,
        router: ModelRouter | None = None,
        diversity_threshold: float = 0.3,
        feasibility_threshold: float = 0.4,
    ):
        self._router = router
        self.diversity_threshold = diversity_threshold
        self.feasibility_threshold = feasibility_threshold

    async def select(
        self,
        frame: ProblemFrame,
        target: TargetDomainPlugin,
        min_domains: int = 18,
        max_domains: int = 30,
    ) -> list[Domain]:
        library = target.get_domain_library()
        all_domains = list(library)

        if not all_domains:
            return []

        if len(all_domains) <= min_domains:
            return all_domains

        ranked = await self._rank_by_relevance(frame, all_domains)

        if ranked is None:
            return all_domains[:max_domains]

        # Stage B: Filter by mapping feasibility
        ranked_domains = [sd.domain for sd in ranked]
        feasible_domains = await self._filter_by_mapping_feasibility(frame, ranked_domains)

        # Re-rank after filtering
        feasible_ids = {d.id for d in feasible_domains}
        ranked_feasible = [sd for sd in ranked if sd.domain.id in feasible_ids]

        if len(ranked_feasible) < min_domains:
            # Not enough after filtering, fall back to ranked
            ranked_feasible = ranked

        selected = self._ensure_diversity(ranked_feasible, max_domains)
        return [sd.domain for sd in selected]

    async def filter_by_mapping_feasibility(
        self,
        frame: ProblemFrame,
        target: TargetDomainPlugin,
    ) -> list[Domain]:
        """Apply Stage-B feasibility filter to all source domains in a target plugin."""
        domains = list(target.get_domain_library())
        if not domains:
            return []
        return await self._filter_by_mapping_feasibility(frame, domains)

    async def _rank_by_relevance(
        self, frame: ProblemFrame, domains: list[Domain]
    ) -> list[ScoredDomain] | None:
        router = self._router or ModelRouter()
        owns_router = self._router is None
        try:
            domain_list = "\n".join(f"- {d.id}: {d.name} — {d.description}" for d in domains)
            prompt = SOURCE_RELEVANCE_PROMPT.format(
                description=frame.description,
                objective=frame.objective or frame.description,
                domain_list=domain_list,
            )
            result = await router.complete(
                task="creativity", messages=[{"role": "user", "content": prompt}], temperature=0.2,
            )
            text = result.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
            scores = json.loads(text)
            score_map = {s["domain_id"]: s["relevance"] for s in scores if isinstance(s, dict)}
            scored = []
            for d in domains:
                rel = score_map.get(d.id, 0.5)
                scored.append(ScoredDomain(domain=d, relevance=rel))
            scored.sort(key=lambda s: s.relevance, reverse=True)
            return scored
        except Exception:
            logger.warning("LLM relevance ranking failed", exc_info=True)
            return None
        finally:
            if owns_router:
                await router.close()

    async def _filter_by_mapping_feasibility(
        self, frame: ProblemFrame, domains: list[Domain]
    ) -> list[Domain]:
        """Stage B: Filter domains by structural mapping feasibility."""
        router = self._router or ModelRouter()
        owns_router = self._router is None
        try:
            from x_creative.answer.prompts import MAPPING_FEASIBILITY_PROMPT

            domain_list = "\n".join(
                f"- {d.id}: {d.name} — {d.description}" for d in domains
            )
            prompt = MAPPING_FEASIBILITY_PROMPT.format(
                description=frame.description,
                objective=frame.objective or frame.description,
                domain_list=domain_list,
            )
            result = await router.complete(
                task="creativity",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = result.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
            scores = json.loads(text)
            score_map = {
                s["domain_id"]: s.get("feasibility", 0.5)
                for s in scores
                if isinstance(s, dict)
            }

            filtered = [
                d
                for d in domains
                if score_map.get(d.id, 0.5) >= self.feasibility_threshold
            ]
            logger.info(
                "Mapping feasibility filter: total=%d, kept=%d",
                len(domains),
                len(filtered),
            )
            return filtered if filtered else domains  # Never return empty
        except Exception:
            logger.warning(
                "Mapping feasibility filtering failed, keeping all domains",
                exc_info=True,
            )
            return domains
        finally:
            if owns_router:
                await router.close()

    def _ensure_diversity(self, ranked: list[ScoredDomain], max_n: int) -> list[ScoredDomain]:
        if not ranked:
            return []
        selected = [ranked[0]]
        for candidate in ranked[1:]:
            if len(selected) >= max_n:
                break
            if not any(self._names_too_similar(candidate.domain, s.domain) for s in selected):
                selected.append(candidate)
        return selected

    @staticmethod
    def _names_too_similar(a: Domain, b: Domain) -> bool:
        a_words = set(a.id.lower().split("_"))
        b_words = set(b.id.lower().split("_"))
        overlap = a_words & b_words
        min_len = min(len(a_words), len(b_words))
        return len(overlap) > min_len / 2 if min_len > 0 else False
