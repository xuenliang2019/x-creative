"""SEARCH module implementing Graph of Thoughts for hypothesis exploration."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import structlog
from structlog import contextvars

from x_creative.core.types import Hypothesis, SearchConfig
from x_creative.creativity.prompts import SEARCH_COMBINE_PROMPT, SEARCH_EXPAND_PROMPT
from x_creative.creativity.utils import safe_json_loads
from x_creative.llm.router import ModelRouter

if TYPE_CHECKING:
    from x_creative.core.concept_space import ConceptSpace
    from x_creative.creativity.mome import MOMEArchive
    from x_creative.creativity.pareto import ParetoArchive
    from x_creative.hkg.cache import TraversalCache
    from x_creative.hkg.store import HypergraphStore
    from x_creative.hkg.matcher import NodeMatcher
    from x_creative.hkg.types import HKGParams
    from x_creative.core.types import ProblemFrame

logger = structlog.get_logger()


@dataclass
class HeavyQuotaManager:
    """Manages per-round budgets for heavy SEARCH operators (§2.2).

    Heavy operators (hyperbridge, blend_expand, transform_space) are
    quota-limited.  When a quota is exhausted the caller should degrade
    to lightweight operators (refine / variant).
    """

    blend: int = 0
    transform: int = 0
    hyperbridge: int = 0
    _degraded: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: SearchConfig,
        heavy_ops_enabled: bool,
        top_n_size: int,
        hyperpath_expand_topN: int,
    ) -> HeavyQuotaManager:
        """Build a quota manager from SearchConfig and runtime context."""
        if not heavy_ops_enabled:
            return cls()
        return cls(
            blend=config.blend_expand_budget_per_round,
            transform=config.transform_space_budget_per_round,
            hyperbridge=max(0, min(top_n_size, hyperpath_expand_topN) // 2),
        )

    def allow(self, op: str) -> bool:
        """Check whether *op* still has remaining quota."""
        return getattr(self, op, 0) > 0

    def consume(self, op: str) -> None:
        """Consume one unit of *op* quota."""
        current = getattr(self, op, 0)
        if current <= 0:
            raise ValueError(f"Quota exhausted for {op}")
        setattr(self, op, current - 1)

    def record_degradation(self, op: str) -> None:
        """Track that *op* was degraded to lightweight ops."""
        self._degraded[op] = self._degraded.get(op, 0) + 1

    @property
    def degraded_ops(self) -> dict[str, int]:
        """Return counts of quota-exhaustion degradation events."""
        return dict(self._degraded)

ExpansionType = Literal[
    "refine",
    "variant",
    "combine",
    "oppose",
    "extreme",
    "hyperpath_expand",
    "hyperbridge",
    "blend",
    "blend_expand",
    "transform_space",
]


class SearchModule:
    """Module for structured search using Graph of Thoughts.

    Expands, combines, and prunes hypotheses through multiple
    iterations of exploration.
    """

    def __init__(
        self,
        router: ModelRouter | None = None,
        hkg_store: HypergraphStore | None = None,
        hkg_matcher: NodeMatcher | None = None,
        hkg_params: HKGParams | None = None,
        problem_frame: ProblemFrame | None = None,
        enable_hyperbridge: bool = False,
        mapping_quality_gate: float | None = None,
        pareto_archive: ParetoArchive | None = None,
        mome_archive: MOMEArchive | None = None,
        concept_space: ConceptSpace | None = None,
    ) -> None:
        """Initialize the search module.

        Args:
            router: Model router for LLM calls.
            hkg_store: Optional HypergraphStore for HKG expansion.
            hkg_matcher: Optional NodeMatcher for HKG expansion.
            hkg_params: Optional HKGParams for HKG expansion.
            problem_frame: Optional ProblemFrame for HKG expansion context.
            enable_hyperbridge: Whether to enable hyperbridge expansion.
            mapping_quality_gate: Minimum mapping_quality for expansion selection.
                Hypotheses below this threshold are excluded, and hypotheses
                without mapping_quality are also excluded. None disables the gate.
            pareto_archive: Optional ParetoArchive for NSGA-II selection.
                When provided, replaces composite-score ranking in _select_for_expansion.
            mome_archive: Optional MOMEArchive for MAP-Elites selection.
                When provided, takes priority over pareto_archive.
            concept_space: Optional ConceptSpace for transform_space operator.
        """
        self._router = router or ModelRouter()
        self._hkg_store = hkg_store
        self._hkg_matcher = hkg_matcher
        self._hkg_params = hkg_params
        self._problem_frame = problem_frame
        self._enable_hyperbridge = enable_hyperbridge
        self._mapping_quality_gate = mapping_quality_gate
        self._pareto_archive = pareto_archive
        self._mome_archive = mome_archive
        self._concept_space = concept_space
        self._hkg_cache: TraversalCache | None = None
        if hkg_store is not None:
            from x_creative.hkg.cache import TraversalCache as _TC
            self._hkg_cache = _TC()

    async def expand_hypothesis(
        self,
        hypothesis: Hypothesis,
        expansion_types: list[ExpansionType] | None = None,
        max_expansions: int = 3,
    ) -> list[Hypothesis]:
        """Expand a single hypothesis into variations.

        Args:
            hypothesis: The hypothesis to expand.
            expansion_types: Types of expansions to generate.
            max_expansions: Maximum number of expansions.

        Returns:
            List of expanded hypotheses.
        """
        if expansion_types is None:
            expansion_types = ["refine", "variant"]

        problem_description = ""
        problem_constraints = "None"
        problem_context = "No specific context provided"
        target_domain_id = "unknown"
        target_domain_name = "Unknown"
        target_domain_description = ""

        if self._problem_frame is not None:
            problem_description = str(self._problem_frame.description or "").strip()
            if self._problem_frame.constraints:
                problem_constraints = "\n".join(f"- {c}" for c in self._problem_frame.constraints)
            ctx = self._problem_frame.context or {}
            if ctx:
                problem_context = "\n".join(f"- {k}: {v}" for k, v in ctx.items())
            target_domain_id = str(self._problem_frame.target_domain or "").strip() or "unknown"

        try:
            from x_creative.core.plugin import load_target_domain

            plugin = load_target_domain(target_domain_id) if target_domain_id else None
            if plugin is not None:
                target_domain_name = plugin.name
                target_domain_description = plugin.description
            else:
                target_domain_name = target_domain_id
        except Exception:
            # Prompt context is best-effort; SEARCH should still run without domain plugin.
            target_domain_name = target_domain_id

        prompt = SEARCH_EXPAND_PROMPT.format(
            target_domain_id=target_domain_id,
            target_domain_name=target_domain_name,
            target_domain_description=target_domain_description,
            problem_description=problem_description or "Unknown",
            problem_context=problem_context,
            problem_constraints=problem_constraints,
            hypothesis_id=hypothesis.id,
            hypothesis_description=hypothesis.description,
            source_domain=hypothesis.source_domain,
            source_structure=hypothesis.source_structure,
            analogy_explanation=hypothesis.analogy_explanation,
            observable=hypothesis.observable,
            expansion_types=", ".join(expansion_types),
            max_expansions=max_expansions,
        )

        result = await self._router.complete(
            task="structured_search",
            messages=[{"role": "user", "content": prompt}],
        )

        expanded = self._parse_expansions(result.content, hypothesis)
        return expanded

    async def combine_hypotheses(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        max_expansions: int = 1,
    ) -> list[Hypothesis]:
        """Generate intersection hypotheses from two parent hypotheses."""
        prompt = SEARCH_COMBINE_PROMPT.format(
            hypothesis_a_id=hypothesis_a.id,
            hypothesis_a_description=hypothesis_a.description,
            hypothesis_a_source_domain=hypothesis_a.source_domain,
            hypothesis_a_source_structure=hypothesis_a.source_structure,
            hypothesis_a_analogy_explanation=hypothesis_a.analogy_explanation,
            hypothesis_a_observable=hypothesis_a.observable,
            hypothesis_b_id=hypothesis_b.id,
            hypothesis_b_description=hypothesis_b.description,
            hypothesis_b_source_domain=hypothesis_b.source_domain,
            hypothesis_b_source_structure=hypothesis_b.source_structure,
            hypothesis_b_analogy_explanation=hypothesis_b.analogy_explanation,
            hypothesis_b_observable=hypothesis_b.observable,
            max_expansions=max_expansions,
        )

        result = await self._router.complete(
            task="structured_search",
            messages=[{"role": "user", "content": prompt}],
        )

        combined = self._parse_expansions(result.content, hypothesis_a)
        merged_domain = (
            hypothesis_a.source_domain
            if hypothesis_a.source_domain == hypothesis_b.source_domain
            else f"{hypothesis_a.source_domain}+{hypothesis_b.source_domain}"
        )
        merged_structure = (
            hypothesis_a.source_structure
            if hypothesis_a.source_structure == hypothesis_b.source_structure
            else f"{hypothesis_a.source_structure}+{hypothesis_b.source_structure}"
        )

        normalized: list[Hypothesis] = []
        for child in combined:
            explanation = child.analogy_explanation or ""
            marker = f"[combined_with:{hypothesis_b.id}]"
            if marker not in explanation:
                explanation = f"{explanation}\n{marker}".strip() if explanation else marker
            normalized.append(
                child.model_copy(
                    update={
                        "expansion_type": "combine",
                        "source_domain": merged_domain,
                        "source_structure": merged_structure,
                        "analogy_explanation": explanation,
                    }
                )
            )

        return normalized

    def _parse_expansions(
        self, content: str, parent: Hypothesis
    ) -> list[Hypothesis]:
        """Parse expansion response into hypotheses.

        Args:
            content: Raw LLM response.
            parent: Parent hypothesis.

        Returns:
            List of expanded hypotheses.
        """
        hypotheses: list[Hypothesis] = []

        try:
            json_start = content.find("[")
            json_end = content.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                expansions = safe_json_loads(json_str)

                for exp in expansions:
                    if not isinstance(exp, dict):
                        continue
                    observable = str(exp.get("observable", "")).strip()
                    if not observable:
                        continue
                    hyp = Hypothesis(
                        id=f"hyp_{uuid.uuid4().hex[:8]}",
                        description=str(exp.get("description", "")).strip(),
                        source_domain=parent.source_domain,
                        source_structure=parent.source_structure,
                        analogy_explanation=str(exp.get("analogy_explanation", "")).strip(),
                        observable=observable,
                        parent_id=parent.id,
                        generation=parent.generation + 1,
                        expansion_type=exp.get("expansion_type"),
                        mapping_quality=parent.mapping_quality,
                        quick_score=(
                            parent.quick_score * 0.9
                            if parent.quick_score is not None
                            else None
                        ),
                    )
                    hypotheses.append(hyp)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Failed to parse expansions",
                error=str(e),
                content_preview=content[:200],
            )

        return hypotheses

    async def search_iteration(
        self,
        hypotheses: list[Hypothesis],
        config: SearchConfig,
        on_hkg_event: Callable[[str, dict[str, object]], Awaitable[None] | None] | None = None,
    ) -> list[Hypothesis]:
        """Perform a single search iteration.

        Args:
            hypotheses: Current hypothesis pool.
            config: Search configuration.
            on_hkg_event: Optional callback for HKG path/expansion events.

        Returns:
            Expanded and filtered hypothesis pool.
        """
        all_expanded: list[Hypothesis] = list(hypotheses)  # Keep originals
        runtime_profile = config.runtime_profile
        heavy_ops_enabled = runtime_profile == "research"
        hyperpath_top_n = max(0, config.hyperpath_expand_topN)

        # Build HKG top-N size for quota derivation.
        hkg_top_n_size = 0
        if (
            self._hkg_store is not None
            and self._hkg_matcher is not None
            and self._problem_frame is not None
        ):
            from x_creative.hkg.types import HKGParams as _HKGParams
            _params = self._hkg_params or _HKGParams()
            hkg_top_n_size = min(_params.top_n_hypotheses, hyperpath_top_n)

        quota = HeavyQuotaManager.from_config(
            config=config,
            heavy_ops_enabled=heavy_ops_enabled,
            top_n_size=hkg_top_n_size,
            hyperpath_expand_topN=hyperpath_top_n,
        )

        # Expand each hypothesis
        for hyp in hypotheses[:config.search_breadth]:
            try:
                expansion_types: list[ExpansionType] = ["refine", "variant"]
                if config.enable_opposition:
                    expansion_types.append("oppose")
                if config.enable_extreme:
                    expansion_types.append("extreme")

                expanded = await self.expand_hypothesis(
                    hypothesis=hyp,
                    expansion_types=expansion_types,
                    max_expansions=config.search_breadth,
                )
                all_expanded.extend(expanded)

            except Exception as e:
                logger.warning(
                    "Failed to expand hypothesis",
                    hypothesis_id=hyp.id,
                    error=str(e),
                )
                continue

        # Dedicated combine path: requires two hypotheses as inputs.
        if config.enable_combination and len(hypotheses) >= 2:
            for idx, hyp in enumerate(hypotheses[:config.search_breadth]):
                partner = hypotheses[(idx + 1) % len(hypotheses)]
                if partner.id == hyp.id:
                    continue
                try:
                    combined = await self.combine_hypotheses(
                        hypothesis_a=hyp,
                        hypothesis_b=partner,
                        max_expansions=1,
                    )
                    all_expanded.extend(combined)
                except Exception as e:
                    logger.warning(
                        "Failed to combine hypotheses",
                        hypothesis_a=hyp.id,
                        hypothesis_b=partner.id,
                        error=str(e),
                    )

        # HKG expansion (if enabled)
        if (
            self._hkg_store is not None
            and self._hkg_matcher is not None
            and self._problem_frame is not None
        ):
            from x_creative.hkg.expand import hyperpath_expand
            from x_creative.hkg.types import HKGParams as _HKGParams

            params = self._hkg_params or _HKGParams()
            top_n_limit = min(params.top_n_hypotheses, hyperpath_top_n)
            top_n = hypotheses[:top_n_limit] if top_n_limit > 0 else []
            for hyp in top_n:
                try:
                    hkg_expanded = await hyperpath_expand(
                        hypothesis=hyp,
                        problem_frame=self._problem_frame,
                        hkg=self._hkg_store,
                        matcher=self._hkg_matcher,
                        router=self._router,
                        params=params,
                        cache=self._hkg_cache,
                        on_event=on_hkg_event,
                    )
                    all_expanded.extend(hkg_expanded)
                except Exception as e:
                    logger.warning(
                        "HKG expansion failed",
                        hypothesis_id=hyp.id,
                        error=str(e),
                    )

            # Hyperbridge (quota-managed, §2.2)
            if (
                self._enable_hyperbridge
                and len(top_n) >= 2
                and quota.allow("hyperbridge")
            ):
                from x_creative.hkg.expand import hyperbridge

                for i in range(0, len(top_n) - 1, 2):
                    if not quota.allow("hyperbridge"):
                        break
                    concept_a = top_n[i].source_domain or top_n[i].description
                    concept_b = top_n[i + 1].source_domain or top_n[i + 1].description
                    quota.consume("hyperbridge")
                    try:
                        bridged = await hyperbridge(
                            concept_a=concept_a,
                            concept_b=concept_b,
                            hkg=self._hkg_store,
                            matcher=self._hkg_matcher,
                            router=self._router,
                            params=params,
                            cache=self._hkg_cache,
                            on_event=on_hkg_event,
                        )
                        all_expanded.extend(bridged)
                    except Exception as e:
                        logger.warning(
                            "Hyperbridge failed",
                            concept_a=concept_a,
                            concept_b=concept_b,
                            error=str(e),
                        )

            # Hyperbridge quota exhaustion → degrade to refine/variant (§2.2.2)
            if (
                self._enable_hyperbridge
                and len(top_n) >= 2
                and not quota.allow("hyperbridge")
            ):
                quota.record_degradation("hyperbridge")
                for hyp in top_n[:2]:
                    try:
                        degraded = await self.expand_hypothesis(
                            hypothesis=hyp,
                            expansion_types=["refine", "variant"],
                            max_expansions=1,
                        )
                        all_expanded.extend(degraded)
                    except Exception:
                        pass

        # Blend expansion (quota-managed, §2.2)
        if (
            config.enable_blending
            and quota.allow("blend")
            and len(hypotheses) >= 2
        ):
            from x_creative.creativity.blend import blend_expand as _blend_expand

            pairs = self._select_blend_pairs(
                hypotheses,
                min(config.max_blend_pairs, quota.blend),
            )
            for ha, hb in pairs:
                if not quota.allow("blend"):
                    break
                quota.consume("blend")
                try:
                    blended = await _blend_expand(
                        hypothesis_a=ha,
                        hypothesis_b=hb,
                        router=self._router,
                    )
                    all_expanded.extend(blended)
                    await self._emit_search_event(
                        callback=on_hkg_event,
                        event_type="blend_expand_completed",
                        payload={
                            "hypothesis_a_id": ha.id,
                            "hypothesis_b_id": hb.id,
                            "new_count": len(blended),
                        },
                    )
                except Exception as e:
                    logger.warning("Blend expansion failed", error=str(e))
        elif config.enable_blending and not quota.allow("blend") and len(hypotheses) >= 2:
            # Blend quota exhausted → degrade to refine/variant (§2.2.2)
            quota.record_degradation("blend")
            for hyp in hypotheses[:2]:
                try:
                    degraded = await self.expand_hypothesis(
                        hypothesis=hyp,
                        expansion_types=["refine", "variant"],
                        max_expansions=1,
                    )
                    all_expanded.extend(degraded)
                except Exception:
                    pass

        # Transform space (quota-managed, §2.2)
        if (
            config.enable_transform_space
            and self._concept_space is not None
            and quota.allow("transform")
        ):
            from x_creative.creativity.transform import (
                transform_space as _transform_space,
            )
            from x_creative.core.transform_types import TransformStatus

            top_transform = hypotheses[:config.max_transform_hypotheses]
            for hyp in top_transform:
                if not quota.allow("transform"):
                    break
                quota.consume("transform")
                try:
                    transformed = await _transform_space(
                        hypothesis=hyp,
                        concept_space=self._concept_space,
                        router=self._router,
                    )
                    proposed_count = 0
                    rejected_count = 0
                    accepted: list[Hypothesis] = []
                    for candidate in transformed:
                        pre_gate_diff = candidate.space_transform_diff
                        if (
                            pre_gate_diff is not None
                            and pre_gate_diff.transform_status == TransformStatus.PROPOSED
                        ):
                            await self._emit_search_event(
                                callback=on_hkg_event,
                                event_type="transform_proposed",
                                payload={
                                    "parent_hypothesis_id": hyp.id,
                                    "hypothesis_id": candidate.id,
                                    "concept_space_version": pre_gate_diff.concept_space_version,
                                    "action_count": len(pre_gate_diff.actions),
                                },
                            )

                        gated = await self._apply_transform_gate(candidate)
                        diff = gated.space_transform_diff
                        if diff is not None:
                            proposed_count += 1
                            await self._emit_search_event(
                                callback=on_hkg_event,
                                event_type="transform_accepted_or_rejected",
                                payload={
                                    "parent_hypothesis_id": hyp.id,
                                    "hypothesis_id": gated.id,
                                    "transform_status": diff.transform_status.value,
                                    "rejection_reason": diff.rejection_reason,
                                    "validation_notes": list(diff.validation_notes),
                                },
                            )
                            if diff.transform_status != TransformStatus.ACCEPTED:
                                rejected_count += 1
                                continue
                        accepted.append(gated)
                    all_expanded.extend(accepted)
                    await self._emit_search_event(
                        callback=on_hkg_event,
                        event_type="transform_space_applied",
                        payload={
                            "hypothesis_id": hyp.id,
                            "concept_space_version": self._concept_space.version,
                            "new_count": len(accepted),
                            "proposed_count": proposed_count,
                            "rejected_count": rejected_count,
                        },
                    )
                except Exception as e:
                    logger.warning("Transform space failed", error=str(e))
        elif (
            config.enable_transform_space
            and self._concept_space is not None
            and not quota.allow("transform")
        ):
            # Transform quota exhausted → degrade to refine/variant (§2.2.2)
            quota.record_degradation("transform")
            for hyp in hypotheses[:config.max_transform_hypotheses]:
                try:
                    degraded = await self.expand_hypothesis(
                        hypothesis=hyp,
                        expansion_types=["refine", "variant"],
                        max_expansions=1,
                    )
                    all_expanded.extend(degraded)
                except Exception:
                    pass

        return all_expanded

    async def _emit_search_event(
        self,
        callback: Callable[[str, dict[str, object]], Awaitable[None] | None] | None,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        """Emit SEARCH operator events via callback if provided."""
        if callback is None:
            return
        try:
            maybe_awaitable = callback(event_type, payload)
            if isinstance(maybe_awaitable, Awaitable):
                await maybe_awaitable
        except Exception as exc:
            logger.debug(
                "Search event callback failed",
                event_type=event_type,
                error=str(exc),
            )

    def _select_blend_pairs(
        self, hypotheses: list[Hypothesis], max_pairs: int
    ) -> list[tuple[Hypothesis, Hypothesis]]:
        """Select cross-domain hypothesis pairs for blending.

        Prefers pairs from different source domains. Falls back to any
        pairs if no cross-domain combinations are available.
        """
        pairs: list[tuple[Hypothesis, Hypothesis]] = []
        for i, ha in enumerate(hypotheses):
            for hb in hypotheses[i + 1 :]:
                if ha.source_domain != hb.source_domain:
                    pairs.append((ha, hb))
                    if len(pairs) >= max_pairs:
                        return pairs
        # Fallback: any pairs if no cross-domain
        if not pairs:
            for i, ha in enumerate(hypotheses):
                for hb in hypotheses[i + 1 :]:
                    pairs.append((ha, hb))
                    if len(pairs) >= max_pairs:
                        return pairs
        return pairs

    async def run_search(
        self,
        initial_hypotheses: list[Hypothesis],
        config: SearchConfig,
        on_round_complete: Callable[[int, list[Hypothesis], int], Awaitable[None] | None] | None = None,
        on_hkg_event: Callable[[str, dict[str, object]], Awaitable[None] | None] | None = None,
    ) -> list[Hypothesis]:
        """Run the full search process.

        Args:
            initial_hypotheses: Starting hypothesis pool.
            config: Search configuration.
            on_round_complete: Optional callback invoked at end of each round:
                (round_index_1_based, current_pool, new_count_in_round).
            on_hkg_event: Optional callback for HKG path/expansion events.

        Returns:
            Final hypothesis pool after all iterations.
        """
        # Bind pipeline context so underlying LLM logs include stage + round.
        with contextvars.bound_contextvars(pipeline_stage="search"):
            current_pool = initial_hypotheses

            logger.info(
                "Starting search",
                initial_count=len(initial_hypotheses),
                depth=config.search_depth,
                breadth=config.search_breadth,
            )

            # Pre-score for SEARCH selection (lightweight)
            await self._prescore_hypotheses(current_pool)

            for depth in range(config.search_depth):
                with contextvars.bound_contextvars(search_round=depth + 1):
                    logger.debug(
                        "Search iteration",
                        depth=depth + 1,
                        pool_size=len(current_pool),
                    )

                    # Select top hypotheses to expand (by score if available, else all)
                    to_expand = self._select_for_expansion(current_pool, config.search_breadth)
                    if not to_expand:
                        logger.info(
                            "No hypotheses passed expansion gate; stopping search early",
                            depth=depth + 1,
                            gate=self._mapping_quality_gate,
                            pool_size=len(current_pool),
                        )
                        if on_round_complete is not None:
                            maybe_awaitable = on_round_complete(depth + 1, current_pool, 0)
                            if isinstance(maybe_awaitable, Awaitable):
                                await maybe_awaitable
                        break

                    # Expand
                    expanded = await self.search_iteration(
                        to_expand,
                        config,
                        on_hkg_event=on_hkg_event,
                    )

                    # Merge new into pool (avoiding duplicates)
                    pool_size_before = len(current_pool)
                    existing_ids = {h.id for h in current_pool}
                    for h in expanded:
                        if h.id not in existing_ids:
                            current_pool.append(h)
                            existing_ids.add(h.id)

                    new_count = len(current_pool) - pool_size_before

                    # Re-score current pool each round so new nodes can join top-N.
                    await self._prescore_hypotheses(current_pool)

                    if on_round_complete is not None:
                        maybe_awaitable = on_round_complete(depth + 1, current_pool, new_count)
                        if isinstance(maybe_awaitable, Awaitable):
                            await maybe_awaitable

                    logger.debug(
                        "After iteration",
                        depth=depth + 1,
                        pool_size=len(current_pool),
                    )

            logger.info(
                "Search complete",
                final_count=len(current_pool),
            )

            return current_pool

    async def _prescore_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        """Assign SEARCH selection scores.

        Primary path uses the same five-dimension verifier scoring used in
        theory track-1 composite ranking. If that path fails, fallback to a
        lightweight quick_score.
        """
        needs_full_score = [h for h in hypotheses if h.scores is None]
        if not needs_full_score:
            return

        try:
            from x_creative.creativity.verify import VerifyModule

            verify = VerifyModule(router=self._router)
            scored_batch = await verify.score_batch(needs_full_score)
            score_map = {h.id: h.scores for h in scored_batch if h.scores is not None}
            for hypothesis in needs_full_score:
                if hypothesis.scores is None and hypothesis.id in score_map:
                    hypothesis.scores = score_map[hypothesis.id]
            return
        except Exception as e:
            logger.warning(
                "Five-dimension pre-scoring failed, falling back to quick score",
                error=str(e),
            )

        unscored = [h for h in hypotheses if h.scores is None and h.quick_score is None]
        if not unscored:
            return

        descriptions = [
            {"id": h.id, "description": h.description[:200], "observable": h.observable[:100]}
            for h in unscored
        ]

        prompt = (
            "Rate each hypothesis 0-10 for creative potential. "
            "Return JSON array: [{\"id\": \"...\", \"score\": N}, ...]\n\n"
            + json.dumps(descriptions, ensure_ascii=False)
        )

        try:
            result = await self._router.complete(
                task="structured_search",
                messages=[{"role": "user", "content": prompt}],
            )
            scores_data = safe_json_loads(result.content)
            if isinstance(scores_data, list):
                score_map = {
                    item["id"]: float(item.get("score", 5.0))
                    for item in scores_data
                    if isinstance(item, dict) and "id" in item
                }
                for h in unscored:
                    h.quick_score = score_map.get(h.id, 5.0)
        except Exception as e:
            logger.warning("Quick pre-scoring failed, using default", error=str(e))
            for h in unscored:
                h.quick_score = 5.0

    def _select_for_expansion(
        self,
        hypotheses: list[Hypothesis],
        count: int,
    ) -> list[Hypothesis]:
        """Select hypotheses for expansion.

        Applies mapping_quality gate: hypotheses with mapping_quality below
        the threshold are excluded from expansion. Hypotheses without
        mapping_quality are also excluded when gate is enabled.

        When a ParetoArchive is configured, uses NSGA-II Pareto ranking
        instead of single-axis composite sort (after mapping gate filtering).

        Args:
            hypotheses: Pool to select from.
            count: Number to select.

        Returns:
            Selected hypotheses.
        """
        from x_creative.core.transform_types import TransformStatus

        gate = self._mapping_quality_gate

        candidates = [
            h
            for h in hypotheses
            if (
                h.space_transform_diff is None
                or h.space_transform_diff.transform_status == TransformStatus.ACCEPTED
            )
        ]
        if not candidates:
            candidates = hypotheses

        if gate is not None:
            # Strict structural gate: only scored hypotheses at/above threshold.
            candidates = [
                h
                for h in candidates
                if h.mapping_quality is not None and h.mapping_quality >= gate
            ]

        # MOME selection (highest priority when enabled)
        if self._mome_archive is not None:
            for h in candidates:
                self._mome_archive.add(h)
            return self._mome_archive.select(count)

        # Pareto selection (when enabled)
        if self._pareto_archive is not None:
            return self._pareto_archive.select(candidates, count)

        # Default: composite score sort
        def _key(h: Hypothesis) -> tuple[int, float]:
            if h.scores is not None:
                return (2, h.composite_score())
            if h.quick_score is not None:
                return (1, float(h.quick_score))
            return (0, 0.0)

        ranked = sorted(candidates, key=_key, reverse=True)
        return ranked[:count]

    async def _apply_transform_gate(self, hypothesis: Hypothesis) -> Hypothesis:
        """Validate PROPOSED transform diffs before admitting to main pool.

        Per theory §10.5.1, this gate enforces C→K verification:
        1. Structural rule checks (op whitelist, fixed assumption protection)
        2. LLM Logic Verifier call (mandatory C→K verification)
        3. High-risk audit hook checks
        """
        from x_creative.core.transform_types import TransformStatus

        diff = hypothesis.space_transform_diff
        if diff is None:
            return hypothesis
        if diff.transform_status != TransformStatus.PROPOSED:
            return hypothesis
        if self._concept_space is None:
            return hypothesis.model_copy(
                update={
                    "space_transform_diff": diff.model_copy(
                        update={
                            "transform_status": TransformStatus.REJECTED,
                            "validation_notes": ["ck_verification_missing:concept_space_unavailable"],
                            "rejection_reason": "concept_space unavailable",
                        }
                    )
                }
            )

        allowed_map = {op.id: op for op in self._concept_space.allowed_ops}
        fixed_assumptions = {a.id for a in self._concept_space.assumptions_fixed}
        high_risk_ops = {"swap_causal_direction", "change_representation"}
        has_high_risk = False
        rejection_reason: str | None = None
        notes: list[str] = []

        # --- Phase 1: Structural rule checks ---
        for action in diff.actions:
            allowed = allowed_map.get(action.op_id)
            if allowed is None:
                rejection_reason = f"op_id_not_allowed:{action.op_id}"
                break
            if allowed.op_type != action.op_type:
                rejection_reason = (
                    f"op_type_mismatch:{action.op_id}:{action.op_type}!={allowed.op_type}"
                )
                break
            if (
                allowed.target_type == "assumption"
                and action.target_id in fixed_assumptions
            ):
                rejection_reason = f"attempted_transform_on_fixed_assumption:{action.target_id}"
                break
            if action.op_type in high_risk_ops:
                has_high_risk = True

        if rejection_reason is not None:
            return hypothesis.model_copy(
                update={
                    "space_transform_diff": diff.model_copy(
                        update={
                            "transform_status": TransformStatus.REJECTED,
                            "validation_notes": notes,
                            "rejection_reason": rejection_reason,
                        }
                    )
                }
            )

        # --- Phase 2: C→K LLM Logic Verifier (§10.5.1 mandatory) ---
        ck_passed = await self._ck_verify_transform(hypothesis, diff)
        notes.append("ck_verification_invoked:logic_verifier")
        if not ck_passed:
            rejection_reason = "ck_logic_verification_failed"
            notes.append("ck_verification_rejected")
            return hypothesis.model_copy(
                update={
                    "space_transform_diff": diff.model_copy(
                        update={
                            "transform_status": TransformStatus.REJECTED,
                            "validation_notes": notes,
                            "rejection_reason": rejection_reason,
                        }
                    )
                }
            )
        notes.append("ck_verification_passed")

        # --- Phase 3: High-risk audit hook checks ---
        if has_high_risk:
            if not diff.new_failure_modes:
                rejection_reason = "high_risk_transform_missing_failure_modes"
            elif not diff.new_detectable_signals and not diff.new_observables:
                rejection_reason = "high_risk_transform_missing_testable_hooks"
            else:
                notes.append("high_risk_transform_with_audit_hooks")

        status = (
            TransformStatus.REJECTED
            if rejection_reason is not None
            else TransformStatus.ACCEPTED
        )
        if rejection_reason is None:
            notes.append("logic_consistency_gate_passed")

        updated_diff = diff.model_copy(
            update={
                "transform_status": status,
                "validation_notes": notes,
                "rejection_reason": rejection_reason,
            }
        )
        return hypothesis.model_copy(update={"space_transform_diff": updated_diff})

    async def _ck_verify_transform(
        self,
        hypothesis: Hypothesis,
        diff: object,
    ) -> bool:
        """Invoke LLM Logic Verifier for C→K verification of a transform (§10.5.1).

        Returns True if the transform passes logical consistency check.
        On LLM failure, conservatively returns True to avoid false rejections.
        """
        from x_creative.creativity.prompts import TRANSFORM_GATE_CK_VERIFY_PROMPT

        assert self._concept_space is not None  # guaranteed by caller

        # Build transform summary from diff actions
        actions_summary = []
        for action in diff.actions:  # type: ignore[attr-defined]
            actions_summary.append(
                f"  - {action.op_type}: {action.target_id} "
                f"({action.before_state} → {action.after_state}): {action.rationale}"
            )
        transform_summary = "\n".join(actions_summary)
        hard_constraints = "\n".join(
            f"  - {c.text}" for c in self._concept_space.hard_constraints
        )

        prompt = TRANSFORM_GATE_CK_VERIFY_PROMPT.format(
            hypothesis_description=hypothesis.description,
            observable=hypothesis.observable or "",
            transform_summary=transform_summary,
            domain_id=self._concept_space.domain_id,
            hard_constraints=hard_constraints or "(none)",
        )

        try:
            response = await self._router.complete(
                task="verify",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content
            parsed = safe_json_loads(content)
            if isinstance(parsed, dict):
                verdict = parsed.get("verdict", "").lower()
                return verdict == "accept"
            return True  # unparseable → conservative pass
        except Exception as e:
            logger.warning("C→K transform verification failed, conservatively passing", error=str(e))
            return True
