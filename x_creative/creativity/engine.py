"""Creativity Engine - the main orchestrator for hypothesis generation."""

import asyncio
import inspect
import time
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from structlog import contextvars

from x_creative.config.settings import get_settings
from x_creative.core.types import (
    Domain,
    Hypothesis,
    LogicVerdict,
    NoveltyVerdict,
    ProblemFrame,
    SearchConfig,
    VerifyStatus,
    VerifiedHypothesis,
)
from x_creative.creativity.biso import BISOModule
from x_creative.creativity.search import SearchModule
from x_creative.creativity.verify import VerifyModule
from x_creative.llm.router import ModelRouter
from x_creative.verify import LogicVerifier, NoveltyVerifier, SearchValidator

logger = structlog.get_logger()

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class CreativityEngine:
    """Main engine for creativity-driven hypothesis generation.

    Orchestrates the BISO -> SEARCH -> VERIFY pipeline:
    1. BISO: Generate analogies from distant domains
    2. SEARCH: Expand and explore the hypothesis space
    3. VERIFY: Score and filter hypotheses
    """

    def __init__(
        self,
        router: ModelRouter | None = None,
        biso: BISOModule | None = None,
        search: SearchModule | None = None,
        verify: VerifyModule | None = None,
        logic_verifier: LogicVerifier | None = None,
        novelty_verifier: NoveltyVerifier | None = None,
        search_validator: SearchValidator | None = None,
    ) -> None:
        """Initialize the Creativity Engine.

        Args:
            router: Shared model router. Creates one if not provided.
            biso: BISO module. Creates one if not provided.
            search: Search module. Creates one if not provided.
            verify: Verify module. Creates one if not provided.
            logic_verifier: Logic verifier for dual-model verification. Creates one if not provided.
            novelty_verifier: Novelty verifier for dual-model verification. Creates one if not provided.
            search_validator: Search validator for web-based novelty verification. Creates one if not provided.
        """
        self._router = router or ModelRouter()
        self._settings = get_settings()
        self._concept_space_cache: dict[str, object | None] = {}
        self._biso = biso or BISOModule(router=self._router)
        mapping_quality_gate = (
            self._settings.mapping_quality_gate_threshold
            if self._settings.mapping_quality_gate_enabled
            else None
        )

        # HKG components (if enabled)
        self._hkg_store = None
        self._hkg_matcher = None
        self._hkg_embedding_client = None
        if self._settings.hkg_enabled and self._settings.hkg_store_path:
            try:
                from x_creative.hkg.store import HypergraphStore
                from x_creative.hkg.matcher import NodeMatcher
                from x_creative.hkg.embeddings import EmbeddingClient, NodeEmbeddingIndex

                self._hkg_store = HypergraphStore.load(self._settings.hkg_store_path)
                embedding_index = None
                embedding_provider = self._settings.hkg_embedding_provider.lower()
                embedding_api_key = ""
                embedding_base_url = ""

                if embedding_provider == "openrouter":
                    embedding_api_key = (
                        self._settings.openrouter.api_key.get_secret_value()
                    )
                    embedding_base_url = self._settings.openrouter.base_url
                elif embedding_provider == "yunwu":
                    embedding_api_key = self._settings.yunwu.api_key.get_secret_value()
                    embedding_base_url = self._settings.yunwu.base_url
                else:
                    logger.warning(
                        "Unknown HKG embedding provider; embedding matcher disabled",
                        provider=embedding_provider,
                    )

                if embedding_api_key:
                    self._hkg_embedding_client = EmbeddingClient(
                        api_key=embedding_api_key,
                        base_url=embedding_base_url,
                        model=self._settings.hkg_embedding_model,
                    )
                    index_path = self._settings.hkg_embedding_index_path
                    if index_path is not None and index_path.exists():
                        embedding_index = NodeEmbeddingIndex.load(index_path)
                    elif index_path is not None:
                        logger.info(
                            "HKG embedding index path not found; NodeMatcher will lazily build index",
                            index_path=str(index_path),
                        )

                self._hkg_matcher = NodeMatcher(
                    self._hkg_store,
                    embedding_client=self._hkg_embedding_client,
                    embedding_index=embedding_index,
                )
                logger.info(
                    "HKG initialized",
                    store_path=str(self._settings.hkg_store_path),
                    embedding_enabled=bool(self._hkg_embedding_client),
                    embedding_index_loaded=bool(embedding_index is not None),
                    **self._hkg_store.stats(),
                )
            except Exception as e:
                logger.warning("Failed to initialize HKG", error=str(e))

        if search is not None:
            self._search = search
        elif self._hkg_store is not None and self._hkg_matcher is not None:
            from x_creative.hkg.types import HKGParams
            self._search = SearchModule(
                router=self._router,
                hkg_store=self._hkg_store,
                hkg_matcher=self._hkg_matcher,
                hkg_params=HKGParams(
                    K=self._settings.hkg_K,
                    IS=self._settings.hkg_IS,
                    max_len=self._settings.hkg_max_len,
                    matcher=self._settings.hkg_matcher,
                    top_n_hypotheses=self._settings.hkg_top_n_hypotheses,
                ),
                enable_hyperbridge=self._settings.hkg_enable_hyperbridge,
                mapping_quality_gate=mapping_quality_gate,
            )
        else:
            self._search = SearchModule(
                router=self._router,
                mapping_quality_gate=mapping_quality_gate,
            )

        # Wire ParetoArchive when feature flag is enabled
        if self._settings.pareto_selection_enabled:
            from x_creative.creativity.pareto import ParetoArchive
            self._search._pareto_archive = ParetoArchive(
                wn_min=self._settings.pareto_wn_min,
                wn_max=self._settings.pareto_wn_max,
                gamma=self._settings.pareto_gamma,
                num_bins=self._settings.pareto_novelty_bins,
            )

        # Wire MOMEArchive when feature flag is enabled
        if self._settings.mome_enabled:
            from x_creative.creativity.mome import MOMEArchive

            self._search._mome_archive = MOMEArchive(
                bd_schema=self._default_bd_schema(),
                cell_capacity=self._settings.mome_cell_capacity,
            )

        self._verify = verify or VerifyModule(router=self._router)

        # Dual-model verification components
        self._logic_verifier = logic_verifier or LogicVerifier(
            router=self._router,
            num_samples=self._settings.multi_sample_evaluations,
            position_bias_confidence_factor=self._settings.position_bias_confidence_factor,
        )
        self._novelty_verifier = novelty_verifier or NoveltyVerifier(router=self._router)
        self._search_validator = search_validator or SearchValidator()

    async def generate(
        self,
        problem: ProblemFrame,
        config: SearchConfig | None = None,
        source_domains: list[Domain] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses for a research problem.

        This is the main entry point for the creativity pipeline.

        Args:
            problem: The research problem framing.
            config: Search configuration. Uses defaults if not provided.

        Returns:
            List of scored and sorted hypotheses.
        """
        if config is None:
            config = SearchConfig(
                num_hypotheses=self._settings.default_num_hypotheses,
                search_depth=self._settings.default_search_depth,
                enable_extreme=self._settings.enable_extreme,
                enable_blending=self._settings.enable_blending,
                enable_transform_space=self._settings.enable_transform_space,
                max_blend_pairs=self._settings.max_blend_pairs,
                max_transform_hypotheses=self._settings.max_transform_hypotheses,
                runtime_profile=(
                    "research"
                    if str(self._settings.runtime_profile).lower() == "research"
                    else "interactive"
                ),
                blend_expand_budget_per_round=self._settings.blend_expand_budget_per_round,
                transform_space_budget_per_round=self._settings.transform_space_budget_per_round,
                hyperpath_expand_topN=self._settings.hyperpath_expand_topN,
            )

        logger.info(
            "Starting creativity pipeline",
            problem=problem.description[:50],
            target_hypotheses=config.num_hypotheses,
        )

        async def _report(event: str, payload: dict[str, Any]) -> None:
            cb = progress_callback
            if cb is None:
                return
            try:
                maybe_awaitable = cb(event, payload)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                logger.debug(
                    "Progress callback failed",
                    event=event,
                    error=str(exc),
                )

        # Phase 1: BISO - Generate raw analogies
        with contextvars.bound_contextvars(pipeline_stage="biso"):
            logger.info("Phase 1: BISO - Generating analogies from distant domains")
            num_per_domain = max(3, config.num_hypotheses // 15)
            started_payload: dict[str, Any] = {
                "pipeline_stage": "biso",
                "num_per_domain": num_per_domain,
            }
            if source_domains is not None:
                started_payload["total_domains"] = len(source_domains)
            await _report("biso_started", started_payload)

            async def _on_domain_complete(
                domain_id: str,
                completed: int,
                total: int,
                generated: int,
            ) -> None:
                await _report(
                    "biso_domain_completed",
                    {
                        "pipeline_stage": "biso",
                        "domain_id": domain_id,
                        "completed": completed,
                        "total": total,
                        "generated": generated,
                    },
                )

            raw_analogies = await self._biso.generate_all_analogies(
                problem=problem,
                num_per_domain=num_per_domain,  # Spread across domains
                max_concurrency=self._settings.biso_max_concurrency,
                source_domains=source_domains,
                on_domain_complete=_on_domain_complete,
            )

            logger.info(
                "BISO complete",
                raw_analogies=len(raw_analogies),
            )
            await _report(
                "biso_completed",
                {
                    "pipeline_stage": "biso",
                    "hypothesis_count": len(raw_analogies),
                },
            )
            await self._score_mapping_quality(raw_analogies)

            if not raw_analogies:
                logger.warning("No analogies generated, returning empty list")
                return []

        # Phase 2: SEARCH - Expand the hypothesis space
        with contextvars.bound_contextvars(pipeline_stage="search"):
            logger.info("Phase 2: SEARCH - Expanding hypothesis space")
            await _report(
                "search_started",
                {
                    "pipeline_stage": "search",
                    "total_rounds": int(getattr(config, "search_depth", 0) or 0),
                    "initial_count": len(raw_analogies),
                },
            )
            self._prepare_search_context(problem, config)
            async def _on_round_complete(round_idx: int, pool: list[Hypothesis], new_count: int) -> None:
                await _report(
                    "search_round_completed",
                    {
                        "pipeline_stage": "search",
                        "round_index": round_idx,
                        "total_rounds": int(getattr(config, "search_depth", 0) or 0),
                        "hypothesis_count": len(pool),
                        "new_count": new_count,
                    },
                )

            expanded = await self._search.run_search(
                initial_hypotheses=raw_analogies,
                config=config,
                on_round_complete=_on_round_complete,
            )

            logger.info(
                "SEARCH complete",
                expanded_count=len(expanded),
            )
            await _report(
                "search_completed",
                {
                    "pipeline_stage": "search",
                    "hypothesis_count": len(expanded),
                },
            )

        # Phase 3: VERIFY - Score hypotheses + dual-model verification
        with contextvars.bound_contextvars(pipeline_stage="verify"):
            logger.info("Phase 3: VERIFY - Scoring hypotheses")
            await _report(
                "verify_started",
                {
                    "pipeline_stage": "verify",
                    "score_total": len(expanded),
                },
            )

            async def _on_score_progress(done: int, total: int, hypothesis_id: str) -> None:
                await _report(
                    "verify_hypothesis_scored",
                    {
                        "pipeline_stage": "verify",
                        "phase": "scoring",
                        "completed": done,
                        "total": total,
                        "hypothesis_id": hypothesis_id,
                    },
                )

            scored = await self._verify.score_batch(
                expanded,
                on_progress=_on_score_progress,
            )
            filtered = self.filter_by_threshold(scored, threshold=config.prune_threshold)

            async def _on_dual_progress(done: int, total: int, hypothesis_id: str) -> None:
                await _report(
                    "verify_hypothesis_scored",
                    {
                        "pipeline_stage": "verify",
                        "phase": "dual_verify",
                        "completed": done,
                        "total": total,
                        "hypothesis_id": hypothesis_id,
                    },
                )

            verified = await self._verify_batch_dual_model(
                filtered,
                problem,
                on_progress=_on_dual_progress,
            )
            enriched = self._merge_dual_verification(filtered, verified)
            # Enforce VERIFY-stage hard gates (e.g., logic hard gate) on merged results.
            enriched = self.filter_by_threshold(enriched, threshold=config.prune_threshold)

            logger.info(
                "VERIFY complete",
                scored=len(scored),
                after_filter=len(filtered),
            )
            await _report(
                "verify_completed",
                {
                    "pipeline_stage": "verify",
                    "scored_count": len(scored),
                    "after_filter": len(filtered),
                },
            )

        # Sort by final_score first, fallback to weighted composite score
        sorted_hypotheses = self.sort_by_score(enriched)

        # Limit to target number
        final = sorted_hypotheses[: config.num_hypotheses]

        logger.info(
            "Creativity pipeline complete",
            final_count=len(final),
        )
        await _report(
            "pipeline_completed",
            {
                "pipeline_stage": "completed",
                "final_count": len(final),
            },
        )

        return final

    def sort_by_score(
        self,
        hypotheses: list[Hypothesis],
        descending: bool = True,
    ) -> list[Hypothesis]:
        """Sort hypotheses by dual-verify score first, then weighted composite.

        Unverified hypotheses use a three-way priority:
        1. MOME (when ``mome_enabled``): MAP-Elites grid with per-cell Pareto fronts
        2. Pareto (when ``pareto_selection_enabled``): NSGA-II rank + crowding distance
        3. Composite score fallback

        Verified hypotheses always rank first by final_score.

        Args:
            hypotheses: Hypotheses to sort.
            descending: If True, highest scores first.

        Returns:
            Sorted list.
        """
        # Partition: verified (has final_score) vs unverified
        verified = [h for h in hypotheses if h.final_score is not None]
        unverified = [h for h in hypotheses if h.final_score is None]

        # Verified: always sorted by final_score descending
        verified.sort(key=lambda h: float(h.final_score or 0.0), reverse=descending)

        # Unverified: MOME > Pareto > composite
        if self._settings.mome_enabled and unverified:
            from x_creative.creativity.mome import MOMEArchive

            archive = MOMEArchive(
                bd_schema=self._default_bd_schema(),
                cell_capacity=self._settings.mome_cell_capacity,
            )
            for h in unverified:
                archive.add(h)
            unverified = archive.select(len(unverified))
        elif self._settings.pareto_selection_enabled and unverified:
            from x_creative.creativity.pareto import ParetoArchive
            archive = ParetoArchive(
                wn_min=self._settings.pareto_wn_min,
                wn_max=self._settings.pareto_wn_max,
                gamma=self._settings.pareto_gamma,
                num_bins=self._settings.pareto_novelty_bins,
            )
            unverified = archive.select(unverified, len(unverified))
        else:
            unverified.sort(
                key=lambda h: self._selection_score(h),
                reverse=descending,
            )

        # Verified always rank first
        return verified + unverified

    async def score_and_verify_batch(
        self,
        hypotheses: list[Hypothesis],
        problem_frame: ProblemFrame | None = None,
        *,
        score_progress_callback: Callable[[int, int, str], Awaitable[None] | None] | None = None,
        dual_verify_progress_callback: Callable[[int, int, str], Awaitable[None] | None] | None = None,
    ) -> list[Hypothesis]:
        """Score hypotheses and optionally run dual-model verification."""
        scored = await self._verify.score_batch(
            hypotheses,
            on_progress=score_progress_callback,
        )
        if problem_frame is None:
            return scored
        verified = await self._verify_batch_dual_model(
            scored,
            problem_frame,
            on_progress=dual_verify_progress_callback,
        )
        return self._merge_dual_verification(scored, verified)

    def filter_by_threshold(
        self,
        hypotheses: list[Hypothesis],
        threshold: float = 5.0,
    ) -> list[Hypothesis]:
        """Filter hypotheses with logic hard gate + score threshold."""
        filtered: list[Hypothesis] = []
        for hypothesis in hypotheses:
            if hypothesis.logic_passed is False:
                continue
            score = (
                hypothesis.final_score
                if hypothesis.final_score is not None
                else self._selection_score(hypothesis)
            )
            if score >= threshold:
                filtered.append(hypothesis)
        return filtered

    @staticmethod
    def _default_bd_schema() -> "BDSchema":
        """Default BDSchema for MOME archive (cross-domain research)."""
        from x_creative.creativity.qd_types import BDSchema, GridConfig

        return BDSchema(
            version="1.0.0",
            grid_dimensions=[
                GridConfig(
                    name="mechanism_family",
                    dim_type="categorical",
                    labels=[
                        "structural_analogy", "process_transfer", "constraint_relaxation",
                        "conceptual_blend", "cross_domain", "other",
                    ],
                ),
                GridConfig(
                    name="data_granularity",
                    dim_type="categorical",
                    labels=["system", "component", "mechanism", "pattern"],
                ),
            ],
        )

    def _selection_score(self, hypothesis: Hypothesis) -> float:
        """Score used for pre-verification ranking/filtering.

        Base score is the five-dimension composite score.
        When HKG structural scoring is enabled, blend in structural evidence
        with a small configurable weight.
        """
        settings = self._settings
        base_score = hypothesis.composite_score(
            w_divergence=settings.score_weight_divergence,
            w_testability=settings.score_weight_testability,
            w_rationale=settings.score_weight_rationale,
            w_robustness=settings.score_weight_robustness,
            w_feasibility=settings.score_weight_feasibility,
        )

        if not settings.hkg_enable_structural_scoring:
            return base_score

        structural_score = 0.0
        if hypothesis.hkg_evidence is not None:
            from x_creative.hkg.scoring import structural_grounding_score

            scored = structural_grounding_score(hypothesis.hkg_evidence)
            structural_score = scored if scored is not None else 0.0

        weight = max(0.0, min(1.0, settings.hkg_structural_score_weight))
        return (1.0 - weight) * base_score + weight * structural_score

    async def _verify_batch_dual_model(
        self,
        hypotheses: list[Hypothesis],
        problem_frame: ProblemFrame,
        concurrency: int = 3,
        on_progress: Callable[[int, int, str], Awaitable[None] | None] | None = None,
    ) -> dict[str, VerifiedHypothesis]:
        """Run dual-model verification with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        verified_by_id: dict[str, VerifiedHypothesis] = {}
        total = len(hypotheses)
        completed = 0
        completed_lock = asyncio.Lock()

        async def _report(hypothesis_id: str) -> None:
            nonlocal completed
            if on_progress is None:
                return
            async with completed_lock:
                completed += 1
                current = completed
            try:
                maybe_awaitable = on_progress(current, total, hypothesis_id)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                logger.debug(
                    "Dual-model verification progress callback failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )

        async def _verify_one(hypothesis: Hypothesis) -> None:
            async with semaphore:
                try:
                    verified = await self.verify_hypothesis(hypothesis, problem_frame)
                    verified_by_id[hypothesis.id] = verified
                except Exception as e:
                    logger.warning(
                        "Dual-model verification failed",
                        hypothesis_id=hypothesis.id,
                        error=str(e),
                    )
                finally:
                    await _report(hypothesis.id)

        await asyncio.gather(*[_verify_one(h) for h in hypotheses])
        return verified_by_id

    def _merge_dual_verification(
        self,
        hypotheses: list[Hypothesis],
        verified_by_id: dict[str, VerifiedHypothesis],
    ) -> list[Hypothesis]:
        """Attach dual-model outputs back to Hypothesis objects."""
        merged: list[Hypothesis] = []
        for hypothesis in hypotheses:
            verified = verified_by_id.get(hypothesis.id)
            if verified is None:
                merged.append(hypothesis)
                continue
            merged.append(
                hypothesis.model_copy(
                    update={
                        "final_score": verified.final_score,
                        "logic_passed": verified.logic_verdict.passed,
                        "verify_status": verified.verify_status,
                        "judge_confidence": verified.judge_confidence,
                        "position_consistency": verified.position_consistency,
                        "injection_detected": verified.injection_detected,
                        "novelty_score": verified.novelty_verdict.score,
                        "structural_grounding_score": verified.structural_grounding_score,
                        "blend_network": verified.blend_network,
                    }
                )
            )
        return merged

    async def _score_mapping_quality(self, hypotheses: list[Hypothesis]) -> None:
        """Populate mapping_quality for hypotheses that carry mapping tables."""
        await self._score_mapping_quality_with_events(hypotheses)

    async def _score_mapping_quality_with_events(
        self,
        hypotheses: list[Hypothesis],
        on_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> None:
        """Populate mapping_quality and optionally emit scorer audit events."""
        candidates = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.mapping_quality is None and hypothesis.mapping_table
        ]
        if not candidates:
            return

        try:
            from x_creative.verify.mapping_scorer import MappingScorer
        except Exception:
            return

        scorer = MappingScorer(router=self._router, event_callback=on_event)
        semaphore = asyncio.Semaphore(4)

        async def _score_one(hypothesis: Hypothesis) -> None:
            async with semaphore:
                try:
                    result = await scorer.score(hypothesis)
                    hypothesis.mapping_quality = result.score
                except Exception as exc:
                    logger.debug(
                        "Mapping quality scoring failed",
                        hypothesis_id=hypothesis.id,
                        error=str(exc),
                    )

        await asyncio.gather(*[_score_one(hypothesis) for hypothesis in candidates])

    def set_hkg_enabled(self, enabled: bool) -> None:
        """Enable or disable HKG operators for subsequent runs.

        Disabling is immediate and always safe. Re-enabling requires the
        engine to have initialized HKG components successfully at startup.
        """
        if enabled:
            return

        self._hkg_store = None
        self._hkg_matcher = None

        if hasattr(self._search, "_hkg_store"):
            self._search._hkg_store = None
        if hasattr(self._search, "_hkg_matcher"):
            self._search._hkg_matcher = None
        if hasattr(self._search, "_hkg_params"):
            self._search._hkg_params = None
        if hasattr(self._search, "_enable_hyperbridge"):
            self._search._enable_hyperbridge = False
        if hasattr(self._search, "_hkg_cache"):
            self._search._hkg_cache = None

    async def generate_quick(
        self,
        problem: ProblemFrame,
        num_hypotheses: int = 10,
    ) -> list[Hypothesis]:
        """Quick generation with minimal exploration.

        Good for rapid prototyping and testing.

        Args:
            problem: The research problem.
            num_hypotheses: Target number of hypotheses.

        Returns:
            List of hypotheses.
        """
        config = SearchConfig(
            num_hypotheses=num_hypotheses,
            search_depth=1,
            search_breadth=3,
            prune_threshold=4.0,
        )
        return await self.generate(problem, config)

    async def generate_hypotheses(
        self,
        problem_frame: ProblemFrame,
        num_hypotheses: int = 50,
        verify: bool = True,
        source_domains: list[Domain] | None = None,
    ) -> list[VerifiedHypothesis]:
        """Generate and optionally verify hypotheses.

        This method generates hypotheses using the BISO phase and then
        verifies them using dual-model verification (logic + novelty).

        Args:
            problem_frame: The research problem framing.
            num_hypotheses: Target number of hypotheses to generate.
            verify: If True, run dual-model verification on each hypothesis.

        Returns:
            List of VerifiedHypothesis objects if verify=True,
            or VerifiedHypothesis with empty verdicts if verify=False.
        """
        logger.info(
            "Generating hypotheses",
            target=num_hypotheses,
            verify=verify,
        )

        # Phase 1: BISO - Generate raw analogies
        logger.info("Phase 1: BISO - Generating analogies from distant domains")
        raw_hypotheses = await self._biso.generate_all_analogies(
            problem=problem_frame,
            num_per_domain=max(3, num_hypotheses // 15),
            max_concurrency=self._settings.biso_max_concurrency,
            source_domains=source_domains,
        )

        logger.info(
            "BISO complete",
            raw_count=len(raw_hypotheses),
        )
        await self._score_mapping_quality(raw_hypotheses)

        if not raw_hypotheses:
            logger.warning("No hypotheses generated, returning empty list")
            return []

        self._prepare_search_context(
            problem_frame,
            SearchConfig(enable_transform_space=self._settings.enable_transform_space),
        )

        # Limit to target number
        raw_hypotheses = raw_hypotheses[:num_hypotheses]

        # Phase 2: Verification (if enabled)
        if verify:
            logger.info("Phase 2: Dual-model verification")
            verified_hypotheses: list[VerifiedHypothesis] = []

            for hypothesis in raw_hypotheses:
                try:
                    verified = await self.verify_hypothesis(hypothesis, problem_frame)
                    verified_hypotheses.append(verified)
                except Exception as e:
                    logger.warning(
                        "Verification failed for hypothesis",
                        hypothesis_id=hypothesis.id,
                        error=str(e),
                    )
                    # Create a failed verification result
                    verified = self._create_failed_verification(hypothesis, str(e))
                    verified_hypotheses.append(verified)

            logger.info(
                "Verification complete",
                verified_count=len(verified_hypotheses),
                passed_count=sum(1 for v in verified_hypotheses if v.logic_verdict.passed),
            )

            # Sort by final score
            verified_hypotheses.sort(key=lambda v: v.final_score, reverse=True)
            return verified_hypotheses

        # Skip verification - create placeholder verified hypotheses
        logger.info("Skipping verification (verify=False)")
        return [
            self._create_unverified_hypothesis(hypothesis)
            for hypothesis in raw_hypotheses
        ]

    async def verify_hypothesis(
        self,
        hypothesis: Hypothesis,
        problem_frame: ProblemFrame,
    ) -> VerifiedHypothesis:
        """Verify a single hypothesis through logic and novelty checks.

        This method runs the dual-model verification pipeline:
        1. LogicVerifier (GPT-5.2) for logical validity
        2. NoveltyVerifier (Gemini 3 Flash Preview) for preliminary novelty
        3. SearchValidator (Brave Search) if novelty score >= 6.0

        Args:
            hypothesis: The hypothesis to verify.
            problem_frame: The problem context.

        Returns:
            VerifiedHypothesis with logic and novelty verdicts.
        """
        start_time = time.time()

        logger.debug(
            "Starting verification",
            hypothesis_id=hypothesis.id,
        )

        # Step 1: Logic verification
        logic_verdict = await self._logic_verifier.verify(hypothesis, problem_frame)

        logger.debug(
            "Logic verification complete",
            hypothesis_id=hypothesis.id,
            passed=logic_verdict.passed,
            analogy_validity=logic_verdict.analogy_validity,
            internal_consistency=logic_verdict.internal_consistency,
            causal_rigor=logic_verdict.causal_rigor,
        )

        # Step 2: Novelty verification (LLM preliminary)
        novelty_verdict = await self._novelty_verifier.verify(hypothesis, problem_frame)

        logger.debug(
            "Novelty verification (LLM) complete",
            hypothesis_id=hypothesis.id,
            score=novelty_verdict.score,
            needs_search=self._novelty_verifier.needs_search(novelty_verdict.score),
        )

        # Step 3: Search validation (if novelty score >= 6.0)
        if self._novelty_verifier.needs_search(novelty_verdict.score):
            logger.debug(
                "Running search validation",
                hypothesis_id=hypothesis.id,
                preliminary_score=novelty_verdict.score,
            )

            novelty_verdict = await self._search_validator.validate(
                hypothesis=hypothesis,
                preliminary_score=novelty_verdict.score,
            )

            logger.debug(
                "Search validation complete",
                hypothesis_id=hypothesis.id,
                final_score=novelty_verdict.score,
                searched=novelty_verdict.searched,
                similar_works_found=len(novelty_verdict.similar_works),
            )

        verify_status = self._determine_verify_status(logic_verdict)
        if verify_status == VerifyStatus.ESCALATED:
            logic_verdict, novelty_verdict, verify_status = (
                await self._run_escalated_verification(
                    hypothesis=hypothesis,
                    problem_frame=problem_frame,
                    logic_verdict=logic_verdict,
                    novelty_verdict=novelty_verdict,
                )
            )

        # Calculate final score from final verification verdicts
        final_score = self._calculate_final_score(logic_verdict, novelty_verdict)
        if verify_status != VerifyStatus.PASSED:
            final_score = 0.0

        blend_score = self._score_blend_consistency(hypothesis)
        if blend_score is not None and hypothesis.blend_network is not None:
            hypothesis = hypothesis.model_copy(
                update={
                    "blend_network": hypothesis.blend_network.model_copy(
                        update={"blend_consistency_score": blend_score}
                    )
                }
            )
            if verify_status == VerifyStatus.PASSED:
                final_score = max(0.0, min(10.0, 0.9 * final_score + 0.1 * blend_score))

        # Compute structural grounding score if enabled and evidence exists
        structural_score = None
        if self._settings.hkg_enable_structural_scoring and hypothesis.hkg_evidence is not None:
            from x_creative.hkg.scoring import structural_grounding_score
            structural_score = structural_grounding_score(hypothesis.hkg_evidence)

        verification_time = time.time() - start_time

        logger.info(
            "Hypothesis verification complete",
            hypothesis_id=hypothesis.id,
            logic_passed=logic_verdict.passed,
            novelty_score=novelty_verdict.score,
            final_score=final_score,
            verify_status=verify_status.value,
            judge_confidence=logic_verdict.judge_confidence,
            position_consistency=logic_verdict.position_consistency,
            structural_grounding_score=structural_score,
            verification_time_s=round(verification_time, 2),
        )

        return VerifiedHypothesis.from_hypothesis(
            hypothesis=hypothesis,
            logic_verdict=logic_verdict,
            novelty_verdict=novelty_verdict,
            final_score=final_score,
            structural_grounding_score=structural_score,
            verify_status=verify_status,
        )

    @staticmethod
    def _determine_verify_status(logic_verdict: LogicVerdict) -> VerifyStatus:
        """Determine VERIFY status from confidence and consistency signals."""
        confidence = logic_verdict.judge_confidence
        if logic_verdict.injection_detected:
            return VerifyStatus.ABSTAINED
        if confidence is not None and confidence < 0.35:
            return VerifyStatus.ABSTAINED
        if not logic_verdict.position_consistency:
            return VerifyStatus.ESCALATED
        if confidence is not None and confidence < 0.6:
            return VerifyStatus.ESCALATED
        if not logic_verdict.passed:
            return VerifyStatus.FAILED
        return VerifyStatus.PASSED

    async def _run_escalated_verification(
        self,
        hypothesis: Hypothesis,
        problem_frame: ProblemFrame,
        logic_verdict: LogicVerdict,
        novelty_verdict: NoveltyVerdict,
    ) -> tuple[LogicVerdict, NoveltyVerdict, VerifyStatus]:
        """Run a higher-cost second-pass verification flow for uncertain cases."""
        logger.info(
            "Running escalated verification flow",
            hypothesis_id=hypothesis.id,
            initial_confidence=logic_verdict.judge_confidence,
            initial_position_consistency=logic_verdict.position_consistency,
        )

        escalated_logic = logic_verdict
        escalated_novelty = novelty_verdict

        # Second-pass logic verification (higher cost by repeating full verifier).
        try:
            escalated_logic = await self._logic_verifier.verify(hypothesis, problem_frame)
        except Exception as exc:
            logger.warning(
                "Escalated logic verification failed; keeping primary verdict",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )

        # Second-pass novelty verification before optional forced search.
        try:
            escalated_novelty = await self._novelty_verifier.verify(hypothesis, problem_frame)
        except Exception as exc:
            logger.warning(
                "Escalated novelty verification failed; keeping primary verdict",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )
            escalated_novelty = novelty_verdict

        # If Brave key is configured, force search validation in escalation mode.
        brave_api_key = self._settings.brave_search.api_key.get_secret_value()
        if brave_api_key:
            try:
                preliminary = max(
                    float(escalated_novelty.score),
                    float(self._novelty_verifier.SEARCH_THRESHOLD),
                )
                escalated_novelty = await self._search_validator.validate(
                    hypothesis=hypothesis,
                    preliminary_score=preliminary,
                )
            except Exception as exc:
                logger.warning(
                    "Escalated search validation failed; keeping secondary novelty verdict",
                    hypothesis_id=hypothesis.id,
                    error=str(exc),
                )

        escalated_status = self._determine_verify_status(escalated_logic)
        logger.info(
            "Escalated verification complete",
            hypothesis_id=hypothesis.id,
            escalated_status=escalated_status.value,
            escalated_confidence=escalated_logic.judge_confidence,
        )
        return escalated_logic, escalated_novelty, escalated_status

    @staticmethod
    def _score_blend_consistency(hypothesis: Hypothesis) -> float | None:
        """Heuristic blend consistency score in [0,10] for VERIFY stage."""
        network = hypothesis.blend_network
        if network is None:
            return None
        mapping_count = len(network.cross_space_mappings)
        emergent = network.emergent_structures
        emergent_count = len(emergent)
        observable_link_count = sum(1 for item in emergent if item.observable_link)

        score = 2.5
        score += min(3.0, 0.8 * mapping_count)
        score += min(3.0, 1.0 * emergent_count)
        score += min(1.5, 0.5 * observable_link_count)
        if emergent_count == 0:
            score -= 1.0
        return round(max(0.0, min(10.0, score)), 1)

    def _calculate_final_score(
        self,
        logic_verdict: LogicVerdict,
        novelty_verdict: NoveltyVerdict,
    ) -> float:
        """Calculate the final composite score for a verified hypothesis.

        The final score combines:
        - Logic verification scores (40%): average of the three dimensions
        - Novelty score (60%): from LLM or search validation

        If logic verification fails, the score is hard-gated to 0.0.

        Args:
            logic_verdict: Result from logic verification.
            novelty_verdict: Result from novelty verification.

        Returns:
            Final composite score (0-10).
        """
        # Logic score: average of the three dimensions
        logic_avg = (
            logic_verdict.analogy_validity
            + logic_verdict.internal_consistency
            + logic_verdict.causal_rigor
        ) / 3.0

        if not logic_verdict.passed:
            return 0.0

        # If novelty verification errored (model unavailable), use logic-only
        if novelty_verdict.error:
            return max(0.0, min(10.0, logic_avg))

        # Weighted combination is configurable via Settings.
        final_score = (
            self._settings.final_score_logic_weight * logic_avg
            + self._settings.final_score_novelty_weight * novelty_verdict.score
        )

        return max(0.0, min(10.0, final_score))

    def _create_failed_verification(
        self,
        hypothesis: Hypothesis,
        error_message: str,
    ) -> VerifiedHypothesis:
        """Create a VerifiedHypothesis for a failed verification.

        Args:
            hypothesis: The hypothesis that failed verification.
            error_message: Description of the failure.

        Returns:
            VerifiedHypothesis with failed verdicts.
        """
        logic_verdict = LogicVerdict(
            passed=False,
            analogy_validity=0.0,
            internal_consistency=0.0,
            causal_rigor=0.0,
            reasoning=f"Verification error: {error_message}",
            issues=[f"Error: {error_message}"],
        )

        novelty_verdict = NoveltyVerdict(
            score=0.0,
            searched=False,
            similar_works=[],
            novelty_analysis=f"Verification error: {error_message}",
        )

        return VerifiedHypothesis.from_hypothesis(
            hypothesis=hypothesis,
            logic_verdict=logic_verdict,
            novelty_verdict=novelty_verdict,
            final_score=0.0,
            verify_status=VerifyStatus.FAILED,
        )

    def _create_unverified_hypothesis(
        self,
        hypothesis: Hypothesis,
    ) -> VerifiedHypothesis:
        """Create a VerifiedHypothesis for an unverified hypothesis (verify=False).

        Args:
            hypothesis: The hypothesis without verification.

        Returns:
            VerifiedHypothesis with placeholder verdicts indicating not verified.
        """
        logic_verdict = LogicVerdict(
            passed=True,  # Assume pass when skipping verification
            analogy_validity=5.0,
            internal_consistency=5.0,
            causal_rigor=5.0,
            reasoning="Verification skipped (verify=False)",
            issues=[],
        )

        novelty_verdict = NoveltyVerdict(
            score=5.0,
            searched=False,
            similar_works=[],
            novelty_analysis="Verification skipped (verify=False)",
        )

        return VerifiedHypothesis.from_hypothesis(
            hypothesis=hypothesis,
            logic_verdict=logic_verdict,
            novelty_verdict=novelty_verdict,
            final_score=5.0,  # Neutral score when not verified
            verify_status=VerifyStatus.PASSED,
        )

    async def close(self) -> None:
        """Close underlying resources.

        This method should be called to properly release resources when done
        using the engine. Consider using try/finally:

            engine = CreativityEngine()
            try:
                result = await engine.generate_hypotheses(...)
            finally:
                await engine.close()
        """
        await self._router.close()
        await self._search_validator.close()
        if self._hkg_embedding_client is not None:
            await self._hkg_embedding_client.close()

    def _prepare_search_context(
        self,
        problem: ProblemFrame,
        config: SearchConfig,
    ) -> None:
        """Refresh dynamic SEARCH context for the current target domain."""
        if hasattr(self._search, "_problem_frame"):
            self._search._problem_frame = problem

        # VERIFY uses ProblemFrame to render domain-aware scoring prompts.
        if hasattr(self._verify, "_problem_frame"):
            self._verify._problem_frame = problem

        if not hasattr(self._search, "_concept_space"):
            return

        if not config.enable_transform_space:
            self._search._concept_space = None
            return

        self._search._concept_space = self._load_concept_space(problem.target_domain)

    def _load_concept_space(self, domain_id: str) -> object | None:
        """Load ConceptSpace from target-domain YAML when available."""
        domain_key = str(domain_id or "").strip()
        if not domain_key:
            return None

        if domain_key in self._concept_space_cache:
            return self._concept_space_cache[domain_key]

        try:
            from x_creative.core.concept_space_compiler import ConceptSpaceCompiler
            from x_creative.core.plugin import TARGET_DOMAINS_DIR, USER_DOMAINS_DIR

            yaml_path: Path | None = None
            for candidate in (
                USER_DOMAINS_DIR / f"{domain_key}.yaml",
                TARGET_DOMAINS_DIR / f"{domain_key}.yaml",
            ):
                if candidate.exists():
                    yaml_path = candidate
                    break

            if yaml_path is None:
                self._concept_space_cache[domain_key] = None
                return None

            compiler = ConceptSpaceCompiler()
            concept_space = compiler.compile_from_yaml(yaml_path=yaml_path, domain_id=domain_key)

            # compile_from_yaml returns a synthetic empty space when section is absent.
            if (
                concept_space.provenance == "llm_inferred"
                and not concept_space.allowed_ops
                and not concept_space.assumptions_mutable
            ):
                self._concept_space_cache[domain_key] = None
                return None

            errors = compiler.validate(concept_space)
            if errors:
                logger.warning(
                    "ConceptSpace validation issues",
                    target_domain=domain_key,
                    issues=errors[:5],
                )

            self._concept_space_cache[domain_key] = concept_space
            return concept_space
        except Exception as exc:
            logger.warning(
                "Failed to load ConceptSpace",
                target_domain=domain_key,
                error=str(exc),
            )
            self._concept_space_cache[domain_key] = None
            return None
