"""Node matching for the HKG (Hypergraph Knowledge Grounding) subsystem.

Two-stage matcher:
1. Recall (wide): exact/alias/embedding produce a shared candidate pool.
2. Rerank (strict): contextual rerank with confidence+rationale chain.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from x_creative.hkg.types import HKGMatchCandidate, HKGMatchResult, MatchType

if TYPE_CHECKING:
    from x_creative.hkg.embeddings import EmbeddingClient, NodeEmbeddingIndex
    from x_creative.hkg.store import HypergraphStore

log = structlog.get_logger(__name__)

_ABSTRACT_SCHEMA_HINTS = {
    "feedback",
    "constraint",
    "threshold",
    "phase",
    "transition",
    "loop",
    "regime",
    "因果",
    "约束",
    "阈值",
    "相变",
    "反馈",
}


@dataclass
class _RecallCandidate:
    node_id: str
    method: MatchType
    base_score: float
    reasons: list[str] = field(default_factory=list)


class NodeMatcher:
    """Two-stage matcher: recall-wide + rerank-strict."""

    def __init__(
        self,
        store: HypergraphStore,
        embedding_client: EmbeddingClient | None = None,
        embedding_index: NodeEmbeddingIndex | None = None,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._embedding_index = embedding_index
        self._match_cache: dict[tuple[str, str, str, str, int], HKGMatchResult] = {}
        self._index_build_lock = asyncio.Lock()

    async def _ensure_embedding_index(self) -> None:
        """Lazily build embedding index when embedding matcher is enabled."""
        if self._embedding_client is None or self._embedding_index is not None:
            return
        async with self._index_build_lock:
            if self._embedding_index is not None:
                return
            try:
                from x_creative.hkg.embeddings import NodeEmbeddingIndex

                index = NodeEmbeddingIndex()
                await index.build(self._store.all_nodes, self._embedding_client)
                self._embedding_index = index
                log.info(
                    "matcher_embedding_index_built",
                    node_count=len(self._store.all_nodes),
                )
            except Exception:
                log.warning("matcher_embedding_index_build_failed", exc_info=True)

    async def match(
        self,
        terms: list[str],
        mode: str = "auto",
        context: str | None = None,
        mechanism_hint: str | None = None,
        top_k: int = 5,
    ) -> list[HKGMatchResult]:
        """Match query terms to graph nodes using recall+rereank."""
        k = max(1, int(top_k))
        results: list[HKGMatchResult] = []
        for term in terms:
            cache_key = self._cache_key(term, mode, context, mechanism_hint, k)
            if cache_key in self._match_cache:
                results.append(self._match_cache[cache_key])
            else:
                result = await self._match_single(
                    term=term,
                    mode=mode,
                    context=context,
                    mechanism_hint=mechanism_hint,
                    top_k=k,
                )
                self._match_cache[cache_key] = result
                results.append(result)
        return results

    def _cache_key(
        self,
        term: str,
        mode: str,
        context: str | None,
        mechanism_hint: str | None,
        top_k: int,
    ) -> tuple[str, str, str, str, int]:
        context_key = self._normalize_text(context or "")[:200]
        mech_key = self._normalize_text(mechanism_hint or "")[:200]
        return (self._normalize_text(term), mode, context_key, mech_key, top_k)

    async def _match_single(
        self,
        term: str,
        mode: str,
        context: str | None,
        mechanism_hint: str | None,
        top_k: int,
    ) -> HKGMatchResult:
        """Match a single term through recall+rereank stages."""
        recalled = await self._recall_candidates(term=term, mode=mode, top_k=top_k)
        reranked = self._rerank_candidates(
            term=term,
            context=context,
            mechanism_hint=mechanism_hint,
            candidates=recalled,
            top_k=top_k,
        )

        if not reranked:
            log.debug("match_none", term=term)
            return HKGMatchResult(
                term=term,
                matched_node_ids=[],
                match_type="none",
                confidence=0.0,
                candidates=[],
                chosen_id=None,
                rationale="No candidate survived recall+rereank",
            )

        chosen = reranked[0]
        matched_ids = [candidate.node_id for candidate in reranked]
        match_type: MatchType = chosen.method

        log.debug(
            "match_selected",
            term=term,
            chosen_id=chosen.node_id,
            method=chosen.method,
            confidence=chosen.score,
        )

        return HKGMatchResult(
            term=term,
            matched_node_ids=matched_ids,
            match_type=match_type,
            confidence=chosen.score,
            candidates=reranked,
            chosen_id=chosen.node_id,
            rationale=chosen.rationale,
        )

    async def _recall_candidates(
        self,
        term: str,
        mode: str,
        top_k: int,
    ) -> list[_RecallCandidate]:
        """Stage-1 recall: merge exact/alias/embedding candidates."""
        candidate_map: dict[str, _RecallCandidate] = {}

        def _upsert(
            node_id: str,
            method: MatchType,
            score: float,
            reason: str,
        ) -> None:
            score = max(0.0, min(1.0, score))
            existing = candidate_map.get(node_id)
            if existing is None or score > existing.base_score:
                candidate_map[node_id] = _RecallCandidate(
                    node_id=node_id,
                    method=method,
                    base_score=score,
                    reasons=[reason],
                )
                return
            existing.reasons.append(reason)
            # Prefer stronger methods when score ties.
            if self._method_priority(method) > self._method_priority(existing.method):
                existing.method = method

        # Layer 1: exact/alias
        if mode in ("auto", "exact"):
            node_ids = self._store.find_nodes_by_name(term)
            for node_id in node_ids:
                node = self._store.get_node(node_id)
                if node is None:
                    continue
                is_exact = self._normalize_text(node.name) == self._normalize_text(term)
                method: MatchType = "exact" if is_exact else "alias"
                _upsert(
                    node_id=node_id,
                    method=method,
                    score=1.0 if is_exact else 0.97,
                    reason="lexical match",
                )

        # Layer 2: embedding fallback
        if mode in ("auto", "embedding"):
            if self._embedding_client is not None:
                await self._ensure_embedding_index()
            if self._embedding_client is not None and self._embedding_index is not None:
                try:
                    query_vec = await self._embedding_client.embed_cached(term)
                    nearest = self._embedding_index.find_nearest(query_vec, top_k=max(top_k, 8))
                    for node_id, score in nearest:
                        _upsert(
                            node_id=node_id,
                            method="embedding",
                            score=score,
                            reason="embedding cosine",
                        )
                except Exception:
                    log.warning("match_embedding_failed", term=term, exc_info=True)

        return list(candidate_map.values())

    def _rerank_candidates(
        self,
        term: str,
        context: str | None,
        mechanism_hint: str | None,
        candidates: list[_RecallCandidate],
        top_k: int,
    ) -> list[HKGMatchCandidate]:
        """Stage-2 rerank: contextual reranking with rationale chain."""
        term_tokens = self._tokenize(term)
        context_tokens = self._tokenize(context or "")
        mechanism_tokens = self._tokenize(mechanism_hint or "")
        term_has_schema_hint = bool(term_tokens & _ABSTRACT_SCHEMA_HINTS)

        reranked: list[HKGMatchCandidate] = []
        for candidate in candidates:
            node = self._store.get_node(candidate.node_id)
            if node is None:
                continue

            node_tokens = self._tokenize(" ".join([node.name, *node.aliases]))
            score = candidate.base_score
            rationale_bits = [
                f"recall={candidate.method}:{candidate.base_score:.2f}",
            ]

            if term_tokens & node_tokens:
                score += 0.08
                rationale_bits.append("term_overlap")
            if context_tokens and (context_tokens & node_tokens):
                score += 0.06
                rationale_bits.append("context_overlap")
            if mechanism_tokens and (mechanism_tokens & node_tokens):
                score += 0.04
                rationale_bits.append("mechanism_overlap")

            if node.type == "schema":
                if term_has_schema_hint:
                    score += 0.08
                    rationale_bits.append("schema_boost")
                else:
                    score -= 0.02
                    rationale_bits.append("schema_penalty")

            score = max(0.0, min(1.0, score))
            reranked.append(
                HKGMatchCandidate(
                    node_id=node.node_id,
                    name=node.name,
                    type=node.type,
                    method=candidate.method,
                    score=score,
                    rationale=", ".join(rationale_bits),
                )
            )

        reranked.sort(
            key=lambda c: (
                c.score,
                self._method_priority(c.method),
                c.type == "schema",
            ),
            reverse=True,
        )
        return reranked[:top_k]

    @staticmethod
    def _method_priority(method: MatchType) -> int:
        priority = {
            "none": 0,
            "embedding": 1,
            "alias": 2,
            "exact": 3,
        }
        return priority.get(method, 0)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        normalized = cls._normalize_text(text)
        return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{1,}", normalized))
