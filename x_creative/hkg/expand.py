"""hyperpath_expand and hyperbridge operations for SEARCH."""
import json
import inspect
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from x_creative.core.types import Hypothesis, ProblemFrame
from x_creative.creativity.utils import safe_json_loads
from x_creative.hkg.cache import TraversalCache
from x_creative.hkg.matcher import NodeMatcher
from x_creative.hkg.prompts import HYPERPATH_EXPAND_PROMPT, HYPERBRIDGE_PROMPT
from x_creative.hkg.store import HypergraphStore
from x_creative.hkg.traversal import k_shortest_hyperpaths
from x_creative.hkg.types import (
    HKGEvidence,
    HKGMatchResult,
    HKGParams,
    Hyperpath,
    HyperedgeSummary,
    HyperpathEvidence,
)
from x_creative.llm.router import ModelRouter

logger = structlog.get_logger()


async def _emit_hkg_event(
    on_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Emit an HKG event via optional callback."""
    if on_event is None:
        return
    maybe_awaitable = on_event(event_type, payload)
    if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable


def _extract_start_terms(hypothesis: Hypothesis) -> list[str]:
    """Extract entity/variable/mechanism terms from a hypothesis."""
    terms: list[str] = []
    if hypothesis.source_domain:
        terms.append(hypothesis.source_domain)
    if hypothesis.source_structure:
        terms.append(hypothesis.source_structure)
    desc_words = re.split(r'[，,、/\s:：→\->]+', hypothesis.description)
    for w in desc_words:
        w = w.strip()
        if len(w) >= 2 and w not in terms:
            terms.append(w)
    return terms[:10]


def _extract_end_terms(problem_frame: ProblemFrame) -> list[str]:
    """Extract target outcome/constraint terms from problem frame."""
    terms: list[str] = []
    desc_words = re.split(r'[，,、/\s:：→\->]+', problem_frame.description)
    for w in desc_words:
        w = w.strip()
        if len(w) >= 2 and w not in terms:
            terms.append(w)
    for constraint in problem_frame.constraints:
        c_words = re.split(r'[，,、/\s:：]+', constraint)
        for w in c_words:
            w = w.strip()
            if len(w) >= 2 and w not in terms:
                terms.append(w)
    return terms[:10]


def _build_structural_context(
    store: HypergraphStore, paths: list[Hyperpath]
) -> str:
    """Build a textual context package from hyperpaths for LLM consumption."""
    lines: list[str] = []
    for i, path in enumerate(paths, 1):
        lines.append(f"### Path {i} (length={path.length}):")
        for j, eid in enumerate(path.edges):
            edge = store.get_edge(eid)
            if edge:
                node_names = []
                for nid in edge.nodes:
                    node = store.get_node(nid)
                    node_names.append(node.name if node else nid)
                prov_refs = [f"{p.doc_id}/{p.chunk_id}" for p in edge.provenance]
                lines.append(
                    f"  Edge {j+1} [{eid}]: {' + '.join(node_names)} "
                    f"| relation: {edge.relation} | refs: {', '.join(prov_refs)}"
                )
        if path.intermediate_nodes:
            int_names = []
            for nid in path.intermediate_nodes:
                node = store.get_node(nid)
                int_names.append(node.name if node else nid)
            lines.append(f"  Bridge nodes: {', '.join(int_names)}")
        lines.append("")
    return "\n".join(lines)


def _build_hyperpath_evidence(
    store: HypergraphStore,
    paths: list[Hyperpath],
    start_node_id: str,
    end_node_id: str,
) -> list[HyperpathEvidence]:
    """Convert Hyperpath objects to HyperpathEvidence for hypothesis attachment."""
    evidence_list: list[HyperpathEvidence] = []
    for rank, path in enumerate(paths, 1):
        edge_summaries: list[HyperedgeSummary] = []
        for eid in path.edges:
            edge = store.get_edge(eid)
            if edge:
                edge_summaries.append(HyperedgeSummary(
                    edge_id=eid,
                    nodes=edge.nodes,
                    relation=edge.relation,
                    provenance_refs=[f"{p.doc_id}/{p.chunk_id}" for p in edge.provenance],
                ))
        evidence_list.append(HyperpathEvidence(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            path_rank=rank,
            path_length=path.length,
            hyperedges=edge_summaries,
            intermediate_nodes=path.intermediate_nodes,
        ))
    return evidence_list


async def hyperpath_expand(
    hypothesis: Hypothesis,
    problem_frame: ProblemFrame,
    hkg: HypergraphStore,
    matcher: NodeMatcher,
    router: ModelRouter,
    params: HKGParams,
    cache: TraversalCache | None = None,
    on_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
) -> list[Hypothesis]:
    """Generate new hypotheses grounded by hypergraph structure evidence.

    Returns empty list if no paths found (PATH_NOT_FOUND).
    Never fabricates evidence.
    """
    # 1-2. Extract terms
    start_terms = _extract_start_terms(hypothesis)
    end_terms = _extract_end_terms(problem_frame)

    # 3. Match terms to nodes
    mechanism_hint = f"{hypothesis.description} {hypothesis.observable}".strip()
    start_results = await matcher.match(
        start_terms,
        mode=params.matcher,
        context=hypothesis.description,
        mechanism_hint=mechanism_hint,
        top_k=5,
    )
    end_results = await matcher.match(
        end_terms,
        mode=params.matcher,
        context=problem_frame.description,
        mechanism_hint=mechanism_hint,
        top_k=5,
    )

    start_nodes = []
    for r in start_results:
        start_nodes.extend(r.matched_node_ids[:2])
    end_nodes = []
    for r in end_results:
        end_nodes.extend(r.matched_node_ids[:2])

    start_nodes = list(dict.fromkeys(start_nodes))
    end_nodes = list(dict.fromkeys(end_nodes))

    # 4. No match -> PATH_NOT_FOUND
    if not start_nodes or not end_nodes:
        logger.info(
            "HKG: PATH_NOT_FOUND (no node match)",
            hypothesis_id=hypothesis.id,
            start_terms=start_terms[:3],
            end_terms=end_terms[:3],
        )
        await _emit_hkg_event(
            on_event,
            "hkg_path_not_found",
            {
                "hypothesis_id": hypothesis.id,
                "reason": "no_node_match",
                "start_terms": start_terms[:3],
                "end_terms": end_terms[:3],
            },
        )
        return []

    start_match = _to_match_coverage(start_results)
    end_match = _to_match_coverage(end_results)

    # 5. Check cache
    cache_key = (
        frozenset(start_nodes), frozenset(end_nodes),
        params.K, params.IS, params.max_len,
    )
    paths: list[Hyperpath] | None = None
    if cache:
        paths = cache.get(cache_key)

    if paths is None:
        paths = k_shortest_hyperpaths(
            hkg, start_nodes, end_nodes,
            K=params.K, IS=params.IS, max_len=params.max_len,
        )
        if cache:
            cache.put(cache_key, paths)

    # 6. No paths -> PATH_NOT_FOUND
    if not paths:
        logger.info(
            "HKG: PATH_NOT_FOUND (no reachable path)",
            hypothesis_id=hypothesis.id,
        )
        await _emit_hkg_event(
            on_event,
            "hkg_path_not_found",
            {
                "hypothesis_id": hypothesis.id,
                "reason": "no_reachable_path",
            },
        )
        return []

    await _emit_hkg_event(
        on_event,
        "hkg_path_found",
        {
            "hypothesis_id": hypothesis.id,
            "path_count": len(paths),
            "min_path_length": min((path.length for path in paths), default=0),
            "start_node_count": len(start_nodes),
            "end_node_count": len(end_nodes),
        },
    )

    # 7. Build structural context
    context = _build_structural_context(hkg, paths)

    # 8. LLM call
    prompt = HYPERPATH_EXPAND_PROMPT.format(
        hypothesis_description=hypothesis.description,
        source_domain=hypothesis.source_domain,
        observable=hypothesis.observable,
        structural_context=context,
        problem_description=problem_frame.description,
        max_expansions=3,
    )

    try:
        result = await router.complete(
            task="hkg_expansion",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logger.warning("HKG LLM call failed", error=str(e))
        return []

    # 9. Parse response
    new_hypotheses = _parse_hkg_expansions(
        result.content, hypothesis, hkg, paths,
        start_nodes[0] if start_nodes else "",
        end_nodes[0] if end_nodes else "",
        params,
        start_match,
        end_match,
    )

    if new_hypotheses:
        logger.info(
            "HKG: EXPANSION_CREATED",
            hypothesis_id=hypothesis.id,
            new_count=len(new_hypotheses),
        )
        await _emit_hkg_event(
            on_event,
            "hkg_expansion_created",
            {
                "hypothesis_id": hypothesis.id,
                "new_count": len(new_hypotheses),
            },
        )

    return new_hypotheses


def _parse_hkg_expansions(
    content: str,
    parent: Hypothesis,
    store: HypergraphStore,
    paths: list[Hyperpath],
    start_node_id: str,
    end_node_id: str,
    params: HKGParams,
    start_match: dict[str, Any],
    end_match: dict[str, Any],
) -> list[Hypothesis]:
    """Parse LLM response into Hypothesis objects with HKG evidence."""
    def _text(value: Any) -> str:
        return str(value).strip()

    def _conditions(value: Any) -> list[str] | None:
        if not isinstance(value, list):
            return None
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if 1 <= len(cleaned) <= 3:
            return cleaned
        return None

    def _supporting_edges(value: Any, allowed: set[str]) -> list[str] | None:
        if not isinstance(value, list):
            return None
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            return None
        if any(edge_id not in allowed for edge_id in cleaned):
            return None
        return cleaned

    hypotheses: list[Hypothesis] = []
    try:
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            expansions = safe_json_loads(content[json_start:json_end])
            evidence_list = _build_hyperpath_evidence(
                store, paths, start_node_id, end_node_id
            )
            allowed_edges = {
                edge_id
                for path in paths
                for edge_id in path.edges
            }
            for exp in expansions:
                if not isinstance(exp, dict):
                    continue
                description = _text(exp.get("description", ""))
                analogy_explanation = _text(exp.get("analogy_explanation", ""))
                observable = _text(exp.get("observable", ""))
                mechanism_chain = _text(exp.get("mechanism_chain", ""))
                testable_conditions = _conditions(exp.get("testable_conditions"))
                supporting_edges = _supporting_edges(
                    exp.get("supporting_edges"),
                    allowed_edges,
                )
                if (
                    not description
                    or not analogy_explanation
                    or not observable
                    or not mechanism_chain
                    or testable_conditions is None
                    or supporting_edges is None
                ):
                    continue
                hkg_evidence = HKGEvidence(
                    hyperpaths=evidence_list,
                    hkg_params=params,
                    coverage={
                        "start_match": start_match,
                        "end_match": end_match,
                        "start_node_id": start_node_id,
                        "end_node_id": end_node_id,
                        "mechanism_chain": mechanism_chain,
                    },
                )
                hyp = Hypothesis(
                    id=f"hyp_{uuid.uuid4().hex[:8]}",
                    description=description,
                    source_domain=parent.source_domain,
                    source_structure=parent.source_structure,
                    analogy_explanation=analogy_explanation,
                    observable=observable,
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    expansion_type="hyperpath_expand",
                    hkg_evidence=hkg_evidence,
                    supporting_edges=supporting_edges,
                    quick_score=(
                        parent.quick_score * 0.9
                        if parent.quick_score is not None
                        else None
                    ),
                )
                hypotheses.append(hyp)
    except Exception as e:
        logger.warning("Failed to parse HKG expansions", error=str(e))
    return hypotheses


def _to_match_coverage(results: list[HKGMatchResult]) -> dict[str, Any]:
    """Convert per-term match results into compact coverage chain metadata."""
    if not results:
        return {
            "method": "none",
            "candidates": [],
            "chosen_id": None,
            "confidence": 0.0,
            "term": None,
            "rationale": "no terms",
        }

    chosen = max(results, key=lambda r: r.confidence)
    candidates: list[dict[str, Any]] = []
    for candidate in chosen.candidates:
        candidates.append(
            {
                "node_id": candidate.node_id,
                "name": candidate.name,
                "type": candidate.type,
                "method": candidate.method,
                "score": round(float(candidate.score), 4),
                "rationale": candidate.rationale,
            }
        )

    return {
        "term": chosen.term,
        "method": chosen.match_type,
        "candidates": candidates,
        "chosen_id": chosen.chosen_id,
        "confidence": round(float(chosen.confidence), 4),
        "rationale": chosen.rationale,
    }


def _parse_hyperbridge_expansions(
    content: str,
    *,
    hkg: HypergraphStore,
    paths: list[Hyperpath],
    nodes_a: list[str],
    nodes_b: list[str],
    bridge_params: HKGParams,
    concept_a: str,
    concept_b: str,
    source_domain: str,
) -> list[Hypothesis]:
    """Parse and validate hyperbridge JSON output with per-path 2~3 constraints."""
    def _text(value: Any) -> str:
        return str(value).strip()

    def _conditions(value: Any) -> list[str] | None:
        if not isinstance(value, list):
            return None
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if 1 <= len(cleaned) <= 3:
            return cleaned
        return None

    def _path_rank(value: Any, path_count: int) -> int | None:
        try:
            rank = int(value)
        except (TypeError, ValueError):
            return None
        if rank < 1 or rank > path_count:
            return None
        return rank

    hypotheses_by_path: dict[int, list[Hypothesis]] = {}
    json_start = content.find("[")
    json_end = content.rfind("]") + 1
    if json_start < 0 or json_end <= json_start:
        return []

    expansions = safe_json_loads(content[json_start:json_end])
    if not isinstance(expansions, list):
        return []

    evidence_list = _build_hyperpath_evidence(
        hkg,
        paths,
        nodes_a[0] if nodes_a else "",
        nodes_b[0] if nodes_b else "",
    )

    path_count = max(1, len(paths))
    for exp in expansions:
        if not isinstance(exp, dict):
            continue
        description = _text(exp.get("description", ""))
        analogy_explanation = _text(exp.get("analogy_explanation", ""))
        observable = _text(exp.get("observable", ""))
        bridge_path = _text(exp.get("bridge_path", ""))
        testable_conditions = _conditions(exp.get("testable_conditions"))
        rank = _path_rank(exp.get("path_rank"), path_count)

        if (
            not description
            or not analogy_explanation
            or not observable
            or not bridge_path
            or testable_conditions is None
            or rank is None
        ):
            continue

        selected_path = paths[rank - 1] if 1 <= rank <= len(paths) else None
        if selected_path is None or not selected_path.edges:
            continue
        supporting_edges = list(selected_path.edges)

        hkg_evidence = HKGEvidence(
            hyperpaths=evidence_list,
            hkg_params=bridge_params,
            coverage={
                "concept_a": concept_a,
                "concept_b": concept_b,
                "path_rank": rank,
                "bridge_path": bridge_path,
                "supporting_edges": supporting_edges,
            },
        )
        hyp = Hypothesis(
            id=f"hyp_{uuid.uuid4().hex[:8]}",
            description=description,
            source_domain=source_domain,
            source_structure="hyperbridge",
            analogy_explanation=analogy_explanation,
            observable=observable,
            expansion_type="hyperbridge",
            hkg_evidence=hkg_evidence,
            supporting_edges=supporting_edges,
        )
        hypotheses_by_path.setdefault(rank, []).append(hyp)

    expected_ranks = list(range(1, path_count + 1))
    returned_ranks = set(hypotheses_by_path)
    missing_ranks = [rank for rank in expected_ranks if rank not in returned_ranks]
    if missing_ranks:
        logger.debug(
            "Discard hyperbridge outputs due to missing path explanations",
            missing_ranks=missing_ranks,
            path_count=path_count,
        )
        return []

    accepted: list[Hypothesis] = []
    for rank in expected_ranks:
        hyps = hypotheses_by_path.get(rank, [])
        if not (2 <= len(hyps) <= 3):
            logger.debug(
                "Discard hyperbridge outputs due to per-path explanation count",
                path_rank=rank,
                count=len(hyps),
            )
            return []
        accepted.extend(hyps)
    return accepted


async def hyperbridge(
    concept_a: str,
    concept_b: str,
    hkg: HypergraphStore,
    matcher: NodeMatcher,
    router: ModelRouter,
    params: HKGParams,
    source_domain: str = "hyperbridge",
    cache: TraversalCache | None = None,
    on_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
) -> list[Hypothesis]:
    """Find structural bridges between two concepts.

    Uses IS=1 and K>1 for maximum path diversity.
    Each path generates 2-3 bridging explanations with observables.
    """
    results_a = await matcher.match([concept_a], mode=params.matcher)
    results_b = await matcher.match([concept_b], mode=params.matcher)

    nodes_a = []
    for r in results_a:
        nodes_a.extend(r.matched_node_ids)
    nodes_b = []
    for r in results_b:
        nodes_b.extend(r.matched_node_ids)

    if not nodes_a or not nodes_b:
        logger.info("HKG hyperbridge: PATH_NOT_FOUND (no match)")
        await _emit_hkg_event(
            on_event,
            "hkg_path_not_found",
            {
                "reason": "no_node_match",
                "concept_a": concept_a,
                "concept_b": concept_b,
            },
        )
        return []

    bridge_params = HKGParams(K=max(params.K, 3), IS=1, max_len=params.max_len)

    cache_key = (
        frozenset(nodes_a), frozenset(nodes_b),
        bridge_params.K, bridge_params.IS, bridge_params.max_len,
    )
    paths: list[Hyperpath] | None = None
    if cache:
        paths = cache.get(cache_key)
    if paths is None:
        paths = k_shortest_hyperpaths(
            hkg, nodes_a, nodes_b,
            K=bridge_params.K, IS=bridge_params.IS, max_len=bridge_params.max_len,
        )
        if cache:
            cache.put(cache_key, paths)

    if not paths:
        logger.info("HKG hyperbridge: PATH_NOT_FOUND (no path)")
        await _emit_hkg_event(
            on_event,
            "hkg_path_not_found",
            {
                "reason": "no_reachable_path",
                "concept_a": concept_a,
                "concept_b": concept_b,
            },
        )
        return []

    await _emit_hkg_event(
        on_event,
        "hkg_path_found",
        {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "path_count": len(paths),
            "min_path_length": min((path.length for path in paths), default=0),
        },
    )

    context = _build_structural_context(hkg, paths)

    prompt = HYPERBRIDGE_PROMPT.format(
        concept_a=concept_a,
        concept_b=concept_b,
        structural_context=context,
    )

    try:
        result = await router.complete(
            task="hkg_expansion",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logger.warning("HKG hyperbridge LLM call failed", error=str(e))
        return []

    try:
        hypotheses = _parse_hyperbridge_expansions(
            result.content,
            hkg=hkg,
            paths=paths,
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            bridge_params=bridge_params,
            concept_a=concept_a,
            concept_b=concept_b,
            source_domain=source_domain,
        )
        if hypotheses:
            await _emit_hkg_event(
                on_event,
                "hkg_expansion_created",
                {
                    "concept_a": concept_a,
                    "concept_b": concept_b,
                    "new_count": len(hypotheses),
                },
            )
        return hypotheses
    except Exception as e:
        logger.warning("Failed to parse hyperbridge response", error=str(e))
        return []
