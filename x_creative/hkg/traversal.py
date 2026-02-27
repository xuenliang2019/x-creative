"""IS-constrained BFS k-shortest hyperpaths traversal for HKG.

Implements a BFS over hyperedge space that respects an intersection-size (IS)
constraint between consecutive edges, returning up to K shortest hyperpaths.
"""

from __future__ import annotations

from collections import deque

import structlog

from x_creative.hkg.store import HypergraphStore
from x_creative.hkg.types import Hyperpath

log = structlog.get_logger(__name__)


def _compute_intermediate_nodes(
    store: HypergraphStore,
    edge_ids: list[str],
    start_set: set[str],
    end_set: set[str],
) -> list[str]:
    """Compute intersection nodes between consecutive edges, excluding start/end.

    For each pair of consecutive edges (edge_ids[i], edge_ids[i+1]), compute the
    set intersection of their nodes.  Collect all such intersection nodes, then
    remove any that belong to *start_set* or *end_set*.

    Returns a deduplicated list (preserving first-seen order).
    """
    seen: set[str] = set()
    result: list[str] = []

    for i in range(len(edge_ids) - 1):
        edge_a = store.get_edge(edge_ids[i])
        edge_b = store.get_edge(edge_ids[i + 1])
        if edge_a is None or edge_b is None:
            continue
        overlap = set(edge_a.nodes) & set(edge_b.nodes)
        for node_id in sorted(overlap):  # sorted for determinism
            if node_id not in start_set and node_id not in end_set and node_id not in seen:
                seen.add(node_id)
                result.append(node_id)

    return result


def _compute_provenance_refs(
    store: HypergraphStore,
    edge_ids: list[str],
) -> list[list[str]]:
    """Compute per-edge provenance refs aligned to edge order."""
    refs: list[list[str]] = []
    for edge_id in edge_ids:
        edge = store.get_edge(edge_id)
        if edge is None:
            refs.append([])
            continue
        edge_refs = sorted({f"{p.doc_id}/{p.chunk_id}" for p in edge.provenance})
        refs.append(edge_refs)
    return refs


def k_shortest_hyperpaths(
    store: HypergraphStore,
    start_nodes: list[str],
    end_nodes: list[str],
    K: int = 3,
    IS: int = 1,
    max_len: int = 6,
) -> list[Hyperpath]:
    """Find up to *K* shortest hyperpaths from *start_nodes* to *end_nodes*.

    Algorithm: BFS over hyperedge space with IS constraint.

    1. Find initial edges: all edges containing at least one start_node.
    2. BFS with state = (current_edge_id, path as tuple of edge_ids).
    3. Goal: any edge containing at least one end_node.
    4. Transition from edge_A to edge_B:
       - For each node in edge_A, get all edges via inverted index.
       - Filter: edge_B not already in current path (prevent cycles).
       - Filter: |nodes(edge_A) intersection nodes(edge_B)| >= IS.
    5. BFS guarantees shortest path (fewest edges).
    6. Collect all paths at minimum depth, return up to K.
    7. If fewer than K at min depth, return what we have (do NOT go deeper).
    8. max_len hard cap on path length.
    """
    if not start_nodes or not end_nodes:
        return []

    start_set = set(start_nodes)
    end_set = set(end_nodes)

    # Step 1: Find all initial edges (containing at least one start node)
    initial_edge_ids: set[str] = set()
    for node_id in start_nodes:
        for edge in store.get_edges_for_node(node_id):
            initial_edge_ids.add(edge.edge_id)

    if not initial_edge_ids:
        return []

    # BFS queue: each entry is a tuple of edge_ids representing a path
    queue: deque[tuple[str, ...]] = deque()
    found_paths: list[Hyperpath] = []
    min_found_depth: int | None = None

    # Seed the BFS with single-edge paths
    for eid in sorted(initial_edge_ids):  # sorted for determinism
        queue.append((eid,))

    # Track visited states to avoid exploring the same (edge, path_set) twice
    # State: (current_edge_id, frozenset_of_path_edge_ids)
    visited: set[tuple[str, frozenset[str]]] = set()

    while queue:
        path = queue.popleft()
        current_eid = path[-1]
        path_len = len(path)

        # If we already found paths at a shorter depth, stop exploring longer ones
        if min_found_depth is not None and path_len > min_found_depth:
            break

        # Hard cap on path length
        if path_len > max_len:
            continue

        # Visited check
        state = (current_eid, frozenset(path))
        if state in visited:
            continue
        visited.add(state)

        # Check if current edge reaches any end node
        current_edge = store.get_edge(current_eid)
        if current_edge is None:
            continue

        current_nodes = set(current_edge.nodes)
        if current_nodes & end_set:
            # Found a goal path
            if min_found_depth is None:
                min_found_depth = path_len
            if path_len == min_found_depth:
                edge_list = list(path)
                intermediate = _compute_intermediate_nodes(store, edge_list, start_set, end_set)
                provenance_refs = _compute_provenance_refs(store, edge_list)
                found_paths.append(
                    Hyperpath(
                        edges=edge_list,
                        intermediate_nodes=intermediate,
                        provenance_refs=provenance_refs,
                    )
                )
                if len(found_paths) >= K:
                    break
            continue  # Don't expand further from goal edges

        # Don't expand if we've reached max_len
        if path_len >= max_len:
            continue

        # Expand: for each node in current edge, find neighbor edges
        path_edge_set = set(path)
        neighbor_candidates: set[str] = set()

        for node_id in current_edge.nodes:
            for neighbor_edge in store.get_edges_for_node(node_id):
                neid = neighbor_edge.edge_id
                if neid in path_edge_set:
                    continue  # prevent cycles
                neighbor_candidates.add(neid)

        # Check IS constraint and enqueue
        for neid in sorted(neighbor_candidates):  # sorted for determinism
            neighbor_edge = store.get_edge(neid)
            if neighbor_edge is None:
                continue
            neighbor_nodes = set(neighbor_edge.nodes)
            overlap_size = len(current_nodes & neighbor_nodes)
            if overlap_size >= IS:
                queue.append(path + (neid,))

    log.debug(
        "traversal_complete",
        start=start_nodes,
        end=end_nodes,
        K=K,
        IS=IS,
        paths_found=len(found_paths),
    )
    return found_paths[:K]
