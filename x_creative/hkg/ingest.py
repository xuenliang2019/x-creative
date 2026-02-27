"""Ingest structured knowledge into the HKG HypergraphStore.

Supports two input formats:
- **YAML**: target-domain definition files (e.g. ``open_source_development.yaml``)
- **JSONL**: one JSON object per line with ``nodes``, ``relation``, ``provenance``
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import structlog
import yaml

from x_creative.hkg.store import HypergraphStore
from x_creative.hkg.types import HKGNode, Hyperedge, Provenance

log = structlog.get_logger(__name__)


def _uid() -> str:
    """Generate a short unique id."""
    return uuid.uuid4().hex[:12]


# ============================================================
# YAML ingest
# ============================================================


def ingest_from_yaml(yaml_path: Path) -> HypergraphStore:
    """Convert a target-domain YAML file into a :class:`HypergraphStore`.

    Conversion rules
    ----------------
    1. Each ``source_domains[]`` entry -> ``HKGNode(type="domain")``.
    2. Each ``structures[].key_variables[]`` -> ``HKGNode(type="variable")``.
       Variable nodes are **deduplicated** by lowercase name.
    3. Each ``DomainStructure`` -> ``Hyperedge`` connecting the domain node
       and its variable nodes, with ``relation = structure.dynamics``.
    4. Each ``target_mappings[]`` -> bridging ``Hyperedge`` connecting the
       source variable nodes of the referenced structure to a new target
       concept node, with ``relation = "target_mapping"``.

    Domain nodes receive ``name_en`` as an alias when available.
    """
    store = HypergraphStore()
    data: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    yaml_filename = yaml_path.name

    # Lookup: variable name (lowercase) -> node_id  (for dedup)
    var_index: dict[str, str] = {}

    def _get_or_create_variable(var_name: str) -> str:
        """Return the node_id for *var_name*, creating the node if needed."""
        key = var_name.lower()
        if key in var_index:
            return var_index[key]
        nid = f"var_{_uid()}"
        store.add_node(HKGNode(node_id=nid, name=var_name, type="variable"))
        var_index[key] = nid
        return nid

    # Lookup: (domain_id, structure_id) -> list of variable node_ids
    structure_vars: dict[tuple[str, str], list[str]] = {}

    for domain in data.get("source_domains", []):
        domain_id: str = domain["id"]

        # --- 1. Domain node ---
        domain_nid = f"domain_{domain_id}"
        aliases: list[str] = []
        if domain.get("name_en"):
            aliases.append(domain["name_en"])
        store.add_node(
            HKGNode(
                node_id=domain_nid,
                name=domain.get("name", domain_id),
                aliases=aliases,
                type="domain",
            )
        )

        # --- 2 & 3. Structures -> variable nodes + structure hyperedge ---
        for structure in domain.get("structures", []):
            structure_id: str = structure["id"]
            var_nids: list[str] = []
            for var_name in structure.get("key_variables", []):
                var_nids.append(_get_or_create_variable(str(var_name)))

            # Remember for target_mappings lookup
            structure_vars[(domain_id, structure_id)] = var_nids

            # Hyperedge: domain node + variable nodes
            edge_nodes = [domain_nid] + var_nids
            if len(edge_nodes) < 2:
                log.warning(
                    "structure_edge_skipped",
                    domain_id=domain_id,
                    structure_id=structure_id,
                    reason="fewer than 2 nodes",
                )
                continue

            store.add_edge(
                Hyperedge(
                    edge_id=f"se_{_uid()}",
                    nodes=edge_nodes,
                    relation=structure.get("dynamics", ""),
                    provenance=[
                        Provenance(
                            doc_id=yaml_filename,
                            chunk_id=f"{domain_id}/{structure_id}",
                        )
                    ],
                )
            )

        # --- 4. Target mappings -> bridging hyperedge ---
        for mapping in domain.get("target_mappings", []):
            ref_structure_id: str = mapping["structure"]
            target_name: str = mapping.get("target", "")

            # Look up variable nodes from the referenced structure
            src_var_nids = structure_vars.get((domain_id, ref_structure_id), [])

            # Create a target concept node
            target_nid = f"target_{_uid()}"
            store.add_node(
                HKGNode(
                    node_id=target_nid,
                    name=target_name,
                    type="concept",
                )
            )

            edge_nodes = src_var_nids + [target_nid]
            if len(edge_nodes) < 2:
                log.warning(
                    "target_mapping_edge_skipped",
                    domain_id=domain_id,
                    structure_id=ref_structure_id,
                    reason="fewer than 2 nodes",
                )
                continue

            store.add_edge(
                Hyperedge(
                    edge_id=f"tm_{_uid()}",
                    nodes=edge_nodes,
                    relation="target_mapping",
                    provenance=[
                        Provenance(
                            doc_id=yaml_filename,
                            chunk_id=f"{domain_id}/{ref_structure_id}",
                        )
                    ],
                )
            )

    log.info("yaml_ingested", path=str(yaml_path), **store.stats())
    return store


# ============================================================
# JSONL ingest
# ============================================================


def ingest_from_jsonl(jsonl_path: Path) -> HypergraphStore:
    """Ingest a JSONL file into a :class:`HypergraphStore`.

    Each line is expected to be a JSON object::

        {"nodes": ["a","b","c"], "relation": "causes",
         "provenance": {"doc_id": "x", "chunk_id": "y"}}

    - Nodes are auto-created and **deduplicated case-insensitively**.
    - Lines with fewer than 2 nodes or invalid JSON are silently skipped.
    """
    store = HypergraphStore()
    # lowercase name -> node_id
    node_index: dict[str, str] = {}

    def _get_or_create_node(name: str) -> str:
        key = name.lower()
        if key in node_index:
            return node_index[key]
        nid = f"n_{_uid()}"
        store.add_node(HKGNode(node_id=nid, name=name, type="entity"))
        node_index[key] = nid
        return nid

    text = jsonl_path.read_text(encoding="utf-8")
    if not text.strip():
        log.info("jsonl_empty", path=str(jsonl_path))
        return store

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # --- Parse JSON ---
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            log.warning("jsonl_invalid_json", line=lineno, path=str(jsonl_path))
            continue

        # --- Validate ---
        node_names: list[str] = obj.get("nodes", [])
        if not isinstance(node_names, list) or len(node_names) < 2:
            log.warning("jsonl_too_few_nodes", line=lineno, count=len(node_names))
            continue

        prov_data: dict[str, str] = obj.get("provenance", {})
        if not isinstance(prov_data, dict):
            log.warning("jsonl_invalid_provenance", line=lineno)
            continue
        doc_id = str(prov_data.get("doc_id", "")).strip()
        chunk_id = str(prov_data.get("chunk_id", "")).strip()
        if not doc_id or not chunk_id:
            log.warning(
                "jsonl_invalid_provenance_ref",
                line=lineno,
                doc_id=doc_id,
                chunk_id=chunk_id,
            )
            continue

        # --- Create nodes (dedup) and edge ---
        nids = [_get_or_create_node(str(n)) for n in node_names]

        store.add_edge(
            Hyperedge(
                edge_id=f"e_{_uid()}",
                nodes=nids,
                relation=obj.get("relation", ""),
                provenance=[
                    Provenance(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                    )
                ],
            )
        )

    log.info("jsonl_ingested", path=str(jsonl_path), **store.stats())
    return store
