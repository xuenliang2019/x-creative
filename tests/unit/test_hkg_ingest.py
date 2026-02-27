"""Tests for HKG ingest: YAML and JSONL ingestion into HypergraphStore."""

import json
from pathlib import Path

import pytest

# Path to the built-in open_source_development YAML target domain file.
OPEN_SOURCE_DEV_YAML = (
    Path(__file__).resolve().parents[2]
    / "x_creative"
    / "config"
    / "target_domains"
    / "open_source_development.yaml"
)


# ============================================================
# YAML ingest
# ============================================================
class TestIngestFromYaml:
    """Tests for ingest_from_yaml."""

    def test_ingest_from_yaml(self) -> None:
        """Load open_source_development.yaml; verify node_count > 0, edge_count > 0,
        and every edge has provenance."""
        from x_creative.hkg.ingest import ingest_from_yaml

        store = ingest_from_yaml(OPEN_SOURCE_DEV_YAML)
        stats = store.stats()

        assert stats["node_count"] > 0
        assert stats["edge_count"] > 0

        # Every edge must carry at least one provenance entry.
        for edge in store.all_edges.values():
            assert len(edge.provenance) >= 1, (
                f"Edge {edge.edge_id} has no provenance"
            )

    def test_nodes_have_correct_types(self) -> None:
        """After YAML ingest, both 'domain' and 'variable' node types must exist."""
        from x_creative.hkg.ingest import ingest_from_yaml

        store = ingest_from_yaml(OPEN_SOURCE_DEV_YAML)
        node_types = {node.type for node in store.all_nodes.values()}

        assert "domain" in node_types, f"Missing 'domain' type; found: {node_types}"
        assert "variable" in node_types, f"Missing 'variable' type; found: {node_types}"


# ============================================================
# JSONL ingest
# ============================================================
class TestIngestFromJsonl:
    """Tests for ingest_from_jsonl."""

    def test_ingest_from_jsonl(self, tmp_path: Path) -> None:
        """Write a temp JSONL file, verify correct node and edge counts."""
        from x_creative.hkg.ingest import ingest_from_jsonl

        lines = [
            json.dumps({
                "nodes": ["alpha", "beta", "gamma"],
                "relation": "causes",
                "provenance": {"doc_id": "doc1", "chunk_id": "c1"},
            }),
            json.dumps({
                "nodes": ["beta", "delta"],
                "relation": "correlates",
                "provenance": {"doc_id": "doc2", "chunk_id": "c2"},
            }),
        ]
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("\n".join(lines) + "\n")

        store = ingest_from_jsonl(jsonl_file)
        stats = store.stats()

        # 4 unique nodes: alpha, beta, gamma, delta
        assert stats["node_count"] == 4
        # 2 edges (one per JSONL line)
        assert stats["edge_count"] == 2

    def test_empty_jsonl(self, tmp_path: Path) -> None:
        """An empty JSONL file produces an empty store."""
        from x_creative.hkg.ingest import ingest_from_jsonl

        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        store = ingest_from_jsonl(jsonl_file)
        stats = store.stats()

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    def test_skip_invalid_lines(self, tmp_path: Path) -> None:
        """Invalid JSON and lines with < 2 nodes are skipped."""
        from x_creative.hkg.ingest import ingest_from_jsonl

        lines = [
            "not valid json at all",
            json.dumps({"nodes": ["only_one"], "relation": "x", "provenance": {"doc_id": "d", "chunk_id": "c"}}),
            json.dumps({"nodes": ["a", "b"], "relation": "ok", "provenance": {"doc_id": "d", "chunk_id": "c"}}),
        ]
        jsonl_file = tmp_path / "mixed.jsonl"
        jsonl_file.write_text("\n".join(lines) + "\n")

        store = ingest_from_jsonl(jsonl_file)
        stats = store.stats()

        # Only the last valid line should produce data: 2 nodes, 1 edge
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1

    def test_case_insensitive_dedup(self, tmp_path: Path) -> None:
        """Nodes with same name but different case should be deduplicated."""
        from x_creative.hkg.ingest import ingest_from_jsonl

        lines = [
            json.dumps({"nodes": ["Alpha", "Beta"], "relation": "r1", "provenance": {"doc_id": "d", "chunk_id": "c1"}}),
            json.dumps({"nodes": ["alpha", "GAMMA"], "relation": "r2", "provenance": {"doc_id": "d", "chunk_id": "c2"}}),
        ]
        jsonl_file = tmp_path / "case.jsonl"
        jsonl_file.write_text("\n".join(lines) + "\n")

        store = ingest_from_jsonl(jsonl_file)
        stats = store.stats()

        # "Alpha" and "alpha" should be the same node -> 3 unique nodes
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2

    def test_skip_lines_with_blank_provenance_refs(self, tmp_path: Path) -> None:
        """JSONL rows with blank doc_id/chunk_id should be rejected."""
        from x_creative.hkg.ingest import ingest_from_jsonl

        lines = [
            json.dumps({
                "nodes": ["a", "b"],
                "relation": "valid",
                "provenance": {"doc_id": "doc_ok", "chunk_id": "c1"},
            }),
            json.dumps({
                "nodes": ["b", "c"],
                "relation": "blank_doc",
                "provenance": {"doc_id": " ", "chunk_id": "c2"},
            }),
            json.dumps({
                "nodes": ["c", "d"],
                "relation": "blank_chunk",
                "provenance": {"doc_id": "doc2", "chunk_id": ""},
            }),
        ]
        jsonl_file = tmp_path / "provenance_blank.jsonl"
        jsonl_file.write_text("\n".join(lines) + "\n")

        store = ingest_from_jsonl(jsonl_file)
        stats = store.stats()

        assert stats["edge_count"] == 1
        assert stats["node_count"] == 2
