"""Tests for HKG CLI commands."""
import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from x_creative.cli.main import app

runner = CliRunner()


class TestHKGIngest:
    def test_ingest_yaml(self, tmp_path: Path) -> None:
        yaml_path = Path(__file__).parent.parent.parent / "x_creative" / "config" / "target_domains" / "open_source_development.yaml"
        if not yaml_path.exists():
            pytest.skip("open_source_development.yaml not found")
        output = tmp_path / "store.json"
        result = runner.invoke(app, ["hkg", "ingest", "--source", "yaml", "--path", str(yaml_path), "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()

    def test_ingest_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "events.jsonl"
        jsonl.write_text(json.dumps({"nodes": ["a", "b"], "relation": "r", "provenance": {"doc_id": "d", "chunk_id": "c"}}) + "\n")
        output = tmp_path / "store.json"
        result = runner.invoke(app, ["hkg", "ingest", "--source", "jsonl", "--path", str(jsonl), "--output", str(output)])
        assert result.exit_code == 0

    def test_ingest_invalid_source(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["hkg", "ingest", "--source", "xml", "--path", "x", "--output", "y"])
        assert result.exit_code != 0


class TestHKGStats:
    def test_stats(self, tmp_path: Path) -> None:
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode, Hyperedge, Provenance
        store = HypergraphStore()
        store.add_node(HKGNode(node_id="n1", name="a"))
        store.add_node(HKGNode(node_id="n2", name="b"))
        store.add_edge(Hyperedge(edge_id="e1", nodes=["n1", "n2"], provenance=[Provenance(doc_id="d", chunk_id="c")]))
        path = tmp_path / "store.json"
        store.save(path)

        result = runner.invoke(app, ["hkg", "stats", "--store", str(path)])
        assert result.exit_code == 0
        assert "node_count" in result.stdout or "2" in result.stdout


class TestHKGBuildIndex:
    def test_build_index_with_embedding(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode

        store = HypergraphStore()
        store.add_node(HKGNode(node_id="n1", name="alpha"))
        store.add_node(HKGNode(node_id="n2", name="beta"))
        store_path = tmp_path / "store.json"
        store.save(store_path)

        class _DummyEmbeddingClient:
            def __init__(self, api_key: str, base_url: str = "", model: str = "") -> None:  # noqa: ARG002
                self._api_key = api_key

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[float(i + 1), 0.0] for i, _ in enumerate(texts)]

            async def close(self) -> None:
                return

        with patch("x_creative.hkg.embeddings.EmbeddingClient", _DummyEmbeddingClient):
            result = runner.invoke(
                app,
                ["hkg", "build-index", "--store", str(store_path), "--embedding"],
                env={"OPENROUTER_API_KEY": "test-key"},
            )

        assert result.exit_code == 0
        embedding_index_path = tmp_path / "store.embeddings.json"
        assert embedding_index_path.exists()


class TestHKGTraverse:
    def test_traverse(self, tmp_path: Path) -> None:
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode, Hyperedge, Provenance
        store = HypergraphStore()
        prov = [Provenance(doc_id="d", chunk_id="c")]
        store.add_node(HKGNode(node_id="n1", name="alpha"))
        store.add_node(HKGNode(node_id="n2", name="beta"))
        store.add_edge(Hyperedge(edge_id="e1", nodes=["n1", "n2"], relation="linked", provenance=prov))
        path = tmp_path / "store.json"
        store.save(path)

        result = runner.invoke(app, ["hkg", "traverse", "--store", str(path), "--start", "alpha", "--end", "beta", "--K", "1"])
        assert result.exit_code == 0
        assert "Path 1" in result.stdout
        assert '"provenance_refs"' in result.stdout

    def test_traverse_no_match(self, tmp_path: Path) -> None:
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode, Hyperedge, Provenance
        store = HypergraphStore()
        store.add_node(HKGNode(node_id="n1", name="alpha"))
        store.add_edge(Hyperedge(edge_id="e1", nodes=["n1", "n1"], provenance=[Provenance(doc_id="d", chunk_id="c")]))
        path = tmp_path / "store.json"
        store.save(path)

        result = runner.invoke(app, ["hkg", "traverse", "--store", str(path), "--start", "unknown", "--end", "beta"])
        assert result.exit_code != 0
