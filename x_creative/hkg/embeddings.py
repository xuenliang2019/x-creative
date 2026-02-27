"""Embedding utilities for the HKG (Hypergraph Knowledge Grounding) subsystem.

Provides:
- ``EmbeddingClient``: thin async wrapper around OpenRouter/OpenAI embeddings API.
- ``NodeEmbeddingIndex``: pre-computed embeddings for all node names with
  cosine-similarity nearest-neighbor search.
- ``_cosine_similarity``: helper for computing cosine similarity between two vectors.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from x_creative.hkg.types import HKGNode

log = structlog.get_logger(__name__)


# ---------- cosine similarity helper ----------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude (avoids division by zero).
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------- EmbeddingClient ----------


class EmbeddingClient:
    """Thin wrapper around OpenRouter/OpenAI embeddings API.

    Uses ``httpx.AsyncClient`` to call the ``/embeddings`` endpoint.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        # In-memory cache for embed_cached (keyed by text string)
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the API. Returns one vector per input text."""
        response = await self._client.post(
            "/embeddings",
            json={"input": texts, "model": self._model},
        )
        response.raise_for_status()
        data = response.json()
        # API returns embeddings sorted by index
        embeddings_data = sorted(data["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in embeddings_data]

    async def embed_cached(self, text: str) -> list[float]:
        """Embed a single text, returning a cached result if available."""
        if text in self._cache:
            return self._cache[text]
        results = await self.embed([text])
        embedding = results[0]
        self._cache[text] = embedding
        return embedding

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


# ---------- NodeEmbeddingIndex ----------


class NodeEmbeddingIndex:
    """Pre-computed embeddings for all node names.

    Call ``build()`` once with the full node dict and an ``EmbeddingClient``
    to generate embeddings. Then use ``find_nearest()`` for cosine-similarity
    nearest-neighbor search.
    """

    def __init__(self) -> None:
        self._node_ids: list[str] = []
        self._embeddings: list[list[float]] = []

    def to_dict(self) -> dict[str, object]:
        """Serialize this index into a JSON-friendly dict."""
        return {
            "node_ids": self._node_ids,
            "embeddings": self._embeddings,
        }

    def save(self, path: Path) -> None:
        """Persist index vectors to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> NodeEmbeddingIndex:
        """Load index vectors from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls()
        idx._node_ids = [str(node_id) for node_id in data.get("node_ids", [])]
        idx._embeddings = [
            [float(x) for x in vector]
            for vector in data.get("embeddings", [])
            if isinstance(vector, list)
        ]
        return idx

    async def build(self, nodes: dict[str, HKGNode], client: EmbeddingClient) -> None:
        """Build the embedding index for all nodes.

        Embeds each node's canonical name. The resulting vectors are stored
        alongside the node IDs for later similarity search.
        """
        if not nodes:
            log.warning("embedding_index_build_empty", msg="No nodes to embed")
            return

        node_ids = list(nodes.keys())
        names = [nodes[nid].name for nid in node_ids]

        log.info("embedding_index_building", node_count=len(node_ids))

        # Batch embed all names
        vectors = await client.embed(names)

        self._node_ids = node_ids
        self._embeddings = vectors

        log.info("embedding_index_built", node_count=len(node_ids))

    def find_nearest(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find the nearest nodes to ``query_embedding`` by cosine similarity.

        Parameters
        ----------
        query_embedding:
            The query vector to compare against all indexed nodes.
        top_k:
            Maximum number of results to return.
        threshold:
            Minimum cosine similarity to include a result.

        Returns
        -------
        list[tuple[str, float]]:
            List of (node_id, similarity_score) tuples, sorted descending by score.
        """
        if not self._node_ids:
            return []

        scored: list[tuple[str, float]] = []
        for nid, emb in zip(self._node_ids, self._embeddings):
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                scored.append((nid, sim))

        # Sort by similarity descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
