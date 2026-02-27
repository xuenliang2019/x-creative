"""Semantic hashing via SimHash for cross-session pattern tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from x_creative.hkg.embeddings import EmbeddingClient


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    if len(hash_a) != len(hash_b):
        raise ValueError(f"Hash lengths differ: {len(hash_a)} vs {len(hash_b)}")
    return sum(a != b for a, b in zip(hash_a, hash_b))


class SemanticHasher:
    """SimHash-based semantic fingerprinting.

    Uses embedding vectors projected through random hyperplanes
    to produce locality-sensitive binary hashes.
    """

    def __init__(
        self,
        embedding_service: EmbeddingClient,
        n_bits: int = 64,
        embedding_dim: int = 1536,
    ) -> None:
        self._embedder = embedding_service
        self._n_bits = n_bits
        rng = np.random.RandomState(42)
        self._planes = rng.randn(n_bits, embedding_dim)

    async def hash(self, text: str) -> str:
        """Compute semantic hash from embedding via SimHash."""
        embedding = await self._embedder.embed_cached(text)
        vec = np.array(embedding)
        projections = self._planes @ vec
        bits = (projections > 0).astype(int)
        return "".join(str(b) for b in bits)
