"""Persistent lightweight pattern memory for SAGA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from x_creative.saga.memory.semantic_hash import SemanticHasher


class PatternMemory:
    """Track repeated hypothesis patterns across runs."""

    def __init__(
        self,
        storage_path: Path | None = None,
        hasher: SemanticHasher | None = None,
    ) -> None:
        self._storage_path = storage_path
        self._hasher = hasher
        self._counts: dict[str, int] = {}
        self._load()

    @staticmethod
    def _normalize(text: str | None) -> str:
        if not text:
            return ""
        return " ".join(text.lower().split())

    @classmethod
    def _fallback_semantic_hash(cls, text: str, n_bits: int = 64) -> str:
        """Deterministic token-level SimHash fallback without external embeddings."""
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", cls._normalize(text))
        if not tokens:
            return "0" * n_bits

        accumulator = [0] * n_bits
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bits = int.from_bytes(digest[:8], byteorder="big", signed=False)
            for idx in range(n_bits):
                if (bits >> idx) & 1:
                    accumulator[idx] += 1
                else:
                    accumulator[idx] -= 1
        return "".join("1" if value >= 0 else "0" for value in accumulator)

    async def fingerprint(self, description: str | None, observable: str | None) -> str:
        """Build a stable fingerprint from hypothesis text.

        Uses semantic hash if hasher is available, otherwise falls back
        to local token-level semantic hashing.
        """
        combined = f"{description or ''} {observable or ''}"
        if self._hasher is not None:
            return await self._hasher.hash(combined)
        return self._fallback_semantic_hash(combined)

    def get_count(self, fingerprint: str) -> int:
        return int(self._counts.get(fingerprint, 0))

    async def record_batch(self, hypotheses: list[dict[str, Any]]) -> None:
        """Record a hypothesis batch into memory and persist."""
        for item in hypotheses:
            description = str(item.get("description", ""))
            observable = str(item.get("observable", ""))
            fp = await self.fingerprint(description, observable)
            if not fp:
                continue
            self._counts[fp] = self.get_count(fp) + 1
        self._save()

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:
            return
        counts = payload.get("counts", {})
        if isinstance(counts, dict):
            for key, value in counts.items():
                try:
                    self._counts[str(key)] = int(value)
                except Exception:
                    continue

    def _save(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"counts": self._counts}
        self._storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
