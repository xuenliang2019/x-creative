"""LRU cache for HKG traversal results."""

from __future__ import annotations

from collections import OrderedDict

from x_creative.hkg.types import Hyperpath

TraversalCacheKey = tuple[frozenset[str], frozenset[str], int, int, int]


class TraversalCache:
    """Simple LRU cache keyed by (frozenset(start), frozenset(end), K, IS, max_len).

    Uses an ``OrderedDict`` to maintain insertion/access order and evict the
    least-recently-used entry when ``max_size`` is exceeded.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[TraversalCacheKey, list[Hyperpath]] = OrderedDict()

    def get(self, key: TraversalCacheKey) -> list[Hyperpath] | None:
        """Return cached paths for *key*, or ``None`` on a miss.

        A successful hit moves the entry to the end (most-recently-used).
        """
        if key not in self._cache:
            return None
        # Move to end to mark as recently used
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: TraversalCacheKey, paths: list[Hyperpath]) -> None:
        """Store *paths* under *key*, evicting the LRU entry if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = paths
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # evict oldest (LRU)

    def invalidate(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
