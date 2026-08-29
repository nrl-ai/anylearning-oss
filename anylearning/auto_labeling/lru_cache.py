"""Small thread-safe least-recently-used cache."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar

Key = TypeVar("Key")
Value = TypeVar("Value")


class LRUCache(Generic[Key, Value]):
    """Bounded cache that promotes values whenever they are read."""

    def __init__(self, maxsize: int = 10) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = maxsize
        self.lock = RLock()
        self._cache: OrderedDict[Key, Value] = OrderedDict()

    def get(self, key: Key, default: Value | None = None) -> Value | None:
        with self.lock:
            try:
                value = self._cache.pop(key)
            except KeyError:
                return default
            self._cache[key] = value
            return value

    def put(self, key: Key, value: Value) -> None:
        with self.lock:
            self._cache.pop(key, None)
            self._cache[key] = value
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def find(self, key: Key) -> bool:
        """Compatibility alias for membership checks."""
        return key in self

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()

    def __contains__(self, key: object) -> bool:
        with self.lock:
            return key in self._cache

    def __len__(self) -> int:
        with self.lock:
            return len(self._cache)
