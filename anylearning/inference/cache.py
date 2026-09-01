"""Small bounded caches used by inference sessions."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar

Key = TypeVar("Key")
Value = TypeVar("Value")


class LRUCache(Generic[Key, Value]):
    """Thread-safe least-recently-used cache with a fixed item bound."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._maxsize = maxsize
        self._items: OrderedDict[Key, Value] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Key) -> Value | None:
        with self._lock:
            try:
                value = self._items.pop(key)
            except KeyError:
                return None
            self._items[key] = value
            return value

    def put(self, key: Key, value: Value) -> None:
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = value
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
