"""Small bounded login limiter that does not trust forwarded client headers."""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from threading import RLock


class LoginRateLimiter:
    """Consume attempts before password hashing to bound parallel guessing."""

    def __init__(
        self,
        *,
        attempts_per_client: int = 5,
        global_attempts: int = 120,
        window_seconds: float = 60,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("attempts_per_client", attempts_per_client),
            ("global_attempts", global_attempts),
            ("max_clients", max_clients),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if attempts_per_client > global_attempts:
            raise ValueError("per-client attempts may not exceed the global limit")
        if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._attempts_per_client = attempts_per_client
        self._global_attempts = global_attempts
        self._window_seconds = float(window_seconds)
        self._max_clients = max_clients
        self._clock = clock
        self._clients: OrderedDict[str, deque[float]] = OrderedDict()
        self._global: deque[float] = deque()
        self._lock = RLock()

    def admit(self, client_key: str) -> int | None:
        """Consume one attempt or return a whole-second Retry-After value."""
        if not isinstance(client_key, str) or not client_key or len(client_key) > 256:
            client_key = "unknown"
        now = self._clock()
        cutoff = now - self._window_seconds
        with self._lock:
            self._discard_expired(self._global, cutoff)
            attempts = self._clients.pop(client_key, deque())
            self._discard_expired(attempts, cutoff)
            self._clients[client_key] = attempts
            retry_after = self._retry_after(now, attempts, self._attempts_per_client)
            global_retry = self._retry_after(now, self._global, self._global_attempts)
            if retry_after is not None or global_retry is not None:
                return max(retry_after or 0, global_retry or 0, 1)
            attempts.append(now)
            self._global.append(now)
            while len(self._clients) > self._max_clients:
                self._clients.popitem(last=False)
        return None

    def authentication_succeeded(self, client_key: str) -> None:
        with self._lock:
            self._clients.pop(client_key, None)

    def _retry_after(
        self, now: float, attempts: deque[float], limit: int
    ) -> int | None:
        if len(attempts) < limit:
            return None
        return max(1, math.ceil(attempts[0] + self._window_seconds - now))

    @staticmethod
    def _discard_expired(values: deque[float], cutoff: float) -> None:
        while values and values[0] <= cutoff:
            values.popleft()
