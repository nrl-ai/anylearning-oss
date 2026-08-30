"""Lightweight inference lifecycle and backend registry interfaces."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from enum import Enum
from threading import Event, RLock
from typing import Any

from .contracts import InferenceRequest, InferenceResult, ModelCapabilities

logger = logging.getLogger(__name__)


class InferenceCancelledError(RuntimeError):
    """Raised when cooperative cancellation stops inference work."""


class SessionLifecycleError(RuntimeError):
    """Raised when a session operation is invalid for its current state."""


class RegistryError(LookupError):
    """Raised for duplicate, missing, or invalid inference backends."""


class CancellationToken:
    """Thread-safe cooperative cancellation with linked-token support."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = RLock()
        self._reason: str | None = None
        self._callbacks: list[Callable[[str | None], None]] = []
        self._links: list[tuple[CancellationToken, Callable[[str | None], None]]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str | None = None) -> bool:
        """Cancel once and notify linked tokens; return whether state changed."""
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            callback(reason)
        self.close()
        return True

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            detail = f": {self.reason}" if self.reason else ""
            raise InferenceCancelledError(f"Inference was cancelled{detail}")

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def _add_callback(self, callback: Callable[[str | None], None]) -> bool:
        with self._lock:
            if self._event.is_set():
                reason = self._reason
            else:
                self._callbacks.append(callback)
                return True
        callback(reason)
        return False

    def _remove_callback(self, callback: Callable[[str | None], None]) -> None:
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def close(self) -> None:
        """Stop listening to parents without changing cancellation state."""
        with self._lock:
            links = tuple(self._links)
            self._links.clear()
        for parent, callback in links:
            parent._remove_callback(callback)

    @classmethod
    def linked(cls, *parents: CancellationToken | None) -> CancellationToken:
        token = cls()
        for parent in parents:
            if parent is not None:
                callback = token.cancel
                if not parent._add_callback(callback):
                    break
                with token._lock:
                    if token._event.is_set():
                        remove_callback = True
                    else:
                        token._links.append((parent, callback))
                        remove_callback = False
                if remove_callback:
                    parent._remove_callback(callback)
                    break
        return token


class SessionState(str, Enum):
    NEW = "new"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    CLOSED = "closed"
    FAILED = "failed"


class InferenceSession(ABC):
    """Runtime contract implemented by every local model session."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> SessionState:
        raise NotImplementedError

    @abstractmethod
    def load(self, cancellation: CancellationToken | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken | None = None,
    ) -> InferenceResult:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError


class BaseInferenceSession(InferenceSession):
    """Serialized, exception-safe lifecycle shared by inference backends."""

    def __init__(self, capabilities: ModelCapabilities) -> None:
        self._capabilities = capabilities
        self._state = SessionState.NEW
        self._state_lock = RLock()
        self._operation_lock = RLock()
        self._shutdown = CancellationToken()

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def state(self) -> SessionState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: SessionState) -> None:
        with self._state_lock:
            self._state = state

    def load(self, cancellation: CancellationToken | None = None) -> None:
        with self._operation_lock:
            if self.state is SessionState.READY:
                return
            if self.state not in (SessionState.NEW, SessionState.FAILED):
                raise SessionLifecycleError(
                    f"Cannot load a session in state {self.state.value!r}"
                )
            token = cancellation or CancellationToken()
            token.raise_if_cancelled()
            self._shutdown = CancellationToken()
            self._set_state(SessionState.LOADING)
            try:
                self._load(token)
                token.raise_if_cancelled()
            except Exception:
                try:
                    self._unload()
                except Exception:
                    logger.exception(
                        "Inference session cleanup failed after load error"
                    )
                self._set_state(SessionState.FAILED)
                raise
            self._set_state(SessionState.READY)

    def predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken | None = None,
    ) -> InferenceResult:
        with self._operation_lock:
            if self.state is not SessionState.READY:
                raise SessionLifecycleError(
                    f"Cannot predict with a session in state {self.state.value!r}"
                )
            if request.model_id != self.capabilities.model_id:
                raise ValueError("Request model_id does not match the loaded session")
            if request.model_revision != self.capabilities.model_revision:
                raise ValueError(
                    "Request model_revision does not match the loaded session"
                )
            token = CancellationToken.linked(cancellation, self._shutdown)
            try:
                token.raise_if_cancelled()
                result = self._predict(request, image, token)
                token.raise_if_cancelled()
                if not isinstance(result, InferenceResult):
                    raise TypeError("Inference backends must return InferenceResult")
                identity_fields = (
                    "protocol_version",
                    "request_id",
                    "source_id",
                    "model_id",
                    "model_revision",
                )
                mismatches = [
                    field
                    for field in identity_fields
                    if getattr(result, field) != getattr(request, field)
                ]
                if mismatches:
                    raise ValueError(
                        "Inference result identity does not match the request: "
                        + ", ".join(mismatches)
                    )
                return result
            finally:
                token.close()

    def unload(self) -> None:
        self._shutdown.cancel("session unload requested")
        with self._operation_lock:
            if self.state is SessionState.CLOSED:
                return
            self._set_state(SessionState.UNLOADING)
            try:
                self._unload()
            finally:
                self._set_state(SessionState.CLOSED)

    @abstractmethod
    def _load(self, cancellation: CancellationToken) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        raise NotImplementedError

    @abstractmethod
    def _unload(self) -> None:
        raise NotImplementedError


class InferenceBackend(ABC):
    """Factory and discovery boundary for one model configuration family."""

    backend_id: str

    @abstractmethod
    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        raise NotImplementedError

    @abstractmethod
    def create_session(self, config: Mapping[str, Any]) -> InferenceSession:
        raise NotImplementedError


BackendFactory = Callable[[], InferenceBackend]
_BACKEND_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class ModelRegistry:
    """Thread-safe lazy registry for model backend factories."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}
        self._backends: dict[str, InferenceBackend] = {}
        self._lock = RLock()

    def register(
        self,
        backend_id: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not _BACKEND_ID.fullmatch(backend_id):
            raise RegistryError(f"Invalid backend id: {backend_id!r}")
        if not callable(factory):
            raise TypeError("Backend factory must be callable")
        with self._lock:
            if backend_id in self._factories and not replace:
                raise RegistryError(f"Backend is already registered: {backend_id}")
            self._factories[backend_id] = factory
            self._backends.pop(backend_id, None)

    def backend_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def get(self, backend_id: str) -> InferenceBackend:
        with self._lock:
            backend = self._backends.get(backend_id)
            if backend is not None:
                return backend
            try:
                factory = self._factories[backend_id]
            except KeyError as error:
                raise RegistryError(
                    f"Unknown inference backend: {backend_id}"
                ) from error
            backend = factory()
            if backend.backend_id != backend_id:
                raise RegistryError(
                    f"Backend factory for {backend_id!r} returned {backend.backend_id!r}"
                )
            self._backends[backend_id] = backend
            return backend

    def create_session(
        self, backend_id: str, config: Mapping[str, Any]
    ) -> InferenceSession:
        return self.get(backend_id).create_session(config)
