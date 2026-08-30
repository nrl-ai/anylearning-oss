"""Bounded single-session scheduling for desktop and server inference."""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import RLock, Thread, current_thread
from typing import Any, Callable

from .contracts import InferenceRequest, InferenceResult
from .runtime import (
    CancellationToken,
    InferenceCancelledError,
    InferenceSession,
    SessionState,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_IMAGE_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_PENDING_BYTES = 512 * 1024 * 1024
_MAX_PENDING_ITEMS = 10_000
_STOP = object()


class InferenceQueueError(RuntimeError):
    """Base class for bounded scheduling failures."""


class InferenceQueueFullError(InferenceQueueError):
    """Raised before an item would exceed a queue count or byte bound."""


class InferenceQueueClosedError(InferenceQueueError):
    """Raised when work is submitted after shutdown starts."""


class DuplicateInferenceRequestError(InferenceQueueError):
    """Raised while the same request identifier is already in flight."""


class InferenceQueueShutdownError(InferenceQueueError):
    """Raised when a backend ignores cancellation during bounded shutdown."""


@dataclass(frozen=True)
class InferenceQueueProgress:
    """A point-in-time queue snapshot containing counts but no image data."""

    queued: int
    running: int
    completed: int
    succeeded: int
    failed: int
    cancelled: int
    pending_bytes: int


@dataclass(frozen=True)
class _QueueEntry:
    request: InferenceRequest
    image: Any
    image_bytes: int
    token: CancellationToken
    future: Future[InferenceResult]


class InferenceJob:
    """Handle for one queued request and its cooperative cancellation token."""

    def __init__(
        self,
        request_id: str,
        future: Future[InferenceResult],
        token: CancellationToken,
    ) -> None:
        self.request_id = request_id
        self._future = future
        self._token = token

    @property
    def future(self) -> Future[InferenceResult]:
        return self._future

    def cancel(self, reason: str = "inference job cancelled") -> bool:
        token_changed = self._token.cancel(reason)
        future_changed = self._future.cancel()
        return token_changed or future_changed

    def result(self, timeout: float | None = None) -> InferenceResult:
        return self._future.result(timeout)

    def done(self) -> bool:
        return self._future.done()


ProgressCallback = Callable[[InferenceQueueProgress], None]


class InferenceQueue:
    """Run one model session through a count- and byte-bounded FIFO.

    A single worker is intentional: ``BaseInferenceSession`` serializes model
    operations, and adding worker threads would retain more decoded images
    without increasing throughput. Multiple model sessions should each own a
    queue so independent devices or runtimes can make progress concurrently.
    """

    def __init__(
        self,
        session: InferenceSession,
        *,
        max_pending: int = 16,
        max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
        max_pending_bytes: int = _DEFAULT_MAX_PENDING_BYTES,
        progress: ProgressCallback | None = None,
        thread_name: str | None = None,
    ) -> None:
        if not 1 <= max_pending <= _MAX_PENDING_ITEMS:
            raise ValueError(f"max_pending must be between 1 and {_MAX_PENDING_ITEMS}")
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")
        if max_pending_bytes < max_image_bytes:
            raise ValueError("max_pending_bytes must be at least max_image_bytes")
        if session.state is not SessionState.READY:
            raise ValueError("Inference queue requires a ready session")

        self._session = session
        self._max_pending = max_pending
        self._max_image_bytes = max_image_bytes
        self._max_pending_bytes = max_pending_bytes
        self._progress_callback = progress
        self._queue: Queue[_QueueEntry | object] = Queue(maxsize=max_pending)
        self._lock = RLock()
        self._entries: dict[str, _QueueEntry] = {}
        self._pending_bytes = 0
        self._running = 0
        self._completed = 0
        self._succeeded = 0
        self._failed = 0
        self._cancelled = 0
        self._closed = False
        self._stop_enqueued = False
        self._worker = Thread(
            target=self._run,
            name=thread_name or f"inference-{session.capabilities.model_id}",
            daemon=True,
        )
        self._worker.start()

    @property
    def progress(self) -> InferenceQueueProgress:
        with self._lock:
            return self._snapshot_locked()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def submit(
        self,
        request: InferenceRequest,
        image: Any,
        *,
        image_bytes: int | None = None,
        cancellation: CancellationToken | None = None,
    ) -> InferenceJob:
        """Submit without blocking; reject work before a resource bound is crossed."""
        size = _resolve_image_bytes(image, image_bytes)
        if size > self._max_image_bytes:
            raise InferenceQueueFullError(
                f"Image requires {size} bytes; per-image limit is "
                f"{self._max_image_bytes} bytes"
            )
        token = CancellationToken.linked(cancellation)
        future: Future[InferenceResult] = Future()
        entry = _QueueEntry(request, image, size, token, future)
        try:
            with self._lock:
                if self._closed:
                    raise InferenceQueueClosedError("Inference queue is closed")
                if request.request_id in self._entries:
                    raise DuplicateInferenceRequestError(
                        "Inference request_id is already pending"
                    )
                if len(self._entries) >= self._max_pending:
                    raise InferenceQueueFullError("Inference queue item limit reached")
                if self._pending_bytes + size > self._max_pending_bytes:
                    raise InferenceQueueFullError("Inference queue byte limit reached")
                self._entries[request.request_id] = entry
                self._pending_bytes += size
                self._queue.put_nowait(entry)
                snapshot = self._snapshot_locked()
        except Exception:
            token.close()
            raise
        self._notify_progress(snapshot)
        return InferenceJob(request.request_id, future, token)

    def close(
        self,
        *,
        cancel_pending: bool = True,
        timeout: float | None = 10.0,
    ) -> None:
        """Stop accepting work and wait a bounded time for the worker to exit."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        with self._lock:
            self._closed = True
            entries = tuple(self._entries.values()) if cancel_pending else ()
            enqueue_stop = not self._stop_enqueued
            if enqueue_stop:
                self._stop_enqueued = True
        for entry in entries:
            entry.token.cancel("inference queue shutdown")
            entry.future.cancel()
        if enqueue_stop:
            self._queue.put(_STOP)
        if current_thread() is self._worker:
            return
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise InferenceQueueShutdownError(
                "Inference worker did not stop within the configured timeout"
            )

    def __enter__(self) -> InferenceQueue:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            queued = self._queue.get()
            try:
                if queued is _STOP:
                    return
                if not isinstance(queued, _QueueEntry):
                    raise AssertionError("Unexpected inference queue item")
                self._execute(queued)
            finally:
                self._queue.task_done()

    def _execute(self, entry: _QueueEntry) -> None:
        if not entry.future.set_running_or_notify_cancel():
            snapshot = self._finish(entry, cancelled=True)
            self._notify_progress(snapshot)
            return
        with self._lock:
            self._running += 1
            snapshot = self._snapshot_locked()
        self._notify_progress(snapshot)

        result: InferenceResult | None = None
        error: Exception | None = None
        cancelled = False
        try:
            entry.token.raise_if_cancelled()
            result = self._session.predict(
                entry.request,
                entry.image,
                cancellation=entry.token,
            )
        except InferenceCancelledError as caught:
            error = caught
            cancelled = True
        except Exception as caught:
            error = caught

        snapshot = self._finish(entry, cancelled=cancelled, succeeded=error is None)
        if error is not None:
            entry.future.set_exception(error)
        elif result is None:  # pragma: no cover - defensive invariant
            entry.future.set_exception(RuntimeError("Inference returned no result"))
        else:
            entry.future.set_result(result)
        self._notify_progress(snapshot)

    def _finish(
        self,
        entry: _QueueEntry,
        *,
        cancelled: bool,
        succeeded: bool = False,
    ) -> InferenceQueueProgress:
        entry.token.close()
        with self._lock:
            removed = self._entries.pop(entry.request.request_id, None)
            if removed is None:  # pragma: no cover - internal invariant
                raise AssertionError("Inference queue entry was already removed")
            self._pending_bytes -= entry.image_bytes
            if entry.future.running():
                self._running -= 1
            self._completed += 1
            if cancelled or entry.future.cancelled():
                self._cancelled += 1
            elif succeeded:
                self._succeeded += 1
            else:
                self._failed += 1
            return self._snapshot_locked()

    def _snapshot_locked(self) -> InferenceQueueProgress:
        return InferenceQueueProgress(
            queued=len(self._entries) - self._running,
            running=self._running,
            completed=self._completed,
            succeeded=self._succeeded,
            failed=self._failed,
            cancelled=self._cancelled,
            pending_bytes=self._pending_bytes,
        )

    def _notify_progress(self, snapshot: InferenceQueueProgress) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(snapshot)
        except Exception:
            logger.exception("Inference queue progress callback failed")


def _resolve_image_bytes(image: Any, configured: int | None) -> int:
    if configured is not None:
        size = configured
    elif isinstance(image, memoryview):
        size = image.nbytes
    elif isinstance(image, (bytes, bytearray)):
        size = len(image)
    else:
        size = getattr(image, "nbytes", None)
        if not isinstance(size, int):
            width = getattr(image, "width", None)
            height = getattr(image, "height", None)
            getbands = getattr(image, "getbands", None)
            if (
                isinstance(width, int)
                and isinstance(height, int)
                and callable(getbands)
            ):
                bands = getbands()
                size = width * height * len(bands)
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(
            "image_bytes must be a positive integer when image size is unknown"
        )
    return size
