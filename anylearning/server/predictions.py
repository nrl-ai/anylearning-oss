"""Bounded shared sessions and opaque asynchronous prediction jobs."""

from __future__ import annotations

import logging
import secrets
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Callable

import numpy as np

from anylearning.inference import (
    DuplicateInferenceRequestError,
    InferenceCancelledError,
    InferenceJob,
    InferenceQueue,
    InferenceQueueError,
    InferenceRequest,
    InferenceResult,
    InferenceSession,
    ModelCapabilities,
    ModelRegistry,
    create_default_registry,
)

from .models import ServerModelDefinition

logger = logging.getLogger(__name__)


class PredictionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class PredictionServiceError(RuntimeError):
    """Base error returned by the bounded prediction manager."""


class PredictionServiceUnavailableError(PredictionServiceError):
    pass


class PredictionCapacityError(PredictionServiceError):
    pass


class PredictionNotFoundError(PredictionServiceError):
    pass


@dataclass(frozen=True)
class PredictionSnapshot:
    job_id: str
    request_id: str
    state: PredictionState
    result: InferenceResult | None
    error: str | None
    expires_in: int


@dataclass
class _ModelRuntime:
    session: InferenceSession
    queue: InferenceQueue | None = None


@dataclass
class _PredictionRecord:
    job_id: str
    owner_token_id: str
    request_id: str
    job: InferenceJob
    deadline: float
    retain_until: float
    state: PredictionState = PredictionState.QUEUED
    result: InferenceResult | None = None
    error: str | None = None


class PredictionService:
    """Own model sessions, queues, and a bounded non-enumerable result store."""

    def __init__(
        self,
        definitions: tuple[ServerModelDefinition, ...],
        *,
        registry: ModelRegistry | None = None,
        max_jobs: int = 256,
        max_pending_per_model: int = 16,
        max_image_bytes: int = 192 * 1024**2,
        max_pending_bytes_per_model: int = 512 * 1024**2,
        prediction_timeout_seconds: int = 120,
        result_ttl_seconds: int = 300,
        max_result_bytes: int = 8 * 1024**2,
        shutdown_timeout_seconds: float = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry or create_default_registry()
        self._max_jobs = max_jobs
        self._max_pending_per_model = max_pending_per_model
        self._max_image_bytes = max_image_bytes
        self._max_pending_bytes_per_model = max_pending_bytes_per_model
        self._prediction_timeout_seconds = prediction_timeout_seconds
        self._result_ttl_seconds = result_ttl_seconds
        self._max_result_bytes = max_result_bytes
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._clock = clock
        self._lock = RLock()
        self._started = False
        self._closed = False
        self._records: dict[str, _PredictionRecord] = {}
        self._models: dict[str, _ModelRuntime] = {}

        for definition in definitions:
            session = self._registry.create_session(
                definition.backend,
                definition.config,
            )
            model_id = session.capabilities.model_id
            if model_id in self._models:
                raise ValueError("server model identifiers must be unique")
            self._models[model_id] = _ModelRuntime(session=session)

    @property
    def capabilities(self) -> tuple[ModelCapabilities, ...]:
        return tuple(
            runtime.session.capabilities
            for _model_id, runtime in sorted(self._models.items())
        )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise PredictionServiceUnavailableError("prediction service is closed")
            if self._started:
                return
        started: list[_ModelRuntime] = []
        try:
            for _model_id, runtime in sorted(self._models.items()):
                runtime.session.load()
                runtime.queue = InferenceQueue(
                    runtime.session,
                    max_pending=self._max_pending_per_model,
                    max_image_bytes=self._max_image_bytes,
                    max_pending_bytes=self._max_pending_bytes_per_model,
                    thread_name=f"anylearning-model-{len(started) + 1}",
                )
                started.append(runtime)
        except Exception:
            self._stop_runtimes(started)
            raise
        with self._lock:
            self._started = True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            records = tuple(self._records.values())
            self._records.clear()
        for record in records:
            record.job.cancel("inference server shutdown")
        self._stop_runtimes(list(self._models.values()))

    def submit(
        self,
        request: InferenceRequest,
        image: np.ndarray,
        *,
        owner_token_id: str,
    ) -> PredictionSnapshot:
        now = self._clock()
        with self._lock:
            self._reap_locked(now)
            if not self._started or self._closed:
                raise PredictionServiceUnavailableError(
                    "prediction service is not ready"
                )
            runtime = self._models.get(request.model_id)
            if runtime is None or runtime.queue is None:
                raise PredictionNotFoundError("model was not found")
            if len(self._records) >= self._max_jobs:
                raise PredictionCapacityError("prediction job capacity reached")
            job_id = self._new_job_id_locked()
            try:
                job = runtime.queue.submit(request, image, image_bytes=image.nbytes)
            except DuplicateInferenceRequestError:
                raise
            except InferenceQueueError as error:
                raise PredictionCapacityError(
                    "model inference queue is full"
                ) from error
            record = _PredictionRecord(
                job_id=job_id,
                owner_token_id=owner_token_id,
                request_id=request.request_id,
                job=job,
                deadline=now + self._prediction_timeout_seconds,
                retain_until=now + self._result_ttl_seconds,
            )
            self._records[job_id] = record
            job.add_done_callback(
                lambda completed, bound_job_id=job_id: self._job_completed(
                    bound_job_id, completed
                )
            )
            return self._snapshot_locked(record, now)

    def get(self, job_id: str, *, owner_token_id: str) -> PredictionSnapshot:
        now = self._clock()
        timed_out: InferenceJob | None = None
        with self._lock:
            self._reap_locked(now)
            record = self._owned_record_locked(job_id, owner_token_id)
            if (
                record.state in {PredictionState.QUEUED, PredictionState.RUNNING}
                and now >= record.deadline
            ):
                record.state = PredictionState.TIMED_OUT
                record.error = "Prediction deadline exceeded"
                timed_out = record.job
            snapshot = self._snapshot_locked(record, now)
        if timed_out is not None:
            timed_out.cancel("prediction deadline exceeded")
        return snapshot

    def cancel(self, job_id: str, *, owner_token_id: str) -> None:
        with self._lock:
            record = self._owned_record_locked(job_id, owner_token_id)
            if record.state in {PredictionState.QUEUED, PredictionState.RUNNING}:
                record.state = PredictionState.CANCELLED
                record.error = "Prediction was cancelled"
            job = record.job
        job.cancel("prediction cancelled by client")

    def remove(self, job_id: str, *, owner_token_id: str) -> None:
        with self._lock:
            record = self._owned_record_locked(job_id, owner_token_id)
            self._records.pop(job_id, None)
        record.job.cancel("prediction removed by client")

    def reap(self) -> None:
        now = self._clock()
        jobs_to_cancel: list[InferenceJob] = []
        with self._lock:
            for record in self._records.values():
                if (
                    record.state in {PredictionState.QUEUED, PredictionState.RUNNING}
                    and now >= record.deadline
                ):
                    record.state = PredictionState.TIMED_OUT
                    record.error = "Prediction deadline exceeded"
                    jobs_to_cancel.append(record.job)
            self._reap_locked(now)
        for job in jobs_to_cancel:
            job.cancel("prediction deadline exceeded")

    def _job_completed(self, job_id: str, job: InferenceJob) -> None:
        try:
            result = job.result()
            error: Exception | None = None
        except (CancelledError, InferenceCancelledError) as caught:
            result = None
            error = caught
        except Exception as caught:  # backend details remain server-side only
            result = None
            error = caught

        now = self._clock()
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.retain_until = now + self._result_ttl_seconds
            if record.state is PredictionState.TIMED_OUT:
                record.result = None
                return
            if record.state is PredictionState.CANCELLED or isinstance(
                error, (CancelledError, InferenceCancelledError)
            ):
                record.state = PredictionState.CANCELLED
                record.error = "Prediction was cancelled"
                record.result = None
                return
            if error is not None:
                record.state = PredictionState.FAILED
                record.error = "Prediction failed"
                record.result = None
                logger.error(
                    "Inference job failed",
                    extra={"job_id": job_id, "error_type": type(error).__name__},
                )
                return
            if result is None:  # pragma: no cover - defensive invariant
                record.state = PredictionState.FAILED
                record.error = "Prediction failed"
                return
            encoded_size = len(result.model_dump_json().encode("utf-8"))
            if encoded_size > self._max_result_bytes:
                record.state = PredictionState.FAILED
                record.error = "Prediction result exceeded the server limit"
                record.result = None
                return
            record.state = PredictionState.SUCCEEDED
            record.result = result
            record.error = None

    def _snapshot_locked(
        self, record: _PredictionRecord, now: float
    ) -> PredictionSnapshot:
        state = record.state
        if state is PredictionState.QUEUED and record.job.running():
            state = PredictionState.RUNNING
            record.state = state
        return PredictionSnapshot(
            job_id=record.job_id,
            request_id=record.request_id,
            state=state,
            result=record.result,
            error=record.error,
            expires_in=max(0, int(record.retain_until - now)),
        )

    def _owned_record_locked(
        self, job_id: str, owner_token_id: str
    ) -> _PredictionRecord:
        record = self._records.get(job_id)
        if record is None or not secrets.compare_digest(
            record.owner_token_id, owner_token_id
        ):
            raise PredictionNotFoundError("prediction job was not found")
        return record

    def _new_job_id_locked(self) -> str:
        for _attempt in range(4):
            job_id = secrets.token_urlsafe(24)
            if job_id not in self._records:
                return job_id
        raise PredictionCapacityError("could not allocate prediction job")

    def _reap_locked(self, now: float) -> None:
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.state
            in {
                PredictionState.SUCCEEDED,
                PredictionState.FAILED,
                PredictionState.CANCELLED,
                PredictionState.TIMED_OUT,
            }
            and record.retain_until <= now
        ]
        for job_id in expired:
            self._records.pop(job_id, None)

    def _stop_runtimes(self, runtimes: list[_ModelRuntime]) -> None:
        deadline = self._clock() + self._shutdown_timeout_seconds
        stopped: set[int] = set()
        for runtime in runtimes:
            if runtime.queue is None:
                continue
            remaining = max(0.0, deadline - self._clock())
            try:
                runtime.queue.close(timeout=remaining)
                stopped.add(id(runtime))
            except Exception as error:
                logger.error(
                    "Inference queue did not stop cleanly",
                    extra={"error_type": type(error).__name__},
                )
            finally:
                runtime.queue = None
        for runtime in runtimes:
            if id(runtime) not in stopped and runtime.session.state.value == "ready":
                continue
            try:
                runtime.session.unload()
            except Exception as error:
                logger.error(
                    "Inference session did not unload cleanly",
                    extra={"error_type": type(error).__name__},
                )


__all__ = [
    "PredictionCapacityError",
    "PredictionNotFoundError",
    "PredictionService",
    "PredictionServiceUnavailableError",
    "PredictionSnapshot",
    "PredictionState",
]
