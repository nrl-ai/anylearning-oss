import gc
import weakref
from concurrent.futures import CancelledError
from threading import Event
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from anylearning.inference import (
    BaseInferenceSession,
    CancellationToken,
    DuplicateInferenceRequestError,
    InferenceCancelledError,
    InferenceQueue,
    InferenceQueueClosedError,
    InferenceQueueFullError,
    InferenceRequest,
    InferenceResult,
    ModelCapabilities,
    ModelTask,
)
from anylearning.inference.backends.yolo_onnx import YoloOnnxBackend

CAPABILITIES = ModelCapabilities(
    model_id="queued-model",
    model_revision="revision-1",
    tasks=(ModelTask.DETECTION,),
    supports_cancellation=True,
)


def _request(index: int) -> InferenceRequest:
    return InferenceRequest(
        request_id=f"request-{index}",
        source_id=f"source-{index}",
        model_id=CAPABILITIES.model_id,
        model_revision=CAPABILITIES.model_revision,
    )


class BlockingSession(BaseInferenceSession):
    def __init__(self) -> None:
        super().__init__(CAPABILITIES)
        self.started = Event()
        self.release = Event()
        self.predicted: list[str] = []

    def _load(self, cancellation: CancellationToken) -> None:
        cancellation.raise_if_cancelled()

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        self.predicted.append(request.request_id)
        self.started.set()
        while not self.release.wait(0.01):
            cancellation.raise_if_cancelled()
        cancellation.raise_if_cancelled()
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
        )

    def _unload(self) -> None:
        self.release.set()


class FailingOnceSession(BaseInferenceSession):
    def __init__(self) -> None:
        super().__init__(CAPABILITIES)
        self.calls = 0

    def _load(self, cancellation: CancellationToken) -> None:
        cancellation.raise_if_cancelled()

    def _predict(self, request, image, cancellation):
        self.calls += 1
        cancellation.raise_if_cancelled()
        if self.calls == 1:
            raise RuntimeError("fixture inference failed")
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
        )

    def _unload(self) -> None:
        pass


@pytest.fixture
def blocking_session():
    session = BlockingSession()
    session.load()
    yield session
    session.release.set()
    session.unload()


def test_queue_bounds_items_and_retained_image_bytes(blocking_session):
    snapshots = []
    queue = InferenceQueue(
        blocking_session,
        max_pending=3,
        max_image_bytes=10,
        max_pending_bytes=15,
        progress=snapshots.append,
    )
    first = queue.submit(_request(1), b"12345678")
    assert blocking_session.started.wait(1)
    second = queue.submit(_request(2), b"1234567")

    assert queue.progress.pending_bytes == 15
    assert queue.progress.running == 1
    assert queue.progress.queued == 1
    with pytest.raises(InferenceQueueFullError, match="byte limit"):
        queue.submit(_request(3), b"1")
    with pytest.raises(InferenceQueueFullError, match="per-image limit"):
        queue.submit(_request(4), b"12345678901")

    blocking_session.release.set()
    assert first.result(1).request_id == "request-1"
    assert second.result(1).request_id == "request-2"
    queue.close()
    assert queue.progress.completed == 2
    assert queue.progress.succeeded == 2
    assert queue.progress.pending_bytes == 0
    assert snapshots[-1] == queue.progress


def test_queue_rejects_duplicates_and_submissions_after_close(blocking_session):
    queue = InferenceQueue(blocking_session, max_pending=2)
    first = queue.submit(_request(1), b"image")
    assert blocking_session.started.wait(1)
    with pytest.raises(DuplicateInferenceRequestError, match="already pending"):
        queue.submit(_request(1), b"image")
    queue.submit(_request(2), b"image")
    with pytest.raises(InferenceQueueFullError, match="item limit"):
        queue.submit(_request(3), b"image")
    blocking_session.release.set()
    first.result(1)
    queue.close()
    with pytest.raises(InferenceQueueClosedError, match="closed"):
        queue.submit(_request(2), b"image")


def test_shutdown_cancels_running_and_queued_work_and_releases_bytes(
    blocking_session,
):
    queue = InferenceQueue(blocking_session, max_pending=2)
    running = queue.submit(_request(1), b"running")
    assert blocking_session.started.wait(1)
    queued = queue.submit(_request(2), b"queued")

    queue.close()

    with pytest.raises(InferenceCancelledError, match="queue shutdown"):
        running.result(1)
    with pytest.raises(CancelledError):
        queued.result(1)
    assert queue.progress.cancelled == 2
    assert queue.progress.pending_bytes == 0


def test_cancelling_queued_work_never_calls_the_backend(blocking_session):
    queue = InferenceQueue(blocking_session, max_pending=2)
    first = queue.submit(_request(1), b"image")
    assert blocking_session.started.wait(1)
    queued = queue.submit(_request(2), b"image")
    assert queued.cancel("no longer visible")
    blocking_session.release.set()

    first.result(1)
    with pytest.raises(CancelledError):
        queued.result(1)
    queue.close()
    assert blocking_session.predicted == ["request-1"]
    assert queue.progress.cancelled == 1


def test_cancelling_running_work_is_linked_to_session_prediction(blocking_session):
    queue = InferenceQueue(blocking_session)
    running = queue.submit(_request(1), b"image")
    assert blocking_session.started.wait(1)
    assert running.cancel("client disconnected")

    with pytest.raises(InferenceCancelledError, match="client disconnected"):
        running.result(1)
    queue.close()
    assert queue.progress.cancelled == 1


def test_queue_requires_explicit_size_for_unknown_image_objects(blocking_session):
    queue = InferenceQueue(blocking_session)
    with pytest.raises(ValueError, match="image_bytes"):
        queue.submit(_request(1), object())
    job = queue.submit(_request(1), object(), image_bytes=7)
    assert blocking_session.started.wait(1)
    blocking_session.release.set()
    assert job.result(1).request_id == "request-1"
    queue.close()


def test_backend_failure_does_not_kill_worker_or_block_retry():
    session = FailingOnceSession()
    session.load()
    queue = InferenceQueue(session, max_pending=2)
    failed = queue.submit(_request(1), b"image")
    recovered = queue.submit(_request(2), b"image")

    with pytest.raises(RuntimeError, match="fixture inference failed"):
        failed.result(1)
    assert recovered.result(1).request_id == "request-2"
    queue.close()
    session.unload()
    assert queue.progress.failed == 1
    assert queue.progress.succeeded == 1


def test_idle_queue_releases_successful_and_failed_image_objects():
    session = FailingOnceSession()
    session.load()
    queue = InferenceQueue(session, max_pending=2)
    failed_image = np.zeros((8, 8, 3), dtype=np.uint8)
    successful_image = np.zeros((8, 8, 3), dtype=np.uint8)
    failed_reference = weakref.ref(failed_image)
    successful_reference = weakref.ref(successful_image)
    failed = queue.submit(_request(1), failed_image)
    succeeded = queue.submit(_request(2), successful_image)
    del failed_image, successful_image

    with pytest.raises(RuntimeError, match="fixture inference failed"):
        failed.result(1)
    succeeded.result(1)
    del failed, succeeded
    gc.collect()

    assert failed_reference() is None
    assert successful_reference() is None
    queue.close()
    session.unload()


def _real_constant_yolo_model(path) -> None:
    predictions = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["stored_predictions"], ["predictions"])],
        "queued-real-onnx",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info(
                "predictions", TensorProto.FLOAT, list(predictions.shape)
            )
        ],
        initializer=[numpy_helper.from_array(predictions, name="stored_predictions")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save_model(model, path)


def test_queue_runs_an_actual_onnx_session_and_preserves_result_identity(tmp_path):
    path = tmp_path / "queued-yolo.onnx"
    _real_constant_yolo_model(path)
    session = YoloOnnxBackend().create_session(
        {
            "name": "queued-real-onnx",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        }
    )
    session.load()
    request = InferenceRequest(
        request_id="real-runtime-request",
        source_id="image-sha256:fixture",
        model_id=session.capabilities.model_id,
        model_revision=session.capabilities.model_revision,
    )
    with InferenceQueue(session) as queue:
        result = queue.submit(request, np.zeros((32, 32, 3), dtype=np.uint8)).result(2)
    session.unload()

    assert result.request_id == request.request_id
    assert result.source_id == request.source_id
    assert [shape.label for shape in result.shapes] == ["cat"]
