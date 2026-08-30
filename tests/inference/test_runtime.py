from collections.abc import Mapping
from threading import Event, Thread
from typing import Any

import pytest

from anylearning.inference import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    InferenceCancelledError,
    InferenceRequest,
    InferenceResult,
    ModelCapabilities,
    ModelRegistry,
    ModelTask,
    RegistryError,
    SessionLifecycleError,
    SessionState,
)

CAPABILITIES = ModelCapabilities(
    model_id="test-model",
    model_revision="revision-1",
    tasks=(ModelTask.DETECTION,),
    supports_cancellation=True,
)


def request(**overrides: str) -> InferenceRequest:
    values = {
        "request_id": "request-1",
        "source_id": "source-1",
        "model_id": CAPABILITIES.model_id,
        "model_revision": CAPABILITIES.model_revision,
    }
    values.update(overrides)
    return InferenceRequest(**values)


class RecordingSession(BaseInferenceSession):
    def __init__(self, *, fail_load: bool = False) -> None:
        super().__init__(CAPABILITIES)
        self.fail_load = fail_load
        self.events: list[str] = []

    def _load(self, cancellation: CancellationToken) -> None:
        self.events.append("load")
        cancellation.raise_if_cancelled()
        if self.fail_load:
            raise RuntimeError("load failed")

    def _predict(
        self,
        inference_request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        self.events.append(f"predict:{image}")
        cancellation.raise_if_cancelled()
        return InferenceResult(
            request_id=inference_request.request_id,
            source_id=inference_request.source_id,
            model_id=inference_request.model_id,
            model_revision=inference_request.model_revision,
        )

    def _unload(self) -> None:
        self.events.append("unload")


class RecordingBackend(InferenceBackend):
    backend_id = "recording"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        return CAPABILITIES

    def create_session(self, config: Mapping[str, Any]) -> RecordingSession:
        return RecordingSession()


def test_session_enforces_load_predict_unload_lifecycle():
    session = RecordingSession()

    with pytest.raises(SessionLifecycleError):
        session.predict(request(), "image")

    session.load()
    assert session.state is SessionState.READY
    result = session.predict(request(), "image")
    assert result.request_id == "request-1"

    session.unload()
    session.unload()
    assert session.state is SessionState.CLOSED
    assert session.events == ["load", "predict:image", "unload"]

    with pytest.raises(SessionLifecycleError):
        session.predict(request(), "image")


def test_failed_load_can_be_retried_without_recreating_session():
    session = RecordingSession(fail_load=True)

    with pytest.raises(RuntimeError, match="load failed"):
        session.load()
    assert session.state is SessionState.FAILED

    session.fail_load = False
    session.load()
    assert session.state is SessionState.READY


def test_cancellation_is_linked_and_preserves_first_reason():
    parent = CancellationToken()
    child = CancellationToken.linked(parent)

    assert parent.cancel("client disconnected")
    assert not parent.cancel("later reason")
    assert child.cancelled
    assert child.reason == "client disconnected"
    with pytest.raises(InferenceCancelledError, match="client disconnected"):
        child.raise_if_cancelled()


def test_successful_prediction_detaches_temporary_cancellation_links():
    session = RecordingSession()
    session.load()
    caller = CancellationToken()

    session.predict(request(), "image", caller)

    assert caller._callbacks == []
    assert session._shutdown._callbacks == []


def test_unload_cooperatively_cancels_an_active_prediction():
    started = Event()
    errors = []

    class BlockingSession(RecordingSession):
        def _predict(self, inference_request, image, cancellation):
            started.set()
            cancellation.wait(2)
            cancellation.raise_if_cancelled()
            raise AssertionError("prediction was not cancelled")

    session = BlockingSession()
    session.load()

    def predict() -> None:
        try:
            session.predict(request(), "image")
        except Exception as error:
            errors.append(error)

    worker = Thread(target=predict)
    worker.start()
    assert started.wait(1)
    session.unload()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], InferenceCancelledError)
    assert session.state is SessionState.CLOSED


def test_session_rejects_stale_model_identity():
    session = RecordingSession()
    session.load()

    with pytest.raises(ValueError, match="model_revision"):
        session.predict(request(model_revision="stale"), "image")


def test_session_rejects_a_backend_result_for_another_request():
    class StaleResultSession(RecordingSession):
        def _predict(self, inference_request, image, cancellation):
            return InferenceResult(
                request_id="different-request",
                source_id=inference_request.source_id,
                model_id=inference_request.model_id,
                model_revision=inference_request.model_revision,
            )

    session = StaleResultSession()
    session.load()

    with pytest.raises(ValueError, match="request_id"):
        session.predict(request(), "image")


def test_session_rejects_a_non_contract_backend_result():
    class InvalidResultSession(RecordingSession):
        def _predict(self, inference_request, image, cancellation):
            return {"request_id": inference_request.request_id}

    session = InvalidResultSession()
    session.load()

    with pytest.raises(TypeError, match="InferenceResult"):
        session.predict(request(), "image")


def test_registry_loads_factories_lazily_and_validates_identity():
    registry = ModelRegistry()
    calls: list[str] = []

    def factory() -> RecordingBackend:
        calls.append("factory")
        return RecordingBackend()

    registry.register("recording", factory)
    assert registry.backend_ids() == ("recording",)
    assert calls == []

    first = registry.get("recording")
    assert registry.get("recording") is first
    assert calls == ["factory"]
    assert isinstance(registry.create_session("recording", {}), RecordingSession)

    with pytest.raises(RegistryError, match="already registered"):
        registry.register("recording", factory)
    with pytest.raises(RegistryError, match="Unknown"):
        registry.get("missing")
    with pytest.raises(RegistryError, match="Invalid"):
        registry.register("Not Valid", factory)
