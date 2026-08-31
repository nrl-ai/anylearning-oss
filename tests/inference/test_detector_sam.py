from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anylearning.inference import (
    BoxPrompt,
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    PointPrompt,
    ShapeType,
)
from anylearning.inference.backends.detector_sam import (
    DetectorSamBackend,
    DetectorSamConfig,
)
from anylearning.inference.runtime import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    InferenceCancelledError,
    ModelRegistry,
    SessionState,
)
from anylearning.server.models import ServerModelDefinition

ResultFactory = Callable[
    [InferenceRequest, Any, CancellationToken, int], InferenceResult
]


class RecordingSession(BaseInferenceSession):
    def __init__(
        self,
        capabilities: ModelCapabilities,
        result_factory: ResultFactory,
        *,
        fail_load: bool = False,
    ) -> None:
        self.requests: list[InferenceRequest] = []
        self.images: list[Any] = []
        self.load_calls = 0
        self.unload_calls = 0
        self._result_factory = result_factory
        self._fail_load = fail_load
        super().__init__(capabilities)

    def _load(self, cancellation: CancellationToken) -> None:
        cancellation.raise_if_cancelled()
        self.load_calls += 1
        if self._fail_load:
            raise RuntimeError("scripted load failure")

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        self.requests.append(request)
        self.images.append(image)
        return self._result_factory(
            request,
            image,
            cancellation,
            len(self.requests) - 1,
        )

    def _unload(self) -> None:
        self.unload_calls += 1


class RecordingBackend(InferenceBackend):
    def __init__(
        self,
        backend_id: str,
        tasks: tuple[ModelTask, ...],
        result_factory: ResultFactory,
        *,
        fail_load: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self.tasks = tasks
        self.result_factory = result_factory
        self.fail_load = fail_load
        self.configs: list[dict[str, Any]] = []
        self.sessions: list[RecordingSession] = []

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=str(config["name"]),
            model_revision=str(config.get("revision", f"{self.backend_id}-revision")),
            tasks=self.tasks,
            supports_cancellation=True,
        )

    def create_session(self, config: Mapping[str, Any]) -> RecordingSession:
        self.configs.append(dict(config))
        session = RecordingSession(
            self.capabilities(config),
            self.result_factory,
            fail_load=self.fail_load,
        )
        self.sessions.append(session)
        return session


def _result(
    request: InferenceRequest,
    *,
    shapes: tuple[InferenceShape, ...] = (),
    warnings: tuple[str, ...] = (),
    timings_ms: dict[str, float] | None = None,
) -> InferenceResult:
    return InferenceResult(
        request_id=request.request_id,
        source_id=request.source_id,
        model_id=request.model_id,
        model_revision=request.model_revision,
        shapes=shapes,
        warnings=warnings,
        timings_ms=timings_ms or {},
    )


def _rectangle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    label: str = "object",
    score: float = 0.9,
    class_id: int = 0,
) -> InferenceShape:
    return InferenceShape(
        type=ShapeType.RECTANGLE,
        points=(Point(x=x1, y=y1), Point(x=x2, y=y2)),
        label=label,
        score=score,
        attributes={"class_id": class_id},
    )


def _polygon(*coordinates: tuple[float, float], score: float | None = None):
    return InferenceShape(
        type=ShapeType.POLYGON,
        points=tuple(Point(x=x, y=y) for x, y in coordinates),
        score=score,
    )


def _registry(
    detector_factory: ResultFactory,
    segmenter_factory: ResultFactory,
    *,
    detector_tasks: tuple[ModelTask, ...] = (ModelTask.DETECTION,),
    segmenter_tasks: tuple[ModelTask, ...] = (ModelTask.PROMPTABLE_SEGMENTATION,),
    segmenter_fail_load: bool = False,
) -> tuple[ModelRegistry, RecordingBackend, RecordingBackend]:
    detector = RecordingBackend(
        "fake_detector",
        detector_tasks,
        detector_factory,
    )
    segmenter = RecordingBackend(
        "fake_segmenter",
        segmenter_tasks,
        segmenter_factory,
        fail_load=segmenter_fail_load,
    )
    registry = ModelRegistry()
    registry.register(detector.backend_id, lambda: detector)
    registry.register(segmenter.backend_id, lambda: segmenter)
    return registry, detector, segmenter


def _config(**updates: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "detector-to-mask",
        "detector": {
            "backend": "fake_detector",
            "config": {"name": "detector", "revision": "detector-r1"},
        },
        "segmenter": {
            "backend": "fake_segmenter",
            "config": {"name": "segmenter", "revision": "segmenter-r1"},
        },
    }
    config.update(updates)
    return config


def _request(session: BaseInferenceSession, **updates: Any) -> InferenceRequest:
    values: dict[str, Any] = {
        "request_id": "outer-request",
        "source_id": "image-sha256:stable-pixels",
        "model_id": session.capabilities.model_id,
        "model_revision": session.capabilities.model_revision,
    }
    values.update(updates)
    return InferenceRequest(**values)


def test_refines_one_box_at_a_time_and_preserves_detector_semantics():
    detections = (
        _rectangle(20, 30, 10, 10, label="dog", score=0.91, class_id=16),
        _rectangle(195, 95, 230, 120, label="truck", score=0.82, class_id=7),
    )

    def detector_result(request, _image, _cancellation, _index):
        return _result(
            request,
            shapes=detections,
            warnings=("detector provider warning",),
        )

    masks = (
        (
            _polygon((5, 5), (25, 5), (25, 35), score=0.88),
            _polygon((8, 8), (9, 8), (9, 9)),
        ),
        (_polygon((190, 90), (200, 90), (200, 100), score=0.77),),
    )

    def segmenter_result(request, _image, _cancellation, index):
        return _result(
            request,
            shapes=masks[index],
            warnings=(("segmenter provider warning",) if index == 0 else ()),
            timings_ms={"encode": 5.0 if index == 0 else 0.0, "decode": 2.0},
        )

    registry, detector, segmenter = _registry(detector_result, segmenter_result)
    session = DetectorSamBackend(registry).create_session(_config(box_padding_pixels=5))
    session.load()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = session.predict(
        _request(session, parameters={"confidence": 0.5, "class_ids": [7, 16]}),
        image,
    )

    assert result.request_id == "outer-request"
    assert result.model_id == "detector-to-mask"
    assert result.source_id == "image-sha256:stable-pixels"
    assert [shape.type for shape in result.shapes] == [
        ShapeType.POLYGON,
        ShapeType.POLYGON,
        ShapeType.POLYGON,
    ]
    assert [shape.label for shape in result.shapes] == ["dog", "dog", "truck"]
    assert [shape.score for shape in result.shapes] == [0.91, 0.91, 0.82]
    assert [shape.group_id for shape in result.shapes] == [0, 0, 1]
    assert result.shapes[0].attributes == {
        "class_id": 16,
        "segmenter_score": 0.88,
    }
    assert result.shapes[2].attributes == {
        "class_id": 7,
        "segmenter_score": 0.77,
    }
    assert result.warnings == (
        "detector: detector provider warning",
        "segmenter[0]: segmenter provider warning",
    )
    assert result.timings_ms["segmenter_encode"] == 5.0
    assert result.timings_ms["segmenter_decode"] == 4.0

    detector_request = detector.sessions[0].requests[0]
    assert detector_request.parameters == {
        "confidence": 0.5,
        "class_ids": (7, 16),
    }
    assert detector_request.prompts == ()
    assert len(segmenter.sessions[0].requests) == 2
    first_prompt = segmenter.sessions[0].requests[0].prompts[0]
    second_prompt = segmenter.sessions[0].requests[1].prompts[0]
    assert isinstance(first_prompt, BoxPrompt)
    assert first_prompt.top_left == Point(x=5, y=5)
    assert first_prompt.bottom_right == Point(x=25, y=35)
    assert isinstance(second_prompt, BoxPrompt)
    assert second_prompt.top_left == Point(x=190, y=90)
    assert second_prompt.bottom_right == Point(x=200, y=100)
    assert all(
        request.source_id == "image-sha256:stable-pixels"
        and request.output_shape is ShapeType.POLYGON
        for request in segmenter.sessions[0].requests
    )
    assert detector.sessions[0].images[0] is image
    assert all(item is image for item in segmenter.sessions[0].images)

    session.unload()
    assert session.state is SessionState.CLOSED
    assert detector.sessions[0].unload_calls == 1
    assert segmenter.sessions[0].unload_calls == 1


def test_bounds_refinements_and_falls_back_to_detection_box():
    detections = tuple(
        _rectangle(index, index, index + 10, index + 10, label=f"class-{index}")
        for index in range(3)
    )

    def detector_result(request, _image, _cancellation, _index):
        return _result(request, shapes=detections)

    def segmenter_result(request, _image, _cancellation, index):
        if index == 0:
            return _result(request)
        return _result(
            request,
            shapes=(_polygon((1, 1), (2, 1), (2, 2)),),
        )

    registry, _detector, segmenter = _registry(
        detector_result,
        segmenter_result,
    )
    session = DetectorSamBackend(registry).create_session(_config(max_refinements=2))
    session.load()

    result = session.predict(
        _request(session),
        np.zeros((30, 30, 3), dtype=np.uint8),
    )

    assert len(segmenter.sessions[0].requests) == 2
    assert [shape.type for shape in result.shapes] == [
        ShapeType.RECTANGLE,
        ShapeType.POLYGON,
    ]
    assert [shape.label for shape in result.shapes] == ["class-0", "class-1"]
    assert result.warnings == (
        "detector returned 3 shapes; refined the first 2",
        "segmenter returned no mask for shape 0; kept detector box",
    )
    session.unload()


def test_drops_empty_or_too_small_refinements_when_fallback_is_disabled():
    def detector_result(request, _image, _cancellation, _index):
        return _result(
            request,
            shapes=(
                _rectangle(1, 1, 2, 2, label="tiny"),
                _rectangle(5, 5, 15, 15, label="empty-mask"),
            ),
        )

    def segmenter_result(request, _image, _cancellation, _index):
        return _result(request)

    registry, _detector, segmenter = _registry(
        detector_result,
        segmenter_result,
    )
    session = DetectorSamBackend(registry).create_session(
        _config(
            minimum_box_area_pixels=4,
            fallback_to_box=False,
        )
    )
    session.load()

    result = session.predict(
        _request(session),
        np.zeros((20, 20, 3), dtype=np.uint8),
    )

    assert result.shapes == ()
    assert len(segmenter.sessions[0].requests) == 1
    assert result.warnings == (
        "detector shape 0 was empty or below the minimum box area",
        "segmenter returned no mask for shape 1; dropped detection",
    )
    session.unload()


def test_rejects_bad_requests_and_recovers_after_detector_shape_error():
    detector_shapes = [
        (_polygon((1, 1), (2, 1), (2, 2)),),
        (_rectangle(1, 1, 10, 10),),
    ]

    def detector_result(request, _image, _cancellation, index):
        return _result(request, shapes=detector_shapes[index])

    def segmenter_result(request, _image, _cancellation, _index):
        return _result(
            request,
            shapes=(_polygon((1, 1), (10, 1), (10, 10)),),
        )

    registry, _detector, _segmenter = _registry(
        detector_result,
        segmenter_result,
    )
    session = DetectorSamBackend(registry).create_session(_config())
    session.load()
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="rectangles"):
        session.predict(_request(session), image)
    with pytest.raises(ValueError, match="does not accept"):
        session.predict(
            _request(
                session,
                prompts=(PointPrompt(point=Point(x=1, y=1)),),
            ),
            image,
        )
    with pytest.raises(ValueError, match="must be polygon"):
        session.predict(
            _request(session, output_shape=ShapeType.RECTANGLE),
            image,
        )
    with pytest.raises(ValueError, match="uint8 RGB"):
        session.predict(_request(session), image.astype(np.float32))

    recovered = session.predict(_request(session, request_id="recovered"), image)
    assert recovered.request_id == "recovered"
    assert len(recovered.shapes) == 1
    assert session.state is SessionState.READY
    session.unload()


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max_shapes": 1}, "max_shapes"),
        ({"max_total_points": 3}, "max_total_points"),
    ],
)
def test_aggregate_result_limits_fail_before_return(limits, message):
    def detector_result(request, _image, _cancellation, _index):
        return _result(request, shapes=(_rectangle(1, 1, 10, 10),))

    def segmenter_result(request, _image, _cancellation, _index):
        return _result(
            request,
            shapes=(
                _polygon((1, 1), (2, 1), (2, 2)),
                _polygon((3, 3), (4, 3), (4, 4)),
            ),
        )

    registry, _detector, _segmenter = _registry(
        detector_result,
        segmenter_result,
    )
    session = DetectorSamBackend(registry).create_session(_config(**limits))
    session.load()

    with pytest.raises(ValueError, match=message):
        session.predict(
            _request(session),
            np.zeros((20, 20, 3), dtype=np.uint8),
        )
    assert session.state is SessionState.READY
    session.unload()


def test_cancellation_stops_pipeline_and_child_load_failure_cleans_up():
    external = CancellationToken()

    def detector_result(request, _image, _cancellation, _index):
        return _result(
            request,
            shapes=(
                _rectangle(1, 1, 10, 10),
                _rectangle(11, 11, 20, 20),
            ),
        )

    def cancelling_segmenter(request, _image, _cancellation, _index):
        external.cancel("test cancellation")
        return _result(
            request,
            shapes=(_polygon((1, 1), (2, 1), (2, 2)),),
        )

    registry, _detector, segmenter = _registry(
        detector_result,
        cancelling_segmenter,
    )
    session = DetectorSamBackend(registry).create_session(_config())
    session.load()
    with pytest.raises(InferenceCancelledError, match="test cancellation"):
        session.predict(
            _request(session),
            np.zeros((30, 30, 3), dtype=np.uint8),
            external,
        )
    assert len(segmenter.sessions[0].requests) == 1
    assert session.state is SessionState.READY
    session.unload()

    failing_registry, detector, failing_segmenter = _registry(
        detector_result,
        cancelling_segmenter,
        segmenter_fail_load=True,
    )
    failing = DetectorSamBackend(failing_registry).create_session(_config())
    with pytest.raises(RuntimeError, match="scripted load failure"):
        failing.load()
    assert failing.state is SessionState.FAILED
    assert detector.sessions[0].state is SessionState.CLOSED
    assert failing_segmenter.sessions[0].state is SessionState.CLOSED

    failing_segmenter.fail_load = False
    failing.load()
    assert failing.state is SessionState.READY
    assert len(detector.sessions) == 2
    assert len(failing_segmenter.sessions) == 2
    failing.unload()


def test_capabilities_validate_roles_revision_and_manifest_anchor(tmp_path: Path):
    def empty_result(request, _image, _cancellation, _index):
        return _result(request)

    registry, detector, segmenter = _registry(empty_result, empty_result)
    backend = DetectorSamBackend(registry)
    manifest = tmp_path / "models.json"
    manifest.write_text("{}", encoding="utf-8")
    config = _config(
        config_file=manifest,
        detector={
            "backend": "fake_detector",
            "config": {
                "name": "detector",
                "revision": "detector-r1",
                "config_file": "untrusted-anchor.json",
            },
        },
    )

    first = backend.capabilities(config)
    second = backend.capabilities({**config, "max_total_points": 200_000})
    assert first.tasks == (ModelTask.INSTANCE_SEGMENTATION,)
    assert first.metadata["processing"] == "encode-once-box-per-object"
    assert first.model_revision != second.model_revision
    session = backend.create_session(config)
    assert detector.configs[0]["config_file"] == manifest
    assert segmenter.configs[0]["config_file"] == manifest
    session.unload()

    explicit = backend.capabilities({**config, "model_revision": "pipeline-v1"})
    assert explicit.model_revision.startswith("detector-sam-sha256:")
    assert explicit.model_revision != first.model_revision


def test_configuration_rejects_recursive_or_wrong_role_backends():
    with pytest.raises(ValueError, match="cannot contain themselves"):
        DetectorSamConfig.model_validate(
            _config(
                detector={
                    "backend": "detector_sam",
                    "config": {"name": "recursive"},
                }
            )
        )
    with pytest.raises(ValueError, match="must be different"):
        DetectorSamConfig.model_validate(
            _config(
                segmenter={
                    "backend": "fake_detector",
                    "config": {"name": "detector", "revision": "detector-r1"},
                }
            )
        )

    def empty_result(request, _image, _cancellation, _index):
        return _result(request)

    wrong_detector_registry, _detector, _segmenter = _registry(
        empty_result,
        empty_result,
        detector_tasks=(ModelTask.CLASSIFICATION,),
    )
    with pytest.raises(ValueError, match="advertise detection"):
        DetectorSamBackend(wrong_detector_registry).capabilities(_config())

    wrong_segmenter_registry, _detector, _segmenter = _registry(
        empty_result,
        empty_result,
        segmenter_tasks=(ModelTask.SEMANTIC_SEGMENTATION,),
    )
    with pytest.raises(ValueError, match="promptable segmentation"):
        DetectorSamBackend(wrong_segmenter_registry).capabilities(_config())


def test_server_accepts_pipeline_but_rejects_nested_credentials():
    definition = ServerModelDefinition(
        backend="detector_sam",
        config=_config(
            detector={"backend": "dfine_onnx", "config": {"name": "detector"}},
            segmenter={
                "backend": "segment_anything",
                "config": {"name": "segmenter"},
            },
        ),
    )
    assert definition.backend == "detector_sam"

    with pytest.raises(ValueError, match="credentials"):
        ServerModelDefinition(
            backend="detector_sam",
            config=_config(
                segmenter={
                    "backend": "segment_anything",
                    "config": {
                        "name": "segmenter",
                        "revision": "segmenter-r1",
                        "api_token": "not-allowed",
                    },
                }
            ),
        )

    with pytest.raises(ValueError, match="detector is not approved"):
        ServerModelDefinition(
            backend="detector_sam",
            config=_config(
                detector={
                    "backend": "unreviewed_backend",
                    "config": {"name": "detector"},
                },
                segmenter={
                    "backend": "segment_anything",
                    "config": {"name": "segmenter"},
                },
            ),
        )

    with pytest.raises(ValueError, match="segmenter is not approved"):
        ServerModelDefinition(
            backend="detector_sam",
            config=_config(
                detector={
                    "backend": "yolo_onnx",
                    "config": {"name": "detector"},
                },
                segmenter={
                    "backend": "unreviewed_backend",
                    "config": {"name": "segmenter"},
                },
            ),
        )
