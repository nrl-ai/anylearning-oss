import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from anylearning.inference import (
    CURRENT_PROTOCOL_VERSION,
    BoxPrompt,
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    PointPrompt,
    ShapeType,
    TextPrompt,
)


def test_inference_package_imports_without_runtime_or_application_frameworks():
    script = """
import json
import sys
import anylearning.inference

forbidden = ("cv2", "fastapi", "onnxruntime", "torch", "webview")
print(json.dumps([name for name in forbidden if name in sys.modules]))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_inference_result_round_trips_through_json():
    result = InferenceResult(
        request_id="request-42",
        source_id="image-sha256:abc123",
        model_id="sam2-small",
        model_revision="2026-08-30",
        shapes=(
            InferenceShape(
                type=ShapeType.POLYGON,
                points=(
                    Point(x=10, y=10),
                    Point(x=20, y=10),
                    Point(x=15, y=20),
                ),
                label="object",
                score=0.95,
                attributes={"prompted": True},
            ),
        ),
        warnings=("result was clipped to the image bounds",),
        timings_ms={"inference": 12.5},
    )

    payload = result.model_dump_json()
    restored = InferenceResult.model_validate_json(payload)

    assert restored == result
    assert restored.protocol_version == CURRENT_PROTOCOL_VERSION
    assert restored.model_dump(mode="json")["shapes"][0]["type"] == "polygon"


def test_inference_request_round_trips_neutral_prompts():
    request = InferenceRequest(
        request_id="request-42",
        source_id="image-sha256:abc123",
        model_id="sam2-small",
        model_revision="sha256:def456",
        prompts=(
            PointPrompt(point=Point(x=10, y=20), foreground=False),
            BoxPrompt(
                top_left=Point(x=1, y=2),
                bottom_right=Point(x=30, y=40),
            ),
            TextPrompt(text="dog"),
        ),
        output_shape=ShapeType.POLYGON,
    )

    restored = InferenceRequest.model_validate_json(request.model_dump_json())

    assert restored == request
    assert isinstance(restored.prompts[0], PointPrompt)
    assert isinstance(restored.prompts[1], BoxPrompt)
    assert isinstance(restored.prompts[2], TextPrompt)


@pytest.mark.parametrize("text", ["", "   ", "dog\x00truck", "x" * 1025])
def test_text_prompt_is_bounded_and_visible(text):
    with pytest.raises(ValidationError):
        TextPrompt(text=text)


def test_request_parameter_filters_round_trip_as_immutable_sequences():
    request = InferenceRequest(
        request_id="filter-request",
        source_id="image-sha256:fixture",
        model_id="detector",
        model_revision="revision-1",
        parameters={"class_ids": (1, 3), "class_names": ("cat", "dog")},
    )

    restored = InferenceRequest.model_validate_json(request.model_dump_json())

    assert restored.parameters["class_ids"] == (1, 3)
    assert restored.parameters["class_names"] == ("cat", "dog")


def test_box_prompt_requires_ordered_positive_area():
    with pytest.raises(ValidationError, match="positive width and height"):
        BoxPrompt(
            top_left=Point(x=10, y=10),
            bottom_right=Point(x=5, y=20),
        )


@pytest.mark.parametrize(
    ("shape_type", "points"),
    [
        (ShapeType.POINT, (Point(x=1, y=1), Point(x=2, y=2))),
        (ShapeType.RECTANGLE, (Point(x=1, y=1),)),
        (
            ShapeType.POLYGON,
            (Point(x=1, y=1), Point(x=2, y=2)),
        ),
        (
            ShapeType.ROTATED_RECTANGLE,
            (Point(x=1, y=1), Point(x=2, y=1), Point(x=2, y=2)),
        ),
    ],
)
def test_shape_rejects_invalid_point_count(shape_type, points):
    with pytest.raises(ValidationError, match="requires"):
        InferenceShape(type=shape_type, points=points)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_shape_rejects_invalid_score(score):
    with pytest.raises(ValidationError):
        InferenceShape(
            type=ShapeType.POINT,
            points=(Point(x=1, y=1),),
            score=score,
        )


def test_point_rejects_non_finite_coordinates():
    with pytest.raises(ValidationError):
        Point(x=float("inf"), y=1)


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Point(x=1, y=2, z=3)


def test_model_capabilities_require_unique_tasks_and_positive_batch_size():
    capabilities = ModelCapabilities(
        model_id="yolo11n",
        model_revision="sha256:123",
        tasks=(ModelTask.DETECTION, ModelTask.INSTANCE_SEGMENTATION),
        supports_batch=True,
        max_batch_size=8,
    )

    assert capabilities.tasks == (
        ModelTask.DETECTION,
        ModelTask.INSTANCE_SEGMENTATION,
    )

    with pytest.raises(ValidationError, match="unique"):
        ModelCapabilities(
            model_id="duplicate",
            model_revision="1",
            tasks=(ModelTask.DETECTION, ModelTask.DETECTION),
        )

    with pytest.raises(ValidationError):
        ModelCapabilities(
            model_id="invalid-batch",
            model_revision="1",
            tasks=(ModelTask.DETECTION,),
            max_batch_size=0,
        )


def test_result_rejects_unsupported_protocol_version_and_negative_timing():
    base = {
        "request_id": "request-1",
        "source_id": "image-1",
        "model_id": "model-1",
        "model_revision": "revision-1",
    }

    with pytest.raises(ValidationError, match="Unsupported protocol version"):
        InferenceResult(protocol_version="99", **base)

    with pytest.raises(ValidationError):
        InferenceResult(timings_ms={"inference": -1}, **base)


def test_contract_collection_sizes_are_bounded():
    with pytest.raises(ValidationError, match="at most 128 items"):
        InferenceShape(
            type=ShapeType.POINT,
            points=(Point(x=1, y=1),),
            attributes={f"key-{index}": index for index in range(129)},
        )

    with pytest.raises(ValidationError, match="at most 64 items"):
        InferenceResult(
            request_id="request-1",
            source_id="image-1",
            model_id="model-1",
            model_revision="revision-1",
            timings_ms={f"phase-{index}": 1 for index in range(65)},
        )


def test_metadata_rejects_nested_or_non_finite_values():
    shape = {
        "type": ShapeType.POINT,
        "points": (Point(x=1, y=1),),
    }

    with pytest.raises(ValidationError):
        InferenceShape(attributes={"nested": ["unbounded"]}, **shape)

    with pytest.raises(ValidationError):
        InferenceShape(attributes={"not-finite": float("inf")}, **shape)
