from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from anylearning.inference import (
    BoxPrompt,
    InferenceRequest,
    Point,
    PointPrompt,
    ShapeType,
    TextPrompt,
)
from anylearning.inference.backends.sam3 import (
    Sam3Backend,
    Sam3Config,
    _geometric_prompts,
    _request_options,
    sam3_shapes,
)
from anylearning.inference.backends.sam3_onnx import Sam3Detections


def _config(tmp_path):
    paths = {}
    for role in ("image_encoder", "language_encoder", "decoder"):
        path = tmp_path / f"{role}.onnx"
        path.write_bytes(role.encode())
        paths[f"{role}_model_path"] = path
    return {"name": "sam3-test", **paths}


def _request(capabilities, *, prompts, parameters=None, output_shape=None):
    return InferenceRequest(
        request_id="request-1",
        source_id="image-sha256:fixture",
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
        prompts=prompts,
        parameters=parameters or {},
        output_shape=output_shape,
    )


def test_sam3_config_binds_all_graphs_and_freezes_external_manifests(tmp_path):
    config = _config(tmp_path)
    config["image_encoder_sha256"] = "a" * 64
    config["language_encoder_sha256"] = "b" * 64
    config["decoder_sha256"] = "c" * 64
    config["image_encoder_external_data_sha256"] = {"weights.data": "d" * 64}

    validated = Sam3Config.model_validate(config)

    assert validated.revision.startswith("onnx-triplet-sha256:")
    with pytest.raises(TypeError, match="immutable"):
        validated.image_encoder_external_data_sha256["other"] = "e" * 64

    config["decoder_model_path"] = config["image_encoder_model_path"]
    with pytest.raises(ValidationError, match="three distinct"):
        Sam3Config.model_validate(config)


def test_sam3_config_accepts_exporter_path_aliases(tmp_path):
    config = _config(tmp_path)
    aliased = {
        "name": config["name"],
        "encoder_model_path": config["image_encoder_model_path"],
        "language_encoder_path": config["language_encoder_model_path"],
        "decoder_model_path": config["decoder_model_path"],
    }

    validated = Sam3Config.model_validate(aliased)

    assert validated.image_encoder_model_path.name == "image_encoder.onnx"
    assert validated.language_encoder_model_path.name == "language_encoder.onnx"


def test_sam3_capabilities_advertise_text_geometry_and_separate_license(tmp_path):
    capabilities = Sam3Backend().capabilities(_config(tmp_path))

    assert capabilities.metadata["prompt_types"] == "text,point,box"
    assert capabilities.metadata["artifact_license"] == "SAM-License"
    assert capabilities.supports_cancellation


def test_sam3_prompt_conversion_supports_text_geometry_and_rejects_ambiguity(tmp_path):
    capabilities = Sam3Backend().capabilities(_config(tmp_path))
    request = _request(
        capabilities,
        prompts=(
            TextPrompt(text="dog"),
            PointPrompt(point=Point(x=10, y=20)),
            BoxPrompt(top_left=Point(x=1, y=2), bottom_right=Point(x=30, y=40)),
        ),
    )

    text, geometry = _geometric_prompts(request)

    assert text == "dog"
    assert [item["type"] for item in geometry] == ["point", "rectangle"]
    with pytest.raises(ValueError, match="at most one text"):
        _geometric_prompts(
            request.model_copy(
                update={"prompts": (TextPrompt(text="dog"), TextPrompt(text="cat"))}
            )
        )
    with pytest.raises(ValueError, match="requires"):
        _geometric_prompts(request.model_copy(update={"prompts": ()}))


def test_sam3_request_options_are_bounded_and_reject_unknown_values(tmp_path):
    config = Sam3Config.model_validate(_config(tmp_path))
    capabilities = Sam3Backend().capabilities(_config(tmp_path))
    request = _request(
        capabilities,
        prompts=(TextPrompt(text="dog"),),
        parameters={"confidence": 0.7, "iou": 0.6, "max_instances": 3},
    )

    assert _request_options(request, config, output_profile="processed") == (
        0.7,
        0.6,
        3,
    )
    with pytest.raises(ValueError, match="configured floor"):
        _request_options(
            request.model_copy(update={"parameters": {"confidence": 0.1}}),
            config,
            output_profile="processed",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        _request_options(
            request.model_copy(update={"parameters": {"unexpected": 1}}),
            config,
            output_profile="raw",
        )


def test_sam3_shapes_preserve_instance_scores_labels_and_groups(tmp_path):
    config = Sam3Config.model_validate(_config(tmp_path))
    masks = np.zeros((2, 1, 20, 30), dtype=np.bool_)
    masks[0, 0, 2:10, 3:15] = True
    masks[1, 0, 11:18, 17:28] = True
    detections = Sam3Detections(
        masks=masks,
        scores=np.array([0.9, 0.8], dtype=np.float32),
        boxes=np.array([[3, 2, 15, 10], [17, 11, 28, 18]], dtype=np.float32),
    )

    polygons = sam3_shapes(
        detections, output_shape=ShapeType.POLYGON, label="dog", config=config
    )
    rectangles = sam3_shapes(
        detections, output_shape=ShapeType.RECTANGLE, label="dog", config=config
    )

    assert {shape.group_id for shape in polygons} == {0, 1}
    assert {shape.label for shape in polygons} == {"dog"}
    assert [shape.score for shape in rectangles] == pytest.approx([0.9, 0.8])
    assert rectangles[1].points[1] == Point(x=28, y=18)


def test_sam3_session_caches_image_and_text_and_releases_runtime(tmp_path, monkeypatch):
    from anylearning.inference.backends import sam3 as module

    class FakePipeline:
        def __init__(self):
            self.decoder = SimpleNamespace(
                output_profile="processed", geometric_prompt_capacity=2
            )
            self.image_encodes = 0
            self.text_encodes = 0

        def encode_image(self, image):
            self.image_encodes += 1
            return {"image": np.asarray(image)}

        def encode_text(self, text):
            self.text_encodes += 1
            return {"text": text}

        def predict(self, **_kwargs):
            mask = np.zeros((1, 1, 20, 30), dtype=np.bool_)
            mask[0, 0, 2:10, 3:15] = True
            return Sam3Detections(
                masks=mask,
                scores=np.array([0.9], dtype=np.float32),
                boxes=np.array([[3, 2, 15, 10]], dtype=np.float32),
            )

    pipeline = FakePipeline()
    memory_releases = 0

    def release_memory():
        nonlocal memory_releases
        memory_releases += 1

    def checked_session(*_args, **_kwargs):
        return object(), object(), ("provider warning",)

    monkeypatch.setattr(module, "create_checked_onnx_session", checked_session)
    monkeypatch.setattr(module, "Sam3OnnxPipeline", lambda *_args, **_kwargs: pipeline)
    monkeypatch.setattr(module, "release_unused_cpu_memory", release_memory)
    session = Sam3Backend().create_session(_config(tmp_path))
    session.load()
    request = _request(
        session.capabilities,
        prompts=(TextPrompt(text="dog"),),
        output_shape=ShapeType.POLYGON,
    )
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    first = session.predict(request, image)
    second = session.predict(
        request.model_copy(update={"request_id": "request-2"}), image
    )

    assert pipeline.image_encodes == 1
    assert pipeline.text_encodes == 1
    assert first.shapes == second.shapes
    assert first.shapes[0].label == "dog"
    assert first.warnings == ("provider warning",)
    session.unload()
    assert session._model is None
    assert memory_releases == 1


def test_sam3_explicit_visual_text_keeps_its_label(tmp_path, monkeypatch):
    from anylearning.inference.backends import sam3 as module

    class FakePipeline:
        decoder = SimpleNamespace(
            output_profile="processed", geometric_prompt_capacity=1
        )

        @staticmethod
        def encode_image(_image):
            return {}

        @staticmethod
        def encode_text(_text):
            return {}

        @staticmethod
        def predict(**_kwargs):
            mask = np.zeros((1, 1, 10, 10), dtype=np.bool_)
            mask[0, 0, 2:8, 2:8] = True
            return Sam3Detections(
                masks=mask,
                scores=np.array([0.9], dtype=np.float32),
                boxes=np.array([[2, 2, 8, 8]], dtype=np.float32),
            )

    monkeypatch.setattr(
        module,
        "create_checked_onnx_session",
        lambda *_args, **_kwargs: (object(), object(), ()),
    )
    monkeypatch.setattr(
        module, "Sam3OnnxPipeline", lambda *_args, **_kwargs: FakePipeline()
    )
    monkeypatch.setattr(module, "release_unused_cpu_memory", lambda: None)
    session = Sam3Backend().create_session(_config(tmp_path))
    session.load()
    request = _request(
        session.capabilities,
        prompts=(TextPrompt(text="visual"),),
        output_shape=ShapeType.RECTANGLE,
    )

    result = session.predict(request, np.zeros((10, 10, 3), dtype=np.uint8))
    session.unload()

    assert result.shapes[0].label == "visual"


def test_sam3_session_rejects_processed_output_allocation_before_inference(
    tmp_path, monkeypatch
):
    from anylearning.inference.backends import sam3 as module

    pipeline = SimpleNamespace(
        decoder=SimpleNamespace(output_profile="processed", geometric_prompt_capacity=1)
    )
    monkeypatch.setattr(
        module,
        "create_checked_onnx_session",
        lambda *_args, **_kwargs: (object(), object(), ()),
    )
    monkeypatch.setattr(module, "Sam3OnnxPipeline", lambda *_args, **_kwargs: pipeline)
    config = _config(tmp_path)
    config.update(
        max_output_elements=100,
        max_raw_queries=2,
        max_nms_candidates=2,
        max_instances=1,
    )
    session = Sam3Backend().create_session(config)
    session.load()
    request = _request(session.capabilities, prompts=(TextPrompt(text="dog"),))

    with pytest.raises(ValueError, match="output bound"):
        session.predict(request, np.zeros((10, 10, 3), dtype=np.uint8))
