import gc
import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from anylearning.inference import (
    BoxPrompt,
    InferenceRequest,
    Point,
    PointPrompt,
    SessionState,
    ShapeType,
    create_default_registry,
)
from anylearning.inference.backends.sam import (
    SegmentAnythingBackend,
    image_source_id,
    mask_shapes,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "inference"


class FakeSAM:
    def __init__(self, encoder: str, decoder: str) -> None:
        self.paths = (encoder, decoder)
        self.encode_calls = 0
        self.prompts = []

    def encode(self, image: np.ndarray) -> dict[str, object]:
        self.encode_calls += 1
        return {"shape": image.shape}

    def predict_masks(
        self, embedding: dict[str, object], prompts: list[dict[str, object]]
    ) -> np.ndarray:
        self.prompts.append(prompts)
        mask = np.zeros((1, 1, 32, 32), dtype=np.float32)
        mask[0, 0, 8:24, 10:22] = 1
        return mask


def config(tmp_path):
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    encoder.touch()
    decoder.touch()
    return {
        "name": "sam-test",
        "model_revision": "revision-1",
        "encoder_model_path": str(encoder),
        "decoder_model_path": str(decoder),
    }


def request(*, source_id: str, output_shape=ShapeType.POLYGON):
    return InferenceRequest(
        request_id="request-1",
        source_id=source_id,
        model_id="sam-test",
        model_revision="revision-1",
        prompts=(
            PointPrompt(point=Point(x=5, y=6)),
            BoxPrompt(
                top_left=Point(x=1, y=2),
                bottom_right=Point(x=20, y=21),
            ),
        ),
        output_shape=output_shape,
    )


def checked_sam_session(*_args, **_kwargs):
    graph = SimpleNamespace(graph=SimpleNamespace(input=[]))
    return SimpleNamespace(), graph, ()


def test_default_registry_keeps_sam_lazy():
    registry = create_default_registry()

    assert registry.backend_ids() == (
        "efficient_sam",
        "segment_anything",
        "yolo_onnx",
    )
    assert "efficient_sam" not in registry._backends
    assert "segment_anything" not in registry._backends
    assert "yolo_onnx" not in registry._backends


def test_sam_backend_import_does_not_require_desktop_config_dependencies():
    """The standalone inference extra must not pull in desktop YAML config."""
    script = """
import sys
sys.modules['yaml'] = None
sys.modules['loguru'] = None
import anylearning.inference.backends.sam
import anylearning.inference.backends.efficient_sam
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_sam_session_preserves_identity_shapes_prompts_and_embedding_cache(tmp_path):
    model_config = config(tmp_path)
    with (
        patch(
            "anylearning.inference.backends.sam.create_checked_onnx_session",
            side_effect=checked_sam_session,
        ),
        patch("anylearning.inference.backends.sam.SegmentAnythingONNX", FakeSAM),
    ):
        session = SegmentAnythingBackend().create_session(model_config)
        session.load()
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        first = session.predict(request(source_id="image-sha256:one"), image)
        second = session.predict(request(source_id="image-sha256:one"), image)
        rectangle = session.predict(
            request(
                source_id="image-sha256:two",
                output_shape=ShapeType.RECTANGLE,
            ),
            image,
        )

        assert first.request_id == "request-1"
        assert first.source_id == "image-sha256:one"
        assert first.model_revision == "revision-1"
        assert first.shapes[0].type is ShapeType.POLYGON
        assert second.shapes == first.shapes
        assert rectangle.shapes[0].type is ShapeType.RECTANGLE
        assert session._model.encode_calls == 2
        assert session._model.prompts[0] == [
            {"type": "point", "data": [5.0, 6.0], "label": 1},
            {
                "type": "rectangle",
                "data": [1.0, 2.0, 20.0, 21.0],
                "label": 1,
            },
        ]

        session.unload()
        assert session.state is SessionState.CLOSED
        assert session._model is None
        assert len(session._embedding_cache) == 0


def test_image_source_id_changes_with_content_and_not_object_identity():
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    same = first.copy()
    changed = first.copy()
    changed[0, 0, 0] = 1

    assert image_source_id(first) == image_source_id(same)
    assert image_source_id(first) != image_source_id(changed)


def test_sam_mask_conversion_matches_golden_fixture():
    fixture = json.loads((FIXTURES / "sam_rectangle.json").read_text())
    height, width = fixture["mask_size"]
    x1, y1, x2, y2 = fixture["filled_box_xyxy_exclusive"]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1

    shapes = mask_shapes(mask, ShapeType.RECTANGLE)

    assert len(shapes) == 1
    assert (
        shapes[0].model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        == fixture["expected_shape"]
    )


def test_sam_mask_conversion_bounds_fragmented_and_complex_results():
    fragmented = np.zeros((64, 64), dtype=np.float32)
    fragmented[::2, ::2] = 1

    with pytest.raises(ValueError, match="contours"):
        mask_shapes(
            fragmented,
            ShapeType.POLYGON,
            max_mask_contours=10,
        )

    rectangle = np.zeros((32, 32), dtype=np.float32)
    rectangle[4:28, 4:28] = 1
    with pytest.raises(ValueError, match="total point limit"):
        mask_shapes(
            rectangle,
            ShapeType.RECTANGLE,
            max_total_shape_points=3,
        )


def test_sam_repeated_load_predict_unload_releases_every_adapter(tmp_path):
    model_config = config(tmp_path)
    references = []
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with (
        patch(
            "anylearning.inference.backends.sam.create_checked_onnx_session",
            side_effect=checked_sam_session,
        ),
        patch("anylearning.inference.backends.sam.SegmentAnythingONNX", FakeSAM),
    ):
        for index in range(100):
            session = SegmentAnythingBackend().create_session(model_config)
            session.load()
            references.append(weakref.ref(session._model))
            result = session.predict(
                request(source_id=f"image-sha256:soak-{index}"), image
            )
            assert result.shapes
            session.unload()
            assert session.state is SessionState.CLOSED
            assert session._model is None

    del session
    gc.collect()
    assert all(reference() is None for reference in references)


def test_sam_config_preserves_legacy_fields_and_freezes_integrity_maps(tmp_path):
    model_config = {
        **config(tmp_path),
        "display_name": "Legacy desktop model",
        "input_size": 1024,
        "max_width": 1024,
        "encoder_external_data_sha256": {"weights.bin": "a" * 64},
    }

    session = SegmentAnythingBackend().create_session(model_config)

    assert session.config.display_name == "Legacy desktop model"
    assert session.config.enable_cpu_mem_arena is False
    with pytest.raises(TypeError, match="immutable"):
        session.config.encoder_external_data_sha256["weights.bin"] = "b" * 64


def test_sam_loader_forwards_independent_integrity_and_provider_bounds(tmp_path):
    model_config = {
        **config(tmp_path),
        "encoder_sha256": "a" * 64,
        "decoder_sha256": "b" * 64,
        "providers": ["CUDAExecutionProvider"],
        "allow_cpu_fallback": False,
        "max_model_bytes": 1234,
        "max_external_data_bytes": 5678,
        "enable_cpu_mem_arena": True,
        "intra_op_threads": 2,
        "inter_op_threads": 3,
    }
    with (
        patch(
            "anylearning.inference.backends.sam.create_checked_onnx_session",
            side_effect=checked_sam_session,
        ) as checked,
        patch("anylearning.inference.backends.sam.SegmentAnythingONNX", FakeSAM),
    ):
        session = SegmentAnythingBackend().create_session(model_config)
        session.load()

    assert checked.call_count == 2
    encoder_call, decoder_call = checked.call_args_list
    assert encoder_call.kwargs["expected_sha256"] == "a" * 64
    assert decoder_call.kwargs["expected_sha256"] == "b" * 64
    for call in (encoder_call, decoder_call):
        assert call.kwargs["providers"] == ("CUDAExecutionProvider",)
        assert call.kwargs["allow_cpu_fallback"] is False
        assert call.kwargs["max_model_bytes"] == 1234
        assert call.kwargs["max_external_data_bytes"] == 5678
        assert call.kwargs["enable_cpu_mem_arena"] is True
        assert call.kwargs["intra_op_threads"] == 2
        assert call.kwargs["inter_op_threads"] == 3


def test_sam_explicit_family_must_match_decoder_graph(tmp_path):
    graph = SimpleNamespace(
        graph=SimpleNamespace(input=[SimpleNamespace(name="high_res_feats_0")])
    )

    def checked_sam2_session(*_args, **_kwargs):
        return SimpleNamespace(), graph, ()

    model_config = {**config(tmp_path), "family": "sam"}
    with patch(
        "anylearning.inference.backends.sam.create_checked_onnx_session",
        side_effect=checked_sam2_session,
    ):
        session = SegmentAnythingBackend().create_session(model_config)
        with pytest.raises(ValueError, match="does not match"):
            session.load()


def test_sam_image_pixel_limit_is_enforced_before_encoding(tmp_path):
    model_config = {**config(tmp_path), "max_image_pixels": 4}
    with (
        patch(
            "anylearning.inference.backends.sam.create_checked_onnx_session",
            side_effect=checked_sam_session,
        ),
        patch("anylearning.inference.backends.sam.SegmentAnythingONNX", FakeSAM),
    ):
        session = SegmentAnythingBackend().create_session(model_config)
        session.load()
        with pytest.raises(ValueError, match="configured limit"):
            session.predict(
                request(source_id="image-sha256:too-large"),
                np.zeros((3, 3, 3), dtype=np.uint8),
            )
