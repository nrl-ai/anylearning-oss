import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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


def test_default_registry_keeps_sam_lazy():
    registry = create_default_registry()

    assert registry.backend_ids() == ("segment_anything", "yolo_onnx")
    assert "segment_anything" not in registry._backends
    assert "yolo_onnx" not in registry._backends


def test_sam_session_preserves_identity_shapes_prompts_and_embedding_cache(tmp_path):
    model_config = config(tmp_path)
    graph = SimpleNamespace(graph=SimpleNamespace(input=[]))
    with (
        patch("anylearning.inference.backends.sam.onnx.load_model", return_value=graph),
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


def test_sam_repeated_load_predict_unload_releases_every_adapter(tmp_path):
    model_config = config(tmp_path)
    graph = SimpleNamespace(graph=SimpleNamespace(input=[]))
    references = []
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with (
        patch("anylearning.inference.backends.sam.onnx.load_model", return_value=graph),
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
