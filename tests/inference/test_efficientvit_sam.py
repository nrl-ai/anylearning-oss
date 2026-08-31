from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from anylearning.inference import (
    BoxPrompt,
    InferenceRequest,
    Point,
    PointPrompt,
    ShapeType,
)
from anylearning.inference.backends.efficientvit_sam import EfficientVitSamBackend
from anylearning.inference.backends.efficientvit_sam_onnx import EfficientViTSAMONNX


def node(name, shape, tensor_type="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=tensor_type)


class EncoderSession:
    def __init__(self, *, size=512):
        self._inputs = [node("input_image", ("batch", 3, size, size))]
        self._outputs = [node("image_embeddings", ("batch", 256, 64, 64))]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names is None
        self.feed = feed
        return [np.zeros((1, 256, 64, 64), dtype=np.float32)]


class DecoderSession:
    def __init__(self):
        self._inputs = [
            node("image_embeddings", (1, 256, 64, 64)),
            node("point_coords", ("batch", "points", 2)),
            node("point_labels", ("batch", "points")),
        ]
        self._outputs = [
            node("masks", ("batch", 4, 256, 256)),
            node("iou_predictions", ("batch", 4)),
        ]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names is None
        self.feed = feed
        masks = np.full((1, 4, 256, 256), -2, dtype=np.float32)
        masks[:, 0] = 5
        masks[:, 2] = 1
        scores = np.asarray([[0.99, 0.2, 0.9, 0.3]], dtype=np.float32)
        return [masks, scores]


def test_efficientvit_sam_preprocesses_rgb_and_uses_native_multimask_policy():
    encoder = EncoderSession()
    decoder = DecoderSession()
    adapter = EfficientViTSAMONNX(encoder, decoder)
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[..., 0] = 255

    embedding = adapter.encode(image)
    masks = adapter.predict_masks(
        embedding, [{"type": "point", "data": [2, 1], "label": 1}]
    )

    encoded = encoder.feed["input_image"]
    assert encoded.shape == (1, 3, 512, 512)
    assert encoded[0, 0, 0, 0] == pytest.approx((255 - 123.675) / 58.395)
    assert encoded[0, 2, 0, 0] == pytest.approx((0 - 103.53) / 57.375)
    assert np.all(encoded[:, :, 256:, :] == 0)
    assert decoder.feed["point_coords"].tolist() == [[[512.0, 256.0], [0.0, 0.0]]]
    assert decoder.feed["point_labels"].tolist() == [[1.0, -1.0]]
    assert masks.shape == (1, 1, 2, 4)
    assert np.all(masks == 1)


def test_efficientvit_sam_box_and_mixed_prompts_do_not_add_padding():
    decoder = DecoderSession()
    adapter = EfficientViTSAMONNX(EncoderSession(), decoder)
    embedding = {
        "image_embedding": np.zeros((1, 256, 64, 64), dtype=np.float32),
        "original_size": (2, 4),
    }

    adapter.predict_masks(
        embedding,
        [{"type": "rectangle", "data": [0, 0, 3, 1], "label": 1}],
    )
    assert decoder.feed["point_coords"].tolist() == [[[0.0, 0.0], [768.0, 256.0]]]
    assert decoder.feed["point_labels"].tolist() == [[2.0, 3.0]]

    adapter.predict_masks(
        embedding,
        [
            {"type": "point", "data": [2, 1], "label": 0},
            {"type": "rectangle", "data": [0, 0, 3, 1], "label": 1},
        ],
    )
    assert decoder.feed["point_coords"].shape == (1, 3, 2)
    assert decoder.feed["point_labels"].tolist() == [[0.0, 2.0, 3.0]]


@pytest.mark.parametrize("size", [256, 768, 2048])
def test_efficientvit_sam_rejects_unknown_encoder_size(size):
    with pytest.raises(ValueError, match="512x512 or 1024x1024"):
        EfficientViTSAMONNX(EncoderSession(size=size), DecoderSession())


def test_efficientvit_sam_rejects_changed_decoder_contract():
    decoder = DecoderSession()
    decoder._outputs[0] = node("masks", ("batch", 3, 256, 256))

    with pytest.raises(ValueError, match="unexpected static shape"):
        EfficientViTSAMONNX(EncoderSession(), decoder)

    decoder = DecoderSession()
    decoder._outputs[0] = node("masks", ("batch", "candidates", 256, 256))
    with pytest.raises(ValueError, match="bounded static dimensions"):
        EfficientViTSAMONNX(EncoderSession(), decoder)


def test_efficientvit_sam_rejects_nonfinite_and_oversized_outputs():
    decoder = DecoderSession()
    original_run = decoder.run

    def nonfinite_run(output_names, feed):
        outputs = original_run(output_names, feed)
        outputs[1][0, 1] = np.nan
        return outputs

    decoder.run = nonfinite_run
    adapter = EfficientViTSAMONNX(EncoderSession(), decoder)
    embedding = {
        "image_embedding": np.zeros((1, 256, 64, 64), dtype=np.float32),
        "original_size": (2, 4),
    }
    with pytest.raises(ValueError, match="finite"):
        adapter.predict_masks(
            embedding, [{"type": "point", "data": [2, 1], "label": 1}]
        )

    bounded = EfficientViTSAMONNX(
        EncoderSession(), DecoderSession(), max_output_elements=1
    )
    with pytest.raises(ValueError, match="configured output"):
        bounded.predict_masks(
            embedding, [{"type": "point", "data": [2, 1], "label": 1}]
        )


def test_efficientvit_sam_bounds_user_prompt_points():
    adapter = EfficientViTSAMONNX(
        EncoderSession(), DecoderSession(), max_prompt_points=1
    )
    embedding = {
        "image_embedding": np.zeros((1, 256, 64, 64), dtype=np.float32),
        "original_size": (2, 4),
    }

    with pytest.raises(ValueError, match="point limit"):
        adapter.predict_masks(
            embedding,
            [
                {"type": "point", "data": [1, 1], "label": 1},
                {"type": "point", "data": [2, 1], "label": 0},
            ],
        )


def test_efficientvit_sam_rejects_invalid_cached_embedding():
    adapter = EfficientViTSAMONNX(EncoderSession(), DecoderSession())

    with pytest.raises(ValueError, match="cached embedding"):
        adapter.predict_masks(
            {
                "image_embedding": np.zeros((1, 1), dtype=np.float32),
                "original_size": (2, 4),
            },
            [{"type": "point", "data": [2, 1], "label": 1}],
        )


class FakeEfficientVitSAM:
    def __init__(
        self,
        encoder,
        decoder,
        *,
        max_prompt_points,
        max_output_elements,
    ):
        self.sessions = (encoder, decoder)
        self.limits = (max_prompt_points, max_output_elements)
        self.encode_calls = 0

    def encode(self, image):
        self.encode_calls += 1
        return {
            "original_size": image.shape[:2],
            "image_embedding": np.zeros(1),
        }

    def predict_masks(self, embedding, prompts):
        del prompts
        height, width = embedding["original_size"]
        mask = np.zeros((1, 1, height, width), dtype=np.float32)
        mask[:, :, 2:-2, 3:-3] = 1
        return mask


def checked_session(*_args, **_kwargs):
    return SimpleNamespace(), SimpleNamespace(), ()


def efficientvit_config(tmp_path):
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    encoder.touch()
    decoder.touch()
    return {
        "name": "efficientvit-test",
        "model_revision": "revision-1",
        "encoder_model_path": str(encoder),
        "decoder_model_path": str(decoder),
    }


def efficientvit_request(source_id):
    return InferenceRequest(
        request_id="request-1",
        source_id=source_id,
        model_id="efficientvit-test",
        model_revision="revision-1",
        prompts=(
            PointPrompt(point=Point(x=5, y=6)),
            BoxPrompt(
                top_left=Point(x=2, y=2),
                bottom_right=Point(x=18, y=14),
            ),
        ),
        output_shape=ShapeType.RECTANGLE,
    )


def test_efficientvit_backend_reuses_embedding_and_releases_model(tmp_path):
    with (
        patch(
            "anylearning.inference.backends.efficientvit_sam.create_checked_onnx_session",
            side_effect=checked_session,
        ) as checked,
        patch(
            "anylearning.inference.backends.efficientvit_sam.EfficientViTSAMONNX",
            FakeEfficientVitSAM,
        ),
    ):
        session = EfficientVitSamBackend().create_session(efficientvit_config(tmp_path))
        session.load()
        assert checked.call_count == 2
        assert all(
            call.kwargs["enable_cpu_mem_arena"] is False
            for call in checked.call_args_list
        )
        assert all(
            call.kwargs["enable_mem_pattern"] is False
            for call in checked.call_args_list
        )
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        first = session.predict(efficientvit_request("image:one"), image)
        second = session.predict(efficientvit_request("image:one"), image)

        assert first.shapes == second.shapes
        assert first.shapes[0].type is ShapeType.RECTANGLE
        assert session._model.encode_calls == 1
        assert session._model.limits == (1_024, 50_000_000)
        with patch(
            "anylearning.inference.backends.efficientvit_sam.release_unused_cpu_memory"
        ) as release_memory:
            session.unload()
        assert session._model is None
        assert len(session._embedding_cache) == 0
        release_memory.assert_called_once_with()


def test_efficientvit_backend_bounds_output_before_encoding(tmp_path):
    config = {
        **efficientvit_config(tmp_path),
        "max_output_elements": 100,
    }
    with (
        patch(
            "anylearning.inference.backends.efficientvit_sam.create_checked_onnx_session",
            side_effect=checked_session,
        ),
        patch(
            "anylearning.inference.backends.efficientvit_sam.EfficientViTSAMONNX",
            FakeEfficientVitSAM,
        ),
    ):
        session = EfficientVitSamBackend().create_session(config)
        session.load()
        with pytest.raises(ValueError, match="output element limit"):
            session.predict(
                efficientvit_request("image:bounded"),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
        assert session._model.encode_calls == 0
