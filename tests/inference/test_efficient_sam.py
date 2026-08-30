from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from anylearning.inference import InferenceRequest, Point, PointPrompt, ShapeType
from anylearning.inference.backends.efficient_sam import EfficientSamBackend
from anylearning.inference.backends.efficient_sam_onnx import EfficientSAMONNX


def node(name, shape, tensor_type="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=tensor_type)


class EncoderSession:
    def __init__(self):
        self._inputs = [node("batched_images", ("batch", 3, "height", "width"))]
        self._outputs = [node("image_embeddings", (1, 256, 64, 64))]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names is None
        self.feed = feed
        return [np.zeros((1, 256, 2, 3), dtype=np.float32)]


class DecoderSession:
    def __init__(self):
        self._inputs = [
            node("image_embeddings", (1, 256, 2, 3)),
            node("batched_point_coords", (1, 1, "points", 2)),
            node("batched_point_labels", (1, 1, "points")),
            node("orig_im_size", (2,), "tensor(int64)"),
        ]
        self._outputs = [
            node("output_masks", (1, 1, 3, 2, 3)),
            node("iou_predictions", (1, 1, 3)),
            node("low_res_masks", (1, 3, 4, 4)),
        ]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names is None
        self.feed = feed
        masks = np.zeros((1, 1, 3, 2, 3), dtype=np.float32)
        masks[0, 0, 2] = 1
        return [
            masks,
            np.asarray([[[0.1, 0.2, 0.9]]], dtype=np.float32),
            np.zeros((1, 3, 4, 4), dtype=np.float32),
        ]


def test_efficient_sam_native_rgb_box_prompt_and_best_candidate():
    encoder = EncoderSession()
    decoder = DecoderSession()
    adapter = EfficientSAMONNX(encoder, decoder)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 255

    embedding = adapter.encode(image)
    masks = adapter.predict_masks(
        embedding, [{"type": "rectangle", "data": [0, 0, 2, 1], "label": 1}]
    )

    encoded = encoder.feed["batched_images"]
    assert encoded.shape == (1, 3, 2, 3)
    assert encoded[0, 0, 0, 0] == 1
    assert encoded[0, 2, 0, 0] == 0
    assert decoder.feed["batched_point_labels"].tolist() == [[[2.0, 3.0]]]
    assert decoder.feed["orig_im_size"].dtype == np.int64
    assert masks.shape == (1, 1, 2, 3)
    assert np.all(masks == 1)


def test_efficient_sam_unknown_decoder_contract_is_rejected():
    decoder = DecoderSession()
    decoder._inputs.pop()

    with pytest.raises(ValueError, match="Unexpected EfficientSAM decoder"):
        EfficientSAMONNX(EncoderSession(), decoder)


def test_efficient_sam_rejects_non_uint8_rgb_image():
    adapter = EfficientSAMONNX(EncoderSession(), DecoderSession())

    with pytest.raises(ValueError, match="uint8 RGB"):
        adapter.encode(np.zeros((2, 3, 3), dtype=np.float32))


def test_efficient_sam_rejects_nonfinite_decoder_output():
    decoder = DecoderSession()
    original_run = decoder.run

    def run(output_names, feed):
        outputs = original_run(output_names, feed)
        outputs[1][0, 0, 0] = np.nan
        return outputs

    decoder.run = run
    adapter = EfficientSAMONNX(EncoderSession(), decoder)
    embedding = adapter.encode(np.zeros((2, 3, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict_masks(
            embedding, [{"type": "point", "data": [1, 1], "label": 1}]
        )


class FakeEfficientSAM:
    def __init__(self, encoder, decoder):
        self.sessions = (encoder, decoder)
        self.encode_calls = 0

    def encode(self, image):
        self.encode_calls += 1
        return {"original_size": image.shape[:2], "image_embedding": np.zeros(1)}

    def predict_masks(self, embedding, prompts):
        height, width = embedding["original_size"]
        mask = np.zeros((1, 1, height, width), dtype=np.float32)
        mask[:, :, 2:-2, 3:-3] = 1
        return mask


def checked_session(*_args, **_kwargs):
    return SimpleNamespace(), SimpleNamespace(), ()


def efficient_config(tmp_path):
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    encoder.touch()
    decoder.touch()
    return {
        "name": "efficient-test",
        "model_revision": "revision-1",
        "encoder_model_path": str(encoder),
        "decoder_model_path": str(decoder),
    }


def efficient_request(source_id):
    return InferenceRequest(
        request_id="request-1",
        source_id=source_id,
        model_id="efficient-test",
        model_revision="revision-1",
        prompts=(PointPrompt(point=Point(x=5, y=6)),),
        output_shape=ShapeType.RECTANGLE,
    )


def test_efficient_backend_reuses_embedding_and_releases_model(tmp_path):
    with (
        patch(
            "anylearning.inference.backends.efficient_sam.create_checked_onnx_session",
            side_effect=checked_session,
        ) as checked,
        patch(
            "anylearning.inference.backends.efficient_sam.EfficientSAMONNX",
            FakeEfficientSAM,
        ),
    ):
        session = EfficientSamBackend().create_session(efficient_config(tmp_path))
        session.load()
        assert checked.call_count == 2
        assert all(
            call.kwargs["enable_cpu_mem_arena"] is False
            for call in checked.call_args_list
        )
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        first = session.predict(efficient_request("image:one"), image)
        second = session.predict(efficient_request("image:one"), image)

        assert first.shapes == second.shapes
        assert first.shapes[0].type is ShapeType.RECTANGLE
        assert session._model.encode_calls == 1
        session.unload()
        assert session._model is None
        assert len(session._embedding_cache) == 0


def test_efficient_backend_bounds_candidate_output_before_encoding(tmp_path):
    config = {
        **efficient_config(tmp_path),
        "max_image_pixels": 1_000,
        "max_output_elements": 100,
    }
    with (
        patch(
            "anylearning.inference.backends.efficient_sam.create_checked_onnx_session",
            side_effect=checked_session,
        ),
        patch(
            "anylearning.inference.backends.efficient_sam.EfficientSAMONNX",
            FakeEfficientSAM,
        ),
    ):
        session = EfficientSamBackend().create_session(config)
        session.load()
        with pytest.raises(ValueError, match="candidate masks"):
            session.predict(
                efficient_request("image:large-output"),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
