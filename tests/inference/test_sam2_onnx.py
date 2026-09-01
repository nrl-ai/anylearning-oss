from types import SimpleNamespace

import numpy as np
import pytest

from anylearning.inference.backends.sam2_onnx import SegmentAnything2ONNX


def node(name, shape, tensor_type="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=tensor_type)


class EncoderSession:
    def __init__(self, shape=(1, 3, 64, 64)):
        self._inputs = [node("image", shape)]
        self._outputs = [
            node("high_res_feats_0", (1, 2, 16, 16)),
            node("high_res_feats_1", (1, 2, 8, 8)),
            node("image_embed", (1, 2, 4, 4)),
        ]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names == [item.name for item in self._outputs]
        self.feed = feed
        return [
            np.zeros((1, 2, 16, 16), dtype=np.float32),
            np.zeros((1, 2, 8, 8), dtype=np.float32),
            np.zeros((1, 2, 4, 4), dtype=np.float32),
        ]


class DecoderSession:
    def __init__(self):
        # Deliberately vary graph order to ensure feeds are bound by name.
        names = (
            "point_labels",
            "image_embed",
            "has_mask_input",
            "high_res_feats_1",
            "point_coords",
            "mask_input",
            "high_res_feats_0",
        )
        self._inputs = [node(name, ()) for name in names]
        self._outputs = [
            node("iou_predictions", (1, 2)),
            node("masks", (1, 2, 16, 16)),
        ]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names == ["iou_predictions", "masks"]
        self.feed = feed
        masks = np.zeros((1, 2, 16, 16), dtype=np.float32)
        masks[:, 1] = 1
        return [np.asarray([[0.1, 0.9]], dtype=np.float32), masks]


def test_sam2_native_rgb_named_feeds_coordinate_scaling_and_best_mask():
    encoder = EncoderSession()
    decoder = DecoderSession()
    adapter = SegmentAnything2ONNX(encoder, decoder)
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[..., 0] = 255

    embedding = adapter.encode(image)
    masks = adapter.predict_masks(
        embedding, [{"type": "point", "data": [20, 10], "label": 1}]
    )

    tensor = encoder.feed["image"]
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229)
    assert tensor[0, 2, 0, 0] == pytest.approx((0 - 0.406) / 0.225)
    assert set(decoder.feed) == {
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
    }
    assert decoder.feed["point_coords"].tolist() == [[[32.0, 32.0]]]
    assert decoder.feed["point_labels"].tolist() == [[1.0]]
    assert masks.shape == (1, 1, 20, 40)
    assert np.allclose(masks, 1)


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 64, 64), (1, 4, 64, 64), (1, 3, "height", 64), (1, 3, 8, 8)],
)
def test_sam2_invalid_encoder_contract_is_rejected(shape):
    with pytest.raises(ValueError):
        SegmentAnything2ONNX(EncoderSession(shape), DecoderSession())


def test_sam2_invalid_decoder_contract_is_rejected():
    decoder = DecoderSession()
    decoder._inputs.pop()

    with pytest.raises(ValueError, match="Unexpected SAM2 decoder"):
        SegmentAnything2ONNX(EncoderSession(), decoder)


def test_sam2_rejects_non_uint8_rgb_input():
    adapter = SegmentAnything2ONNX(EncoderSession(), DecoderSession())

    with pytest.raises(ValueError, match="uint8 RGB"):
        adapter.encode(np.zeros((20, 40, 3), dtype=np.float32))
