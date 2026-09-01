from types import SimpleNamespace

import numpy as np
import pytest

from anylearning.inference.backends.sam_onnx import SegmentAnythingONNX


def node(name, shape, tensor_type="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=tensor_type)


class EncoderSession:
    def __init__(self, shape):
        self._inputs = [node("image", shape)]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feed):
        assert output_names is None
        self.feed = feed
        return [np.zeros((1, 256, 64, 64), dtype=np.float32)]


class DecoderSession:
    def __init__(self, outputs):
        self._inputs = [
            node("image_embeddings", (1, 256, 64, 64)),
            node("point_coords", (1, None, 2)),
            node("point_labels", (1, None)),
            node("mask_input", (1, 1, 256, 256)),
            node("has_mask_input", (1,)),
            node("orig_im_size", (2,)),
        ]
        self._outputs = [node(name, value.shape) for name, value in outputs]
        self.outputs = [value for _, value in outputs]
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        assert output_names is None
        self.feed = feed
        return self.outputs


def decoder_with_low_res(masks=None, scores=None):
    if masks is None:
        masks = np.zeros((1, 1, 256, 256), dtype=np.float32)
    if scores is None:
        scores = np.ones((1, masks.shape[1]), dtype=np.float32)
    return DecoderSession(
        [("masks", masks), ("iou_predictions", scores), ("low_res_masks", masks)]
    )


def test_hwc_encoder_uses_native_rgb_and_official_portrait_resize():
    encoder = EncoderSession(("height", "width", 3))
    adapter = SegmentAnythingONNX(encoder, decoder_with_low_res())
    image = np.zeros((1200, 600, 3), dtype=np.uint8)
    image[..., 0] = 255

    embedding = adapter.encode(image)

    encoder_input = encoder.feed["image"]
    assert encoder_input.shape == (1024, 512, 3)
    assert encoder_input.dtype == np.float32
    assert np.all(encoder_input[..., 0] == 255)
    assert np.all(encoder_input[..., 2] == 0)
    assert embedding["original_size"] == (1200, 600)
    assert embedding["resized_size"] == (1024, 512)


def test_raw_encoder_normalizes_and_pads_nchw_without_swapping_rgb():
    encoder = EncoderSession((1, 3, 1024, 1024))
    adapter = SegmentAnythingONNX(encoder, decoder_with_low_res())
    image = np.zeros((600, 1200, 3), dtype=np.uint8)
    image[..., 0] = 255

    adapter.encode(image)

    encoder_input = encoder.feed["image"]
    assert encoder_input.shape == (1, 3, 1024, 1024)
    assert encoder_input.dtype == np.float32
    assert encoder_input[0, 0, 0, 0] == pytest.approx((255 - 123.675) / 58.395)
    assert encoder_input[0, 2, 0, 0] == pytest.approx((0 - 103.53) / 57.375)
    assert np.all(encoder_input[:, :, 512:, :] == 0)


def test_low_resolution_masks_use_aspect_ratio_crop_and_best_iou_candidate():
    encoder = EncoderSession(("height", "width", 3))
    low_res = np.zeros((1, 3, 256, 256), dtype=np.float32)
    low_res[0, 0, :, :128] = 1
    low_res[0, 1] = 0.25
    low_res[0, 2] = 0.75
    decoder = decoder_with_low_res(
        low_res, np.asarray([[0.1, 0.9, 0.2]], dtype=np.float32)
    )
    adapter = SegmentAnythingONNX(encoder, decoder)
    embedding = adapter.encode(np.zeros((1200, 600, 3), dtype=np.uint8))

    masks = adapter.predict_masks(
        embedding, [{"type": "point", "data": [300, 600], "label": 1}]
    )

    assert masks.shape == (1, 1, 1200, 600)
    assert np.allclose(masks, 0.25)
    assert decoder.feed["point_coords"].tolist() == [[[256.0, 512.0], [0.0, 0.0]]]
    assert decoder.feed["point_labels"].tolist() == [[1.0, -1.0]]
    assert decoder.feed["orig_im_size"].tolist() == [1200.0, 600.0]


def test_portrait_crop_does_not_stretch_landscape_export_padding():
    low_res = np.zeros((1, 1, 256, 256), dtype=np.float32)
    # Keep the interpolation transition outside the portrait crop. The right
    # half is export padding and must never be stretched into the real image.
    low_res[..., :130] = 1
    adapter = SegmentAnythingONNX(
        EncoderSession(("height", "width", 3)), decoder_with_low_res(low_res)
    )

    masks = adapter.postprocess_masks(low_res, (1200, 600), (1024, 512))

    assert masks.shape == (1, 1, 1200, 600)
    assert np.allclose(masks, 1)


@pytest.mark.parametrize(
    "encoder_shape",
    [(1, 4, 1024, 1024), (1, 3, 640, 640), (1, 3), (1, 3, 1, 1, 1)],
)
def test_invalid_encoder_contract_is_rejected(encoder_shape):
    with pytest.raises(ValueError):
        SegmentAnythingONNX(EncoderSession(encoder_shape), decoder_with_low_res())


def test_invalid_decoder_contract_is_rejected():
    decoder = decoder_with_low_res()
    decoder._inputs.pop()

    with pytest.raises(ValueError, match="Unexpected SAM decoder inputs"):
        SegmentAnythingONNX(EncoderSession(("height", "width", 3)), decoder)


def test_non_uint8_or_non_rgb_images_are_rejected():
    adapter = SegmentAnythingONNX(
        EncoderSession(("height", "width", 3)), decoder_with_low_res()
    )

    with pytest.raises(ValueError, match="uint8 RGB"):
        adapter.encode(np.zeros((32, 32, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8 RGB"):
        adapter.encode(np.zeros((32, 32), dtype=np.uint8))
