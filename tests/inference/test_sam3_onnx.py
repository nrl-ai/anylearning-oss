from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from anylearning.inference.backends.clip_tokenizer import ClipTokenizer
from anylearning.inference.backends.sam3_onnx import (
    Sam3Decoder,
    Sam3Detections,
    Sam3ImageEncoder,
    Sam3OnnxPipeline,
    resize_sam3_detections,
    select_sam3_detections,
)


@dataclass
class TensorMetadata:
    name: str
    shape: list[Any]
    type: str


class FakeSession:
    def __init__(self, inputs, outputs, result):
        self._inputs = [TensorMetadata(*item) for item in inputs]
        self._outputs = [TensorMetadata(*item) for item in outputs]
        self.result = result
        self.calls = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, inputs):
        self.calls.append((tuple(output_names), inputs))
        if callable(self.result):
            return self.result(output_names, inputs)
        by_name = dict(
            zip((item.name for item in self._outputs), self.result, strict=True)
        )
        return [by_name[name] for name in output_names]


IMAGE_OUTPUTS = [
    ("vision_pos_enc_0", [1, 2, 4, 4], "tensor(float)"),
    ("vision_pos_enc_1", [1, 2, 2, 2], "tensor(float)"),
    ("vision_pos_enc_2", [1, 2, 1, 1], "tensor(float)"),
    ("backbone_fpn_0", [1, 2, 4, 4], "tensor(float)"),
    ("backbone_fpn_1", [1, 2, 2, 2], "tensor(float)"),
    ("backbone_fpn_2", [1, 2, 1, 1], "tensor(float)"),
]
LANGUAGE_OUTPUTS = [
    ("text_attention_mask", [1, 8], "tensor(bool)"),
    ("text_memory", [8, 1, 4], "tensor(float)"),
    ("text_embeds", [8, 1, 6], "tensor(float)"),
]
DECODER_INPUTS = [
    ("original_height", [], "tensor(int64)"),
    ("original_width", [], "tensor(int64)"),
    ("vision_pos_enc_2", [1, 2, 1, 1], "tensor(float)"),
    ("backbone_fpn_0", [1, 2, 4, 4], "tensor(float)"),
    ("backbone_fpn_1", [1, 2, 2, 2], "tensor(float)"),
    ("backbone_fpn_2", [1, 2, 1, 1], "tensor(float)"),
    ("language_mask", [1, 8], "tensor(bool)"),
    ("language_features", [8, 1, 4], "tensor(float)"),
    ("box_coords", [2, 1, 4], "tensor(float)"),
    ("box_labels", [2, 1], "tensor(int64)"),
    ("box_masks", [1, 2], "tensor(bool)"),
]
PROCESSED_OUTPUTS = [
    ("boxes", ["queries", 4], "tensor(float)"),
    ("scores", ["queries"], "tensor(float)"),
    ("masks", ["queries", 1, "height", "width"], "tensor(bool)"),
]
RAW_OUTPUTS = [
    ("pred_boxes", [1, 3, 4], "tensor(float)"),
    ("pred_logits", [1, 3, 1], "tensor(float)"),
    ("pred_masks", [1, 3, 4, 4], "tensor(float)"),
    ("presence_logit_dec", [1, 1], "tensor(float)"),
]


def _image_result():
    return [np.zeros(item[1], dtype=np.float32) for item in IMAGE_OUTPUTS]


def _language_result():
    return [
        np.zeros((1, 8), dtype=np.bool_),
        np.zeros((8, 1, 4), dtype=np.float32),
        np.zeros((8, 1, 6), dtype=np.float32),
    ]


def _pipeline(decoder_result):
    image_session = FakeSession(
        [("image", [3, 16, 16], "tensor(uint8)")],
        IMAGE_OUTPUTS,
        _image_result(),
    )
    language_session = FakeSession(
        [("tokens", [1, 8], "tensor(int64)")],
        LANGUAGE_OUTPUTS,
        _language_result(),
    )
    decoder_session = FakeSession(
        DECODER_INPUTS,
        PROCESSED_OUTPUTS,
        decoder_result,
    )
    return (
        Sam3OnnxPipeline(
            image_session,
            language_session,
            decoder_session,
            max_text_bytes=4096,
            max_raw_queries=8,
            max_output_elements=100_000,
            max_nms_candidates=4,
            max_feature_elements=1000,
        ),
        image_session,
        language_session,
        decoder_session,
    )


def test_clip_tokenizer_matches_pinned_clip_vocabulary():
    tokenizer = ClipTokenizer()

    tokens = tokenizer.tokenize("a red truck", context_length=8)

    assert tokens.tolist() == [[49406, 320, 736, 4629, 49407, 0, 0, 0]]


def test_clip_tokenizer_rejects_byte_and_token_overflow():
    byte_limited = ClipTokenizer(max_text_bytes=16)

    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        byte_limited.tokenize("x" * 17, context_length=32)

    tokenizer = ClipTokenizer()
    with pytest.raises(ValueError, match="model capacity"):
        tokenizer.tokenize("one two three four five six", context_length=3)


def test_image_encoder_preserves_rgb_and_validates_feature_budget():
    session = FakeSession(
        [("image", [3, 16, 16], "tensor(uint8)")],
        IMAGE_OUTPUTS,
        _image_result(),
    )
    encoder = Sam3ImageEncoder(session, max_feature_elements=1000)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 255

    encoder(image)

    tensor = session.calls[0][1]["image"]
    assert tensor.shape == (3, 16, 16)
    assert np.all(tensor[0] == 255)
    assert np.all(tensor[2] == 0)
    with pytest.raises(ValueError, match="element limit"):
        Sam3ImageEncoder(session, max_feature_elements=1)

    invalid_dimension = list(IMAGE_OUTPUTS)
    invalid_dimension[0] = (
        "vision_pos_enc_0",
        [1, 2, -1, 4],
        "tensor(float)",
    )
    with pytest.raises(ValueError, match="dimension"):
        Sam3ImageEncoder(
            FakeSession(
                [("image", [3, 16, 16], "tensor(uint8)")],
                invalid_dimension,
                _image_result(),
            ),
            max_feature_elements=1000,
        )


def test_pipeline_encodes_text_and_normalizes_geometric_prompts():
    masks = np.zeros((1, 1, 20, 40), dtype=np.bool_)
    masks[0, 0, 2:10, 3:15] = True
    pipeline, _, language_session, decoder_session = _pipeline(
        [
            np.array([[3, 2, 15, 10]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            masks,
        ]
    )
    image_features = pipeline.encode_image(np.zeros((20, 40, 3), dtype=np.uint8))
    language_features = pipeline.encode_text("dog")

    result = pipeline.predict(
        image_features=image_features,
        language_features=language_features,
        geometric_prompts=[{"type": "rectangle", "data": [10, 4, 30, 16]}],
        original_size=(20, 40),
        confidence_threshold=0.5,
        nms_threshold=0.7,
        max_instances=1,
    )

    assert language_session.calls[0][1]["tokens"].tolist()[:1] == [
        [49406, 1929, 49407, 0, 0, 0, 0, 0]
    ]
    decoder_inputs = decoder_session.calls[0][1]
    np.testing.assert_allclose(decoder_inputs["box_coords"][0, 0], [0.5, 0.5, 0.5, 0.6])
    assert not decoder_inputs["box_masks"][0, 0]
    assert decoder_inputs["box_masks"][0, 1]
    assert result.masks.shape == (1, 1, 20, 40)
    assert result.scores.tolist() == pytest.approx([0.9])


def test_pipeline_accepts_text_only_and_enforces_exported_prompt_capacity():
    pipeline, _, _, decoder_session = _pipeline(
        [
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 1, 20, 40), dtype=np.bool_),
        ]
    )
    image_features = pipeline.encode_image(np.zeros((20, 40, 3), dtype=np.uint8))
    language_features = pipeline.encode_text("dog")

    pipeline.predict(
        image_features=image_features,
        language_features=language_features,
        geometric_prompts=[],
        original_size=(20, 40),
        confidence_threshold=0.5,
        nms_threshold=0.7,
        max_instances=1,
    )

    assert decoder_session.calls[0][1]["box_masks"].all()
    with pytest.raises(ValueError, match="at most 2"):
        pipeline.geometric_inputs(
            [
                {"type": "point", "data": [1, 1], "label": 1},
                {"type": "point", "data": [2, 2], "label": 1},
                {"type": "point", "data": [3, 3], "label": 1},
            ],
            image_size=(20, 40),
        )


def test_raw_decoder_scores_and_resizes_only_retained_native_masks():
    masks = np.full((1, 3, 4, 4), -5.0, dtype=np.float32)
    masks[0, 0, :2, :2] = 5
    masks[0, 1, :2, :2] = 5
    masks[0, 2, 2:, 2:] = 5
    session = FakeSession(
        DECODER_INPUTS,
        RAW_OUTPUTS,
        [
            np.array(
                [
                    [
                        [0.25, 0.25, 0.5, 0.5],
                        [0.25, 0.25, 0.5, 0.5],
                        [0.75, 0.75, 0.5, 0.5],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([[[3.0], [2.0], [4.0]]], dtype=np.float32),
            masks,
            np.array([[3.0]], dtype=np.float32),
        ],
    )
    decoder = Sam3Decoder(
        session,
        context_length=8,
        max_raw_queries=3,
        max_output_elements=1000,
    )
    image_features = {
        name: np.zeros(shape, dtype=np.float32) for name, shape, _ in IMAGE_OUTPUTS
    }
    language_features = {
        "language_mask": np.zeros((1, 8), dtype=np.bool_),
        "language_features": np.zeros((8, 1, 4), dtype=np.float32),
        "language_embeds": np.zeros((8, 1, 6), dtype=np.float32),
    }

    raw = decoder.run(
        original_size=(20, 40),
        image_features=image_features,
        language_features=language_features,
        box_coords=np.zeros((2, 1, 4), dtype=np.float32),
        box_labels=np.ones((2, 1), dtype=np.int64),
        box_masks=np.ones((1, 2), dtype=np.bool_),
    )
    selected = select_sam3_detections(
        raw,
        confidence_threshold=0.5,
        nms_threshold=0.5,
        max_instances=2,
        max_nms_candidates=3,
    )
    resized = resize_sam3_detections(selected, original_size=(20, 40))

    assert decoder.output_profile == "raw"
    assert resized.masks.shape == (2, 1, 20, 40)
    assert resized.scores[0] > resized.scores[1]
    np.testing.assert_allclose(resized.boxes[0], [20, 10, 40, 20])


def test_selection_rejects_non_finite_out_of_range_and_oversized_contracts():
    valid = Sam3Detections(
        masks=np.zeros((1, 4, 4), dtype=np.bool_),
        scores=np.array([1.1], dtype=np.float32),
        boxes=np.zeros((1, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="between zero and one"):
        select_sam3_detections(
            valid,
            confidence_threshold=0.5,
            nms_threshold=0.7,
            max_instances=1,
            max_nms_candidates=1,
        )

    decoder_session = FakeSession(
        DECODER_INPUTS,
        PROCESSED_OUTPUTS,
        [
            np.zeros((9, 4), dtype=np.float32),
            np.zeros((9,), dtype=np.float32),
            np.zeros((9, 1, 20, 40), dtype=np.bool_),
        ],
    )
    decoder = Sam3Decoder(
        decoder_session,
        context_length=8,
        max_raw_queries=8,
        max_output_elements=100_000,
    )
    with pytest.raises(ValueError, match="query count"):
        decoder._processed_detections(
            dict(
                zip(
                    (item[0] for item in PROCESSED_OUTPUTS),
                    decoder_session.result,
                    strict=True,
                )
            ),
            (20, 40),
        )


def test_pipeline_rejects_cross_graph_shape_mismatch():
    mismatched_inputs = list(DECODER_INPUTS)
    mismatched_inputs[2] = ("vision_pos_enc_2", [1, 3, 1, 1], "tensor(float)")
    image_session = FakeSession(
        [("image", [3, 16, 16], "tensor(uint8)")], IMAGE_OUTPUTS, _image_result()
    )
    language_session = FakeSession(
        [("tokens", [1, 8], "tensor(int64)")], LANGUAGE_OUTPUTS, _language_result()
    )
    decoder_session = FakeSession(mismatched_inputs, PROCESSED_OUTPUTS, [])

    with pytest.raises(ValueError, match="does not match the decoder"):
        Sam3OnnxPipeline(
            image_session,
            language_session,
            decoder_session,
            max_text_bytes=4096,
            max_raw_queries=8,
            max_output_elements=100_000,
            max_nms_candidates=4,
            max_feature_elements=1000,
        )
