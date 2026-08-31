"""ONNX Runtime adapter for official EfficientViT-SAM model pairs.

EfficientViT-SAM uses a 512- or 1024-pixel image encoder while retaining the
original SAM 1024-pixel prompt and mask coordinate frame.  AnyLearning's image
boundary is uint8 RGB, matching the Apache-2.0 reference implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .sam_onnx import _static_dimension, _validate_finite, runtime_session
from .sam_prompts import prompt_arrays

_DECODER_INPUTS = ("image_embeddings", "point_coords", "point_labels")
_DECODER_OUTPUTS = ("masks", "iou_predictions")
_ENCODER_SIZES = frozenset((512, 1024))
_PROMPT_FRAME_SIZE = 1024
_RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def _validate_static_shape(
    actual: tuple[Any, ...],
    expected: tuple[int | None, ...],
    description: str,
) -> None:
    if len(actual) != len(expected):
        raise ValueError(f"{description} must be rank {len(expected)}")
    for value, required in zip(actual, expected, strict=True):
        static = _static_dimension(value)
        if required is not None and static not in (None, required):
            raise ValueError(f"{description} has an unexpected static shape")


def _require_static_dimensions(
    shape: tuple[Any, ...],
    required: Mapping[int, int],
    description: str,
) -> None:
    if any(
        _static_dimension(shape[index]) != value for index, value in required.items()
    ):
        raise ValueError(f"{description} must expose bounded static dimensions")


class EfficientViTSAMONNX:
    """Run one validated EfficientViT-SAM encoder/decoder pair."""

    prompt_frame_size = _PROMPT_FRAME_SIZE

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        *,
        max_prompt_points: int = 1_024,
        max_output_elements: int = 50_000_000,
    ) -> None:
        if max_prompt_points < 1 or max_output_elements < 1:
            raise ValueError("EfficientViT-SAM resource limits must be positive")
        self.max_prompt_points = max_prompt_points
        self.max_output_elements = max_output_elements
        self.encoder_session = runtime_session(encoder)
        self.decoder_session = runtime_session(decoder)

        encoder_inputs = self.encoder_session.get_inputs()
        if len(encoder_inputs) != 1 or encoder_inputs[0].name != "input_image":
            raise ValueError(
                "EfficientViT-SAM encoder must expose one input_image input"
            )
        encoder_input = encoder_inputs[0]
        if getattr(encoder_input, "type", None) != "tensor(float)":
            raise ValueError("EfficientViT-SAM encoder input must be float32")
        encoder_shape = tuple(encoder_input.shape)
        _validate_static_shape(
            encoder_shape,
            (1, 3, None, None),
            "EfficientViT-SAM encoder input",
        )
        height = _static_dimension(encoder_shape[2])
        width = _static_dimension(encoder_shape[3])
        if (
            height is None
            or width is None
            or height != width
            or height not in _ENCODER_SIZES
        ):
            raise ValueError(
                "EfficientViT-SAM encoder spatial input must be 512x512 or 1024x1024"
            )
        self.encoder_size = height
        encoder_outputs = self.encoder_session.get_outputs()
        if len(encoder_outputs) != 1 or encoder_outputs[0].name != "image_embeddings":
            raise ValueError(
                "EfficientViT-SAM encoder must expose one image_embeddings output"
            )
        if getattr(encoder_outputs[0], "type", None) != "tensor(float)":
            raise ValueError("EfficientViT-SAM encoder output must be float32")
        _validate_static_shape(
            tuple(encoder_outputs[0].shape),
            (1, 256, 64, 64),
            "EfficientViT-SAM encoder output",
        )
        decoder_inputs = self.decoder_session.get_inputs()
        if tuple(item.name for item in decoder_inputs) != _DECODER_INPUTS:
            raise ValueError("Unexpected EfficientViT-SAM decoder input contract")
        if any(
            getattr(item, "type", None) != "tensor(float)" for item in decoder_inputs
        ):
            raise ValueError("EfficientViT-SAM decoder inputs must be float32")
        _validate_static_shape(
            tuple(decoder_inputs[0].shape),
            (1, 256, 64, 64),
            "EfficientViT-SAM decoder image_embeddings input",
        )
        _require_static_dimensions(
            tuple(decoder_inputs[0].shape),
            {1: 256, 2: 64, 3: 64},
            "EfficientViT-SAM decoder image_embeddings input",
        )
        _validate_static_shape(
            tuple(decoder_inputs[1].shape),
            (1, None, 2),
            "EfficientViT-SAM decoder point_coords input",
        )
        _require_static_dimensions(
            tuple(decoder_inputs[1].shape),
            {2: 2},
            "EfficientViT-SAM decoder point_coords input",
        )
        _validate_static_shape(
            tuple(decoder_inputs[2].shape),
            (1, None),
            "EfficientViT-SAM decoder point_labels input",
        )

        decoder_outputs = self.decoder_session.get_outputs()
        if tuple(item.name for item in decoder_outputs) != _DECODER_OUTPUTS:
            raise ValueError("Unexpected EfficientViT-SAM decoder output contract")
        if any(
            getattr(item, "type", None) != "tensor(float)" for item in decoder_outputs
        ):
            raise ValueError("EfficientViT-SAM decoder outputs must be float32")
        _validate_static_shape(
            tuple(decoder_outputs[0].shape),
            (1, 4, 256, 256),
            "EfficientViT-SAM decoder masks output",
        )
        _require_static_dimensions(
            tuple(decoder_outputs[0].shape),
            {1: 4, 2: 256, 3: 256},
            "EfficientViT-SAM decoder masks output",
        )
        _validate_static_shape(
            tuple(decoder_outputs[1].shape),
            (1, 4),
            "EfficientViT-SAM decoder IoU output",
        )
        _require_static_dimensions(
            tuple(decoder_outputs[1].shape),
            {1: 4},
            "EfficientViT-SAM decoder IoU output",
        )

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> tuple[int, int]:
        if oldh <= 0 or oldw <= 0 or long_side_length <= 0:
            raise ValueError("EfficientViT-SAM image dimensions must be positive")
        scale = long_side_length / max(oldh, oldw)
        return int(oldh * scale + 0.5), int(oldw * scale + 0.5)

    @classmethod
    def apply_coords(
        cls,
        coords: np.ndarray,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = cls.get_preprocess_shape(old_h, old_w, cls.prompt_frame_size)
        scaled = np.asarray(coords, dtype=np.float64).copy()
        scaled[..., 0] *= new_w / old_w
        scaled[..., 1] *= new_h / old_h
        return scaled.astype(np.float32)

    def encode(self, image: np.ndarray) -> dict[str, Any]:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("EfficientViT-SAM expects an H x W x 3 uint8 RGB image")
        original_size = image.shape[:2]
        resized_size = self.get_preprocess_shape(*original_size, self.encoder_size)
        if resized_size == original_size:
            resized = image
        else:
            resized = np.asarray(
                Image.fromarray(image).resize(
                    (resized_size[1], resized_size[0]),
                    resample=Image.Resampling.BILINEAR,
                )
            )
        normalized = (resized.astype(np.float32) - _RGB_MEAN) / _RGB_STD
        chw = normalized.transpose(2, 0, 1)
        input_image = np.pad(
            chw,
            (
                (0, 0),
                (0, self.encoder_size - resized_size[0]),
                (0, self.encoder_size - resized_size[1]),
            ),
        )[None].astype(np.float32, copy=False)
        outputs = self.encoder_session.run(None, {"input_image": input_image})
        if len(outputs) != 1:
            raise ValueError("EfficientViT-SAM encoder output count changed at runtime")
        image_embedding = np.asarray(outputs[0])
        if image_embedding.shape != (1, 256, 64, 64):
            raise ValueError(
                "EfficientViT-SAM encoder embedding must have shape 1x256x64x64"
            )
        _validate_finite("EfficientViT-SAM encoder embedding", image_embedding)
        return {
            "image_embedding": image_embedding.astype(np.float32, copy=False),
            "original_size": original_size,
        }

    def predict_masks(
        self,
        embedding: Mapping[str, Any],
        prompt: Iterable[Mapping[str, Any]],
    ) -> np.ndarray:
        original_size = tuple(embedding["original_size"])
        if len(original_size) != 2 or any(
            not isinstance(value, (int, np.integer)) or value <= 0
            for value in original_size
        ):
            raise ValueError("EfficientViT-SAM original size must be positive H x W")
        required_output_elements = (
            4 * 256 * 256
            + 4
            + self.prompt_frame_size * self.prompt_frame_size
            + int(original_size[0]) * int(original_size[1])
        )
        if required_output_elements > self.max_output_elements:
            raise ValueError(
                "EfficientViT-SAM mask processing exceeds the configured output "
                "element limit"
            )
        marks = tuple(prompt)
        points, labels = prompt_arrays(marks)
        if len(points) > self.max_prompt_points:
            raise ValueError(
                "EfficientViT-SAM prompts exceed the configured point limit"
            )
        has_box = any(mark.get("type") == "rectangle" for mark in marks)
        if not has_box:
            points = np.concatenate(
                (points, np.zeros((1, 2), dtype=np.float32)), axis=0
            )
            labels = np.concatenate(
                (labels, np.asarray([-1], dtype=np.float32)), axis=0
            )
        coords = self.apply_coords(points, original_size)[None]
        image_embedding = np.asarray(embedding["image_embedding"])
        if image_embedding.shape != (1, 256, 64, 64):
            raise ValueError(
                "EfficientViT-SAM cached embedding must have shape 1x256x64x64"
            )
        _validate_finite("EfficientViT-SAM cached embedding", image_embedding)
        decoder_inputs = {
            "image_embeddings": image_embedding.astype(np.float32, copy=False),
            "point_coords": coords,
            "point_labels": labels[None].astype(np.float32, copy=False),
        }
        outputs = self.decoder_session.run(None, decoder_inputs)
        if len(outputs) != 2:
            raise ValueError("EfficientViT-SAM decoder output count changed at runtime")
        masks = np.asarray(outputs[0])
        scores = np.asarray(outputs[1])
        if masks.shape != (1, 4, 256, 256) or scores.shape != (1, 4):
            raise ValueError("EfficientViT-SAM decoder output shapes changed")
        if masks.size + scores.size > self.max_output_elements:
            raise ValueError(
                "EfficientViT-SAM decoder outputs exceed the configured limit"
            )
        _validate_finite("EfficientViT-SAM decoder masks", masks)
        _validate_finite("EfficientViT-SAM decoder IoU scores", scores)

        # Token zero is the single-mask candidate. Native evaluation requests
        # multimask output and chooses the highest-IoU candidate from tokens 1-3.
        best_index = int(np.argmax(scores[0, 1:4])) + 1
        selected = masks[0, best_index]
        processed = self.postprocess_mask(selected, original_size)
        _validate_finite("EfficientViT-SAM postprocessed mask", processed)
        return processed[None, None].astype(np.float32, copy=False)

    @classmethod
    def postprocess_mask(
        cls, mask: np.ndarray, original_size: tuple[int, int]
    ) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.shape != (256, 256):
            raise ValueError("EfficientViT-SAM low-resolution mask must be 256x256")
        resized_size = cls.get_preprocess_shape(*original_size, cls.prompt_frame_size)
        upscaled = cv2.resize(
            mask,
            (cls.prompt_frame_size, cls.prompt_frame_size),
            interpolation=cv2.INTER_LINEAR,
        )
        cropped = upscaled[: resized_size[0], : resized_size[1]]
        return cv2.resize(
            cropped,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )


__all__ = ["EfficientViTSAMONNX"]
