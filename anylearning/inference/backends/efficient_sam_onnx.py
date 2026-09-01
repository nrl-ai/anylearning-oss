"""ONNX Runtime adapter for official EfficientSAM-Ti and EfficientSAM-S pairs.

The processing contract is based on the Apache-2.0 EfficientSAM reference ONNX
example. AnyLearning receives native uint8 RGB arrays, so no channel swap is
performed at this boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .sam_onnx import runtime_session
from .sam_prompts import prompt_arrays

_DECODER_INPUTS = frozenset(
    {
        "image_embeddings",
        "batched_point_coords",
        "batched_point_labels",
        "orig_im_size",
    }
)


class EfficientSAMONNX:
    """Run an official split EfficientSAM encoder and prompt decoder."""

    def __init__(self, encoder: Any, decoder: Any) -> None:
        self.encoder_session = runtime_session(encoder)
        self.decoder_session = runtime_session(decoder)

        encoder_inputs = self.encoder_session.get_inputs()
        if len(encoder_inputs) != 1:
            raise ValueError("EfficientSAM encoder must expose exactly one input")
        encoder_input = encoder_inputs[0]
        if len(encoder_input.shape) != 4:
            raise ValueError("EfficientSAM encoder input must be rank-4 NCHW")
        batch, channels, height, width = encoder_input.shape
        if isinstance(batch, int) and batch != 1:
            raise ValueError("EfficientSAM encoder batch size must be one or dynamic")
        if isinstance(channels, int) and channels != 3:
            raise ValueError("EfficientSAM encoder input must have three RGB channels")
        for dimension in (height, width):
            if isinstance(dimension, int) and not 16 <= dimension <= 16_384:
                raise ValueError(
                    "EfficientSAM encoder spatial dimensions must be dynamic or "
                    "between 16 and 16384"
                )
        if getattr(encoder_input, "type", "tensor(float)") != "tensor(float)":
            raise ValueError("EfficientSAM encoder input must be float32")
        self.encoder_input_name = encoder_input.name
        encoder_outputs = self.encoder_session.get_outputs()
        if len(encoder_outputs) != 1 or encoder_outputs[0].name != "image_embeddings":
            raise ValueError(
                "EfficientSAM encoder must expose one image_embeddings output"
            )

        decoder_inputs = {item.name: item for item in self.decoder_session.get_inputs()}
        if set(decoder_inputs) != _DECODER_INPUTS:
            raise ValueError("Unexpected EfficientSAM decoder input contract")
        if getattr(decoder_inputs["orig_im_size"], "type", "tensor(int64)") != (
            "tensor(int64)"
        ):
            raise ValueError("EfficientSAM orig_im_size input must be int64")
        self.decoder_output_names = tuple(
            item.name for item in self.decoder_session.get_outputs()
        )
        if "output_masks" not in self.decoder_output_names or (
            "iou_predictions" not in self.decoder_output_names
        ):
            raise ValueError(
                "EfficientSAM decoder must expose output_masks and iou_predictions"
            )

    def encode(self, image: np.ndarray) -> dict[str, Any]:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("EfficientSAM expects an H x W x 3 uint8 RGB image")
        input_image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outputs = self.encoder_session.run(None, {self.encoder_input_name: input_image})
        if len(outputs) != 1:
            raise ValueError("EfficientSAM encoder output count changed at runtime")
        image_embedding = np.asarray(outputs[0])
        if image_embedding.ndim != 4 or image_embedding.shape[0] != 1:
            raise ValueError(
                "EfficientSAM image embedding must be a rank-4 batch of one"
            )
        if not np.isfinite(image_embedding).all():
            raise ValueError("EfficientSAM image embedding contains NaN or infinity")
        return {
            "image_embedding": image_embedding,
            "original_size": image.shape[:2],
        }

    def predict_masks(
        self,
        embedding: Mapping[str, Any],
        prompt: Iterable[Mapping[str, Any]],
    ) -> np.ndarray:
        points, labels = prompt_arrays(prompt)
        original_size = tuple(embedding["original_size"])
        if len(original_size) != 2 or any(
            not isinstance(value, (int, np.integer)) or value <= 0
            for value in original_size
        ):
            raise ValueError("EfficientSAM original size must be positive H x W")
        decoder_inputs = {
            "image_embeddings": np.asarray(embedding["image_embedding"]),
            "batched_point_coords": points[None, None],
            "batched_point_labels": labels[None, None],
            "orig_im_size": np.asarray(original_size, dtype=np.int64),
        }
        outputs = self.decoder_session.run(None, decoder_inputs)
        if len(outputs) != len(self.decoder_output_names):
            raise ValueError("EfficientSAM decoder output count does not match graph")
        by_name = {
            name: np.asarray(output)
            for name, output in zip(self.decoder_output_names, outputs, strict=True)
        }
        masks = by_name["output_masks"]
        scores = by_name["iou_predictions"]
        if masks.ndim != 5 or scores.ndim != 3:
            raise ValueError(
                "EfficientSAM decoder outputs must have shapes BxQxCxHxW and BxQxC"
            )
        if masks.shape[:3] != scores.shape or masks.shape[0] != 1:
            raise ValueError("EfficientSAM scores must correspond to every mask")
        if masks.shape[-2:] != original_size:
            raise ValueError("EfficientSAM decoder mask size does not match the image")
        if (
            masks.size == 0
            or not np.isfinite(masks).all()
            or not np.isfinite(scores).all()
        ):
            raise ValueError("EfficientSAM decoder output is empty or non-finite")
        best_indices = np.argmax(scores[0], axis=-1)
        best_masks = np.stack(
            [masks[0, query, best] for query, best in enumerate(best_indices)]
        )
        return best_masks[:, None].astype(np.float32, copy=False)


__all__ = ["EfficientSAMONNX"]
