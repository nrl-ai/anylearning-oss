"""ONNX Runtime adapter for SAM and MobileSAM encoder/decoder pairs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .sam_prompts import prompt_arrays

_DECODER_INPUTS = frozenset(
    {
        "image_embeddings",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
        "orig_im_size",
    }
)
_RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def runtime_session(value: Any) -> Any:
    """Keep legacy path construction while allowing prevalidated sessions."""
    if hasattr(value, "get_inputs") and hasattr(value, "run"):
        return value
    if not isinstance(value, (str, Path)):
        raise TypeError("ONNX session must be a runtime session or model path")
    import onnxruntime

    return onnxruntime.InferenceSession(str(value), providers=["CPUExecutionProvider"])


def _static_dimension(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_finite(name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"SAM {name} must contain only finite numeric values")


class SegmentAnythingONNX:
    """Run an official SAM or MobileSAM ONNX encoder/decoder pair.

    AnyLearning's image boundary is uint8 RGB. Encoders exported with embedded
    preprocessing accept resized HWC input, while raw encoders accept official
    normalized and padded NCHW input.
    """

    target_size = 1024

    def __init__(self, encoder: Any, decoder: Any) -> None:
        self.encoder_session = runtime_session(encoder)
        self.decoder_session = runtime_session(decoder)

        encoder_inputs = self.encoder_session.get_inputs()
        if len(encoder_inputs) != 1:
            raise ValueError("SAM encoder must expose exactly one input")
        encoder_input = encoder_inputs[0]
        self.encoder_input_name = encoder_input.name
        self.encoder_input_shape = tuple(encoder_input.shape)
        self.encoder_input_rank = len(self.encoder_input_shape)
        if getattr(encoder_input, "type", "tensor(float)") != "tensor(float)":
            raise ValueError("SAM encoder input must be float32")
        self._validate_encoder_shape()

        decoder_inputs = {item.name for item in self.decoder_session.get_inputs()}
        if decoder_inputs != _DECODER_INPUTS:
            missing = sorted(_DECODER_INPUTS - decoder_inputs)
            extra = sorted(decoder_inputs - _DECODER_INPUTS)
            raise ValueError(
                f"Unexpected SAM decoder inputs (missing={missing}, extra={extra})"
            )
        self.decoder_output_names = tuple(
            item.name for item in self.decoder_session.get_outputs()
        )
        if not ({"masks", "low_res_masks"} & set(self.decoder_output_names)):
            raise ValueError("SAM decoder must output masks or low_res_masks")

    def _validate_encoder_shape(self) -> None:
        if self.encoder_input_rank == 3:
            channels = _static_dimension(self.encoder_input_shape[2])
            if channels not in (None, 3):
                raise ValueError("SAM HWC encoder input must have three RGB channels")
            return
        if self.encoder_input_rank == 4:
            batch = _static_dimension(self.encoder_input_shape[0])
            channels = _static_dimension(self.encoder_input_shape[1])
            height = _static_dimension(self.encoder_input_shape[2])
            width = _static_dimension(self.encoder_input_shape[3])
            if batch not in (None, 1) or channels not in (None, 3):
                raise ValueError("SAM NCHW encoder input must have shape 1x3xHxW")
            if height not in (None, self.target_size) or width not in (
                None,
                self.target_size,
            ):
                raise ValueError("SAM raw encoder spatial input must be 1024x1024")
            return
        raise ValueError(
            f"Unsupported SAM encoder input rank: {self.encoder_input_rank}"
        )

    @staticmethod
    def get_input_points(
        prompt: Iterable[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        return prompt_arrays(prompt)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> tuple[int, int]:
        if oldh <= 0 or oldw <= 0:
            raise ValueError("Image dimensions must be positive")
        scale = long_side_length / max(oldh, oldw)
        return int(oldh * scale + 0.5), int(oldw * scale + 0.5)

    @classmethod
    def apply_coords(
        cls,
        coords: np.ndarray,
        original_size: tuple[int, int],
        target_length: int,
    ) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = cls.get_preprocess_shape(old_h, old_w, target_length)
        scaled = np.asarray(coords, dtype=np.float64).copy()
        scaled[..., 0] *= new_w / old_w
        scaled[..., 1] *= new_h / old_h
        return scaled

    def encode(self, image: np.ndarray) -> dict[str, Any]:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("SAM expects an H x W x 3 uint8 RGB image")
        original_size = image.shape[:2]
        resized_size = self.get_preprocess_shape(*original_size, self.target_size)
        resized = np.asarray(
            Image.fromarray(image).resize(
                (resized_size[1], resized_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
        )

        if self.encoder_input_rank == 3:
            input_image = resized.astype(np.float32)
        else:
            normalized = (resized.astype(np.float32) - _RGB_MEAN) / _RGB_STD
            chw = normalized.transpose(2, 0, 1)
            input_image = np.pad(
                chw,
                (
                    (0, 0),
                    (0, self.target_size - resized_size[0]),
                    (0, self.target_size - resized_size[1]),
                ),
            )[None].astype(np.float32, copy=False)

        outputs = self.encoder_session.run(None, {self.encoder_input_name: input_image})
        if len(outputs) != 1:
            raise ValueError("SAM encoder must produce exactly one embedding")
        image_embedding = np.asarray(outputs[0])
        if image_embedding.ndim != 4 or image_embedding.shape[0] != 1:
            raise ValueError("SAM encoder embedding must be a rank-4 batch of one")
        _validate_finite("encoder embedding", image_embedding)
        return {
            "image_embedding": image_embedding,
            "original_size": original_size,
            "resized_size": resized_size,
        }

    def predict_masks(
        self,
        embedding: Mapping[str, Any],
        prompt: Iterable[Mapping[str, Any]],
    ) -> np.ndarray:
        original_size = tuple(embedding["original_size"])
        resized_size = tuple(embedding["resized_size"])
        if len(original_size) != 2 or len(resized_size) != 2:
            raise ValueError("SAM embedding sizes must contain height and width")
        points, labels = prompt_arrays(prompt)
        coords = np.concatenate((points, np.zeros((1, 2), dtype=np.float32)))[None]
        labels = np.concatenate((labels, np.asarray([-1], dtype=np.float32)))[None]
        coords = self.apply_coords(coords, original_size, self.target_size).astype(
            np.float32
        )

        decoder_inputs = {
            "image_embeddings": np.asarray(embedding["image_embedding"]),
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.asarray(original_size, dtype=np.float32),
        }
        outputs = self.decoder_session.run(None, decoder_inputs)
        if len(outputs) != len(self.decoder_output_names):
            raise ValueError("SAM decoder output count does not match its graph")
        by_name = {
            name: np.asarray(output)
            for name, output in zip(self.decoder_output_names, outputs, strict=True)
        }
        masks = by_name.get("low_res_masks")
        if masks is not None:
            masks = self.postprocess_masks(masks, original_size, resized_size)
        else:
            masks = by_name["masks"]
            if masks.shape[-2:] != original_size:
                masks = self.resize_masks(masks, original_size)
        if masks.ndim != 4 or masks.shape[0] != 1 or masks.shape[1] < 1:
            raise ValueError("SAM decoder masks must have shape 1xNxHxW")
        _validate_finite("decoder masks", masks)

        scores = by_name.get("iou_predictions")
        if scores is not None:
            scores = np.asarray(scores)
            _validate_finite("decoder IoU scores", scores)
            flattened = scores.reshape(scores.shape[0], -1)
            if flattened.shape != masks.shape[:2]:
                raise ValueError("SAM IoU scores must correspond to every mask")
            best = int(np.argmax(flattened[0]))
            masks = masks[:, best : best + 1]
        return masks.astype(np.float32, copy=False)

    @classmethod
    def postprocess_masks(
        cls,
        masks: np.ndarray,
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
    ) -> np.ndarray:
        """Apply official resize-square, crop-aspect, resize-original flow."""
        masks = np.asarray(masks)
        if masks.ndim != 4:
            raise ValueError("SAM low-resolution masks must be rank 4")
        processed: list[list[np.ndarray]] = []
        for batch in masks:
            batch_masks: list[np.ndarray] = []
            for mask in batch:
                upscaled = cv2.resize(
                    mask,
                    (cls.target_size, cls.target_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                cropped = upscaled[: resized_size[0], : resized_size[1]]
                batch_masks.append(
                    cv2.resize(
                        cropped,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
            processed.append(batch_masks)
        return np.asarray(processed, dtype=np.float32)

    @staticmethod
    def resize_masks(masks: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        masks = np.asarray(masks)
        if masks.ndim != 4:
            raise ValueError("SAM masks must be rank 4")
        return np.asarray(
            [
                [
                    cv2.resize(
                        mask,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for mask in batch
                ]
                for batch in masks
            ],
            dtype=np.float32,
        )


__all__ = ["SegmentAnythingONNX", "runtime_session"]
