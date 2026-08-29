"""ONNX adapter for Segment Anything image encoders and mask decoders."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import onnxruntime

from .prompts import prompt_arrays


class SegmentAnythingONNX:
    """Run a MobileSAM-compatible encoder/decoder pair."""

    target_size = 1024
    input_size = (684, 1024)

    def __init__(self, encoder_model_path: str, decoder_model_path: str) -> None:
        self.encoder_session = onnxruntime.InferenceSession(encoder_model_path)
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = onnxruntime.InferenceSession(decoder_model_path)

    @staticmethod
    def get_input_points(prompt: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
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
        scaled = coords.astype(np.float64, copy=True)
        scaled[..., 0] *= new_w / old_w
        scaled[..., 1] *= new_h / old_h
        return scaled

    def encode(self, image: np.ndarray) -> dict[str, Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SAM expects an H x W x 3 RGB image")
        original_size = image.shape[:2]
        scale = min(
            self.input_size[1] / original_size[1],
            self.input_size[0] / original_size[0],
        )
        transform = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        resized = cv2.warpAffine(
            image,
            transform[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )
        embedding = self.encoder_session.run(
            None, {self.encoder_input_name: resized.astype(np.float32)}
        )[0]
        return {
            "image_embedding": embedding,
            "original_size": original_size,
            "transform_matrix": transform,
        }

    def predict_masks(
        self, embedding: dict[str, Any], prompt: list[dict[str, Any]]
    ) -> np.ndarray:
        points, labels = prompt_arrays(prompt)
        coords = np.concatenate((points, np.zeros((1, 2), dtype=np.float32)))[None]
        labels = np.concatenate((labels, np.array([-1], dtype=np.float32)))[None]
        coords = self.apply_coords(coords, self.input_size, self.target_size).astype(
            np.float32
        )
        homogeneous = np.concatenate(
            (coords, np.ones((*coords.shape[:2], 1), dtype=np.float32)), axis=2
        )
        transform = embedding["transform_matrix"]
        coords = (homogeneous @ transform.T)[..., :2].astype(np.float32)

        decoder_inputs = {
            "image_embeddings": embedding["image_embedding"],
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.asarray(self.input_size, dtype=np.float32),
        }
        masks = self.decoder_session.run(None, decoder_inputs)[0]
        return self.transform_masks(
            masks,
            embedding["original_size"],
            np.linalg.inv(transform),
        )

    @staticmethod
    def transform_masks(
        masks: np.ndarray,
        original_size: tuple[int, int],
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        height, width = original_size
        return np.asarray(
            [
                [
                    cv2.warpAffine(
                        mask,
                        transform_matrix[:2],
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                    )
                    for mask in batch
                ]
                for batch in masks
            ]
        )
