"""ONNX Runtime adapter for Segment Anything 2.

Originally developed in ``vietanhdev/samexporter`` and relicensed here by its
author under the repository's Apache-2.0 license.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from numpy import ndarray

from .sam_onnx import runtime_session
from .sam_prompts import prompt_arrays

logger = logging.getLogger(__name__)


class SegmentAnything2ONNX:
    """Segmentation model using Segment Anything 2 (SAM2)"""

    def __init__(self, encoder: Any, decoder: Any) -> None:
        self.encoder = SAM2ImageEncoder(encoder)
        self.decoder = SAM2ImageDecoder(decoder, self.encoder.input_shape[2:])

    def encode(self, cv_image: np.ndarray) -> dict[str, Any]:
        if cv_image.ndim != 3 or cv_image.shape[2] != 3 or cv_image.dtype != np.uint8:
            raise ValueError("SAM2 expects an H x W x 3 uint8 RGB image")
        original_size = cv_image.shape[:2]
        high_res_feats_0, high_res_feats_1, image_embed = self.encoder(cv_image)
        return {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embedding": image_embed,
            "original_size": original_size,
        }

    def predict_masks(
        self, embedding: dict[str, Any], prompt: list[dict[str, Any]]
    ) -> np.ndarray:
        points, labels = prompt_arrays(prompt)

        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]
        self.decoder.set_image_size(original_size)
        masks, _ = self.decoder(
            image_embedding,
            high_res_feats_0,
            high_res_feats_1,
            points,
            labels,
        )

        return masks


class SAM2ImageEncoder:
    def __init__(self, session: Any) -> None:
        self.session = runtime_session(session)
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode_image(image)

    def encode_image(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        outputs = self.infer(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        # SegmentAnything loads files with PIL, whose NumPy representation is
        # already RGB.  This used to apply BGR -> RGB conversion a second time,
        # swapping the red and blue channels before every SAM 2 embedding.  It
        # was easy to miss on neutral images, but produced visibly fragmented
        # masks on real photographs.  Keep the RGB contract explicit here;
        # MobileSAM has its own preprocessing path and is unaffected.
        input_img = cv2.resize(image, (self.input_width, self.input_height))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        start = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        logger.debug(
            "SAM2 encoder inference took %.2f ms",
            (time.perf_counter() - start) * 1000,
        )
        return outputs

    def process_output(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = tuple(np.asarray(output) for output in outputs)
        if any(array.ndim != 4 or array.shape[0] != 1 for array in arrays):
            raise ValueError("SAM2 encoder features must be rank-4 batches of one")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("SAM2 encoder features contain NaN or infinity")
        return arrays

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        if len(model_inputs) != 1:
            raise ValueError("SAM2 encoder must expose exactly one input")
        model_input = model_inputs[0]
        if len(model_input.shape) != 4:
            raise ValueError("SAM2 encoder input must be rank-4 NCHW")
        batch, channels, height, width = model_input.shape
        if isinstance(batch, int) and batch != 1:
            raise ValueError("SAM2 encoder batch size must be one")
        if isinstance(channels, int) and channels != 3:
            raise ValueError("SAM2 encoder input must have three RGB channels")
        if not all(
            isinstance(value, int) and 16 <= value <= 16_384
            for value in (height, width)
        ):
            raise ValueError(
                "SAM2 encoder spatial dimensions must be static and bounded"
            )
        if getattr(model_input, "type", "tensor(float)") != "tensor(float)":
            raise ValueError("SAM2 encoder input must be float32")
        self.input_names = [model_input.name]
        self.input_shape = model_inputs[0].shape
        self.input_height = height
        self.input_width = width

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        if len(model_outputs) != 3:
            raise ValueError("SAM2 encoder must expose exactly three feature outputs")
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class SAM2ImageDecoder:
    def __init__(
        self,
        session: Any,
        encoder_input_size: tuple[int, int],
        orig_im_size: tuple[int, int] | None = None,
        mask_threshold: float = 0.0,
    ) -> None:
        self.session = runtime_session(session)

        self.orig_im_size = (
            orig_im_size if orig_im_size is not None else encoder_input_size
        )
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
    ) -> tuple[list[np.ndarray], ndarray]:
        return self.predict(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

    def predict(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
    ) -> tuple[list[np.ndarray], ndarray]:
        inputs = self.prepare_inputs(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

        outputs = self.infer(inputs)

        return self.process_output(outputs)

    def prepare_inputs(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        input_point_coords, input_point_labels = self.prepare_points(
            point_coords, point_labels
        )

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)

        return {
            "image_embed": image_embed,
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "point_coords": input_point_coords,
            "point_labels": input_point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
        }

    def prepare_points(
        self,
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...].copy()
            input_point_labels = np.asarray(point_labels)[np.newaxis, ...].copy()
        else:
            if not point_coords:
                raise ValueError("At least one point batch is required")
            max_num_points = max(coords.shape[0] for coords in point_coords)
            # We need to make sure that all inputs have the same number of points
            # Add invalid points to pad the input (0, 0) with -1 value for labels
            input_point_coords = np.zeros(
                (len(point_coords), max_num_points, 2), dtype=np.float32
            )
            input_point_labels = (
                np.ones((len(point_coords), max_num_points), dtype=np.float32) * -1
            )

            for i, (coords, labels) in enumerate(zip(point_coords, point_labels)):
                input_point_coords[i, : coords.shape[0], :] = coords
                input_point_labels[i, : labels.shape[0]] = labels

        input_point_coords[..., 0] = (
            input_point_coords[..., 0]
            / self.orig_im_size[1]
            * self.encoder_input_size[1]
        )  # Normalize x
        input_point_coords[..., 1] = (
            input_point_coords[..., 1]
            / self.orig_im_size[0]
            * self.encoder_input_size[0]
        )  # Normalize y

        return input_point_coords.astype(np.float32), input_point_labels.astype(
            np.float32
        )

    def infer(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        start = time.perf_counter()

        outputs = self.session.run(self.output_names, dict(inputs))

        logger.debug(
            "SAM2 decoder inference took %.2f ms",
            (time.perf_counter() - start) * 1000,
        )
        return outputs

    def process_output(
        self, outputs: list[np.ndarray]
    ) -> tuple[list[ndarray | Any], ndarray[Any, Any]]:
        if len(outputs) != len(self.output_names):
            raise ValueError("SAM2 decoder output count does not match its graph")
        by_name = dict(zip(self.output_names, outputs, strict=True))
        masks_output = np.asarray(by_name["masks"])
        scores_output = np.asarray(by_name["iou_predictions"])
        if masks_output.ndim != 4 or masks_output.shape[0] != 1:
            raise ValueError("SAM2 decoder masks must have shape 1xNxHxW")
        scores = scores_output.reshape(-1)
        masks = masks_output[0]
        if masks.shape[0] != scores.size or scores.size == 0:
            raise ValueError("SAM2 decoder scores must correspond to every mask")
        if not np.isfinite(masks).all() or not np.isfinite(scores).all():
            raise ValueError("SAM2 decoder output contains NaN or infinity")

        # Select the best masks based on the scores
        best_mask = masks[np.argmax(scores)]
        best_mask = cv2.resize(best_mask, (self.orig_im_size[1], self.orig_im_size[0]))
        return (
            np.array([[best_mask]]),
            scores,
        )

    def set_image_size(self, orig_im_size: tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        expected = {
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        }
        received = {item.name for item in model_inputs}
        if received != expected:
            raise ValueError("Unexpected SAM2 decoder input contract")
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        if {item.name for item in model_outputs} != {"masks", "iou_predictions"}:
            raise ValueError("SAM2 decoder must expose masks and iou_predictions")
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
