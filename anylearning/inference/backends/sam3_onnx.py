"""Bounded ONNX Runtime pipeline for SAM3 image segmentation.

The runtime contract is independently implemented from the exported graph
metadata. It supports both the current raw-head exporter and the earlier
processor-based graph while keeping filtering, NMS, and output resizing bounded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from .clip_tokenizer import ClipTokenizer

_IMAGE_OUTPUTS = (
    "vision_pos_enc_0",
    "vision_pos_enc_1",
    "vision_pos_enc_2",
    "backbone_fpn_0",
    "backbone_fpn_1",
    "backbone_fpn_2",
)
_LANGUAGE_OUTPUTS = (
    "text_attention_mask",
    "text_memory",
    "text_embeds",
)
_DECODER_REQUIRED_INPUTS = frozenset(
    {
        "original_height",
        "original_width",
        "vision_pos_enc_2",
        "backbone_fpn_0",
        "backbone_fpn_1",
        "backbone_fpn_2",
        "language_mask",
        "language_features",
        "box_coords",
        "box_labels",
        "box_masks",
    }
)
_DECODER_OPTIONAL_INPUTS = frozenset(
    {"vision_pos_enc_0", "vision_pos_enc_1", "language_embeds"}
)
_RAW_OUTPUTS = frozenset(
    {"pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec"}
)
_PROCESSED_OUTPUTS = frozenset({"boxes", "scores", "masks"})
_BIT_COUNTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _static_dimension(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"SAM3 {name} must be a static integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"SAM3 {name} must be between {minimum} and {maximum}")
    return value


def _metadata_by_name(items: Sequence[Any], *, role: str) -> dict[str, Any]:
    result = {item.name: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"SAM3 {role} contains duplicate tensor names")
    return result


def _require_tensor(
    item: Any,
    *,
    dtype: str,
    rank: int,
    description: str,
) -> tuple[Any, ...]:
    if getattr(item, "type", None) != dtype:
        raise ValueError(f"SAM3 {description} must have type {dtype}")
    shape = tuple(item.shape)
    if len(shape) != rank:
        raise ValueError(f"SAM3 {description} must have rank {rank}")
    return shape


def _finite_float(value: np.ndarray, *, description: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"SAM3 {description} contains NaN or infinity")
    return array


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class Sam3Detections:
    masks: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    normalized_boxes: bool = False


class Sam3ImageEncoder:
    def __init__(self, session: Any, *, max_feature_elements: int) -> None:
        self.session = session
        inputs = list(session.get_inputs())
        if len(inputs) != 1 or inputs[0].name != "image":
            raise ValueError("SAM3 image encoder must expose only the 'image' input")
        input_item = inputs[0]
        if input_item.type not in {"tensor(uint8)", "tensor(float)"}:
            raise ValueError("SAM3 image encoder input must be uint8 or float32")
        shape = tuple(input_item.shape)
        if len(shape) == 3:
            channels, height, width = shape
            self._batched = False
        elif len(shape) == 4:
            batch, channels, height, width = shape
            if batch != 1:
                raise ValueError("SAM3 image encoder batch size must be one")
            self._batched = True
        else:
            raise ValueError("SAM3 image encoder input must be CHW or 1xCHW")
        if channels != 3:
            raise ValueError("SAM3 image encoder input must have three RGB channels")
        self.height = _static_dimension(
            height, name="image encoder height", minimum=16, maximum=4096
        )
        self.width = _static_dimension(
            width, name="image encoder width", minimum=16, maximum=4096
        )
        self._input_type = input_item.type
        outputs = _metadata_by_name(session.get_outputs(), role="image encoder outputs")
        if set(outputs) != set(_IMAGE_OUTPUTS):
            raise ValueError("Unexpected SAM3 image encoder output contract")
        self.output_shapes: dict[str, tuple[Any, ...]] = {}
        static_elements = 0
        for name, output in outputs.items():
            shape = _require_tensor(
                output,
                dtype="tensor(float)",
                rank=4,
                description=f"image encoder output {name!r}",
            )
            if isinstance(shape[0], int) and shape[0] != 1:
                raise ValueError("SAM3 image encoder features must have batch one")
            output_elements = 1
            static_shape = True
            for dimension in shape:
                if isinstance(dimension, int):
                    _static_dimension(
                        dimension,
                        name=f"image encoder output {name!r} dimension",
                        minimum=1,
                        maximum=1_000_000_000,
                    )
                    if output_elements > max_feature_elements // dimension:
                        raise ValueError(
                            "SAM3 image features exceed the configured element limit"
                        )
                    output_elements *= dimension
                else:
                    static_shape = False
            if static_shape:
                static_elements += output_elements
            self.output_shapes[name] = shape
        if static_elements > max_feature_elements:
            raise ValueError("SAM3 image features exceed the configured element limit")
        self._output_names = tuple(name for name in _IMAGE_OUTPUTS)
        self._max_feature_elements = max_feature_elements

    def prepare(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("SAM3 expects an H x W x 3 uint8 RGB image")
        resized = cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)
        if self._input_type == "tensor(float)":
            tensor = resized.astype(np.float32) / 127.5 - 1.0
        else:
            tensor = resized.astype(np.uint8, copy=False)
        return tensor[None] if self._batched else tensor

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.session.run(
            list(self._output_names), {"image": self.prepare(image)}
        )
        if len(outputs) != len(self._output_names):
            raise ValueError("SAM3 image encoder output count changed at runtime")
        result: dict[str, np.ndarray] = {}
        total_elements = 0
        for name, output in zip(self._output_names, outputs, strict=True):
            array = _finite_float(output, description=f"image feature {name!r}")
            if array.ndim != 4 or array.shape[0] != 1:
                raise ValueError("SAM3 image features must be rank-4 batches of one")
            total_elements += array.size
            if total_elements > self._max_feature_elements:
                raise ValueError(
                    "SAM3 image features exceed the configured element limit"
                )
            result[name] = array
        return result


class Sam3LanguageEncoder:
    def __init__(
        self,
        session: Any,
        *,
        tokenizer: ClipTokenizer,
    ) -> None:
        self.session = session
        inputs = list(session.get_inputs())
        if len(inputs) != 1 or inputs[0].name != "tokens":
            raise ValueError("SAM3 language encoder must expose only 'tokens'")
        shape = _require_tensor(
            inputs[0],
            dtype="tensor(int64)",
            rank=2,
            description="language tokens",
        )
        if shape[0] != 1:
            raise ValueError("SAM3 language encoder batch size must be one")
        self.context_length = _static_dimension(
            shape[1], name="language token capacity", minimum=2, maximum=256
        )
        outputs = _metadata_by_name(
            session.get_outputs(), role="language encoder outputs"
        )
        if set(outputs) != set(_LANGUAGE_OUTPUTS):
            raise ValueError("Unexpected SAM3 language encoder output contract")
        _require_tensor(
            outputs["text_attention_mask"],
            dtype="tensor(bool)",
            rank=2,
            description="language attention mask",
        )
        memory_shape = _require_tensor(
            outputs["text_memory"],
            dtype="tensor(float)",
            rank=3,
            description="language memory",
        )
        embedding_shape = _require_tensor(
            outputs["text_embeds"],
            dtype="tensor(float)",
            rank=3,
            description="language embeddings",
        )
        if memory_shape[:2] != (self.context_length, 1):
            raise ValueError("SAM3 language memory metadata has an unexpected shape")
        if embedding_shape[:2] != (self.context_length, 1):
            raise ValueError("SAM3 language embedding metadata has an unexpected shape")
        for description, dimension in (
            ("language memory width", memory_shape[2]),
            ("language embedding width", embedding_shape[2]),
        ):
            _static_dimension(dimension, name=description, minimum=1, maximum=8192)
        self.output_shapes = {
            "language_mask": (1, self.context_length),
            "language_features": memory_shape,
            "language_embeds": embedding_shape,
        }
        self._tokenizer = tokenizer

    def __call__(self, text: str) -> dict[str, np.ndarray]:
        tokens = self._tokenizer.tokenize(text, context_length=self.context_length)
        outputs = self.session.run(list(_LANGUAGE_OUTPUTS), {"tokens": tokens})
        if len(outputs) != len(_LANGUAGE_OUTPUTS):
            raise ValueError("SAM3 language encoder output count changed at runtime")
        result = dict(zip(_LANGUAGE_OUTPUTS, outputs, strict=True))
        attention = np.asarray(result["text_attention_mask"])
        memory = _finite_float(result["text_memory"], description="language memory")
        embeddings = _finite_float(
            result["text_embeds"], description="language embeddings"
        )
        if attention.dtype != np.bool_ or attention.shape != (1, self.context_length):
            raise ValueError("SAM3 language attention mask has an unexpected shape")
        if memory.ndim != 3 or memory.shape[:2] != (self.context_length, 1):
            raise ValueError("SAM3 language memory has an unexpected shape")
        if embeddings.ndim != 3 or embeddings.shape[:2] != (
            self.context_length,
            1,
        ):
            raise ValueError("SAM3 language embeddings have an unexpected shape")
        return {
            "language_mask": attention,
            "language_features": memory,
            "language_embeds": embeddings,
        }


class Sam3Decoder:
    def __init__(
        self,
        session: Any,
        *,
        context_length: int,
        max_raw_queries: int,
        max_output_elements: int,
    ) -> None:
        self.session = session
        inputs = _metadata_by_name(session.get_inputs(), role="decoder inputs")
        received = set(inputs)
        if not _DECODER_REQUIRED_INPUTS.issubset(received) or not received.issubset(
            _DECODER_REQUIRED_INPUTS | _DECODER_OPTIONAL_INPUTS
        ):
            raise ValueError("Unexpected SAM3 decoder input contract")
        for name in ("original_height", "original_width"):
            _require_tensor(
                inputs[name],
                dtype="tensor(int64)",
                rank=0,
                description=name,
            )
        for name in received & {
            "vision_pos_enc_0",
            "vision_pos_enc_1",
            "vision_pos_enc_2",
            "backbone_fpn_0",
            "backbone_fpn_1",
            "backbone_fpn_2",
        }:
            _require_tensor(
                inputs[name],
                dtype="tensor(float)",
                rank=4,
                description=f"decoder feature {name!r}",
            )
        self.input_shapes = {name: tuple(item.shape) for name, item in inputs.items()}
        language_mask_shape = _require_tensor(
            inputs["language_mask"],
            dtype="tensor(bool)",
            rank=2,
            description="decoder language mask",
        )
        language_features_shape = _require_tensor(
            inputs["language_features"],
            dtype="tensor(float)",
            rank=3,
            description="decoder language features",
        )
        if language_mask_shape != (1, context_length):
            raise ValueError(
                "SAM3 decoder and language encoder token capacities differ"
            )
        if language_features_shape[:2] != (context_length, 1):
            raise ValueError("SAM3 decoder language feature shape is incompatible")
        if "language_embeds" in inputs:
            language_embeds_shape = _require_tensor(
                inputs["language_embeds"],
                dtype="tensor(float)",
                rank=3,
                description="decoder language embeddings",
            )
            if language_embeds_shape[:2] != (context_length, 1):
                raise ValueError(
                    "SAM3 decoder language embedding shape is incompatible"
                )

        coordinate_shape = _require_tensor(
            inputs["box_coords"],
            dtype="tensor(float)",
            rank=3,
            description="geometric prompt coordinates",
        )
        label_shape = _require_tensor(
            inputs["box_labels"],
            dtype="tensor(int64)",
            rank=2,
            description="geometric prompt labels",
        )
        mask_shape = _require_tensor(
            inputs["box_masks"],
            dtype="tensor(bool)",
            rank=2,
            description="geometric prompt mask",
        )
        self.geometric_prompt_capacity = _static_dimension(
            coordinate_shape[0],
            name="geometric prompt capacity",
            minimum=1,
            maximum=1024,
        )
        expected_coordinate_shape = (self.geometric_prompt_capacity, 1, 4)
        if coordinate_shape != expected_coordinate_shape:
            raise ValueError("SAM3 geometric coordinate tensor has an invalid shape")
        if label_shape != (self.geometric_prompt_capacity, 1):
            raise ValueError("SAM3 geometric label tensor has an invalid shape")
        if mask_shape != (1, self.geometric_prompt_capacity):
            raise ValueError("SAM3 geometric mask tensor has an invalid shape")

        outputs = _metadata_by_name(session.get_outputs(), role="decoder outputs")
        output_names = frozenset(outputs)
        if output_names == _RAW_OUTPUTS:
            self.output_profile: Literal["raw", "processed"] = "raw"
            _require_tensor(
                outputs["pred_boxes"],
                dtype="tensor(float)",
                rank=3,
                description="raw boxes",
            )
            _require_tensor(
                outputs["pred_logits"],
                dtype="tensor(float)",
                rank=3,
                description="raw logits",
            )
            _require_tensor(
                outputs["pred_masks"],
                dtype="tensor(float)",
                rank=4,
                description="raw masks",
            )
            presence_shape = tuple(outputs["presence_logit_dec"].shape)
            if (
                outputs["presence_logit_dec"].type != "tensor(float)"
                or len(presence_shape) > 2
            ):
                raise ValueError("SAM3 presence logits have an invalid contract")
        elif output_names == _PROCESSED_OUTPUTS:
            self.output_profile = "processed"
            _require_tensor(
                outputs["boxes"],
                dtype="tensor(float)",
                rank=2,
                description="processed boxes",
            )
            _require_tensor(
                outputs["scores"],
                dtype="tensor(float)",
                rank=1,
                description="processed scores",
            )
            if outputs["masks"].type not in {"tensor(bool)", "tensor(float)"}:
                raise ValueError("SAM3 processed masks must be bool or float32")
            if len(tuple(outputs["masks"].shape)) != 4:
                raise ValueError("SAM3 processed masks must have rank four")
        else:
            raise ValueError("Unexpected SAM3 decoder output contract")
        self._input_names = tuple(item.name for item in session.get_inputs())
        self._output_names = tuple(item.name for item in session.get_outputs())
        self._max_raw_queries = max_raw_queries
        self._max_output_elements = max_output_elements

    def run(
        self,
        *,
        original_size: tuple[int, int],
        image_features: Mapping[str, np.ndarray],
        language_features: Mapping[str, np.ndarray],
        box_coords: np.ndarray,
        box_labels: np.ndarray,
        box_masks: np.ndarray,
    ) -> Sam3Detections:
        height, width = original_size
        available: dict[str, np.ndarray] = {
            "original_height": np.asarray(height, dtype=np.int64),
            "original_width": np.asarray(width, dtype=np.int64),
            **image_features,
            **language_features,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        }
        try:
            inputs = {name: available[name] for name in self._input_names}
        except KeyError as error:
            raise ValueError(
                f"SAM3 decoder input {error.args[0]!r} is unavailable"
            ) from error
        outputs = self.session.run(list(self._output_names), inputs)
        if len(outputs) != len(self._output_names):
            raise ValueError("SAM3 decoder output count changed at runtime")
        by_name = dict(zip(self._output_names, outputs, strict=True))
        if self.output_profile == "raw":
            return self._raw_detections(by_name)
        return self._processed_detections(by_name, original_size)

    def _raw_detections(self, output: Mapping[str, np.ndarray]) -> Sam3Detections:
        boxes = _finite_float(output["pred_boxes"], description="raw boxes")
        logits = _finite_float(output["pred_logits"], description="raw logits")
        mask_logits = _finite_float(output["pred_masks"], description="raw masks")
        presence = _finite_float(
            output["presence_logit_dec"], description="presence logits"
        )
        if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
            raise ValueError("SAM3 raw boxes must have shape 1xNx4")
        if mask_logits.ndim != 4 or mask_logits.shape[:2] != boxes.shape[:2]:
            raise ValueError("SAM3 raw masks must have shape 1xNxHxW")
        query_count = boxes.shape[1]
        if query_count > self._max_raw_queries:
            raise ValueError("SAM3 raw query count exceeds the configured limit")
        if logits.size != query_count:
            raise ValueError("SAM3 raw logits must provide one value per query")
        if presence.size != 1:
            raise ValueError("SAM3 presence output must contain exactly one logit")
        if mask_logits.size > self._max_output_elements:
            raise ValueError("SAM3 raw masks exceed the configured element limit")
        scores = _sigmoid(logits.reshape(-1)) * float(_sigmoid(presence.reshape(-1))[0])
        return Sam3Detections(
            masks=mask_logits[0],
            scores=scores.astype(np.float32),
            boxes=boxes[0],
            normalized_boxes=True,
        )

    def _processed_detections(
        self,
        output: Mapping[str, np.ndarray],
        original_size: tuple[int, int],
    ) -> Sam3Detections:
        boxes = _finite_float(output["boxes"], description="processed boxes")
        scores = _finite_float(output["scores"], description="processed scores")
        masks = np.asarray(output["masks"])
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("SAM3 processed boxes must have shape Nx4")
        scores = scores.reshape(-1)
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError("SAM3 processed masks must have shape Nx1xHxW")
        if not (len(boxes) == len(scores) == len(masks)):
            raise ValueError("SAM3 processed outputs must have equal query counts")
        if len(masks) > self._max_raw_queries:
            raise ValueError("SAM3 query count exceeds the configured limit")
        if masks.size > self._max_output_elements:
            raise ValueError("SAM3 processed masks exceed the configured element limit")
        if masks.dtype != np.bool_:
            masks = _finite_float(masks, description="processed masks") > 0
        if masks.shape[-2:] != original_size:
            raise ValueError("SAM3 processed masks must match the requested image size")
        return Sam3Detections(
            masks=masks[:, 0],
            scores=scores,
            boxes=boxes,
            normalized_boxes=False,
        )


class Sam3OnnxPipeline:
    def __init__(
        self,
        image_session: Any,
        language_session: Any,
        decoder_session: Any,
        *,
        max_text_bytes: int,
        max_raw_queries: int,
        max_output_elements: int,
        max_nms_candidates: int,
        max_feature_elements: int,
    ) -> None:
        tokenizer = ClipTokenizer(max_text_bytes=max_text_bytes)
        self.image_encoder = Sam3ImageEncoder(
            image_session, max_feature_elements=max_feature_elements
        )
        self.language_encoder = Sam3LanguageEncoder(
            language_session, tokenizer=tokenizer
        )
        self.decoder = Sam3Decoder(
            decoder_session,
            context_length=self.language_encoder.context_length,
            max_raw_queries=max_raw_queries,
            max_output_elements=max_output_elements,
        )
        for name in set(_IMAGE_OUTPUTS) & set(self.decoder.input_shapes):
            if (
                self.image_encoder.output_shapes[name]
                != self.decoder.input_shapes[name]
            ):
                raise ValueError(
                    f"SAM3 image encoder output {name!r} does not match the decoder"
                )
        for name, shape in self.language_encoder.output_shapes.items():
            if (
                name in self.decoder.input_shapes
                and shape != self.decoder.input_shapes[name]
            ):
                raise ValueError(
                    f"SAM3 language encoder output {name!r} does not match the decoder"
                )
        self.max_nms_candidates = max_nms_candidates

    def encode_image(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return self.image_encoder(image)

    def encode_text(self, text: str) -> dict[str, np.ndarray]:
        return self.language_encoder(text)

    def geometric_inputs(
        self,
        prompts: Sequence[Mapping[str, Any]],
        *,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image_size
        capacity = self.decoder.geometric_prompt_capacity
        if len(prompts) > capacity:
            raise ValueError(
                f"SAM3 decoder accepts at most {capacity} geometric prompts"
            )
        coordinates = np.zeros((capacity, 1, 4), dtype=np.float32)
        labels = np.ones((capacity, 1), dtype=np.int64)
        padding = np.ones((1, capacity), dtype=np.bool_)
        for index, prompt in enumerate(prompts):
            kind = prompt.get("type")
            data = prompt.get("data")
            if kind == "rectangle":
                if not isinstance(data, (list, tuple)) or len(data) != 4:
                    raise ValueError("SAM3 rectangle prompt must contain four values")
                x1, y1, x2, y2 = (float(value) for value in data)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("SAM3 rectangle prompt must be inside the image")
                coordinates[index, 0] = (
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                )
                labels[index, 0] = 1
            elif kind == "point":
                if not isinstance(data, (list, tuple)) or len(data) != 2:
                    raise ValueError("SAM3 point prompt must contain two values")
                x, y = (float(value) for value in data)
                label = prompt.get("label")
                if label not in (0, 1):
                    raise ValueError("SAM3 point prompt label must be 0 or 1")
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("SAM3 point prompt must be inside the image")
                coordinates[index, 0] = (x / width, y / height, 0.01, 0.01)
                labels[index, 0] = int(label)
            else:
                raise ValueError(f"Unsupported SAM3 geometric prompt: {kind!r}")
            padding[0, index] = False
        return coordinates, labels, padding

    def predict(
        self,
        *,
        image_features: Mapping[str, np.ndarray],
        language_features: Mapping[str, np.ndarray],
        geometric_prompts: Sequence[Mapping[str, Any]],
        original_size: tuple[int, int],
        confidence_threshold: float,
        nms_threshold: float,
        max_instances: int,
    ) -> Sam3Detections:
        coords, labels, masks = self.geometric_inputs(
            geometric_prompts, image_size=original_size
        )
        raw = self.decoder.run(
            original_size=original_size,
            image_features=image_features,
            language_features=language_features,
            box_coords=coords,
            box_labels=labels,
            box_masks=masks,
        )
        selected = select_sam3_detections(
            raw,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            max_instances=max_instances,
            max_nms_candidates=self.max_nms_candidates,
        )
        return resize_sam3_detections(selected, original_size=original_size)


def _binary_masks(masks: np.ndarray) -> np.ndarray:
    array = np.asarray(masks)
    if array.ndim != 3:
        raise ValueError("SAM3 selection masks must have shape NxHxW")
    if array.dtype == np.bool_:
        return array
    return _finite_float(array, description="selection masks") > 0


def _mask_nms(masks: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    binary = _binary_masks(masks)
    if len(binary) <= 1:
        return np.arange(len(binary), dtype=np.int64)
    step = max(1, int(np.ceil(max(binary.shape[-2:]) / 288)))
    sampled = binary[:, ::step, ::step].reshape(len(binary), -1)
    packed = np.packbits(sampled, axis=1)
    areas = _BIT_COUNTS[packed].sum(axis=1, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        intersections = _BIT_COUNTS[
            np.bitwise_and(packed[remaining], packed[current])
        ].sum(axis=1, dtype=np.int64)
        unions = areas[current] + areas[remaining] - intersections
        overlaps = np.divide(
            intersections,
            unions,
            out=np.zeros(len(remaining), dtype=np.float64),
            where=unions > 0,
        )
        order = remaining[overlaps <= threshold]
    return np.asarray(kept, dtype=np.int64)


def select_sam3_detections(
    detections: Sam3Detections,
    *,
    confidence_threshold: float,
    nms_threshold: float,
    max_instances: int,
    max_nms_candidates: int,
) -> Sam3Detections:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("SAM3 confidence threshold must be between 0 and 1")
    if not 0 <= nms_threshold <= 1:
        raise ValueError("SAM3 NMS threshold must be between 0 and 1")
    if max_instances < 1 or max_nms_candidates < max_instances:
        raise ValueError("SAM3 instance and NMS candidate limits are inconsistent")
    masks = np.asarray(detections.masks)
    scores = _finite_float(detections.scores, description="scores").reshape(-1)
    boxes = _finite_float(detections.boxes, description="boxes")
    if masks.ndim != 3 or boxes.shape != (len(scores), 4):
        raise ValueError("SAM3 detection arrays have incompatible shapes")
    if len(masks) != len(scores):
        raise ValueError("SAM3 masks and scores have incompatible lengths")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("SAM3 scores must be between zero and one")
    candidates = np.flatnonzero(scores >= confidence_threshold)
    if not len(candidates):
        return Sam3Detections(
            masks=np.zeros((0,) + masks.shape[1:], dtype=masks.dtype),
            scores=np.zeros((0,), dtype=np.float32),
            boxes=np.zeros((0, 4), dtype=np.float32),
            normalized_boxes=detections.normalized_boxes,
        )
    candidates = candidates[
        np.argsort(-scores[candidates], kind="stable")[:max_nms_candidates]
    ]
    masks = masks[candidates]
    scores = scores[candidates]
    boxes = boxes[candidates]
    keep = _mask_nms(masks, scores, nms_threshold)[:max_instances]
    return Sam3Detections(
        masks=masks[keep],
        scores=scores[keep],
        boxes=boxes[keep],
        normalized_boxes=detections.normalized_boxes,
    )


def resize_sam3_detections(
    detections: Sam3Detections,
    *,
    original_size: tuple[int, int],
) -> Sam3Detections:
    height, width = original_size
    source_masks = np.asarray(detections.masks)
    if source_masks.ndim != 3:
        raise ValueError("SAM3 resize masks must have shape NxHxW")
    if source_masks.shape[-2:] == original_size:
        masks = _binary_masks(source_masks)[:, None]
    else:
        resized = [
            cv2.resize(
                np.asarray(mask, dtype=np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            > (0.5 if source_masks.dtype == np.bool_ else 0.0)
            for mask in source_masks
        ]
        masks = (
            np.stack(resized)[:, None]
            if resized
            else np.zeros((0, 1, height, width), dtype=np.bool_)
        )
    if detections.normalized_boxes:
        cx, cy, box_width, box_height = detections.boxes.T
        boxes = np.stack(
            (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ),
            axis=1,
        ).astype(np.float32)
    else:
        boxes = detections.boxes.astype(np.float32, copy=True)
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
    return Sam3Detections(
        masks=masks,
        scores=detections.scores,
        boxes=boxes,
        normalized_boxes=False,
    )


__all__ = [
    "Sam3Decoder",
    "Sam3Detections",
    "Sam3ImageEncoder",
    "Sam3LanguageEncoder",
    "Sam3OnnxPipeline",
    "resize_sam3_detections",
    "select_sam3_detections",
]
