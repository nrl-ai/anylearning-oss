"""Promptable EfficientSAM inference behind the shared runtime boundary."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import Field

from ..cache import LRUCache
from ..contracts import (
    InferenceRequest,
    InferenceResult,
    ModelCapabilities,
    ModelTask,
    ShapeType,
)
from ..runtime import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    SessionLifecycleError,
    SessionState,
)
from .efficient_sam_onnx import EfficientSAMONNX
from .onnx_session import create_checked_onnx_session
from .sam import SamOnnxConfig, _legacy_prompts, mask_shapes

_MAX_IMAGE_PIXELS = 16_000_000
_MAX_OUTPUT_ELEMENTS = 50_000_000


class EfficientSamConfig(SamOnnxConfig):
    """Bounded configuration for official split EfficientSAM ONNX models."""

    max_image_pixels: int = Field(default=_MAX_IMAGE_PIXELS, ge=1, le=100_000_000)
    max_output_elements: int = Field(default=_MAX_OUTPUT_ELEMENTS, ge=1, le=300_000_000)


class EfficientSamSession(BaseInferenceSession):
    """One loaded EfficientSAM pair with a revision-aware embedding cache."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = EfficientSamConfig.model_validate(config)
        self._model: EfficientSAMONNX | None = None
        self._embedding_cache: LRUCache[tuple[str, str], dict[str, Any]] = LRUCache(10)
        self._provider_warnings: tuple[str, ...] = ()
        super().__init__(EfficientSamBackend().capabilities(config))

    def _load(self, cancellation: CancellationToken) -> None:
        encoder_session, _encoder_graph, encoder_warnings = create_checked_onnx_session(
            self.config.resolved_path("encoder_model_path"),
            providers=self.config.providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_model_bytes=self.config.max_model_bytes,
            expected_sha256=self.config.encoder_sha256,
            external_data_sha256=self.config.encoder_external_data_sha256,
            max_external_data_bytes=self.config.max_external_data_bytes,
            intra_op_threads=self.config.intra_op_threads,
            inter_op_threads=self.config.inter_op_threads,
            cancellation=cancellation,
        )
        decoder_session, _decoder_graph, decoder_warnings = create_checked_onnx_session(
            self.config.resolved_path("decoder_model_path"),
            providers=self.config.providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_model_bytes=self.config.max_model_bytes,
            expected_sha256=self.config.decoder_sha256,
            external_data_sha256=self.config.decoder_external_data_sha256,
            max_external_data_bytes=self.config.max_external_data_bytes,
            intra_op_threads=self.config.intra_op_threads,
            inter_op_threads=self.config.inter_op_threads,
            cancellation=cancellation,
        )
        cancellation.raise_if_cancelled()
        self._model = EfficientSAMONNX(encoder_session, decoder_session)
        self._provider_warnings = tuple(
            dict.fromkeys((*encoder_warnings, *decoder_warnings))
        )

    def _validated_image(self, image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("EfficientSAM expects an H x W x 3 uint8 RGB image")
        pixels = int(array.shape[0]) * int(array.shape[1])
        if pixels <= 0 or pixels > self.config.max_image_pixels:
            raise ValueError(
                f"EfficientSAM image has {pixels} pixels; configured limit is "
                f"{self.config.max_image_pixels}"
            )
        # Official split graphs emit three full-resolution candidate masks.
        if pixels * 3 > self.config.max_output_elements:
            raise ValueError(
                "EfficientSAM candidate masks exceed the configured output limit"
            )
        return array

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        if self._model is None:
            raise SessionLifecycleError("EfficientSAM runtime is not loaded")
        image_array = self._validated_image(image)
        prompts = _legacy_prompts(request)
        output_shape = request.output_shape or ShapeType.POLYGON
        cache_key = (request.model_revision, request.source_id)
        started = time.perf_counter()
        embedding = self._embedding_cache.get(cache_key)
        encode_ms = 0.0
        if embedding is None:
            encode_started = time.perf_counter()
            embedding = self._model.encode(image_array)
            encode_ms = (time.perf_counter() - encode_started) * 1000
            cancellation.raise_if_cancelled()
            self._embedding_cache.put(cache_key, embedding)
        decode_started = time.perf_counter()
        masks = np.asarray(self._model.predict_masks(embedding, prompts))
        decode_ms = (time.perf_counter() - decode_started) * 1000
        cancellation.raise_if_cancelled()
        if masks.shape != (1, 1, image_array.shape[0], image_array.shape[1]):
            raise ValueError("EfficientSAM returned an unexpected mask shape")
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=mask_shapes(
                masks[0, 0],
                output_shape,
                max_mask_contours=self.config.max_mask_contours,
                max_shapes=self.config.max_shapes,
                max_polygon_points=self.config.max_polygon_points,
                max_total_shape_points=self.config.max_total_shape_points,
            ),
            timings_ms={
                "encode": encode_ms,
                "decode": decode_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    def preload(
        self,
        image: Any,
        source_id: str,
        cancellation: CancellationToken | None = None,
    ) -> None:
        token = CancellationToken.linked(cancellation, self._shutdown)
        try:
            with self._operation_lock:
                if self.state is not SessionState.READY:
                    raise SessionLifecycleError(
                        f"Cannot preload with a session in state {self.state.value!r}"
                    )
                token.raise_if_cancelled()
                cache_key = (self.capabilities.model_revision, source_id)
                if cache_key not in self._embedding_cache:
                    if self._model is None:
                        raise SessionLifecycleError(
                            "EfficientSAM runtime is not loaded"
                        )
                    embedding = self._model.encode(self._validated_image(image))
                    token.raise_if_cancelled()
                    self._embedding_cache.put(cache_key, embedding)
        finally:
            token.close()

    def _unload(self) -> None:
        self._embedding_cache.clear()
        self._model = None
        self._provider_warnings = ()


class EfficientSamBackend(InferenceBackend):
    backend_id = "efficient_sam"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        validated = EfficientSamConfig.model_validate(config)
        return ModelCapabilities(
            model_id=validated.name,
            model_revision=validated.revision,
            tasks=(ModelTask.PROMPTABLE_SEGMENTATION,),
            supports_cancellation=True,
            metadata={"backend": self.backend_id, "embedding_cache_items": 10},
        )

    def create_session(self, config: Mapping[str, Any]) -> EfficientSamSession:
        return EfficientSamSession(config)


__all__ = ["EfficientSamBackend", "EfficientSamConfig", "EfficientSamSession"]
