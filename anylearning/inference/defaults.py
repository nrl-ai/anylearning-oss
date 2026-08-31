"""Built-in inference backends, registered without eager runtime imports."""

from __future__ import annotations

from threading import Lock

from .runtime import InferenceBackend, ModelRegistry

_registry: ModelRegistry | None = None
_registry_lock = Lock()


def _sam_backend() -> InferenceBackend:
    from .backends.sam import SegmentAnythingBackend

    return SegmentAnythingBackend()


def _efficient_sam_backend() -> InferenceBackend:
    from .backends.efficient_sam import EfficientSamBackend

    return EfficientSamBackend()


def _efficientvit_sam_backend() -> InferenceBackend:
    from .backends.efficientvit_sam import EfficientVitSamBackend

    return EfficientVitSamBackend()


def _sam3_backend() -> InferenceBackend:
    from .backends.sam3 import Sam3Backend

    return Sam3Backend()


def _yolo_onnx_backend() -> InferenceBackend:
    from .backends.yolo_onnx import YoloOnnxBackend

    return YoloOnnxBackend()


def create_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("efficient_sam", _efficient_sam_backend)
    registry.register("efficientvit_sam", _efficientvit_sam_backend)
    registry.register("sam3", _sam3_backend)
    registry.register("segment_anything", _sam_backend)
    registry.register("yolo_onnx", _yolo_onnx_backend)
    return registry


def get_default_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = create_default_registry()
    return _registry
