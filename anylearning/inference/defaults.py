"""Built-in inference backends, registered without eager runtime imports."""

from __future__ import annotations

from threading import Lock

from .runtime import InferenceBackend, ModelRegistry

_registry: ModelRegistry | None = None
_registry_lock = Lock()


def _sam_backend() -> InferenceBackend:
    from .backends.sam import SegmentAnythingBackend

    return SegmentAnythingBackend()


def create_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("segment_anything", _sam_backend)
    return registry


def get_default_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = create_default_registry()
    return _registry
