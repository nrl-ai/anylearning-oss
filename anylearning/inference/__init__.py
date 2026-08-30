"""Headless inference contracts and runtime interfaces.

Importing this package must remain safe without optional model runtimes or
application frameworks installed.
"""

from .contracts import (
    CURRENT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    ShapeType,
)

__all__ = [
    "CURRENT_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "InferenceResult",
    "InferenceShape",
    "ModelCapabilities",
    "ModelTask",
    "Point",
    "ShapeType",
]
