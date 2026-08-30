"""Headless inference contracts and runtime interfaces.

Importing this package must remain safe without optional model runtimes or
application frameworks installed.
"""

from .contracts import (
    CURRENT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    BoxPrompt,
    InferencePrompt,
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    PointPrompt,
    ShapeType,
    TextPrompt,
)
from .defaults import create_default_registry, get_default_registry
from .queue import (
    DuplicateInferenceRequestError,
    InferenceJob,
    InferenceQueue,
    InferenceQueueClosedError,
    InferenceQueueError,
    InferenceQueueFullError,
    InferenceQueueProgress,
    InferenceQueueShutdownError,
)
from .runtime import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    InferenceCancelledError,
    InferenceSession,
    ModelRegistry,
    RegistryError,
    SessionLifecycleError,
    SessionState,
)

__all__ = [
    "CURRENT_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "BaseInferenceSession",
    "BoxPrompt",
    "CancellationToken",
    "create_default_registry",
    "DuplicateInferenceRequestError",
    "get_default_registry",
    "InferenceBackend",
    "InferenceCancelledError",
    "InferencePrompt",
    "InferenceJob",
    "InferenceQueue",
    "InferenceQueueClosedError",
    "InferenceQueueError",
    "InferenceQueueFullError",
    "InferenceQueueProgress",
    "InferenceQueueShutdownError",
    "InferenceRequest",
    "InferenceSession",
    "ModelRegistry",
    "InferenceResult",
    "InferenceShape",
    "ModelCapabilities",
    "ModelTask",
    "Point",
    "PointPrompt",
    "RegistryError",
    "SessionLifecycleError",
    "SessionState",
    "ShapeType",
    "TextPrompt",
]
