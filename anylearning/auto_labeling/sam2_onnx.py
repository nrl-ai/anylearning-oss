"""Compatibility imports for the shared SAM2 ONNX adapter."""

from anylearning.inference.backends.sam2_onnx import (
    SAM2ImageDecoder,
    SAM2ImageEncoder,
    SegmentAnything2ONNX,
)

__all__ = ["SAM2ImageDecoder", "SAM2ImageEncoder", "SegmentAnything2ONNX"]
