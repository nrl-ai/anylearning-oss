"""Compact request metadata and bounded image decoding for HTTP inference."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import warnings
from typing import Final

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from anylearning.inference import InferenceRequest

_MAX_REQUEST_METADATA_BYTES: Final = 8 * 1024
_MAX_ENCODED_HEADER_BYTES: Final = 12 * 1024
_BASE64URL = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)
_MEDIA_FORMATS: Final = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class InvalidPredictionPayloadError(ValueError):
    """Raised for malformed metadata or encoded images without parser details."""


def encode_request_header(request: InferenceRequest) -> str:
    """Encode canonical request JSON for ``X-AnyLearning-Request``."""
    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be InferenceRequest")
    payload = request.model_dump_json(exclude_defaults=False).encode("utf-8")
    if len(payload) > _MAX_REQUEST_METADATA_BYTES:
        raise ValueError("inference request metadata exceeds the transport limit")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def decode_request_header(value: str) -> InferenceRequest:
    """Decode strict base64url metadata without accepting alternate spellings."""
    try:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _MAX_ENCODED_HEADER_BYTES
            or any(character not in _BASE64URL for character in value)
        ):
            raise InvalidPredictionPayloadError("Invalid inference request metadata")
        padding = "=" * (-len(value) % 4)
        payload = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        if len(payload) > _MAX_REQUEST_METADATA_BYTES:
            raise InvalidPredictionPayloadError("Invalid inference request metadata")
        return InferenceRequest.model_validate_json(payload)
    except InvalidPredictionPayloadError:
        raise
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise InvalidPredictionPayloadError(
            "Invalid inference request metadata"
        ) from error


def decode_image(
    encoded: bytes,
    media_type: str,
    *,
    max_pixels: int,
    max_decoded_bytes: int,
    max_decompression_ratio: int,
) -> np.ndarray:
    """Decode one still RGB image after header-level allocation checks."""
    expected_format = _MEDIA_FORMATS.get(media_type.lower())
    if expected_format is None:
        raise InvalidPredictionPayloadError(
            "Content-Type must be image/jpeg, image/png, or image/webp"
        )
    if not encoded:
        raise InvalidPredictionPayloadError("Image body is empty")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(encoded)) as image:
                if image.format != expected_format:
                    raise InvalidPredictionPayloadError(
                        "Image bytes do not match Content-Type"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidPredictionPayloadError(
                        "Animated or multi-frame images are not supported"
                    )
                width, height = image.size
                pixels = width * height
                decoded_bytes = pixels * 3
                if (
                    width < 1
                    or height < 1
                    or pixels > max_pixels
                    or decoded_bytes > max_decoded_bytes
                    or decoded_bytes > len(encoded) * max_decompression_ratio
                ):
                    raise InvalidPredictionPayloadError(
                        "Decoded image exceeds configured resource limits"
                    )
                image.load()
                oriented = ImageOps.exif_transpose(image).convert("RGB")
                array = np.array(oriented, dtype=np.uint8, copy=True)
    except InvalidPredictionPayloadError:
        raise
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        UnidentifiedImageError,
    ) as error:
        raise InvalidPredictionPayloadError("Image could not be decoded") from error
    if array.ndim != 3 or array.shape[2] != 3:
        raise InvalidPredictionPayloadError("Image could not be decoded")
    return array


def encoded_image_source_id(encoded: bytes) -> str:
    """Bind a remote request to the exact encoded bytes sent over HTTP."""
    if not isinstance(encoded, bytes) or not encoded:
        raise TypeError("encoded image must be non-empty bytes")
    return f"content-sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "InvalidPredictionPayloadError",
    "decode_image",
    "decode_request_header",
    "encoded_image_source_id",
    "encode_request_header",
]
