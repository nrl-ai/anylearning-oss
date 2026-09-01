"""ASGI limits enforced before FastAPI parses attacker-controlled bodies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TOO_LARGE = b'{"detail":"Request body too large"}'
_INVALID_FRAMING = b'{"detail":"Invalid request framing"}'


class RequestBodyLimitMiddleware:
    """Buffer one small API body up to a hard byte limit, then replay it once."""

    def __init__(
        self,
        app: Any,
        *,
        max_bytes: int,
        streaming_paths: Mapping[str, int] | None = None,
    ) -> None:
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        self._app = app
        self._max_bytes = max_bytes
        self._streaming_paths = dict(streaming_paths or {})
        if any(
            not isinstance(path, str)
            or not path.startswith("/")
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            for path, limit in self._streaming_paths.items()
        ):
            raise ValueError("streaming path limits must be positive integers")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        content_lengths = [
            value
            for name, value in scope.get("headers", ())
            if name.lower() == b"content-length"
        ]
        has_transfer_encoding = any(
            name.lower() == b"transfer-encoding"
            for name, _value in scope.get("headers", ())
        )
        if len(content_lengths) > 1 or (content_lengths and has_transfer_encoding):
            await _send_json(send, 400, _INVALID_FRAMING)
            return
        path = scope.get("path", "")
        streaming_limit = self._streaming_paths.get(path)
        body_limit = streaming_limit or self._max_bytes
        if content_lengths:
            try:
                encoded_length = content_lengths[0].decode("ascii")
            except (UnicodeError, ValueError):
                await _send_json(send, 400, _INVALID_FRAMING)
                return
            if (
                len(encoded_length) > 20
                or not encoded_length.isdecimal()
                or not encoded_length.isascii()
            ):
                await _send_json(send, 400, _INVALID_FRAMING)
                return
            declared = int(encoded_length)
            if declared > body_limit:
                await _send_json(send, 413, _TOO_LARGE)
                return

        # Large prediction bodies are consumed explicitly by the authenticated
        # endpoint. Passing through here ensures bearer validation finishes
        # before an unauthenticated client can make the process retain an image.
        if streaming_limit is not None:
            state = scope.setdefault("state", {})
            state["request_body_limit"] = streaming_limit
            await self._app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":

                async def disconnected() -> dict[str, str]:
                    return {"type": "http.disconnect"}

                await self._app(scope, disconnected, send)
                return
            if message.get("type") != "http.request":
                await _send_json(send, 400, _INVALID_FRAMING)
                return
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes) or len(body) + len(chunk) > body_limit:
                await _send_json(send, 413, _TOO_LARGE)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self._app(scope, replay, send)


async def _send_json(send: Any, status_code: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
