"""Password verification and compact signed bearer tokens for the server."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from argon2 import PasswordHasher, extract_parameters
from argon2.exceptions import VerificationError
from argon2.low_level import Type

_MAX_PASSWORD_BYTES = 1_024
_MIN_PASSWORD_BYTES = 12
_MAX_ENCODED_HASH_BYTES = 1_024
_MAX_TOKEN_BYTES = 4_096
_TOKEN_VERSION = "v1"
_TOKEN_ISSUER = "anylearning-server"
_TOKEN_SUBJECT = "inference-client"
_MAX_CLOCK_SKEW_SECONDS = 30
_PAYLOAD_KEYS = frozenset({"exp", "iat", "iss", "jti", "sub", "v"})

# argon2-cffi's current RFC 9106 low-memory defaults: 64 MiB, three iterations,
# and four lanes. The public endpoint separately caps concurrent verification,
# so stronger password storage cannot create unbounded CPU or memory fan-out.
_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65_536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)


class AuthenticationConfigurationError(ValueError):
    """Raised when server credential material is absent or malformed."""


class InvalidTokenError(ValueError):
    """Raised without revealing which bearer-token check failed."""


@dataclass(frozen=True)
class TokenClaims:
    subject: str
    issued_at: int
    expires_at: int
    token_id: str


def _password_bytes(password: str) -> bytes:
    if not isinstance(password, str):
        raise TypeError("password must be text")
    encoded = password.encode("utf-8")
    if not _MIN_PASSWORD_BYTES <= len(encoded) <= _MAX_PASSWORD_BYTES:
        raise ValueError(
            f"password must contain {_MIN_PASSWORD_BYTES} to "
            f"{_MAX_PASSWORD_BYTES} UTF-8 bytes"
        )
    return encoded


def hash_password(password: str) -> str:
    """Create a salted Argon2id hash suitable for the server environment."""
    return _PASSWORD_HASHER.hash(_password_bytes(password))


class PasswordAuthenticator:
    """Verify one operator-configured Argon2id password hash."""

    def __init__(self, encoded_hash: str) -> None:
        if not isinstance(encoded_hash, str):
            raise AuthenticationConfigurationError("password hash must be text")
        if (
            not encoded_hash.startswith("$argon2id$")
            or len(encoded_hash.encode("utf-8")) > _MAX_ENCODED_HASH_BYTES
        ):
            raise AuthenticationConfigurationError(
                "password hash must be a bounded Argon2id encoded hash"
            )
        try:
            parameters = extract_parameters(encoded_hash)
        except ValueError as error:
            raise AuthenticationConfigurationError(
                "password hash is not a valid Argon2id encoded hash"
            ) from error
        if (
            parameters.type is not Type.ID
            or parameters.version != 19
            or not 19_456 <= parameters.memory_cost <= 262_144
            or not 2 <= parameters.time_cost <= 10
            or not 1 <= parameters.parallelism <= 16
            or not 16 <= parameters.hash_len <= 64
            or not 16 <= parameters.salt_len <= 64
        ):
            raise AuthenticationConfigurationError(
                "password hash parameters are outside server safety policy"
            )
        self._encoded_hash = encoded_hash

    def verify(self, password: str) -> bool:
        try:
            encoded = _password_bytes(password)
        except (TypeError, ValueError):
            return False
        try:
            return bool(_PASSWORD_HASHER.verify(self._encoded_hash, encoded))
        except VerificationError:
            return False


class TokenSigner:
    """Issue and verify short-lived HMAC-SHA-256 bearer tokens."""

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise AuthenticationConfigurationError(
                "token signing secret must contain at least 32 bytes"
            )
        if (
            not isinstance(ttl_seconds, int)
            or isinstance(ttl_seconds, bool)
            or not 30 <= ttl_seconds <= 3_600
        ):
            raise AuthenticationConfigurationError(
                "token TTL must be between 30 and 3600 seconds"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._secret = secret
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def issue(self) -> str:
        now = int(self._clock())
        payload = {
            "exp": now + self._ttl_seconds,
            "iat": now,
            "iss": _TOKEN_ISSUER,
            "jti": secrets.token_hex(16),
            "sub": _TOKEN_SUBJECT,
            "v": 1,
        }
        encoded_payload = _base64url(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        signed = f"{_TOKEN_VERSION}.{encoded_payload}".encode("ascii")
        signature = _base64url(hmac.digest(self._secret, signed, hashlib.sha256))
        return f"{_TOKEN_VERSION}.{encoded_payload}.{signature}"

    def verify(self, token: str) -> TokenClaims:
        try:
            if (
                not isinstance(token, str)
                or len(token.encode("ascii")) > _MAX_TOKEN_BYTES
            ):
                raise InvalidTokenError("Invalid bearer token")
            version, encoded_payload, encoded_signature = token.split(".")
            if version != _TOKEN_VERSION:
                raise InvalidTokenError("Invalid bearer token")
            signature = _base64url_decode(encoded_signature, maximum_bytes=64)
            signed = f"{version}.{encoded_payload}".encode("ascii")
            expected = hmac.digest(self._secret, signed, hashlib.sha256)
            if not hmac.compare_digest(signature, expected):
                raise InvalidTokenError("Invalid bearer token")
            payload_bytes = _base64url_decode(encoded_payload, maximum_bytes=2_048)
            payload = json.loads(payload_bytes)
        except InvalidTokenError:
            raise
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise InvalidTokenError("Invalid bearer token") from error

        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise InvalidTokenError("Invalid bearer token")
        if (
            payload.get("v") != 1
            or payload.get("iss") != _TOKEN_ISSUER
            or payload.get("sub") != _TOKEN_SUBJECT
            or not isinstance(payload.get("jti"), str)
            or len(payload["jti"]) != 32
            or any(character not in "0123456789abcdef" for character in payload["jti"])
            or type(payload.get("iat")) is not int
            or type(payload.get("exp")) is not int
        ):
            raise InvalidTokenError("Invalid bearer token")
        issued_at = payload["iat"]
        expires_at = payload["exp"]
        now = int(self._clock())
        if (
            issued_at > now + _MAX_CLOCK_SKEW_SECONDS
            or expires_at <= now
            or expires_at <= issued_at
            or expires_at - issued_at != self._ttl_seconds
        ):
            raise InvalidTokenError("Invalid bearer token")
        return TokenClaims(
            subject=payload["sub"],
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=payload["jti"],
        )


def generate_token_secret() -> str:
    """Return a base64url-encoded 256-bit signing secret for configuration."""
    return _base64url(secrets.token_bytes(32))


def decode_token_secret(value: str) -> bytes:
    try:
        secret = _base64url_decode(value, maximum_bytes=64)
    except (TypeError, ValueError) as error:
        raise AuthenticationConfigurationError(
            "token secret must be base64url-encoded"
        ) from error
    if len(secret) < 32:
        raise AuthenticationConfigurationError(
            "token signing secret must decode to at least 32 bytes"
        )
    return secret


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str, *, maximum_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid base64url value")
    if len(value) > maximum_bytes * 2:
        raise ValueError("base64url value exceeds limit")
    if any(character not in _BASE64URL_CHARACTERS for character in value):
        raise ValueError("invalid base64url value")
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    if len(decoded) > maximum_bytes:
        raise ValueError("decoded value exceeds limit")
    return decoded


_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)
