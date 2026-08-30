"""Strict environment-backed settings for the public inference service."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .auth import AuthenticationConfigurationError, decode_token_secret

PASSWORD_HASH_ENV = "ANYLEARNING_SERVER_PASSWORD_HASH"
TOKEN_SECRET_ENV = "ANYLEARNING_SERVER_TOKEN_SECRET"
TOKEN_TTL_ENV = "ANYLEARNING_SERVER_TOKEN_TTL_SECONDS"
CORS_ORIGINS_ENV = "ANYLEARNING_SERVER_CORS_ORIGINS"


@dataclass(frozen=True)
class ServerSettings:
    password_hash: str = field(repr=False)
    token_secret: bytes = field(repr=False)
    token_ttl_seconds: int = 300
    cors_origins: tuple[str, ...] = ()
    login_attempts_per_client: int = 5
    global_login_attempts: int = 120
    login_window_seconds: int = 60
    max_concurrent_password_checks: int = 2
    max_request_body_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.password_hash, str) or not self.password_hash:
            raise AuthenticationConfigurationError("password hash is required")
        if not isinstance(self.token_secret, bytes) or len(self.token_secret) < 32:
            raise AuthenticationConfigurationError(
                "token signing secret must contain at least 32 bytes"
            )
        for name, value, minimum, maximum in (
            ("token_ttl_seconds", self.token_ttl_seconds, 30, 3_600),
            ("login_attempts_per_client", self.login_attempts_per_client, 1, 100),
            ("global_login_attempts", self.global_login_attempts, 1, 10_000),
            ("login_window_seconds", self.login_window_seconds, 1, 3_600),
            (
                "max_concurrent_password_checks",
                self.max_concurrent_password_checks,
                1,
                32,
            ),
            ("max_request_body_bytes", self.max_request_body_bytes, 1_024, 1_048_576),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.login_attempts_per_client > self.global_login_attempts:
            raise ValueError(
                "per-client login attempts may not exceed the global limit"
            )
        normalized_origins = tuple(
            _validate_origin(value) for value in self.cors_origins
        )
        if len(normalized_origins) != len(set(normalized_origins)):
            raise ValueError("CORS origins must be unique")
        object.__setattr__(self, "cors_origins", normalized_origins)

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> ServerSettings:
        values = os.environ if environment is None else environment
        password_hash = values.get(PASSWORD_HASH_ENV, "")
        encoded_secret = values.get(TOKEN_SECRET_ENV, "")
        if not password_hash:
            raise AuthenticationConfigurationError(
                f"{PASSWORD_HASH_ENV} must contain an Argon2id hash"
            )
        if not encoded_secret:
            raise AuthenticationConfigurationError(
                f"{TOKEN_SECRET_ENV} must contain a base64url signing secret"
            )
        ttl = _environment_integer(values, TOKEN_TTL_ENV, 300)
        origins = _environment_origins(values.get(CORS_ORIGINS_ENV, "[]"))
        return cls(
            password_hash=password_hash,
            token_secret=decode_token_secret(encoded_secret),
            token_ttl_seconds=ttl,
            cors_origins=origins,
        )


def _environment_integer(
    environment: Mapping[str, str], name: str, default: int
) -> int:
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    if not raw_value.isascii() or not raw_value.isdecimal():
        raise ValueError(f"{name} must be a decimal integer")
    return int(raw_value)


def _environment_origins(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{CORS_ORIGINS_ENV} must be a JSON array") from error
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"{CORS_ORIGINS_ENV} must be a JSON string array")
    if len(parsed) > 64:
        raise ValueError("At most 64 CORS origins may be configured")
    return tuple(parsed)


def _validate_origin(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ValueError("CORS origin is invalid")
    if value == "*":
        raise ValueError("Wildcard CORS origins are not accepted")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("CORS origins must be exact HTTP(S) origins")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("CORS origin port is invalid") from error
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    authority = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{authority}"
