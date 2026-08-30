import base64
import json

import pytest

from anylearning.server.auth import (
    AuthenticationConfigurationError,
    InvalidTokenError,
    PasswordAuthenticator,
    TokenSigner,
    decode_token_secret,
    generate_token_secret,
    hash_password,
)
from anylearning.server.rate_limit import LoginRateLimiter


def test_argon2id_hash_is_salted_and_verifies_without_plaintext_storage():
    password = "correct horse battery staple"
    first = hash_password(password)
    second = hash_password(password)

    assert first.startswith("$argon2id$")
    assert first != second
    assert password not in first
    verifier = PasswordAuthenticator(first)
    assert verifier.verify(password)
    assert not verifier.verify("wrong password value")


def test_password_policy_and_configured_hash_fail_closed():
    with pytest.raises(ValueError, match="12 to 1024"):
        hash_password("short")
    with pytest.raises(ValueError, match="12 to 1024"):
        hash_password("a" * 1_025)
    with pytest.raises(AuthenticationConfigurationError, match="Argon2id"):
        PasswordAuthenticator("plaintext is not accepted")
    with pytest.raises(AuthenticationConfigurationError, match="not a valid"):
        PasswordAuthenticator("$argon2id$malformed")
    excessive = hash_password("long enough password").replace("m=65536", "m=999999999")
    with pytest.raises(AuthenticationConfigurationError, match="safety policy"):
        PasswordAuthenticator(excessive)


def test_signed_tokens_expire_and_reject_tampering_or_wrong_keys():
    now = [1_000.0]
    signer = TokenSigner(b"a" * 32, ttl_seconds=60, clock=lambda: now[0])
    token = signer.issue()
    claims = signer.verify(token)
    assert claims.subject == "inference-client"
    assert claims.issued_at == 1_000
    assert claims.expires_at == 1_060
    assert len(claims.token_id) == 32

    version, payload, signature = token.split(".")
    with pytest.raises(InvalidTokenError, match="Invalid bearer token"):
        signer.verify(f"{version}.{payload}A.{signature}")
    with pytest.raises(InvalidTokenError, match="Invalid bearer token"):
        TokenSigner(b"b" * 32, ttl_seconds=60, clock=lambda: now[0]).verify(token)
    now[0] = 1_060
    with pytest.raises(InvalidTokenError, match="Invalid bearer token"):
        signer.verify(token)


def test_signed_token_rejects_validly_signed_noncanonical_claims():
    now = [1_000.0]
    secret = b"a" * 32
    signer = TokenSigner(secret, ttl_seconds=60, clock=lambda: now[0])
    token = signer.issue()
    version, encoded_payload, _signature = token.split(".")
    payload = json.loads(_decode(encoded_payload))
    payload["admin"] = True
    replacement = _encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    import hashlib
    import hmac

    signature = _encode(
        hmac.digest(secret, f"{version}.{replacement}".encode(), hashlib.sha256)
    )
    with pytest.raises(InvalidTokenError, match="Invalid bearer token"):
        signer.verify(f"{version}.{replacement}.{signature}")


def test_generated_token_secret_has_256_bits_and_strict_decoding():
    encoded = generate_token_secret()
    assert len(decode_token_secret(encoded)) == 32
    with pytest.raises(AuthenticationConfigurationError, match="base64url"):
        decode_token_secret("not valid!")
    with pytest.raises(AuthenticationConfigurationError, match="at least 32"):
        decode_token_secret(_encode(b"too short"))


def test_rate_limiter_consumes_attempts_before_work_and_recovers_after_window():
    now = [100.0]
    limiter = LoginRateLimiter(
        attempts_per_client=2,
        global_attempts=3,
        window_seconds=10,
        clock=lambda: now[0],
    )
    assert limiter.admit("client-a") is None
    assert limiter.admit("client-a") is None
    assert limiter.admit("client-a") == 10
    assert limiter.admit("client-b") is None
    assert limiter.admit("client-c") == 10

    now[0] = 111
    assert limiter.admit("client-a") is None
    limiter.authentication_succeeded("client-a")
    assert limiter.admit("client-a") is None


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
