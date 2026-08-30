import argparse
import io

import pytest

from anylearning.server.auth import PasswordAuthenticator, decode_token_secret
from anylearning.server.cli import (
    _validate_serve_arguments,
    build_parser,
    main,
)
from anylearning.server.config import (
    CORS_ORIGINS_ENV,
    PASSWORD_HASH_ENV,
    TOKEN_SECRET_ENV,
    AuthenticationConfigurationError,
    ServerSettings,
)


def test_settings_require_hash_and_secret_and_parse_explicit_cors_json():
    with pytest.raises(AuthenticationConfigurationError, match=PASSWORD_HASH_ENV):
        ServerSettings.from_environment({})
    with pytest.raises(AuthenticationConfigurationError, match=TOKEN_SECRET_ENV):
        ServerSettings.from_environment({PASSWORD_HASH_ENV: "$argon2id$fixture"})

    settings = ServerSettings.from_environment(
        {
            PASSWORD_HASH_ENV: "$argon2id$fixture",
            TOKEN_SECRET_ENV: "c3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3M",
            CORS_ORIGINS_ENV: '["https://Label.Example:443"]',
        }
    )
    assert settings.cors_origins == ("https://label.example:443",)
    assert settings.token_secret == b"s" * 32
    assert "token_secret" not in repr(settings)
    assert "password_hash" not in repr(settings)


@pytest.mark.parametrize(
    "origin",
    ["*", "file:///tmp/app", "https://user:password@example.com", "https://a/b"],
)
def test_settings_reject_unsafe_cors_origins(origin):
    with pytest.raises(ValueError, match="CORS|origin"):
        ServerSettings(
            password_hash="$argon2id$fixture",
            token_secret=b"s" * 32,
            cors_origins=(origin,),
        )


def test_cli_generates_secret_and_hashes_password_from_stdin(monkeypatch, capsys):
    assert main(["generate-token-secret"]) == 0
    encoded_secret = capsys.readouterr().out.strip()
    assert len(decode_token_secret(encoded_secret)) == 32

    monkeypatch.setattr("sys.stdin", io.StringIO("correct horse battery staple\n"))
    assert main(["hash-password", "--password-stdin"]) == 0
    encoded_hash = capsys.readouterr().out.strip()
    assert PasswordAuthenticator(encoded_hash).verify("correct horse battery staple")


def test_non_loopback_cli_requires_explicit_transport_protection():
    parser = build_parser()
    arguments = parser.parse_args(["serve", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        _validate_serve_arguments(parser, arguments)
    protected = parser.parse_args(["serve", "--host", "0.0.0.0", "--behind-tls-proxy"])
    _validate_serve_arguments(parser, protected)


def test_cli_rejects_half_configured_tls(tmp_path):
    certificate = tmp_path / "certificate.pem"
    certificate.write_text("fixture", encoding="utf-8")
    arguments = argparse.Namespace(
        host="0.0.0.0",
        port=8000,
        ssl_certificate=certificate,
        ssl_key=None,
        behind_tls_proxy=False,
    )
    with pytest.raises(SystemExit):
        _validate_serve_arguments(build_parser(), arguments)
