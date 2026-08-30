"""Command-line entry point for credential setup and the headless server."""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import sys
from collections.abc import Sequence
from pathlib import Path

from .app import create_server_app
from .auth import generate_token_secret, hash_password
from .config import ServerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anylearning-server",
        description="Authenticated AnyLearning inference service",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    hash_command = commands.add_parser(
        "hash-password", help="prompt for a password and print its Argon2id hash"
    )
    hash_command.add_argument(
        "--password-stdin",
        action="store_true",
        help="read one password line from standard input instead of prompting",
    )
    commands.add_parser(
        "generate-token-secret",
        help="print a new base64url 256-bit bearer-token signing secret",
    )
    serve = commands.add_parser("serve", help="run the public inference API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--ssl-certificate", type=Path)
    serve.add_argument("--ssl-key", type=Path)
    serve.add_argument(
        "--behind-tls-proxy",
        action="store_true",
        help="confirm that a trusted reverse proxy provides HTTPS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "hash-password":
        try:
            password = _read_password(arguments.password_stdin)
            encoded_hash = hash_password(password)
        except ValueError as error:
            parser.error(str(error))
        print(encoded_hash)
        return 0
    if arguments.command == "generate-token-secret":
        print(generate_token_secret())
        return 0
    if arguments.command == "serve":
        _validate_serve_arguments(parser, arguments)
        import uvicorn

        settings = ServerSettings.from_environment()
        app = create_server_app(settings)
        uvicorn.run(
            app,
            host=arguments.host,
            port=arguments.port,
            ssl_certfile=(
                str(arguments.ssl_certificate)
                if arguments.ssl_certificate is not None
                else None
            ),
            ssl_keyfile=str(arguments.ssl_key)
            if arguments.ssl_key is not None
            else None,
            proxy_headers=False,
        )
        return 0
    parser.error("unknown command")


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        password = sys.stdin.readline()
        if not password:
            raise ValueError("standard input did not contain a password")
        return password.rstrip("\r\n")
    password = getpass.getpass("Server password: ")
    confirmation = getpass.getpass("Confirm server password: ")
    if password != confirmation:
        raise ValueError("password confirmation does not match")
    return password


def _validate_serve_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> None:
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    certificate_configured = arguments.ssl_certificate is not None
    key_configured = arguments.ssl_key is not None
    if certificate_configured != key_configured:
        parser.error("--ssl-certificate and --ssl-key must be supplied together")
    if certificate_configured:
        if not arguments.ssl_certificate.is_file():
            parser.error("--ssl-certificate must be an existing file")
        if not arguments.ssl_key.is_file():
            parser.error("--ssl-key must be an existing file")
    if (
        not _is_loopback_host(arguments.host)
        and not certificate_configured
        and not arguments.behind_tls_proxy
    ):
        parser.error("non-loopback serving requires TLS files or --behind-tls-proxy")
    if arguments.behind_tls_proxy and certificate_configured:
        parser.error("choose direct TLS or --behind-tls-proxy, not both")


def _is_loopback_host(value: str) -> bool:
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False
