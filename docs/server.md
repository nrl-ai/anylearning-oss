# Authenticated inference server

`anylearning.server` is a separate headless application boundary. It does not
mount the desktop project's database, training, filesystem, frontend, window,
or development routes. The initial API provides health, password-to-token
authentication, and protected model capability discovery. Prediction jobs are
added through the same versioned inference contracts and bounded queue; do not
expose the desktop application's legacy `--server` mode as a substitute.

## Create credentials

Generate an Argon2id password hash interactively. The plaintext password is not
accepted as a command argument and is never stored by the server.

```shell
anylearning-server hash-password
anylearning-server generate-token-secret
```

For automation, `hash-password --password-stdin` reads exactly one line. Supply
the resulting values through a secret manager as
`ANYLEARNING_SERVER_PASSWORD_HASH` and
`ANYLEARNING_SERVER_TOKEN_SECRET`. The token secret is base64url-encoded random
data, not another password hash. Rotate it to invalidate every outstanding
token.

Optional settings:

```text
ANYLEARNING_SERVER_TOKEN_TTL_SECONDS=300
ANYLEARNING_SERVER_CORS_ORIGINS=["https://label.example"]
```

CORS origins are a JSON array of exact HTTP(S) origins. Wildcards, embedded
credentials, paths, queries, and fragments are rejected.

## Run safely

Loopback development:

```shell
anylearning-server serve --host 127.0.0.1 --port 8000
```

A non-loopback bind fails unless transport protection is explicit. Supply a
certificate and key for direct TLS, or confirm that a trusted reverse proxy
terminates TLS:

```shell
anylearning-server serve --host 0.0.0.0 \
  --ssl-certificate /run/secrets/server.crt \
  --ssl-key /run/secrets/server.key

anylearning-server serve --host 0.0.0.0 --behind-tls-proxy
```

With a reverse proxy, keep the server on a private network, enforce HTTPS and a
request-body limit at the proxy, and add a proxy-side login limit. The server
does not trust forwarded client-IP headers; proxy clients therefore share a
stricter server-side login bucket rather than being able to spoof addresses.

## Authenticate

`GET /v1/health` is public and deliberately reveals no model inventory.
Exchange the configured password at `POST /v1/auth/token`, then place the
short-lived result only in the HTTP header:

```text
Authorization: Bearer <token>
```

Tokens in query strings are ignored. `/v1/models` and `/v1/models/{model_id}`
require a valid token. Login work is bounded by per-client and global attempt
windows plus a small concurrency gate before Argon2id runs. Request validation
never echoes rejected password input, and API responses carry `no-store`,
request-ID, MIME-sniffing, and referrer-policy headers.

Interactive OpenAPI documentation is available at `/docs`. It contains no real
password, token, filesystem path, or model secret examples.
