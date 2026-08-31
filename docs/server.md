# Authenticated inference server

`anylearning.server` is a separate headless application boundary. It does not
mount the desktop project's database, training, filesystem, frontend, window,
or development routes. The API provides health, password-to-token
authentication, protected model discovery, and bounded asynchronous ONNX
prediction jobs. Do not expose the desktop application's legacy `--server`
mode as a substitute.

## Configure local ONNX models

The server never accepts model paths or backend settings from an HTTP client.
Give it a bounded startup manifest instead:

```json
{
  "version": 1,
  "models": [
    {
      "backend": "yolo_onnx",
      "config": {
        "name": "shared-detector",
        "model_path": "models/detector.onnx",
        "sha256": "<64-character SHA-256>",
        "format": "yolov8",
        "task": "detection",
        "class_names": ["person", "vehicle"],
        "providers": ["CPUExecutionProvider"]
      }
    }
  ]
}
```

Relative artifact paths are resolved from the manifest. The manifest must be a
regular non-link JSON file and cannot contain credential-like fields. Public
serving permits the safety-checked `yolo_onnx`, `rfdetr_onnx`, `dfine_onnx`,
`segment_anything`, `efficient_sam`, `efficientvit_sam`, and `sam3` backends.
Promptable pairs use
separate encoder and decoder paths plus `encoder_sha256` and `decoder_sha256`;
each graph may also
declare an exact external-data digest map. SAM3 instead requires
`image_encoder_model_path`, `language_encoder_model_path`, and
`decoder_model_path`, independent graph hashes, and exact external-data maps
for the two encoders. Its separately licensed weights remain deployment-owned
and are not part of the server package.

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
anylearning-server serve --host 127.0.0.1 --port 8000 \
  --model-manifest /srv/anylearning/models.json
```

A non-loopback bind fails unless transport protection is explicit. Supply a
certificate and key for direct TLS, or confirm that a trusted reverse proxy
terminates TLS:

```shell
anylearning-server serve --host 0.0.0.0 \
  --model-manifest /srv/anylearning/models.json \
  --ssl-certificate /run/secrets/server.crt \
  --ssl-key /run/secrets/server.key

anylearning-server serve --host 0.0.0.0 \
  --model-manifest /srv/anylearning/models.json \
  --behind-tls-proxy
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

## Submit and poll predictions

`POST /v1/predictions` uses the encoded image as its body and a compact
base64url form of `InferenceRequest` in `X-AnyLearning-Request`. The public
helper `anylearning.server.encode_request_header()` produces that value.
Accepted media types are JPEG, PNG, and WebP.

Set the request's `source_id` to
`content-sha256:<SHA-256 of the exact encoded body>`. The server verifies this
identity before image decoding, preventing a result from being attached to the
wrong image. Authentication also completes before the large body is read.

The response is `202 Accepted` with an opaque `job_id`. Poll
`GET /v1/predictions/{job_id}` with the same bearer token until the state is
`succeeded`, `failed`, `cancelled`, or `timed_out`. A job belongs to the exact
short-lived token that submitted it; another valid token receives `404`.
`DELETE /v1/predictions/{job_id}` cancels queued/running work and removes any
retained result.

Encoded bytes, decoded pixels, decompression ratio, decoder concurrency,
per-model queue items, retained image bytes, total jobs, result bytes,
deadlines, and result retention all have independent bounds. Each model owns a
single shared session worker so concurrent clients do not multiply model
memory or run an unsafe session concurrently.

Interactive OpenAPI documentation is available at `/docs`. It contains no real
password, token, filesystem path, or model secret examples.
