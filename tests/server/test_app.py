import logging

import pytest
from fastapi.testclient import TestClient

from anylearning.inference import ModelCapabilities, ModelTask
from anylearning.server import ServerSettings, create_server_app, hash_password

PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="module")
def settings():
    return ServerSettings(
        password_hash=hash_password(PASSWORD),
        token_secret=b"s" * 32,
        token_ttl_seconds=60,
        cors_origins=("https://label.example",),
        login_attempts_per_client=3,
        global_login_attempts=20,
    )


@pytest.fixture(scope="module")
def capabilities():
    return ModelCapabilities(
        model_id="detector",
        model_revision="sha256:fixture",
        tasks=(ModelTask.DETECTION,),
        supports_cancellation=True,
    )


def _login(client: TestClient) -> str:
    response = client.post("/v1/auth/token", json={"password": PASSWORD})
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] == 60
    return payload["access_token"]


def test_public_health_is_minimal_and_models_require_header_token(
    settings, capabilities
):
    app = create_server_app(settings, models=[capabilities])
    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "protocol_version": "1.0"}
        assert "detector" not in health.text
        assert health.headers["cache-control"] == "no-store"
        assert len(health.headers["x-request-id"]) == 32

        assert client.get("/v1/models").status_code == 401
        token = _login(client)
        headers = {"Authorization": f"Bearer {token}"}
        models = client.get("/v1/models", headers=headers)
        assert models.status_code == 200
        assert models.json()["models"][0]["model_id"] == "detector"
        assert client.get("/v1/models/detector", headers=headers).status_code == 200
        assert client.get("/v1/models/missing", headers=headers).status_code == 404


def test_tokens_in_query_strings_and_malformed_headers_are_rejected(
    settings, capabilities
):
    app = create_server_app(settings, models=[capabilities])
    with TestClient(app) as client:
        token = _login(client)
        assert client.get(f"/v1/models?access_token={token}").status_code == 401
        assert (
            client.get(
                "/v1/models", headers={"Authorization": f"Basic {token}"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer invalid"}
            ).status_code
            == 401
        )


def test_rejected_password_input_is_not_echoed_or_logged(
    settings, capabilities, caplog
):
    secret_input = "never echo this password value"
    app = create_server_app(settings, models=[capabilities])
    with caplog.at_level(logging.INFO), TestClient(app) as client:
        rejected = client.post(
            "/v1/auth/token",
            json={"password": secret_input, "unexpected": secret_input},
        )
        assert rejected.status_code == 422
        assert rejected.json() == {"detail": "Invalid request"}
        wrong = client.post("/v1/auth/token", json={"password": secret_input})
        assert wrong.status_code == 401
    assert secret_input not in rejected.text
    assert secret_input not in caplog.text


def test_request_body_limit_runs_before_json_or_password_parsing(
    settings, capabilities
):
    limited = ServerSettings(
        password_hash=settings.password_hash,
        token_secret=b"s" * 32,
        token_ttl_seconds=60,
        max_request_body_bytes=1_024,
    )
    app = create_server_app(limited, models=[capabilities])
    with TestClient(app) as client:
        response = client.post(
            "/v1/auth/token",
            content=b"x" * 1_025,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json() == {"detail": "Request body too large"}
        assert response.headers["cache-control"] == "no-store"
        assert len(response.headers["x-request-id"]) == 32


def test_login_attempts_are_rate_limited_before_more_password_checks(capabilities):
    limited_settings = ServerSettings(
        password_hash=hash_password(PASSWORD),
        token_secret=b"s" * 32,
        token_ttl_seconds=60,
        login_attempts_per_client=2,
        global_login_attempts=10,
    )
    app = create_server_app(limited_settings, models=[capabilities])
    with TestClient(app) as client:
        for _ in range(2):
            assert (
                client.post(
                    "/v1/auth/token", json={"password": "wrong password value"}
                ).status_code
                == 401
            )
        blocked = client.post(
            "/v1/auth/token", json={"password": "wrong password value"}
        )
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1


def test_cors_is_explicit_and_app_contains_no_desktop_routes(settings, capabilities):
    app = create_server_app(settings, models=[capabilities])
    paths = {route.path for route in app.routes}
    assert "/api/projects" not in paths
    assert "/api/window" not in paths
    assert {"/v1/health", "/v1/auth/token", "/v1/models"} <= paths
    schema = app.openapi()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    assert schema["paths"]["/v1/models"]["get"]["security"] == [{"HTTPBearer": []}]
    with TestClient(app) as client:
        allowed = client.options(
            "/v1/predictions",
            headers={
                "Origin": "https://label.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "Authorization, Content-Type, X-AnyLearning-Request"
                ),
            },
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == (
            "https://label.example"
        )
        assert (
            "x-anylearning-request"
            in allowed.headers["access-control-allow-headers"].lower()
        )
        denied = client.options(
            "/v1/models",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert denied.status_code == 400


def test_model_catalog_rejects_duplicates_and_excessive_entries(settings, capabilities):
    with pytest.raises(ValueError, match="unique"):
        create_server_app(settings, models=[capabilities, capabilities])
    models = [
        capabilities.model_copy(update={"model_id": f"model-{index}"})
        for index in range(257)
    ]
    with pytest.raises(ValueError, match="at most 256"):
        create_server_app(settings, models=models)
