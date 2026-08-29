"""The health endpoints.

These exist for one job: telling you a *packaged* build is broken. A Nuitka
build that excluded torch._dynamo compiled, linked, produced a 696 MB binary
and then died on startup, because torchvision.ops imports the excluded module.
Nothing in the build reported it.

So the properties worth pinning are that the probe reports per-module instead
of dying on the first failure, and that it stays reachable when the thing it is
diagnosing is broken.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def api():
    from anylearning.routers.health import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


def test_health_is_cheap_and_says_ok(api):
    response = api.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["python"]


def test_import_probe_reports_every_critical_module(api):
    from anylearning.routers.health import CRITICAL_MODULES, TRAINER_MODULES

    body = api.get("/api/health/imports").json()

    assert set(body["critical"]) == set(CRITICAL_MODULES)
    assert set(body["trainers"]) == set(TRAINER_MODULES)


def test_import_probe_passes_in_a_working_install(api):
    """If this fails here, the development environment is broken too."""
    body = api.get("/api/health/imports").json()
    assert body["ok"], f"modules failed to import: {body['broken']}"


def test_a_broken_module_is_reported_without_taking_the_endpoint_down(api, monkeypatch):
    """The regression this exists for: report the failure, do not become it.

    An endpoint that 500s when an import fails tells you less than one that
    names the module -- and it is unreachable exactly when you need it.
    """
    from anylearning.routers import health

    real_import = health.importlib.import_module

    def explode(name, *args, **kwargs):
        if name == "torchvision.ops":
            raise ImportError("Module 'torch._dynamo' was actively excluded")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(health.importlib, "import_module", explode)

    response = api.get("/api/health/imports")

    assert response.status_code == 200, "the probe must survive what it reports"
    body = response.json()
    assert body["ok"] is False
    assert body["broken"] == ["torchvision.ops"]
    assert "torch._dynamo" in body["critical"]["torchvision.ops"]["error"]


def test_the_probe_does_not_stop_at_the_first_failure(api, monkeypatch):
    from anylearning.routers import health

    real_import = health.importlib.import_module
    broken = {"torch", "cv2"}

    def explode(name, *args, **kwargs):
        if name in broken:
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(health.importlib, "import_module", explode)

    body = api.get("/api/health/imports").json()

    assert set(body["broken"]) == broken
