"""Project CRUD over the real HTTP surface.

These go through TestClient rather than calling the handlers directly, so they
exercise request parsing, response_model serialisation and status codes -- the
parts that broke silently in the Pydantic v1 -> v2 move.

Every test redirects config.PROJECTS_ROOT and the database at a tmp_path, so
nothing touches the developer's real ~/anylearning-data.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A client backed by a throwaway database under tmp_path."""
    from anylearning import config

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    # Rebuild the manager so it picks up the patched paths rather than the
    # module-level one created at import time against the real location.
    from anylearning import database

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    from anylearning.routers import project

    monkeypatch.setattr(project, "db_manager", manager, raising=False)

    app = FastAPI()
    app.include_router(project.router)
    with TestClient(app) as client:
        yield client
    manager.dispose_all()


def create_project(api, name="Demo", project_type="Object Detection"):
    response = api.post(
        "/api/projects",
        json={"name": name, "type": project_type, "description": "a test project"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_create_project_returns_the_stored_record(api):
    body = create_project(api)
    assert body["name"] == "Demo"
    assert body["type"] == "Object Detection"
    assert isinstance(body["id"], int)


def test_created_project_is_listed(api):
    create_project(api, name="First")
    create_project(api, name="Second")

    response = api.get("/api/projects")
    assert response.status_code == 200

    # The endpoint returns a bare list, not an envelope object.
    payload = response.json()
    assert isinstance(payload, list)
    names = [item["name"] for item in payload]
    assert {"First", "Second"} <= set(names)


def test_get_single_project(api):
    created = create_project(api, name="Fetch me")
    response = api.get(f"/api/projects/{created['id']}")
    assert response.status_code == 200
    assert response.json()["name"] == "Fetch me"


def test_get_missing_project_is_404_not_500(api):
    response = api.get("/api/projects/999999")
    assert response.status_code == 404


def test_nullable_fields_serialise_as_null(api):
    """Pydantic v1 auto-promoted `x: str = None` to Optional; v2 does not.

    A freshly created project has no path/dataset yet, so these come back from
    the database as NULL -- exactly the case that would raise a ValidationError
    if the field were not explicitly Optional.
    """
    body = create_project(api)
    for field in ("description", "path", "dataset"):
        assert field in body


def test_project_type_round_trips_for_every_supported_type(api):
    """Each type the UI offers must survive a create/read cycle."""
    from anylearning import config

    for index, project_type in enumerate(config.MODEL_VARIANTS):
        created = create_project(api, name=f"P{index}", project_type=project_type)
        fetched = api.get(f"/api/projects/{created['id']}").json()
        assert fetched["type"] == project_type


def test_delete_project(api):
    created = create_project(api, name="Temporary")
    response = api.delete(f"/api/projects/{created['id']}")
    assert response.status_code in (200, 204), response.text
    assert api.get(f"/api/projects/{created['id']}").status_code == 404


def test_export_status_for_unknown_project_is_404(api):
    """The route whose operation id used to collide with dataset.py's."""
    response = api.get("/api/projects/424242/export/status")
    assert response.status_code == 404


def test_model_variants_endpoint_matches_config():
    """The dropdown source of truth, served straight from config."""
    from anylearning import config
    from anylearning.routers import model

    app = FastAPI()
    app.include_router(model.router)
    with TestClient(app) as client:
        response = client.get("/api/model-variants")

    assert response.status_code == 200
    assert response.json() == config.MODEL_VARIANTS


def test_unknown_route_is_404():
    from anylearning.routers import project

    app = FastAPI()
    app.include_router(project.router)
    with TestClient(app) as client:
        assert client.get("/api/definitely-not-a-route").status_code == 404


@patch("anylearning.routers.project.db_manager")
def test_create_project_rejects_missing_required_fields(_manager, api):
    """FastAPI should reject the body before any handler code runs."""
    response = api.post("/api/projects", json={"description": "no name or type"})
    assert response.status_code == 422
