"""Window controls and the auto-labelling API.

The window routes drive the frameless desktop chrome. They run in a browser too
(dev mode, or the app served over HTTP), where `webview.windows` is empty -- so
the "no window" branch is a real path, not defensive padding, and it must return
cleanly rather than raise.

The auto-labelling routes are covered only as far as the model files allow: the
SAM checkpoints are ~100 MB downloads the suite deliberately avoids, so these
assert the behaviour when a model is *not* loaded, which is also what the UI
sees on first launch.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def window_api():
    from anylearning.routers import window

    app = FastAPI()
    app.include_router(window.router)
    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------
# Window controls -- no window attached
# --------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["maximize", "restore", "minimize"])
def test_window_control_without_a_window_responds_cleanly(window_api, route):
    """Served in a browser there is no webview window; this must not 500."""
    with patch("anylearning.window_chrome.webview") as webview:
        webview.windows = []
        response = window_api.post(f"/window/{route}")

    assert response.status_code == 200
    assert "message" in response.json()


def test_close_without_a_window_does_not_exit_the_process(window_api):
    """os._exit in the happy path makes the empty branch worth pinning down."""
    with patch("anylearning.window_chrome.webview") as webview:
        webview.windows = []
        response = window_api.post("/window/close")

    assert response.status_code == 200
    assert response.json() == {"message": "No window to close."}


# --------------------------------------------------------------------------
# Window controls -- window attached
# --------------------------------------------------------------------------


def test_maximize_maximizes_rather_than_going_fullscreen(window_api):
    """This route used to call toggle_fullscreen, which is a different window."""
    window = MagicMock()
    with patch("anylearning.window_chrome.webview") as webview:
        webview.windows = [window]
        response = window_api.post("/window/maximize")

    window.maximize.assert_called_once()
    window.toggle_fullscreen.assert_not_called()
    assert response.status_code == 200


def test_restore_calls_restore(window_api):
    window = MagicMock()
    with patch("anylearning.window_chrome.webview") as webview:
        webview.windows = [window]
        response = window_api.post("/window/restore")

    window.restore.assert_called_once()
    assert response.status_code == 200


def test_minimize_calls_minimize(window_api):
    window = MagicMock()
    with patch("anylearning.window_chrome.webview") as webview:
        webview.windows = [window]
        response = window_api.post("/window/minimize")

    window.minimize.assert_called_once()
    assert response.status_code == 200


def test_close_destroys_the_window(window_api):
    """os._exit is patched out; without that it would kill the test runner."""
    window = MagicMock()
    with patch("anylearning.window_chrome.webview") as webview:
        webview.windows = [window]
        with patch("anylearning.window_chrome.os._exit") as exit_process:
            window_api.post("/window/close")

    window.destroy.assert_called_once()
    exit_process.assert_called_once_with(0)


# --------------------------------------------------------------------------
# Auto-labelling
# --------------------------------------------------------------------------


@pytest.fixture
def labeling_api(tmp_path, monkeypatch):
    from anylearning import config, database

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    # Project 1's directory must exist: get_project_engine opens
    # <PROJECTS_ROOT>/<id>/database.db and SQLite will not create the parent, so
    # without this the request fails on a missing directory rather than on the
    # thing under test.
    (projects_root / "1").mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    from anylearning.routers import labeling

    monkeypatch.setattr(labeling, "db_manager", manager, raising=False)

    app = FastAPI()
    app.include_router(labeling.router)
    with TestClient(app) as client:
        yield client
    manager.dispose_all()


def test_available_models_are_listed_without_downloading_any(labeling_api):
    """The picker is populated from config, not from what is on disk."""
    response = labeling_api.get("/api/projects/1/auto_labeling/models")
    assert response.status_code == 200
    payload = response.json()
    models = payload if isinstance(payload, list) else payload.get("models", [])
    assert models, "no auto-labeling models advertised"


def test_advertised_models_match_the_shipped_config(labeling_api):
    import yaml

    from anylearning.utils.resources import resource_path

    with open(resource_path("anylearning", "configs/auto_labeling/models.yaml")) as f:
        configured = yaml.safe_load(f)

    response = labeling_api.get("/api/projects/1/auto_labeling/models")
    payload = response.json()
    models = payload if isinstance(payload, list) else payload.get("models", [])
    assert len(models) == len(configured)


def test_status_before_any_model_is_loaded(labeling_api):
    """What the UI sees on first launch."""
    response = labeling_api.get("/api/projects/1/auto_labeling/status")
    assert response.status_code == 200


def test_inference_without_a_loaded_model_fails_cleanly(labeling_api):
    response = labeling_api.post(
        "/api/projects/1/auto_labeling/inference",
        json={"model_name": "not_loaded", "data_item_id": 1, "marks": []},
    )
    # A clear client/server error, never an unhandled traceback.
    assert response.status_code >= 400
    assert response.status_code != 500 or "detail" in response.json()


def test_inference_rejects_a_body_missing_required_fields(labeling_api):
    response = labeling_api.post(
        "/api/projects/1/auto_labeling/inference", json={"marks": []}
    )
    assert response.status_code == 422


def test_load_model_rejects_an_unknown_model_name(labeling_api):
    response = labeling_api.post(
        "/api/projects/1/auto_labeling/load_model",
        json={"model_name": "definitely-not-a-model"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Model definitely-not-a-model not found."


def test_load_model_rejects_a_missing_body(labeling_api):
    response = labeling_api.post("/api/projects/1/auto_labeling/load_model")
    assert response.status_code == 422
