"""The model and training-session APIs.

These are the endpoints the UI polls while a job runs, so their failure modes
matter more than their happy paths: asking about a project that no longer
exists, a session id from a different project, or a model whose file has been
deleted underneath the database row.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated config paths plus a fresh DatabaseManager bound to them."""
    from anylearning import config, database

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    from anylearning.routers import model, project, training

    for module in (model, project, training):
        monkeypatch.setattr(module, "db_manager", manager, raising=False)

    yield {
        "manager": manager,
        "projects_root": projects_root,
        "modules": (model, project, training),
    }
    manager.dispose_all()


@pytest.fixture
def api(env):
    model, project, training = env["modules"]
    app = FastAPI()
    for module in (project, model, training):
        app.include_router(module.router)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def project_id(api):
    response = api.post(
        "/api/projects",
        json={"name": "Demo", "type": "Image Classification", "description": ""},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def test_listing_models_of_a_new_project_is_empty_not_an_error(api, project_id):
    response = api.get(f"/api/projects/{project_id}/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert body["total_count"] == 0


def test_model_list_reports_pagination_parameters(api, project_id):
    response = api.get(f"/api/projects/{project_id}/models?offset=0&limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["offset"] == 0
    assert body["limit"] == 5


def test_model_list_for_unknown_project_is_404(api):
    assert api.get("/api/projects/999999/models").status_code == 404


def test_single_model_not_found_is_404(api, project_id):
    assert api.get(f"/api/projects/{project_id}/models/12345").status_code == 404


def test_delete_missing_model_is_404(api, project_id):
    assert api.delete(f"/api/projects/{project_id}/models/12345").status_code == 404


def test_update_missing_model_is_404(api, project_id):
    response = api.put(
        f"/api/projects/{project_id}/models/12345", json={"name": "renamed"}
    )
    assert response.status_code == 404


def test_download_missing_model_is_404(api, project_id):
    assert api.get(f"/api/projects/{project_id}/models/1/download").status_code == 404


def test_model_search_filters_accept_all_variants(api, project_id):
    """Every architecture/size in the UI must be a valid filter value."""
    from anylearning import config

    for variants in config.MODEL_VARIANTS.values():
        for variant in variants:
            response = api.get(
                f"/api/projects/{project_id}/models",
                params={
                    "model_architecture": variant["model_architecture"],
                    "model_size": variant["model_size"],
                },
            )
            assert response.status_code == 200, response.text


def test_model_filters_are_optional(api, project_id):
    """They are Optional[str] = None; omitting them must not 422."""
    assert api.get(f"/api/projects/{project_id}/models").status_code == 200


# --------------------------------------------------------------------------
# Training sessions
# --------------------------------------------------------------------------


def test_training_sessions_of_a_new_project_is_empty(api, project_id):
    response = api.get(f"/api/projects/{project_id}/training_sessions")
    assert response.status_code == 200
    payload = response.json()
    sessions = payload if isinstance(payload, list) else payload.get("training_sessions", [])
    assert sessions == []


def test_training_sessions_for_unknown_project_is_404(api):
    assert api.get("/api/projects/999999/training_sessions").status_code == 404


def test_single_training_session_not_found_is_404(api, project_id):
    response = api.get(f"/api/projects/{project_id}/training_sessions/4321")
    assert response.status_code == 404


def test_last_training_session_when_there_are_none(api, project_id):
    """The UI polls this immediately after project creation."""
    response = api.get(f"/api/projects/{project_id}/last_training_session")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.json() in (None, {}, [])


def test_terminate_unknown_session_is_404(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/training_sessions/4321/terminate"
    )
    assert response.status_code == 404


def test_start_training_rejects_an_incomplete_body(api, project_id):
    """TrainingParams requires every field; FastAPI should 422 before handling."""
    response = api.post(
        f"/api/projects/{project_id}/training_sessions",
        json={"model_architecture": "resnet18"},
    )
    assert response.status_code == 422


def test_start_training_accepts_a_complete_body_shape(api, project_id):
    """The model_ fields are in Pydantic v2's protected namespace.

    This asserts the request is *parsed*, not that training runs -- a 422 here
    would mean the protected-namespace opt-out regressed.
    """
    response = api.post(
        f"/api/projects/{project_id}/training_sessions",
        json={
            "model_architecture": "resnet18",
            "model_size": "lightweight",
            "model_variant": "resnet18_lightweight",
            "batch_size": 1,
            "epochs": 1,
            "learning_rate": 0.001,
            "pretrained_model": "default",
        },
    )
    assert response.status_code != 422, response.text


# --------------------------------------------------------------------------
# Stale-session reconciliation
#
# Training runs in its own process and the PID is stored on the row. If the app
# is killed mid-run, nothing writes a terminal status -- so listing the sessions
# is what reconciles them. Without this the UI shows a run as "training"
# forever and the project can never start another one.
# --------------------------------------------------------------------------


def _add_session(manager, project_id, status, pid=None):
    from anylearning.database import TrainingProcess, TrainingSession
    from sqlalchemy.orm import Session as OrmSession

    with OrmSession(manager.get_project_engine(project_id)) as session:
        training_session = TrainingSession(name="run", description="", status=status)
        session.add(training_session)
        session.flush()
        if pid is not None:
            session.add(
                TrainingProcess(
                    training_session_id=training_session.id, pid=pid, status="running"
                )
            )
        session.commit()
        return training_session.id


def test_session_left_training_without_a_process_is_reported_as_error(
    api, env, project_id
):
    session_id = _add_session(env["manager"], project_id, "training")

    response = api.get(f"/api/projects/{project_id}/training_sessions")

    assert response.status_code == 200, response.text
    statuses = {s["id"]: s["status"] for s in response.json()}
    assert statuses[session_id] == "error"


def test_session_whose_process_is_gone_is_reported_as_error(api, env, project_id):
    """A PID that no longer exists means the run died with the app."""
    import psutil

    # Pick a PID that is definitely free rather than a fixed sentinel: what PID
    # 0 means differs by platform, and reusing a live PID would invert the test.
    dead_pid = max(psutil.pids()) + 1000
    assert not psutil.pid_exists(dead_pid)
    session_id = _add_session(env["manager"], project_id, "training", pid=dead_pid)

    response = api.get(f"/api/projects/{project_id}/training_sessions")

    assert response.status_code == 200, response.text
    statuses = {s["id"]: s["status"] for s in response.json()}
    assert statuses[session_id] == "error"


def test_finished_sessions_are_left_alone(api, env, project_id):
    """Reconciliation must only touch runs that claim to be in progress."""
    session_id = _add_session(env["manager"], project_id, "finished")

    response = api.get(f"/api/projects/{project_id}/training_sessions")

    statuses = {s["id"]: s["status"] for s in response.json()}
    assert statuses[session_id] == "finished"


def test_training_sessions_are_newest_first_and_capped_at_ten(api, env, project_id):
    """The endpoint silently returns only the 10 most recent runs."""
    ids = [_add_session(env["manager"], project_id, "finished") for _ in range(12)]

    body = api.get(f"/api/projects/{project_id}/training_sessions").json()

    assert [s["id"] for s in body] == sorted(ids, reverse=True)[:10]


# --------------------------------------------------------------------------
# Terminate
# --------------------------------------------------------------------------


def test_terminating_a_finished_session_is_rejected(api, env, project_id):
    """Only a run that is actually in progress can be stopped."""
    session_id = _add_session(env["manager"], project_id, "finished")

    response = api.post(
        f"/api/projects/{project_id}/training_sessions/{session_id}/terminate"
    )

    assert response.status_code == 400
    assert "not in progress" in response.json()["detail"]


def test_terminating_a_session_whose_process_is_gone_still_succeeds(
    api, env, project_id
):
    """The row must reach a terminal state even if the process died first.

    Otherwise a run that crashed can never be cleared, and the project stays
    blocked because a session is still considered in progress.
    """
    import psutil

    dead_pid = max(psutil.pids()) + 1000
    assert not psutil.pid_exists(dead_pid)
    session_id = _add_session(env["manager"], project_id, "training", pid=dead_pid)

    response = api.post(
        f"/api/projects/{project_id}/training_sessions/{session_id}/terminate"
    )

    assert response.status_code == 200, response.text
    detail = api.get(
        f"/api/projects/{project_id}/training_sessions/{session_id}"
    ).json()
    assert detail["status"] == "terminated"


def test_terminating_a_session_without_a_process_still_succeeds(api, env, project_id):
    session_id = _add_session(env["manager"], project_id, "not_started")

    response = api.post(
        f"/api/projects/{project_id}/training_sessions/{session_id}/terminate"
    )

    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------
# Last training session
# --------------------------------------------------------------------------


def test_last_training_session_returns_the_newest(api, env, project_id):
    _add_session(env["manager"], project_id, "finished")
    newest = _add_session(env["manager"], project_id, "finished")

    body = api.get(f"/api/projects/{project_id}/last_training_session").json()

    assert body["id"] == newest


def test_last_training_session_reconciles_a_dead_process(api, env, project_id):
    import psutil

    dead_pid = max(psutil.pids()) + 1000
    assert not psutil.pid_exists(dead_pid)
    _add_session(env["manager"], project_id, "training", pid=dead_pid)

    body = api.get(f"/api/projects/{project_id}/last_training_session").json()

    assert body["status"] == "error"
