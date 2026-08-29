import io
import json
import pathlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def structured_env(tmp_path, monkeypatch):
    from anylearning import config, database

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))
    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    from anylearning.routers import project, structured
    from anylearning.structured import store

    for module in (project, structured, store):
        monkeypatch.setattr(module, "db_manager", manager, raising=False)
    app = FastAPI()
    app.include_router(project.router)
    app.include_router(structured.router)
    with TestClient(app) as client:
        yield client, projects_root, manager
    manager.dispose_all()


def create_project(client, project_type="Tabular AI"):
    response = client.post(
        "/api/projects",
        json={"name": "Structured", "type": project_type, "description": ""},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def classification_csv(rows=60):
    output = io.StringIO()
    output.write("age,job,balance,target\n")
    for index in range(rows):
        target = "yes" if index % 3 == 0 else "no"
        job = "engineer" if index % 2 else "teacher"
        output.write(f"{20 + index % 40},{job},{100 + index * 17},{target}\n")
    return output.getvalue().encode()


def upload(client, project_id, content=None):
    response = client.post(
        f"/api/projects/{project_id}/structured/upload",
        files={"file": ("people.csv", content or classification_csv(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def configure(client, project_id, **overrides):
    body = {
        "type": "classification",
        "target": "target",
        "text_column": None,
        "ignored_columns": [],
        "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 7},
        **overrides,
    }
    response = client.put(f"/api/projects/{project_id}/structured/config", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_profiles_and_preserves_provenance(structured_env):
    client, projects_root, _ = structured_env
    project_id = create_project(client)
    metadata = upload(client, project_id)
    assert metadata["source"]["rows"] == 60
    assert len(metadata["source"]["sha256"]) == 64
    assert {item["name"] for item in metadata["profile"]} == {
        "age",
        "job",
        "balance",
        "target",
    }
    assert (projects_root / str(project_id) / "structured" / "source.csv").is_file()
    assert (projects_root / str(project_id) / "structured" / "rows.parquet").is_file()
    exported = client.get(f"/api/projects/{project_id}/structured/export")
    assert exported.status_code == 200
    assert exported.content.startswith(b"age,job,balance,target")
    assert not list(
        (projects_root / str(project_id) / "structured").glob(".export-*.csv")
    )


@pytest.mark.parametrize("suffix", ["tsv", "xlsx", "parquet", "jsonl"])
def test_supported_structured_formats(structured_env, suffix):
    import pandas as pd

    client, _, _ = structured_env
    project_id = create_project(client)
    frame = pd.DataFrame({"feature": [1, 2, 3], "label": ["a", "b", "a"]})
    buffer = io.BytesIO()
    if suffix == "tsv":
        content = frame.to_csv(index=False, sep="\t").encode()
    elif suffix == "xlsx":
        frame.to_excel(buffer, index=False)
        content = buffer.getvalue()
    elif suffix == "parquet":
        frame.to_parquet(buffer, index=False)
        content = buffer.getvalue()
    else:
        content = frame.to_json(orient="records", lines=True).encode()
    response = client.post(
        f"/api/projects/{project_id}/structured/upload",
        files={"file": (f"example.{suffix}", content, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["source"]["rows"] == 3
    assert response.json()["source"]["format"] == suffix


def test_wrong_project_type_and_invalid_files_are_rejected(structured_env):
    client, _, _ = structured_env
    image_project = create_project(client, "Image Classification")
    assert client.get(f"/api/projects/{image_project}/structured").status_code == 400
    table_project = create_project(client)
    response = client.post(
        f"/api/projects/{table_project}/structured/upload",
        files={"file": ("notes.txt", b"not a table", "text/plain")},
    )
    assert response.status_code == 415


def test_huggingface_inspection_and_bounded_parquet_import(structured_env, monkeypatch):
    import pandas as pd

    client, _, _ = structured_env
    project_id = create_project(client, "Text AI")
    parquet_buffer = io.BytesIO()
    pd.DataFrame(
        {
            "text": ["card delivery", "cash withdrawal", "pending transfer"],
            "intent": ["card_arrival", "cash_withdrawal", "transfer_pending"],
            "tokens": [["card", "delivery"], ["cash"], ["pending", "transfer"]],
        }
    ).to_parquet(parquet_buffer, index=False)

    repository = {
        "prettyName": "Example intents",
        # License metadata is retained as provenance, but it never gates an
        # import or requires an acknowledgement in the open-source app.
        "cardData": {"license": "custom", "pretty_name": "Example intents"},
        "downloads": 12,
        "likes": 3,
        "gated": False,
    }
    shard_url = "https://huggingface.co/api/datasets/example/intents/parquet/default/train/0.parquet"
    parquet_listing = {"default": {"train": [shard_url]}}

    class Response:
        def __init__(self, payload):
            self.payload = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            return self.payload.read(size)

    def fake_urlopen(request, timeout=0):
        url = request.full_url
        if url == "https://huggingface.co/api/datasets/example/intents":
            return Response(json.dumps(repository).encode())
        if url == "https://huggingface.co/api/datasets/example/intents/parquet":
            return Response(json.dumps(parquet_listing).encode())
        if url.endswith("/parquet/default/train"):
            return Response(json.dumps([shard_url]).encode())
        if url == shard_url:
            return Response(parquet_buffer.getvalue())
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(
        "anylearning.routers.structured.urllib.request.urlopen", fake_urlopen
    )
    inspected = client.get(
        "/api/structured/huggingface/inspect", params={"dataset_id": "example/intents"}
    )
    assert inspected.status_code == 200, inspected.text
    assert inspected.json()["licenses"] == ["custom"]
    assert inspected.json()["configs"]["default"]["train"]["files"] == 1

    imported = client.post(
        f"/api/projects/{project_id}/structured/huggingface",
        json={
            "dataset_id": "example/intents",
            "config": "default",
            "split": "train",
            "row_limit": 100,
        },
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["source"]["rows"] == 3
    page = client.get(f"/api/projects/{project_id}/structured/rows")
    assert page.status_code == 200, page.text
    assert page.json()["rows"][0]["tokens"] == ["card", "delivery"]

    repository["gated"] = True
    gated = client.post(
        f"/api/projects/{project_id}/structured/huggingface",
        json={
            "dataset_id": "example/intents",
            "config": "default",
            "split": "train",
            "row_limit": 100,
        },
    )
    assert gated.status_code == 409
    assert "gated or private" in gated.json()["detail"]


def test_configuration_rows_edits_search_and_export(structured_env):
    client, projects_root, _ = structured_env
    project_id = create_project(client)
    upload(client, project_id)
    configured = configure(client, project_id)
    assert configured["configured"] is True
    page = client.get(
        f"/api/projects/{project_id}/structured/rows?query=engineer"
    ).json()
    assert page["total"] == 30
    row_id = page["rows"][0]["_row_id"]
    updated = client.patch(
        f"/api/projects/{project_id}/structured/rows/{row_id}",
        json={"values": {"target": "reviewed"}},
    )
    assert updated.status_code == 200
    assert updated.json()["target"] == "reviewed"
    exported = client.get(f"/api/projects/{project_id}/structured/export")
    assert exported.status_code == 200
    assert b"reviewed" in exported.content
    structured_root = projects_root / str(project_id) / "structured"
    assert not list(structured_root.glob(".export-*.csv"))


def test_training_controls_are_validated_and_saved(structured_env):
    client, _, _ = structured_env
    project_id = create_project(client)
    upload(client, project_id)
    configured = configure(
        client,
        project_id,
        id_column="age",
        ignored_columns=["balance"],
        primary_metric="Macro F1",
        class_balance="natural",
        split={"train": 0.6, "validation": 0.2, "test": 0.2, "seed": 91},
    )
    assert configured["task"]["id_column"] == "age"
    assert configured["task"]["ignored_columns"] == ["balance"]
    assert configured["task"]["primary_metric"] == "Macro F1"
    assert configured["task"]["class_balance"] == "natural"
    assert configured["split"] == {
        "train": 0.6,
        "validation": 0.2,
        "test": 0.2,
        "seed": 91,
    }

    invalid_metric = client.put(
        f"/api/projects/{project_id}/structured/config",
        json={
            "type": "classification",
            "target": "target",
            "primary_metric": "RMSE",
        },
    )
    assert invalid_metric.status_code == 422
    assert "primary metric" in invalid_metric.json()["detail"]

    no_holdout = client.put(
        f"/api/projects/{project_id}/structured/config",
        json={
            "type": "classification",
            "target": "target",
            "split": {"train": 1, "validation": 0, "test": 0, "seed": 1},
        },
    )
    assert no_holdout.status_code == 422
    assert "held-out" in no_holdout.json()["detail"]


def test_large_table_uses_partial_pages_samples_and_batches(
    structured_env, monkeypatch
):
    from anylearning.structured import store

    client, _, _ = structured_env
    project_id = create_project(client)
    total_rows = 120_123
    source = io.StringIO()
    source.write("record_id,group,value,target,notes\n")
    for index in range(total_rows):
        notes = "last-row-marker" if index == total_rows - 1 else f"record {index}"
        source.write(
            f"{index},g{index % 17},{index * 3},{'yes' if index % 5 == 0 else 'no'},{notes}\n"
        )

    # CSV conversion must not fall back to a whole pandas frame.
    monkeypatch.setattr(
        store,
        "read_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("whole read")),
    )
    metadata = upload(client, project_id, source.getvalue().encode())
    assert metadata["source"]["rows"] == total_rows
    assert metadata["performance"]["paged_loading"] is True
    assert metadata["performance"]["profile_is_sampled"] is True
    assert metadata["performance"]["profile_rows"] == 100_000
    assert all(column["profile_rows"] == 100_000 for column in metadata["profile"])

    page = client.get(
        f"/api/projects/{project_id}/structured/rows?offset=100000&limit=25"
        "&columns=record_id&columns=target"
    )
    assert page.status_code == 200, page.text
    assert page.json()["paged"] is True
    assert page.json()["dataset_total"] == total_rows
    assert page.json()["rows"][0]["_row_id"] == 100_000
    assert set(page.json()["rows"][0]) == {"_row_id", "record_id", "target"}
    assert len(page.json()["rows"]) == 25

    filtered = client.get(
        f"/api/projects/{project_id}/structured/rows?query=last-row-marker&limit=10"
    )
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["total"] == 1
    assert filtered.json()["rows"][0]["_row_id"] == total_rows - 1

    configured = configure(client, project_id)
    assert configured["configured"] is True
    changed = client.patch(
        f"/api/projects/{project_id}/structured/rows/{total_rows - 1}",
        json={"values": {"target": "reviewed"}},
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["target"] == "reviewed"

    batches = list(
        store.iter_project_batches(project_id, columns=["record_id"], batch_size=8192)
    )
    assert sum(len(batch) for batch in batches) == total_rows
    assert max(map(len, batches)) <= 8192
    sample = store.project_frame(
        project_id, columns=["record_id", "target"], max_rows=5_000
    )
    assert len(sample) == 5_000
    assert sample.index.nunique() == 5_000


def test_llm_evaluation_bounds_detail_rows_while_aggregating_all_rows():
    import pandas as pd

    from anylearning.structured.modeling import evaluate_llm_batches

    batches = (
        pd.DataFrame(
            {
                "prompt": [f"prompt {offset + index}" for index in range(4_000)],
                "response": ["answer"] * 4_000,
                "reference": ["answer"] * 4_000,
            },
            index=range(offset, offset + 4_000),
        )
        for offset in (0, 4_000, 8_000)
    )
    report = evaluate_llm_batches(
        batches, "prompt", "response", "reference", detail_limit=500
    )
    assert report["metrics"]["rows"] == 12_000
    assert report["metrics"]["exact_match"] == 1
    assert len(report["rows"]) == 500
    assert report["rows_truncated"] is True


def test_split_can_assign_the_entire_holdout_to_validation_or_test():
    import pandas as pd

    from anylearning.structured.modeling import split_indices

    frame = pd.DataFrame({"target": ["a", "b"] * 20})
    train, validation, test = split_indices(
        frame,
        "target",
        "classification",
        {"train": 0.8, "validation": 0, "test": 0.2, "seed": 4},
    )
    assert len(train) == 32
    assert len(validation) == 0
    assert len(test) == 8

    train, validation, test = split_indices(
        frame,
        "target",
        "classification",
        {"train": 0.8, "validation": 0.2, "test": 0, "seed": 4},
    )
    assert len(train) == 32
    assert len(validation) == 8
    assert len(test) == 0


def test_project_archive_round_trip_preserves_structured_bundle(structured_env):
    client, _, _ = structured_env
    project_id = create_project(client)
    upload(client, project_id)
    configure(client, project_id)
    changed = client.patch(
        f"/api/projects/{project_id}/structured/rows/2",
        json={"values": {"target": "human_reviewed"}},
    )
    assert changed.status_code == 200, changed.text

    started = client.post(f"/api/projects/{project_id}/export")
    assert started.status_code == 200, started.text
    status = client.get(f"/api/projects/{project_id}/export/status")
    assert status.status_code == 200, status.text
    assert status.json()["status"] == "completed"
    archive = client.get(f"/api/projects/{project_id}/export/download")
    assert archive.status_code == 200, archive.text

    imported = client.post(
        "/api/projects/import",
        files={
            "import_file": (
                "structured-project.tar.gz",
                archive.content,
                "application/gzip",
            )
        },
    )
    assert imported.status_code == 200, imported.text
    import_id = imported.json()["import_id"]
    imported_status = client.get(f"/api/projects/import/{import_id}/status")
    assert imported_status.status_code == 200, imported_status.text
    assert imported_status.json()["status"] == "completed"
    imported_id = imported_status.json()["project_id"]
    metadata = client.get(f"/api/projects/{imported_id}/structured")
    assert metadata.status_code == 200, metadata.text
    assert metadata.json()["configured"] is True
    rows_page = client.get(
        f"/api/projects/{imported_id}/structured/rows?offset=2&limit=1"
    )
    assert rows_page.status_code == 200, rows_page.text
    assert rows_page.json()["rows"][0]["target"] == "human_reviewed"


def test_llm_evaluation_and_lexical_search(structured_env):
    client, _, _ = structured_env
    project_id = create_project(client, "Text AI")
    data = (
        "prompt,response,reference\n"
        '"What is two plus two?","Four","four"\n'
        '"How do I reset a card PIN?","Open settings and choose Change PIN","Open settings then choose Change PIN"\n'
        '"Where is my transfer?","","Check transfer status"\n'
    ).encode()
    upload(client, project_id, data)
    configured = configure(
        client,
        project_id,
        type="llm_evaluation",
        target=None,
        prompt_column="prompt",
        response_column="response",
        reference_column="reference",
    )
    assert configured["task"]["prompt_column"] == "prompt"
    report = client.post(f"/api/projects/{project_id}/structured/evaluate")
    assert report.status_code == 200, report.text
    assert report.json()["metrics"]["completion_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert report.json()["metrics"]["exact_match"] == pytest.approx(1 / 3, abs=1e-6)

    configure(
        client, project_id, type="lexical_search", target=None, text_column="prompt"
    )
    search = client.post(
        f"/api/projects/{project_id}/structured/search",
        json={"query": "forgot card pin", "limit": 2},
    )
    assert search.status_code == 200, search.text
    assert len(search.json()["results"]) == 2
    assert "PIN" in search.json()["results"][0]["text"]


def test_preview_text_name_and_search_task_remain_compatible(structured_env):
    """Projects saved before the terminology correction must still open."""
    client, _, _ = structured_env
    project_id = create_project(client, "Text & LLM")
    upload(client, project_id, b'text\n"card delivery"\n"pending transfer"\n')
    configured = configure(
        client,
        project_id,
        type="semantic_search",
        target=None,
        text_column="text",
    )
    assert configured["task"]["type"] == "semantic_search"
    search = client.post(
        f"/api/projects/{project_id}/structured/search",
        json={"query": "transfer", "limit": 1},
    )
    assert search.status_code == 200, search.text
    assert search.json()["results"][0]["text"] == "pending transfer"


class Logger:
    def __init__(self):
        self.lines = []
        self.metrics = []

    def write(self, message):
        self.lines.append(message)

    def write_metrics(self, metrics):
        self.metrics.append(metrics)


def params(**changes):
    from anylearning.database import TrainingParams

    defaults = {
        "model_architecture": "catboost",
        "model_size": "balanced",
        "model_variant": "catboost_balanced",
        "batch_size": 16,
        "epochs": 25,
        "learning_rate": 0.08,
        "pretrained_model": "default",
        "device": "cpu",
    }
    defaults.update(changes)
    return TrainingParams(**defaults)


def test_tabular_trainer_model_report_review_and_prediction(structured_env, tmp_path):
    client, _, _ = structured_env
    project_id = create_project(client)
    upload(client, project_id)
    configure(client, project_id)

    from anylearning.structured.modeling import predict_artifact
    from anylearning.training.trainers.structured_trainer import StructuredTrainer

    logger = Logger()
    trainer = StructuredTrainer(tmp_path / "run", logger, project_id, params())
    trainer.prepare_data()
    config_data = trainer.prepare_config()
    trainer.train()
    found, model_path = trainer.get_model_path()
    assert found
    assert trainer.export_onnx() is None
    assert trainer.report["engine"] == "CatBoost"
    assert 0 <= trainer.report["metrics"]["Accuracy"] <= 1
    assert trainer.report["baseline"]["metrics"]
    assert trainer.report["primary_metric"] == "Balanced Accuracy"
    assert trainer.report["evaluation"]["confusion_matrix"]
    assert trainer.report["evaluation"]["per_class"]
    assert trainer.report["configuration"]["class_balance"] == "balanced"
    assert trainer.report["review_queue"][0]["reason"].startswith("uncertainty")
    predictions = predict_artifact(
        pathlib.Path(model_path),
        config_data,
        [{"age": 44, "job": "teacher", "balance": 900}],
    )
    assert predictions[0]["prediction"] in {"yes", "no"}
    assert 0 <= predictions[0]["confidence"] <= 1


def test_text_trainer_is_multiclass_and_predictable(structured_env, tmp_path):
    client, _, _ = structured_env
    project_id = create_project(client, "Text AI")
    lines = ["text,intent"]
    examples = {
        "card_arrival": [
            "where is my card",
            "card delivery status",
            "when will the card arrive",
        ],
        "cash_withdrawal": [
            "cash machine failed",
            "atm did not pay",
            "withdraw money from atm",
        ],
        "transfer_pending": [
            "transfer still pending",
            "where is my transfer",
            "bank transfer has not arrived",
        ],
    }
    for repeat in range(5):
        for label, texts in examples.items():
            for text in texts:
                lines.append(f'"{text} {repeat}",{label}')
    upload(client, project_id, ("\n".join(lines) + "\n").encode())
    configure(
        client,
        project_id,
        type="text_classification",
        target="intent",
        text_column="text",
        primary_metric="Macro F1",
        class_balance="natural",
        text_features="word_character",
    )

    from anylearning.structured.modeling import predict_artifact
    from anylearning.training.trainers.structured_trainer import StructuredTrainer

    logger = Logger()
    trainer = StructuredTrainer(
        tmp_path / "text-run",
        logger,
        project_id,
        params(model_architecture="tfidf-logreg", model_size="lightweight"),
    )
    trainer.prepare_data()
    config_data = trainer.prepare_config()
    saved_config = json.loads(config_data)
    assert saved_config["features"] == ["text"]
    assert saved_config["categorical_features"] == []
    trainer.train()
    _, path = trainer.get_model_path()
    result = predict_artifact(
        pathlib.Path(path), config_data, [{"text": "my card has not arrived"}]
    )
    assert result[0]["prediction"] == "card_arrival"
    assert trainer.report["classes"] == [
        "card_arrival",
        "cash_withdrawal",
        "transfer_pending",
    ]
    assert trainer.report["primary_metric"] == "Macro F1"
    assert trainer.report["configuration"]["class_balance"] == "natural"
    assert trainer.report["configuration"]["text_features"] == "word_character"
    assert trainer.report["evaluation"]["per_class"]
