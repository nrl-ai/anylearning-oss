"""API for Tabular AI, Text AI, search, and response-evaluation projects."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote, urlparse

import yaml
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from anylearning import config
from anylearning.database import Model, db_manager
from anylearning.structured import is_text_project
from anylearning.structured.modeling import (
    evaluate_llm_batches,
    lexical_search_batches,
    predict_artifact,
)
from anylearning.structured.store import (
    atomic_json,
    bundle_download,
    configure,
    iter_project_batches,
    load_metadata,
    parquet_sample,
    parquet_summary,
    performance_metadata,
    project_and_root,
    rows,
    save_upload,
    update_row,
)
from anylearning.utils.resources import resource_path

router = APIRouter(prefix="/api", tags=["Structured Data"])


class StructuredConfig(BaseModel):
    type: str
    target: str | None = None
    text_column: str | None = None
    id_column: str | None = None
    ignored_columns: list[str] = Field(default_factory=list)
    prompt_column: str | None = None
    response_column: str | None = None
    reference_column: str | None = None
    primary_metric: str | None = None
    class_balance: str | None = None
    text_features: str | None = None
    split: dict[str, float | int] | None = None
    attribution: dict[str, str] | None = None


class RowUpdate(BaseModel):
    values: dict[str, Any]


class PredictionRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    limit: int = Field(default=10, ge=1, le=100)


class HuggingFaceImportRequest(BaseModel):
    dataset_id: str = Field(min_length=3, max_length=200)
    config: str = Field(min_length=1, max_length=200)
    split: str = Field(min_length=1, max_length=200)
    row_limit: int = Field(default=50_000, ge=100, le=200_000)


HF_DATASET_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def json_url(url: str, timeout: int = 15) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "AnyLearning Hugging Face connector/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise HTTPException(
                status_code=409,
                detail="This Hugging Face dataset is gated or private. Public Parquet access is required.",
            ) from exc
        if exc.code == 404:
            raise HTTPException(
                status_code=404,
                detail="Hugging Face dataset or converted Parquet data was not found.",
            ) from exc
        if exc.code == 400:
            raise HTTPException(
                status_code=409,
                detail="Hugging Face could not provide a safe Parquet conversion for this dataset.",
            ) from exc
        raise HTTPException(
            status_code=502, detail=f"Hugging Face returned HTTP {exc.code}."
        ) from exc


def inspect_huggingface(dataset_id: str) -> dict[str, Any]:
    if not HF_DATASET_ID.fullmatch(dataset_id):
        raise HTTPException(
            status_code=422,
            detail="Use a Hugging Face dataset ID such as owner/dataset-name.",
        )
    encoded = quote(dataset_id, safe="/")
    repository = json_url(f"https://huggingface.co/api/datasets/{encoded}")
    parquet = json_url(f"https://huggingface.co/api/datasets/{encoded}/parquet")
    if not isinstance(parquet, dict) or not parquet:
        raise HTTPException(
            status_code=409,
            detail="The Dataset Viewer has no safe Parquet conversion for this dataset.",
        )
    card = repository.get("cardData") or {}
    license_value = card.get("license")
    if isinstance(license_value, list):
        licenses = [str(item) for item in license_value]
    elif license_value:
        licenses = [str(license_value)]
    else:
        licenses = []
    grouped: dict[str, dict[str, dict[str, int | None]]] = {}
    for config_name, splits in parquet.items():
        if not isinstance(splits, dict):
            continue
        for split_name, urls in splits.items():
            if isinstance(urls, list) and urls:
                grouped.setdefault(str(config_name), {})[str(split_name)] = {
                    "files": len(urls),
                    "bytes": None,
                }
    if not grouped:
        raise HTTPException(
            status_code=409,
            detail="The Dataset Viewer has no safe Parquet conversion for this dataset.",
        )
    return {
        "dataset_id": dataset_id,
        "name": card.get("pretty_name") or repository.get("prettyName") or dataset_id,
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "licenses": licenses,
        "gated": bool(repository.get("gated")),
        "configs": grouped,
        "downloads": repository.get("downloads"),
        "likes": repository.get("likes"),
    }


@router.get("/structured/huggingface/inspect")
def inspect_huggingface_dataset(dataset_id: str):
    return inspect_huggingface(dataset_id.strip())


@router.post("/projects/{project_id}/structured/huggingface")
def import_huggingface_dataset(project_id: int, body: HuggingFaceImportRequest):
    import pyarrow.parquet as pq

    project, root = project_and_root(project_id)
    info = inspect_huggingface(body.dataset_id)
    if info["gated"]:
        raise HTTPException(
            status_code=409,
            detail="This Hugging Face dataset is gated or private. Public Parquet access is required.",
        )
    if (
        body.config not in info["configs"]
        or body.split not in info["configs"][body.config]
    ):
        raise HTTPException(
            status_code=422, detail="Choose a config and split reported by the dataset."
        )
    encoded = quote(body.dataset_id, safe="/")
    urls = json_url(
        f"https://huggingface.co/api/datasets/{encoded}/parquet/{quote(body.config, safe='')}/{quote(body.split, safe='')}"
    )
    if not isinstance(urls, list) or not urls:
        raise HTTPException(
            status_code=409,
            detail="No Parquet shards are available for that config and split.",
        )
    canonical_temporary = root / ".rows-huggingface.parquet"
    canonical_temporary.unlink(missing_ok=True)
    writer = None
    canonical_schema = None
    rows_written = 0
    downloaded = 0
    shards_downloaded = 0
    digest = hashlib.sha256()
    completed = False
    try:
        for shard_index, url in enumerate(urls):
            parsed = urlparse(str(url))
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
                raise HTTPException(
                    status_code=502,
                    detail="Hugging Face returned an unexpected download host.",
                )
            request = urllib.request.Request(
                str(url), headers={"User-Agent": "AnyLearning Hugging Face connector/1"}
            )
            temporary = root / f".huggingface-{shard_index}.parquet"
            try:
                try:
                    response = urllib.request.urlopen(request, timeout=60)
                except urllib.error.HTTPError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Could not download Hugging Face Parquet shard {shard_index + 1}.",
                    ) from exc
                with response, temporary.open("wb") as target:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > 2 * 1024 * 1024 * 1024:
                            raise HTTPException(
                                status_code=413,
                                detail="Selected Hugging Face data exceeds the 2 GB project limit.",
                            )
                        digest.update(chunk)
                        target.write(chunk)
                remaining = body.row_limit - rows_written
                # Stream record batches straight into the canonical writer. No
                # pandas frame or shard list grows with the selected row limit.
                with pq.ParquetFile(temporary) as parquet_file:
                    for batch in parquet_file.iter_batches(
                        batch_size=min(8192, remaining)
                    ):
                        batch = batch.slice(0, remaining)
                        if writer is None:
                            canonical_schema = batch.schema
                            writer = pq.ParquetWriter(
                                canonical_temporary,
                                canonical_schema,
                                compression="zstd",
                            )
                        elif batch.schema != canonical_schema:
                            batch = batch.cast(canonical_schema)
                        writer.write_batch(batch, row_group_size=100_000)
                        rows_written += batch.num_rows
                        remaining -= batch.num_rows
                        if remaining <= 0:
                            break
                shards_downloaded += 1
            finally:
                temporary.unlink(missing_ok=True)
            if rows_written >= body.row_limit:
                break
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if not completed:
            canonical_temporary.unlink(missing_ok=True)
    if not rows_written:
        canonical_temporary.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="The selected Hugging Face split has no rows."
        )
    os.replace(canonical_temporary, root / "rows.parquet")
    row_count, columns = parquet_summary(root / "rows.parquet")
    manifest = {
        "dataset_id": body.dataset_id,
        "config": body.config,
        "split": body.split,
        "row_limit": body.row_limit,
        "downloaded_shards": shards_downloaded,
        "available_shards": len(urls),
        "licenses": info["licenses"],
        "url": info["url"],
    }
    atomic_json(root / "huggingface-source.json", manifest)
    from anylearning.structured.store import profile_frame, save_metadata

    default_task = (
        "text_classification" if is_text_project(project.type) else "classification"
    )
    metadata = {
        "version": 1,
        "configured": False,
        "project_type": project.type,
        "source": {
            "filename": f"huggingface://{body.dataset_id}/{body.config}/{body.split}",
            "stored_name": "huggingface-source.json",
            "format": "parquet",
            "bytes": downloaded,
            "sha256": digest.hexdigest(),
            "rows": row_count,
            "columns": len(columns),
            "partial": row_count >= body.row_limit or shards_downloaded < len(urls),
        },
        "task": {
            "type": default_task,
            "target": None,
            "text_column": None,
            "id_column": None,
        },
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": 42},
        "profile": profile_frame(
            parquet_sample(root / "rows.parquet", 100_000), total_rows=row_count
        ),
        "performance": performance_metadata(row_count),
        "attribution": {
            "name": info["name"],
            "url": info["url"],
            "license": ", ".join(info["licenses"])
            or "Not declared — manually acknowledged",
            "citation": f"Hugging Face dataset {body.dataset_id}",
        },
        "huggingface": manifest,
    }
    save_metadata(project_id, metadata)
    atomic_json(root / "overrides.json", {})
    return metadata


@router.get("/projects/{project_id}/structured")
def get_structured_project(project_id: int):
    project_and_root(project_id)
    metadata = load_metadata(project_id)
    if not metadata.get("source"):
        return metadata
    # The profile is already stored; summary reads never scan the dataset.
    return metadata


@router.post("/projects/{project_id}/structured/upload")
async def upload_structured_dataset(project_id: int, file: UploadFile = File(...)):
    return await save_upload(project_id, file)


@router.put("/projects/{project_id}/structured/config")
def configure_structured_dataset(project_id: int, body: StructuredConfig):
    project_and_root(project_id)
    metadata = load_metadata(project_id, required=True)
    columns = {str(column["name"]) for column in metadata.get("profile", [])}
    if body.type == "llm_evaluation":
        missing = [
            name
            for name in (body.prompt_column, body.response_column)
            if name not in columns
        ]
        if missing:
            raise HTTPException(
                status_code=422, detail="Choose existing prompt and response columns."
            )
        if body.reference_column is not None and body.reference_column not in columns:
            raise HTTPException(
                status_code=422, detail="The reference column does not exist."
            )
    return configure(project_id, body.model_dump())


@router.get("/projects/{project_id}/structured/rows")
def list_structured_rows(
    project_id: int,
    offset: int = 0,
    limit: int = 50,
    query: str | None = None,
    columns: list[str] | None = Query(default=None),
):
    project_and_root(project_id)
    return rows(project_id, max(0, offset), max(1, min(200, limit)), query, columns)


@router.patch("/projects/{project_id}/structured/rows/{row_id}")
def patch_structured_row(project_id: int, row_id: int, body: RowUpdate):
    project_and_root(project_id)
    return update_row(project_id, row_id, body.values)


@router.get("/projects/{project_id}/structured/export")
def export_structured_dataset(project_id: int):
    path = bundle_download(project_id)
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"anylearning-project-{project_id}.csv",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


def model_files(
    project_id: int, model_id: int
) -> tuple[Model, pathlib.Path, pathlib.Path]:
    project_and_root(project_id)
    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.get(Model, model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        path = model.path
        config_data = model.config_file
        session.expunge(model)
    artifact = pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "models" / path
    if not artifact.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found")
    report = artifact.parent / "report.json"
    model.config_file = config_data
    return model, artifact, report


@router.post("/projects/{project_id}/structured/models/{model_id}/predict")
def predict_structured_rows(project_id: int, model_id: int, body: PredictionRequest):
    model, artifact, _ = model_files(project_id, model_id)
    try:
        predictions = predict_artifact(artifact, model.config_file, body.rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": predictions}


@router.get("/projects/{project_id}/structured/models/{model_id}/report")
def structured_model_report(project_id: int, model_id: int):
    _, _, report = model_files(project_id, model_id)
    if not report.exists():
        raise HTTPException(status_code=404, detail="Model report not found")
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Model report is damaged: {exc}"
        ) from exc


@router.post("/projects/{project_id}/structured/evaluate")
def evaluate_llm_dataset(project_id: int):
    _, root = project_and_root(project_id)
    metadata = load_metadata(project_id, required=True)
    task = metadata.get("task", {})
    if task.get("type") != "llm_evaluation":
        raise HTTPException(
            status_code=409,
            detail="Configure this dataset for Response evaluation first.",
        )
    selected_columns = [task["prompt_column"], task["response_column"]]
    if task.get("reference_column"):
        selected_columns.append(task["reference_column"])
    report = evaluate_llm_batches(
        iter_project_batches(project_id, columns=selected_columns),
        task["prompt_column"],
        task["response_column"],
        task.get("reference_column"),
    )
    atomic_json(root / "llm-evaluation.json", report)
    return report


@router.post("/projects/{project_id}/structured/search")
def lexical_search(project_id: int, body: SearchRequest):
    metadata = load_metadata(project_id, required=True)
    task = metadata.get("task", {})
    # ``semantic_search`` was the inaccurate preview name. Existing projects
    # keep working, while newly configured ones persist the corrected term.
    if task.get("type") not in {"lexical_search", "semantic_search"}:
        raise HTTPException(
            status_code=409,
            detail="Configure this dataset as Lexical & fuzzy search first.",
        )
    text_column = task["text_column"]
    return lexical_search_batches(
        iter_project_batches(project_id, columns=[text_column]),
        text_column,
        body.query,
        body.limit,
    )


def fallback_catalog() -> list[dict[str, Any]]:
    path = resource_path("anylearning", "configs/structured_datasets.yml")
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["datasets"]


def load_catalog() -> tuple[str, list[dict[str, Any]]]:
    url = "https://cdn.anylearning.nrl.ai/datasets/structured/catalog-v1.json"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AnyLearning structured catalog"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
        datasets = payload.get("datasets")
        if not isinstance(datasets, list):
            raise ValueError("catalog has no datasets list")
        for entry in datasets:
            parsed = urlparse(str(entry.get("url", "")))
            checksum = str(entry.get("sha256", ""))
            if parsed.scheme != "https" or parsed.hostname != "cdn.anylearning.nrl.ai":
                raise ValueError("catalog dataset URL is outside the AnyLearning CDN")
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum.lower()
            ):
                raise ValueError("catalog dataset checksum is invalid")
        return "cdn", datasets
    except Exception:
        return "bundled", fallback_catalog()


@router.get("/structured/catalog")
def structured_dataset_catalog():
    """Curated, license-reviewed examples. Remote additions need no app update."""
    source, datasets = load_catalog()
    return {"source": source, "datasets": datasets}


@router.post("/projects/{project_id}/structured/catalog/{slug}")
async def install_catalog_dataset(project_id: int, slug: str):
    _, catalog = load_catalog()
    entry = next((item for item in catalog if item.get("slug") == slug), None)
    if entry is None:
        raise HTTPException(
            status_code=404, detail="Dataset is not in the curated catalog."
        )
    _, root = project_and_root(project_id)
    temporary = root / f".catalog-{slug}{pathlib.Path(entry['url']).suffix}"
    try:
        request = urllib.request.Request(
            entry["url"], headers={"User-Agent": "AnyLearning dataset installer"}
        )
        digest = hashlib.sha256()
        total = 0
        expected_bytes = int(entry.get("bytes", 0))
        limit = max(
            expected_bytes + 1024 * 1024, int(expected_bytes * 1.1), 10 * 1024 * 1024
        )
        with (
            urllib.request.urlopen(request, timeout=30) as response,
            temporary.open("wb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise ValueError("download is larger than its catalog entry")
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest().lower() != str(entry["sha256"]).lower():
            raise ValueError("download checksum does not match the curated catalog")
        with temporary.open("rb") as stream:
            upload = UploadFile(file=stream, filename=entry["filename"])
            metadata = await save_upload(project_id, upload)
        metadata["attribution"] = {
            "name": entry["name"],
            "url": entry["source_url"],
            "license": entry["license"],
            "citation": entry.get("citation", ""),
        }
        metadata["catalog_slug"] = slug
        from anylearning.structured.store import save_metadata

        save_metadata(project_id, metadata)
        return configure(
            project_id,
            {
                "type": entry["task"],
                "target": entry.get("target"),
                "text_column": entry.get("text_column"),
                "id_column": entry.get("id_column"),
                "ignored_columns": entry.get("ignored_columns", []),
                "prompt_column": entry.get("prompt_column"),
                "response_column": entry.get("response_column"),
                "reference_column": entry.get("reference_column"),
                "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": 42},
                "attribution": metadata["attribution"],
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not download the dataset: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
