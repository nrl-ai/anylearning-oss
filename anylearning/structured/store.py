"""Versioned, archive-friendly storage for row and document projects.

The image workflow has a mature per-item SQLite schema.  A row is not an image,
and squeezing arbitrary cells into ``DataItem.annotation`` would make both
workflows harder to evolve.  Structured projects therefore keep an explicit
bundle below ``projects/<id>/structured``.  Project export already archives the
whole directory, so data, configuration, attribution and review decisions move
together without a database migration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from anylearning import config
from anylearning.database import Project, db_manager
from anylearning.structured import is_structured_project, is_text_project

BUNDLE_VERSION = 1
ALLOWED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".jsonl"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
PARQUET_ROW_GROUP_ROWS = 100_000
PROFILE_SAMPLE_ROWS = 100_000
DEFAULT_BATCH_ROWS = 8_192
TABULAR_TRAINING_ROWS = 500_000
TEXT_TRAINING_ROWS = 250_000
REVIEW_SAMPLE_ROWS = 100_000

PRIMARY_METRICS = {
    "classification": {"Accuracy", "Balanced Accuracy", "Macro F1", "Log Loss"},
    "text_classification": {
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Log Loss",
    },
    "regression": {"RMSE", "MAE", "R²"},
}
DEFAULT_PRIMARY_METRIC = {
    "classification": "Balanced Accuracy",
    "text_classification": "Macro F1",
    "regression": "RMSE",
}


def project_and_root(project_id: int) -> tuple[Project, pathlib.Path]:
    with Session(db_manager.main_engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if not is_structured_project(project.type):
            raise HTTPException(
                status_code=400,
                detail="This endpoint is only available for Tabular AI and Text AI projects.",
            )
        session.expunge(project)
    root = pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "structured"
    root.mkdir(parents=True, exist_ok=True)
    return project, root


def metadata_path(project_id: int) -> pathlib.Path:
    return (
        pathlib.Path(config.PROJECTS_ROOT)
        / str(project_id)
        / "structured"
        / "metadata.json"
    )


def load_metadata(project_id: int, required: bool = False) -> dict[str, Any]:
    path = metadata_path(project_id)
    if not path.exists():
        if required:
            raise HTTPException(
                status_code=409, detail="Upload a dataset before using this project."
            )
        return {"version": BUNDLE_VERSION, "configured": False}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Structured project metadata is damaged: {exc}"
        ) from exc
    if value.get("version") != BUNDLE_VERSION:
        raise HTTPException(
            status_code=409,
            detail=f"Structured bundle version {value.get('version')} is not supported by this app.",
        )
    # Bundles created by the first structured-data preview remain readable.
    # Expose the new execution contract without rewriting their archive merely
    # because the user opened it.
    if value.get("source") and not value.get("performance"):
        value["performance"] = performance_metadata(int(value["source"].get("rows", 0)))
    return value


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_metadata(project_id: int, value: dict[str, Any]) -> None:
    value["version"] = BUNDLE_VERSION
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(metadata_path(project_id), value)


async def save_upload(project_id: int, upload: UploadFile) -> dict[str, Any]:
    project, root = project_and_root(project_id)
    original = pathlib.Path(upload.filename or "dataset.csv").name
    suffix = pathlib.Path(original).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Use CSV, TSV, XLSX, XLS, Parquet or JSON Lines data.",
        )

    incoming = root / f".incoming{suffix}"
    digest = hashlib.sha256()
    size = 0
    try:
        with incoming.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Dataset exceeds the 2 GB project limit.",
                    )
                digest.update(chunk)
                target.write(chunk)
        # Conversion is CPU and disk intensive. Keep it off FastAPI's event
        # loop so other projects stay responsive during a multi-GB import.
        from starlette.concurrency import run_in_threadpool

        dataset = await run_in_threadpool(finalize_upload, root, incoming, suffix)
    except HTTPException:
        incoming.unlink(missing_ok=True)
        raise
    except Exception as exc:
        incoming.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail=f"Could not read {original}: {exc}"
        ) from exc

    original_path = dataset["original_path"]
    profile = dataset["profile"]
    default_task = (
        "text_classification" if is_text_project(project.type) else "classification"
    )
    metadata = {
        "version": BUNDLE_VERSION,
        "configured": False,
        "project_type": project.type,
        "source": {
            "filename": original,
            "stored_name": original_path.name,
            "format": suffix.lstrip("."),
            "bytes": size,
            "sha256": digest.hexdigest(),
            "rows": dataset["rows"],
            "columns": len(dataset["columns"]),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        },
        "task": {
            "type": default_task,
            "target": None,
            "text_column": None,
            "id_column": None,
            "ignored_columns": [],
            "primary_metric": DEFAULT_PRIMARY_METRIC[default_task],
            "class_balance": "balanced",
            "text_features": "word_character",
        },
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": 42},
        "profile": profile,
        "performance": performance_metadata(dataset["rows"]),
        "attribution": {},
    }
    save_metadata(project_id, metadata)
    atomic_json(root / "overrides.json", {})
    return metadata


def sql_string(value: pathlib.Path | str) -> str:
    """A DuckDB string literal for an app-owned path, not user SQL."""
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def duckdb_connection():
    import duckdb

    connection = duckdb.connect(database=":memory:")
    temporary = pathlib.Path(config.DATA_ROOT) / "duckdb-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute("SET preserve_insertion_order = true")
    connection.execute("SET enable_progress_bar = false")
    connection.execute("SET memory_limit = '1GB'")
    connection.execute(f"SET threads = {max(1, min(8, os.cpu_count() or 1))}")
    connection.execute(f"SET temp_directory = {sql_string(temporary)}")
    return connection


def canonical_query(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    literal = sql_string(path)
    # Deliberately exclude BOOLEAN from inference. Business datasets commonly
    # use yes/no as classification labels; silently converting those labels to
    # True/False changes the task semantics and differs from pandas imports.
    candidates = "['BIGINT', 'DOUBLE', 'DATE', 'TIMESTAMP', 'VARCHAR']"
    if suffix == ".csv":
        return f"SELECT * FROM read_csv_auto({literal}, sample_size=100000, auto_type_candidates={candidates})"
    if suffix == ".tsv":
        return (
            f"SELECT * FROM read_csv_auto({literal}, delim='\\t', sample_size=100000, "
            f"auto_type_candidates={candidates})"
        )
    if suffix == ".parquet":
        return f"SELECT * FROM read_parquet({literal})"
    if suffix == ".jsonl":
        return f"SELECT * FROM read_json_auto({literal}, format='newline_delimited')"
    raise ValueError(f"Unsupported streaming dataset extension: {suffix}")


def parquet_summary(path: pathlib.Path) -> tuple[int, list[str]]:
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as parquet:
        return int(parquet.metadata.num_rows), [
            str(name) for name in parquet.schema_arrow.names
        ]


def finalize_upload(
    root: pathlib.Path, incoming: pathlib.Path, suffix: str
) -> dict[str, Any]:
    """Create canonical Parquet without materializing streamable inputs."""
    canonical_temporary = root / ".rows.parquet"
    canonical_temporary.unlink(missing_ok=True)
    try:
        if suffix in {".xlsx", ".xls"}:
            # Spreadsheet containers require workbook materialization. The 2 GB
            # upload ceiling still applies; users with larger data should use
            # CSV or Parquet, which take the streaming path above.
            frame = read_frame(incoming)
            frame.to_parquet(
                canonical_temporary,
                index=False,
                compression="zstd",
                row_group_size=PARQUET_ROW_GROUP_ROWS,
            )
        else:
            with duckdb_connection() as connection:
                connection.execute(
                    f"COPY ({canonical_query(incoming)}) TO {sql_string(canonical_temporary)} "
                    f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {PARQUET_ROW_GROUP_ROWS})"
                )
        row_count, columns = parquet_summary(canonical_temporary)
        if not columns:
            raise HTTPException(status_code=422, detail="The dataset has no columns.")
        if not row_count:
            raise HTTPException(status_code=422, detail="The dataset has no rows.")
        profile_sample = parquet_sample(canonical_temporary, PROFILE_SAMPLE_ROWS)
        original_path = root / f"source{suffix}"
        for old in root.glob("source.*"):
            old.unlink(missing_ok=True)
        os.replace(incoming, original_path)
        os.replace(canonical_temporary, root / "rows.parquet")
        return {
            "original_path": original_path,
            "rows": row_count,
            "columns": columns,
            "profile": profile_frame(profile_sample, total_rows=row_count),
        }
    finally:
        canonical_temporary.unlink(missing_ok=True)


def performance_metadata(
    total_rows: int, profile_rows: int | None = None
) -> dict[str, Any]:
    sampled = min(total_rows, profile_rows or PROFILE_SAMPLE_ROWS)
    return {
        "storage_engine": "DuckDB + Parquet",
        "memory_limit": "1 GB with disk spill",
        "paged_loading": True,
        "batch_rows": DEFAULT_BATCH_ROWS,
        "parquet_row_group_rows": PARQUET_ROW_GROUP_ROWS,
        "profile_rows": sampled,
        "profile_is_sampled": total_rows > sampled,
        "tabular_training_row_limit": TABULAR_TRAINING_ROWS,
        "text_training_row_limit": TEXT_TRAINING_ROWS,
        "review_row_limit": REVIEW_SAMPLE_ROWS,
    }


def read_frame(path: pathlib.Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset extension: {suffix}")


def canonical_path(project_id: int) -> pathlib.Path:
    _, root = project_and_root(project_id)
    canonical = root / "rows.parquet"
    if not canonical.exists():
        raise HTTPException(
            status_code=409, detail="Upload a dataset before using this project."
        )
    return canonical


def parquet_sample(
    path: pathlib.Path, limit: int, seed: int = 42, columns: Sequence[str] | None = None
):
    rows_count, available = parquet_summary(path)
    selected = list(columns) if columns is not None else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Unknown columns: {', '.join(unknown)}"
        )
    projection = ", ".join(quote_identifier(name) for name in selected)
    base = (
        f"SELECT file_row_number AS _row_id, {projection} "
        "FROM read_parquet(?, file_row_number=true)"
    )
    if rows_count > limit:
        statement = (
            f"SELECT * FROM ({base}) USING SAMPLE reservoir ({int(limit)} ROWS) "
            f"REPEATABLE ({int(seed)}) ORDER BY _row_id"
        )
    else:
        statement = base
    with duckdb_connection() as connection:
        frame = connection.execute(statement, [str(path)]).fetchdf()
    return frame.set_index("_row_id", drop=True)


def project_frame(
    project_id: int,
    apply_overrides: bool = True,
    *,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    seed: int = 42,
):
    """Load a bounded training/sample frame; whole reads remain compatible.

    Interactive table paging uses :func:`rows` and never calls this function.
    Callers working on potentially large datasets should always pass max_rows.
    """
    canonical = canonical_path(project_id)
    total, _ = parquet_summary(canonical)
    frame = parquet_sample(
        canonical, min(total, max_rows) if max_rows else total, seed, columns
    )
    if apply_overrides:
        apply_overrides_to_frame(frame, load_overrides(project_id))
    return frame


def apply_overrides_to_frame(frame, overrides: dict[str, dict[str, Any]]) -> None:
    widened: set[str] = set()
    for row_id, cells in overrides.items():
        try:
            index = int(row_id)
        except ValueError:
            continue
        if index not in frame.index:
            continue
        for column, value in cells.items():
            if column in frame.columns:
                # Human corrections may intentionally change the inferred
                # physical type (for example yes/no to a review-state string).
                if column not in widened:
                    frame[column] = frame[column].astype(object)
                    widened.add(column)
                frame.at[index, column] = value


def iter_project_batches(
    project_id: int,
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_ROWS,
    apply_overrides: bool = True,
) -> Iterator[Any]:
    """Yield pandas batches with stable source row IDs and bounded memory."""
    import pyarrow.parquet as pq

    canonical = canonical_path(project_id)
    _, available = parquet_summary(canonical)
    selected = list(columns) if columns is not None else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Unknown columns: {', '.join(unknown)}"
        )
    overrides = load_overrides(project_id) if apply_overrides else {}
    start = 0
    with pq.ParquetFile(canonical) as parquet:
        for batch in parquet.iter_batches(
            batch_size=max(1, batch_size), columns=selected
        ):
            frame = batch.to_pandas()
            frame.index = range(start, start + len(frame))
            frame.index.name = "_row_id"
            start += len(frame)
            if overrides:
                apply_overrides_to_frame(frame, overrides)
            yield frame


def load_overrides(project_id: int) -> dict[str, dict[str, Any]]:
    path = metadata_path(project_id).parent / "overrides.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def update_row(project_id: int, row_id: int, values: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_path(project_id)
    total, columns = parquet_summary(canonical)
    if not 0 <= row_id < total:
        raise HTTPException(status_code=404, detail="Row not found")
    unknown = sorted(set(values) - set(columns))
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Unknown columns: {', '.join(unknown)}"
        )
    overrides = load_overrides(project_id)
    overrides.setdefault(str(row_id), {}).update(values)
    atomic_json(metadata_path(project_id).parent / "overrides.json", overrides)
    return rows(project_id, row_id, 1)["rows"][0]


def configure(project_id: int, changes: dict[str, Any]) -> dict[str, Any]:
    metadata = load_metadata(project_id, required=True)
    columns = {str(column["name"]) for column in metadata.get("profile", [])}
    task_type = changes.get("type")
    supported = {
        "classification",
        "regression",
        "text_classification",
        "lexical_search",
        "semantic_search",  # legacy preview name
        "llm_evaluation",
    }
    if task_type not in supported:
        raise HTTPException(
            status_code=422, detail=f"Unsupported task type: {task_type}"
        )
    target = changes.get("target")
    text_column = changes.get("text_column")
    id_column = changes.get("id_column")
    if (
        task_type not in {"lexical_search", "semantic_search", "llm_evaluation"}
        and target not in columns
    ):
        raise HTTPException(
            status_code=422, detail="Choose a target column that exists in the dataset."
        )
    if (
        task_type in {"text_classification", "lexical_search", "semantic_search"}
        and text_column not in columns
    ):
        raise HTTPException(
            status_code=422, detail="Choose a text column that exists in the dataset."
        )
    if id_column is not None and id_column not in columns:
        raise HTTPException(
            status_code=422, detail="The ID column does not exist in the dataset."
        )

    if task_type == "llm_evaluation":
        prompt_column = changes.get("prompt_column")
        response_column = changes.get("response_column")
        reference_column = changes.get("reference_column")
        if prompt_column not in columns or response_column not in columns:
            raise HTTPException(
                status_code=422,
                detail="Choose prompt and response columns that exist in the dataset.",
            )
        if reference_column is not None and reference_column not in columns:
            raise HTTPException(
                status_code=422,
                detail="The reference column does not exist in the dataset.",
            )

    ignored = [name for name in changes.get("ignored_columns", []) if name in columns]
    if target in ignored or text_column in ignored:
        raise HTTPException(
            status_code=422, detail="The target or text column cannot also be ignored."
        )
    if id_column and id_column in {target, text_column}:
        raise HTTPException(
            status_code=422,
            detail="The row identifier cannot also be the target or text column.",
        )

    primary_metric = changes.get("primary_metric") or DEFAULT_PRIMARY_METRIC.get(
        task_type
    )
    if (
        task_type in PRIMARY_METRICS
        and primary_metric not in PRIMARY_METRICS[task_type]
    ):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported primary metric for {task_type}: {primary_metric}",
        )
    class_balance = changes.get("class_balance") or "balanced"
    if class_balance not in {"balanced", "natural"}:
        raise HTTPException(
            status_code=422,
            detail="Class balance must be balanced or natural.",
        )
    text_features = changes.get("text_features") or "word_character"
    if text_features not in {"word_character", "word", "character"}:
        raise HTTPException(
            status_code=422,
            detail="Text features must be word_character, word or character.",
        )
    metadata["task"] = {
        "type": task_type,
        "target": target,
        "text_column": text_column,
        "id_column": id_column,
        "ignored_columns": ignored,
        "prompt_column": changes.get("prompt_column"),
        "response_column": changes.get("response_column"),
        "reference_column": changes.get("reference_column"),
        "primary_metric": primary_metric,
        "class_balance": class_balance,
        "text_features": text_features,
    }
    split = changes.get("split") or metadata.get("split") or {}
    train = float(split.get("train", 0.70))
    validation = float(split.get("validation", 0.15))
    test = float(split.get("test", 0.15))
    if min(train, validation, test) < 0 or not math.isclose(
        train + validation + test, 1.0, abs_tol=0.001
    ):
        raise HTTPException(
            status_code=422,
            detail="Train, validation and test proportions must add up to 1.",
        )
    if task_type in PRIMARY_METRICS and (train <= 0 or validation + test <= 0):
        raise HTTPException(
            status_code=422,
            detail="Training must be above 0% and at least one held-out split is required.",
        )
    metadata["split"] = {
        "train": train,
        "validation": validation,
        "test": test,
        "seed": int(split.get("seed", 42)),
    }
    attribution = changes.get("attribution")
    if isinstance(attribution, dict):
        metadata["attribution"] = {
            key: str(value).strip()
            for key, value in attribution.items()
            if key in {"name", "url", "license", "citation"} and value
        }
    metadata["configured"] = True
    save_metadata(project_id, metadata)
    return metadata


def rows(
    project_id: int,
    offset: int,
    limit: int,
    query: str | None = None,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    canonical = canonical_path(project_id)
    total_rows, available = parquet_summary(canonical)
    selected = list(columns) if columns else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Unknown columns: {', '.join(unknown)}"
        )
    scan_columns = available if query and query.strip() else selected
    projection = ", ".join(quote_identifier(name) for name in scan_columns)
    page_projection = ", ".join(
        ["_row_id", *(quote_identifier(name) for name in selected)]
    )
    source = (
        f"SELECT file_row_number AS _row_id, {projection} "
        "FROM read_parquet(?, file_row_number=true)"
    )
    parameters: list[Any] = [str(canonical)]
    where = ""
    if query and query.strip():
        clauses = [
            f"contains(lower(coalesce(CAST({quote_identifier(name)} AS VARCHAR), '')), lower(?))"
            for name in available
        ]
        where = " WHERE " + " OR ".join(clauses)
        parameters.extend([query.strip()] * len(available))
    with duckdb_connection() as connection:
        if where:
            selected_total = int(
                connection.execute(
                    f"SELECT count(*) FROM ({source}) source{where}", parameters
                ).fetchone()[0]
            )
        else:
            selected_total = total_rows
        page_parameters = [*parameters, int(limit), int(offset)]
        page = connection.execute(
            f"SELECT {page_projection} FROM ({source}) source{where} "
            "ORDER BY _row_id LIMIT ? OFFSET ?",
            page_parameters,
        ).fetchdf()
    if len(page):
        page = page.set_index("_row_id", drop=True)
        apply_overrides_to_frame(page, load_overrides(project_id))
    return {
        "offset": offset,
        "limit": limit,
        "total": selected_total,
        "dataset_total": total_rows,
        "paged": True,
        "rows": [row_as_json(row, int(index)) for index, row in page.iterrows()],
    }


def row_as_json(row, row_id: int) -> dict[str, Any]:
    return {
        "_row_id": row_id,
        **{str(key): json_value(value) for key, value in row.items()},
    }


def json_value(value: Any) -> Any:
    import pandas as pd

    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, (dict, list, tuple)):
            return json_value(converted)
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def profile_frame(frame, total_rows: int | None = None) -> list[dict[str, Any]]:
    import pandas as pd

    result = []
    source_rows = len(frame)
    total_rows = source_rows if total_rows is None else total_rows
    sampled = total_rows > source_rows
    for name in frame.columns:
        series = frame[name]
        sample_missing = int(series.isna().sum())
        missing = (
            round(sample_missing * total_rows / max(1, source_rows))
            if sampled
            else sample_missing
        )
        comparable = series
        if pd.api.types.is_object_dtype(series):
            comparable = series.map(
                lambda value: (
                    json.dumps(json_value(value), ensure_ascii=False, sort_keys=True)
                    if value is not None
                    else None
                )
            )
        unique = int(comparable.nunique(dropna=True))
        kind = (
            "boolean"
            if pd.api.types.is_bool_dtype(series)
            else "datetime"
            if pd.api.types.is_datetime64_any_dtype(series)
            else "numeric"
            if pd.api.types.is_numeric_dtype(series)
            else "text"
        )
        column: dict[str, Any] = {
            "name": str(name),
            "type": kind,
            "missing": missing,
            "missing_percent": round(100 * sample_missing / max(1, source_rows), 2),
            "unique": unique,
            "examples": [
                json_value(value) for value in series.dropna().head(3).tolist()
            ],
            "profile_rows": source_rows,
            "estimated": sampled,
        }
        if kind == "numeric" and len(series.dropna()):
            numeric = pd.to_numeric(series, errors="coerce")
            column.update(
                minimum=json_value(numeric.min()),
                maximum=json_value(numeric.max()),
                mean=round(float(numeric.mean()), 6),
                median=round(float(numeric.median()), 6),
            )
        elif unique <= 20:
            column["top_values"] = {
                str(key): int(value)
                for key, value in comparable.value_counts(dropna=False).head(10).items()
            }
        result.append(column)
    return result


def bundle_download(project_id: int) -> pathlib.Path:
    """Create a CSV snapshot containing edits; callers stream and remove it."""
    _, root = project_and_root(project_id)
    handle, temporary = tempfile.mkstemp(prefix=".export-", suffix=".csv", dir=root)
    os.close(handle)
    path = pathlib.Path(temporary)
    _, columns = parquet_summary(canonical_path(project_id))
    overrides = load_overrides(project_id)
    if not overrides:
        # DuckDB owns creation of its output path. mkstemp supplied the
        # collision-safe name; remove the empty placeholder first.
        path.unlink()
        with duckdb_connection() as connection:
            connection.execute(
                f"COPY (SELECT * FROM read_parquet({sql_string(canonical_path(project_id))})) "
                f"TO {sql_string(path)} (FORMAT CSV, HEADER true)"
            )
        return path
    with path.open("w", encoding="utf-8", newline="") as stream:
        first = True
        for frame in iter_project_batches(
            project_id, columns=columns, apply_overrides=True
        ):
            frame.to_csv(stream, index=False, header=first, lineterminator="\n")
            first = False
    return path
