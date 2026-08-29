"""Training, evaluation, prediction and Smart Review for structured projects."""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter, deque
from typing import Any

import numpy as np

from anylearning.structured.store import (
    REVIEW_SAMPLE_ROWS,
    TABULAR_TRAINING_ROWS,
    TEXT_TRAINING_ROWS,
    atomic_json,
    json_value,
    project_frame,
)

TRAINING_CELL_BUDGET = 20_000_000
REPORT_INDEX_LIMIT = 10_000


def bounded_training_rows(row_limit: int, column_count: int) -> int:
    """Bound both tall and very wide datasets to a predictable RAM envelope."""
    return max(10_000, min(row_limit, TRAINING_CELL_BUDGET // max(1, column_count)))


def split_report(train_idx, val_idx, test_idx, split: dict[str, Any]) -> dict[str, Any]:
    result = {
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "test_count": len(test_idx),
        **split,
    }
    if len(train_idx) + len(val_idx) + len(test_idx) <= REPORT_INDEX_LIMIT:
        result.update(
            train_rows=train_idx.tolist(),
            validation_rows=val_idx.tolist(),
            test_rows=test_idx.tolist(),
        )
    return result


def review_sample(frame, seed: int):
    if len(frame) <= REVIEW_SAMPLE_ROWS:
        return frame
    return frame.sample(n=REVIEW_SAMPLE_ROWS, random_state=seed).sort_index()


def split_indices(frame, target: str, task_type: str, split: dict[str, Any]):
    """Deterministic train/validation/test indexes, stratified when safe."""
    from sklearn.model_selection import train_test_split

    labelled = frame[target].notna() & (
        frame[target].astype("string").str.strip() != ""
    )
    indexes = np.asarray(frame.index[labelled])
    if len(indexes) < 6:
        raise ValueError(
            "At least 6 labelled rows are required to train and evaluate a model."
        )
    seed = int(split.get("seed", 42))
    test_size = float(split.get("test", 0.15))
    val_size = float(split.get("validation", 0.15))
    y = frame.loc[indexes, target]
    stratify = None
    if task_type != "regression":
        counts = y.astype("string").value_counts()
        if len(counts) >= 2 and int(counts.min()) >= 3:
            stratify = y
    holdout_size = min(0.8, max(test_size + val_size, 2 / len(indexes)))
    train_idx, holdout_idx = train_test_split(
        indexes,
        test_size=holdout_size,
        random_state=seed,
        stratify=stratify,
    )
    if not len(holdout_idx):
        return train_idx, np.asarray([], dtype=int), np.asarray([], dtype=int)
    if val_size == 0:
        return train_idx, np.asarray([], dtype=int), np.asarray(holdout_idx)
    if test_size == 0:
        return train_idx, np.asarray(holdout_idx), np.asarray([], dtype=int)
    relative_test = test_size / max(test_size + val_size, 1e-9)
    holdout_y = frame.loc[holdout_idx, target]
    holdout_stratify = None
    if task_type != "regression":
        counts = holdout_y.astype("string").value_counts()
        if len(counts) >= 2 and int(counts.min()) >= 2:
            holdout_stratify = holdout_y
    try:
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=relative_test,
            random_state=seed + 1,
            stratify=holdout_stratify,
        )
    except ValueError:
        cut = max(1, round(len(holdout_idx) * (1 - relative_test)))
        val_idx, test_idx = holdout_idx[:cut], holdout_idx[cut:]
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def duplicate_groups(frame, feature_columns: list[str]) -> dict[int, str]:
    import pandas as pd

    normalized = frame[feature_columns].astype("string").fillna("")
    for name in feature_columns:
        normalized[name] = (
            normalized[name]
            .str.casefold()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    hashes = pd.util.hash_pandas_object(normalized, index=False)
    counts = hashes.value_counts()
    duplicates = counts[counts > 1].index
    duplicate_hashes = set(int(value) for value in duplicates)
    return {
        int(row_id): f"{int(value):016x}"[:12]
        for row_id, value in hashes.items()
        if int(value) in duplicate_hashes
    }


def normalize_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def smart_review_rows(
    frame,
    probabilities: np.ndarray | None,
    predictions,
    target: str | None,
    feature_columns: list[str],
    limit: int = 200,
    uncertainty_values: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Transparent uncertainty + diversity queue with duplicate warnings.

    We first rank by uncertainty, then interleave predicted classes.  This is a
    deliberately explainable active-learning policy: users can always see why
    a row was selected, and the saved report also includes a seeded-random
    baseline so a future experiment can measure whether it actually helps.
    """
    duplicates = duplicate_groups(frame, feature_columns)
    if uncertainty_values is not None:
        uncertainty = np.asarray(uncertainty_values, dtype=float)
    elif probabilities is None:
        uncertainty = np.zeros(len(frame), dtype=float)
    else:
        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])
        uncertainty = 1.0 - probabilities.max(axis=1)
    labels = [str(value) for value in predictions]
    candidates = np.argsort(-uncertainty, kind="stable").tolist()
    buckets: dict[str, deque[int]] = {}
    for pos in candidates:
        buckets.setdefault(labels[pos], deque()).append(pos)
    ordered: list[int] = []
    while buckets and len(ordered) < limit:
        for label in sorted(list(buckets)):
            bucket = buckets[label]
            if bucket:
                ordered.append(bucket.popleft())
            if not bucket:
                del buckets[label]
            if len(ordered) >= limit:
                break
    result = []
    for rank, pos in enumerate(ordered, start=1):
        row_id = int(frame.index[pos])
        actual = frame.iloc[pos][target] if target else None
        predicted = labels[pos]
        reasons = [f"uncertainty {uncertainty[pos]:.3f}"]
        if row_id in duplicates:
            reasons.append(f"duplicate group {duplicates[row_id]}")
        if actual is not None and str(actual).strip() and str(actual) != predicted:
            reasons.append("prediction disagrees with current label")
        result.append(
            {
                "rank": rank,
                "row_id": row_id,
                "prediction": predicted,
                "actual": json_value(actual),
                "uncertainty": round(float(uncertainty[pos]), 6),
                "confidence": round(float(1 - uncertainty[pos]), 6),
                "duplicate_group": duplicates.get(row_id),
                "reason": "; ".join(reasons),
            }
        )
    return result


def classification_metrics(y_true, y_pred, probabilities=None) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
    )

    result = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "Balanced Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "Macro F1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
    }
    if probabilities is not None:
        try:
            result["Log Loss"] = round(float(log_loss(y_true, probabilities)), 6)
        except ValueError:
            pass
    return result


def classification_details(y_true, y_pred) -> dict[str, Any]:
    """Serializable confusion matrix and class-level precision/recall/F1."""
    from sklearn.metrics import classification_report, confusion_matrix

    labels = sorted({str(value) for value in y_true} | {str(value) for value in y_pred})
    truth = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    report = classification_report(
        truth,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "per_class": [
            {
                "label": label,
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
            for label in labels
        ],
    }


def primary_first(metrics: dict[str, float], primary: str | None) -> dict[str, float]:
    if not primary or primary not in metrics:
        return metrics
    return {
        primary: metrics[primary],
        **{key: value for key, value in metrics.items() if key != primary},
    }


def catboost_metric(task_type: str, primary: str, multiclass: bool = False) -> str:
    if task_type == "regression":
        return {"MAE": "MAE", "R²": "R2"}.get(primary, "RMSE")
    return {
        "Accuracy": "Accuracy",
        "Balanced Accuracy": "BalancedAccuracy",
        "Macro F1": "TotalF1:average=Macro",
        "Log Loss": "MultiClass" if multiclass else "Logloss",
    }.get(primary, "MultiClass" if multiclass else "Logloss")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "RMSE": round(float(rmse), 6),
        "R²": round(float(r2_score(y_true, y_pred)), 6),
    }


def train_tabular(
    project_id: int, output: pathlib.Path, params, logger
) -> tuple[pathlib.Path, dict[str, Any]]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    from pandas.api.types import is_numeric_dtype
    from sklearn.dummy import DummyClassifier, DummyRegressor

    from anylearning.structured.store import load_metadata

    metadata = load_metadata(project_id, required=True)
    task = metadata["task"]
    task_type = task["type"]
    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"This project is configured for {task_type}, not tabular prediction."
        )
    target = task["target"]
    ignored = set(task.get("ignored_columns") or []) | {target}
    if task.get("id_column"):
        ignored.add(task["id_column"])
    available = [str(column["name"]) for column in metadata.get("profile", [])]
    features = [name for name in available if name not in ignored]
    if not features:
        raise ValueError(
            "No feature columns remain after the target and ignored columns are removed."
        )
    dataset_rows = int(metadata["source"]["rows"])
    training_limit = bounded_training_rows(TABULAR_TRAINING_ROWS, len(features) + 1)
    seed = int(metadata["split"].get("seed", 42))
    frame = project_frame(
        project_id,
        columns=[*features, target],
        max_rows=training_limit,
        seed=seed,
    )
    if dataset_rows > len(frame):
        logger.write(
            f"Performance guard: sampled {len(frame):,} of {dataset_rows:,} rows for training."
        )
    train_idx, val_idx, test_idx = split_indices(
        frame, target, task_type, metadata["split"]
    )
    X = frame[features].copy()
    categorical = [name for name in features if not is_numeric_dtype(X[name])]
    for name in categorical:
        X[name] = X[name].fillna("<missing>").astype(str)
    y = frame[target]
    primary_metric = task.get("primary_metric") or (
        "Balanced Accuracy" if task_type == "classification" else "RMSE"
    )
    multiclass = task_type == "classification" and y.loc[train_idx].nunique() > 2
    iterations = max(20, min(int(getattr(params, "epochs", 300)), 5000))
    common = {
        "iterations": iterations,
        "learning_rate": float(getattr(params, "learning_rate", 0.05)),
        "depth": 6 if getattr(params, "model_size", "balanced") != "accurate" else 8,
        "random_seed": int(metadata["split"].get("seed", 42)),
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "eval_metric": catboost_metric(task_type, primary_metric, multiclass),
    }
    if task_type == "classification":
        if task.get("class_balance", "balanced") == "balanced":
            common["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(
            **common,
            loss_function="MultiClass" if multiclass else "Logloss",
        )
        baseline = DummyClassifier(strategy="prior")
    else:
        model = CatBoostRegressor(**common, loss_function="RMSE")
        baseline = DummyRegressor(strategy="mean")
    train_pool = Pool(X.loc[train_idx], y.loc[train_idx], cat_features=categorical)
    val_pool = (
        Pool(X.loc[val_idx], y.loc[val_idx], cat_features=categorical)
        if len(val_idx)
        else None
    )
    logger.write(
        f"Structured split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test; "
        f"{len(features)} features ({len(categorical)} categorical)."
    )
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50 if val_pool is not None else None,
    )
    artifact = output / "model.cbm"
    model.save_model(artifact)

    baseline.fit(
        X.loc[train_idx].astype(str) if categorical else X.loc[train_idx],
        y.loc[train_idx],
    )
    evaluation_idx = test_idx if len(test_idx) else val_idx
    predictions = model.predict(X.loc[evaluation_idx]).reshape(-1)
    review_frame = review_sample(frame, seed)
    review_X = X.loc[review_frame.index]
    review_predictions = model.predict(review_X).reshape(-1)
    probabilities = (
        model.predict_proba(review_X) if task_type == "classification" else None
    )
    if task_type == "classification":
        evaluation_probabilities = model.predict_proba(X.loc[evaluation_idx])
        metrics = classification_metrics(
            y.loc[evaluation_idx].astype(str),
            predictions.astype(str),
            evaluation_probabilities,
        )
        baseline_predictions = baseline.predict(
            X.loc[evaluation_idx].astype(str) if categorical else X.loc[evaluation_idx]
        )
        baseline_metrics = classification_metrics(
            y.loc[evaluation_idx].astype(str), baseline_predictions.astype(str)
        )
        evaluation_details = classification_details(
            y.loc[evaluation_idx].astype(str), predictions.astype(str)
        )
    else:
        metrics = regression_metrics(
            y.loc[evaluation_idx].astype(float), predictions.astype(float)
        )
        baseline_predictions = baseline.predict(
            X.loc[evaluation_idx].astype(str) if categorical else X.loc[evaluation_idx]
        )
        baseline_metrics = regression_metrics(
            y.loc[evaluation_idx].astype(float), baseline_predictions.astype(float)
        )
        evaluation_details = None
    metrics = primary_first(metrics, primary_metric)
    importance = sorted(
        (
            {"feature": name, "importance": round(float(value), 6)}
            for name, value in zip(
                features, model.get_feature_importance(), strict=True
            )
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    regression_uncertainty = None
    if task_type == "regression":
        # For labelled regression rows, large residuals are the most useful
        # review signal. Normalize to a stable 0..1 score for the UI.
        residuals = np.abs(
            y.loc[review_frame.index].astype(float).to_numpy()
            - review_predictions.astype(float)
        )
        scale = float(np.max(residuals))
        regression_uncertainty = residuals / scale if scale > 0 else residuals
    review = smart_review_rows(
        review_frame,
        probabilities,
        review_predictions,
        target,
        features,
        uncertainty_values=regression_uncertainty,
    )
    report = {
        "version": 1,
        "engine": "CatBoost",
        "task": task_type,
        "target": target,
        "features": features,
        "categorical_features": categorical,
        "dataset_rows": dataset_rows,
        "training_rows": len(frame),
        "training_sampled": dataset_rows > len(frame),
        "training_row_limit": training_limit,
        "split": split_report(train_idx, val_idx, test_idx, metadata["split"]),
        "metrics": metrics,
        "primary_metric": primary_metric,
        "evaluation": evaluation_details,
        "configuration": {
            "class_balance": (
                task.get("class_balance", "balanced")
                if task_type == "classification"
                else None
            ),
            "iterations": iterations,
            "learning_rate": common["learning_rate"],
            "depth": common["depth"],
        },
        "baseline": {"engine": type(baseline).__name__, "metrics": baseline_metrics},
        "feature_importance": importance,
        "review_queue": review,
        "random_review_baseline": np.random.default_rng(seed)
        .choice(frame.index, size=min(len(review), len(frame)), replace=False)
        .astype(int)
        .tolist(),
    }
    atomic_json(output / "report.json", report)
    logger.write(f"Test metrics: {metrics}")
    logger.write(f"Baseline metrics: {baseline_metrics}")
    return artifact, report


def train_text(
    project_id: int, output: pathlib.Path, params, logger
) -> tuple[pathlib.Path, dict[str, Any]]:
    import pickle

    from scipy.sparse import hstack
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from anylearning.structured.store import load_metadata

    metadata = load_metadata(project_id, required=True)
    task = metadata["task"]
    if task["type"] != "text_classification":
        raise ValueError(
            f"This project is configured for {task['type']}, not text classification."
        )
    target, text_column = task["target"], task["text_column"]
    primary_metric = task.get("primary_metric") or "Macro F1"
    text_features = task.get("text_features") or "word_character"
    available = [str(column["name"]) for column in metadata.get("profile", [])]
    training_limit = bounded_training_rows(TEXT_TRAINING_ROWS, max(2, len(available)))
    dataset_rows = int(metadata["source"]["rows"])
    seed = int(metadata["split"].get("seed", 42))
    frame = project_frame(
        project_id,
        columns=[text_column, target],
        max_rows=training_limit,
        seed=seed,
    )
    if dataset_rows > len(frame):
        logger.write(
            f"Performance guard: sampled {len(frame):,} of {dataset_rows:,} rows for training."
        )
    train_idx, val_idx, test_idx = split_indices(
        frame, target, "text_classification", metadata["split"]
    )
    texts = frame[text_column].fillna("").astype(str)
    vectorizers = []
    if text_features in {"word_character", "word"}:
        vectorizers.append(
            (
                "word",
                TfidfVectorizer(
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=120_000,
                ),
            )
        )
    if text_features in {"word_character", "character"}:
        vectorizers.append(
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    sublinear_tf=True,
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=120_000,
                ),
            )
        )
    train_matrix = hstack(
        [
            vectorizer.fit_transform(texts.loc[train_idx])
            for _, vectorizer in vectorizers
        ]
    )
    class_weight = (
        "balanced" if task.get("class_balance", "balanced") == "balanced" else None
    )
    feature_label = {
        "word_character": "word + character",
        "word": "word",
        "character": "character",
    }[text_features]
    if len(train_idx) > 100_000:
        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier(
            loss="log_loss",
            max_iter=max(50, int(getattr(params, "epochs", 300))),
            class_weight=class_weight,
            average=True,
            random_state=seed,
            tol=1e-3,
            n_jobs=-1,
        )
        engine = f"{feature_label} TF-IDF / SGD log-loss"
    else:
        model = LogisticRegression(
            max_iter=max(200, int(getattr(params, "epochs", 300))),
            class_weight=class_weight,
            l1_ratio=0,
        )
        engine = f"{feature_label} TF-IDF / Logistic Regression"
    model.fit(train_matrix, frame.loc[train_idx, target].astype(str))
    evaluation_idx = test_idx if len(test_idx) else val_idx
    evaluation_texts = texts.loc[evaluation_idx]
    evaluation_matrix = hstack(
        [vectorizer.transform(evaluation_texts) for _, vectorizer in vectorizers]
    )
    evaluation_predictions = model.predict(evaluation_matrix)
    evaluation_probabilities = model.predict_proba(evaluation_matrix)
    metrics = classification_metrics(
        frame.loc[evaluation_idx, target].astype(str),
        evaluation_predictions,
        evaluation_probabilities,
    )
    metrics = primary_first(metrics, primary_metric)
    evaluation_details = classification_details(
        frame.loc[evaluation_idx, target].astype(str), evaluation_predictions
    )
    dummy = DummyClassifier(strategy="prior").fit(
        np.zeros((len(train_idx), 1)), frame.loc[train_idx, target].astype(str)
    )
    baseline = classification_metrics(
        frame.loc[evaluation_idx, target].astype(str),
        dummy.predict(np.zeros((len(evaluation_idx), 1))),
    )
    artifact = output / "model.pkl"
    with artifact.open("wb") as stream:
        pickle.dump(
            {
                "vectorizers": vectorizers,
                "model": model,
                "text_column": text_column,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    feature_columns = [text_column]
    review_frame = review_sample(frame, seed)
    review_texts = review_frame[text_column].fillna("").astype(str)
    review_matrix = hstack(
        [vectorizer.transform(review_texts) for _, vectorizer in vectorizers]
    )
    review_predictions = model.predict(review_matrix)
    review_probabilities = model.predict_proba(review_matrix)
    report = {
        "version": 1,
        "engine": engine,
        "task": "text_classification",
        "target": target,
        "text_column": text_column,
        "classes": model.classes_.tolist(),
        "dataset_rows": dataset_rows,
        "training_rows": len(frame),
        "training_sampled": dataset_rows > len(frame),
        "training_row_limit": training_limit,
        "split": split_report(train_idx, val_idx, test_idx, metadata["split"]),
        "metrics": metrics,
        "primary_metric": primary_metric,
        "evaluation": evaluation_details,
        "configuration": {
            "class_balance": task.get("class_balance", "balanced"),
            "text_features": text_features,
            "maximum_features_per_analyzer": 120_000,
        },
        "baseline": {"engine": "most frequent class", "metrics": baseline},
        "review_queue": smart_review_rows(
            review_frame,
            review_probabilities,
            review_predictions,
            target,
            feature_columns,
        ),
        "random_review_baseline": np.random.default_rng(seed)
        .choice(frame.index, size=min(200, len(frame)), replace=False)
        .astype(int)
        .tolist(),
    }
    atomic_json(output / "report.json", report)
    logger.write(f"Test metrics: {metrics}")
    logger.write(f"Baseline metrics: {baseline}")
    return artifact, report


def predict_artifact(
    artifact: pathlib.Path, config_data: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    config = json.loads(config_data)
    task_type = config["task"]["type"]
    if task_type in {"classification", "regression"}:
        import pandas as pd
        from catboost import CatBoostClassifier, CatBoostRegressor

        features = config["features"]
        categorical = config.get("categorical_features", [])
        frame = pd.DataFrame(rows)
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing)}")
        for name in categorical:
            frame[name] = frame[name].fillna("<missing>").astype(str)
        model = (
            CatBoostClassifier()
            if task_type == "classification"
            else CatBoostRegressor()
        )
        model.load_model(artifact)
        predictions = model.predict(frame[features]).reshape(-1)
        probabilities = (
            model.predict_proba(frame[features])
            if task_type == "classification"
            else None
        )
    elif task_type == "text_classification":
        import pickle

        from scipy.sparse import hstack

        # Artifacts are created and loaded inside the same local AnyLearning
        # project. The app has no model-upload endpoint, so untrusted pickle
        # input is outside this path's trust boundary.
        with artifact.open("rb") as stream:
            bundle = pickle.load(stream)
        text_column = bundle["text_column"]
        texts = [str(row.get(text_column, "")) for row in rows]
        vectorizers = bundle.get("vectorizers")
        if vectorizers is None:
            # Earlier releases used fixed word + character keys. Keep those
            # local artifacts predictable after an upgrade.
            vectorizers = [("word", bundle["word"]), ("character", bundle["char"])]
        matrix = hstack([vectorizer.transform(texts) for _, vectorizer in vectorizers])
        predictions = bundle["model"].predict(matrix)
        probabilities = bundle["model"].predict_proba(matrix)
    else:
        raise ValueError(f"Models cannot predict task type {task_type}")
    result = []
    for index, prediction in enumerate(predictions):
        item = {"prediction": json_value(prediction)}
        if probabilities is not None:
            values = probabilities[index].tolist()
            item["confidence"] = round(float(max(values)), 6)
            item["probabilities"] = [round(float(value), 6) for value in values]
        result.append(item)
    return result


def token_f1(reference: str, response: str) -> float:
    reference_tokens = re.findall(r"\w+", reference.casefold())
    response_tokens = re.findall(r"\w+", response.casefold())
    if not reference_tokens and not response_tokens:
        return 1.0
    if not reference_tokens or not response_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(response_tokens)).values())
    precision = overlap / len(response_tokens)
    recall = overlap / len(reference_tokens)
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def evaluate_llm_rows(
    frame, prompt_column: str, response_column: str, reference_column: str | None
) -> dict[str, Any]:
    return evaluate_llm_batches(
        [frame], prompt_column, response_column, reference_column
    )


def evaluate_llm_batches(
    batches,
    prompt_column: str,
    response_column: str,
    reference_column: str | None,
    detail_limit: int = 10_000,
) -> dict[str, Any]:
    """Aggregate every row while retaining only a bounded audit sample."""
    import pandas as pd

    details = []
    total = nonempty = response_words = 0
    token_f1_total = 0.0
    exact_total = 0
    for frame in batches:
        for row_id, row in frame.iterrows():
            response_value = row.get(response_column)
            reference_value = row.get(reference_column) if reference_column else None
            response = "" if pd.isna(response_value) else str(response_value)
            reference = (
                ""
                if reference_column is None or pd.isna(reference_value)
                else str(reference_value)
            )
            word_count = len(re.findall(r"\w+", response))
            score = token_f1(reference, response) if reference_column else None
            exact = (
                normalize_cell(reference) == normalize_cell(response)
                if reference_column
                else None
            )
            item = {
                "row_id": int(row_id),
                "prompt": json_value(row.get(prompt_column)),
                "response": response,
                "reference": reference if reference_column else None,
                "response_words": word_count,
                "empty": not bool(response.strip()),
                "token_f1": round(score, 6) if score is not None else None,
                "exact_match": exact,
            }
            total += 1
            nonempty += int(not item["empty"])
            response_words += word_count
            token_f1_total += score or 0.0
            exact_total += int(bool(exact))
            if len(details) < detail_limit:
                details.append(item)
    metrics = {
        "rows": total,
        "completion_rate": round(nonempty / max(1, total), 6),
        "average_response_words": round(response_words / max(1, total), 3),
    }
    if reference_column:
        metrics["token_f1"] = round(token_f1_total / max(1, total), 6)
        metrics["exact_match"] = round(exact_total / max(1, total), 6)
    return {
        "version": 1,
        "engine": "streaming lexical evaluation",
        "metrics": metrics,
        "rows": details,
        "detail_row_limit": detail_limit,
        "rows_truncated": total > len(details),
    }


def lexical_search_batches(
    batches, text_column: str, query: str, limit: int
) -> dict[str, Any]:
    """Bounded-memory character n-gram search over arbitrarily many rows."""
    import heapq

    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
    )
    query_vector = vectorizer.transform([query])
    best: list[tuple[float, int, str]] = []
    scanned = 0
    for frame in batches:
        texts = frame[text_column].fillna("").astype(str).tolist()
        scores = (vectorizer.transform(texts) @ query_vector.T).toarray().reshape(-1)
        for position, score in enumerate(scores):
            row_id = int(frame.index[position])
            candidate = (float(score), -row_id, texts[position])
            if len(best) < limit:
                heapq.heappush(best, candidate)
            elif candidate[:2] > best[0][:2]:
                heapq.heapreplace(best, candidate)
        scanned += len(frame)
    ordered = sorted(best, key=lambda item: (-item[0], -item[1]))
    return {
        "engine": "streaming character hashing",
        "rows_scanned": scanned,
        "results": [
            {"row_id": -negative_row_id, "score": round(score, 6), "text": text}
            for score, negative_row_id, text in ordered
        ],
    }
