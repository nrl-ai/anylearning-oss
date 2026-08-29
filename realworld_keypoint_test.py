#!/usr/bin/env python3
"""Train packaged RF-DETR keypoints on prepared real-world COCO archives.

Use ``prepare_vertebral_keypoints.py`` to create the two input archives. This
harness drives the packaged application over HTTP, including project creation,
COCO import, training, ONNX-gated registration, and inference on a held-out
validation image.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import shutil
import tempfile
import time
import zipfile

import feature_test


def _image_from_archive(path: pathlib.Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        candidates = sorted(
            name
            for name in archive.namelist()
            if pathlib.PurePosixPath(name).suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )
        if not candidates:
            raise ValueError(f"{path} contains no image.")
        name = candidates[0]
        return pathlib.PurePosixPath(name).name, archive.read(name)


def _wait_for_training(
    app: feature_test.App, project_id: int, session_id: int, budget: int
) -> dict:
    deadline = time.time() + budget
    detail = {}
    while time.time() < deadline:
        detail = app.get(
            f"/api/projects/{project_id}/training_sessions/{session_id}", timeout=180
        ).ok("training session")
        status = detail.get("status")
        if status in {"finished", "error", "terminated"}:
            return detail
        logs = (detail.get("training_logs") or "").strip().splitlines()
        print(f"  {status}: {logs[-1] if logs else 'waiting'}", flush=True)
        time.sleep(10)
    raise TimeoutError(f"Training did not finish within {budget} seconds.")


def _save_preview(reply: dict, path: pathlib.Path) -> None:
    value = reply.get("visualization_image", "")
    if not value.startswith("data:image/") or "," not in value:
        raise ValueError("Inference returned no visualization image.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(value.split(",", 1)[1]))


def run(args) -> None:
    data_root = tempfile.mkdtemp(prefix="anylearning-real-keypoints-")
    log_path = str(pathlib.Path(data_root) / "app.log")
    port = feature_test.free_port()
    base = f"http://127.0.0.1:{port}"
    process = feature_test.start_binary(str(args.binary), port, data_root, log_path)
    app = feature_test.App(base, data_root, log_path)

    try:
        feature_test.wait_for_server(base, process)
        project_id = app.project("Real vertebral keypoints", "Keypoint Detection")
        print(f"project {project_id}: importing {args.train.name}", flush=True)
        train_status = app.upload_images(
            project_id,
            [(args.train.name, args.train.read_bytes())],
            subset=0,
        )
        print(f"  {train_status}", flush=True)
        print(f"project {project_id}: importing {args.valid.name}", flush=True)
        valid_status = app.upload_images(
            project_id,
            [(args.valid.name, args.valid.read_bytes())],
            subset=1,
        )
        print(f"  {valid_status}", flush=True)

        project = app.get(f"/api/projects/{project_id}").ok("project")
        labels = [label["name"] for label in project.get("labels", [])]
        expected = list(args.keypoint_names)
        if labels != expected:
            raise ValueError(f"Imported labels are {labels}, expected {expected}.")
        train_count = len(app.items(project_id, subset=0, limit=10_000))
        valid_count = len(app.items(project_id, subset=1, limit=10_000))
        print(f"imported {train_count} train and {valid_count} valid images")

        params = {
            "model_architecture": "rfdetr-keypoint",
            "model_size": "preview",
            "model_variant": "RF-DETR-Keypoint-Preview",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": 0.0001,
            "pretrained_model": "default",
            "device": args.device,
            "image_size": args.image_size,
        }
        started = app.post(
            f"/api/projects/{project_id}/training_sessions", params, timeout=180
        ).ok("start training")
        session_id = started["session_id"]
        print(f"training session {session_id}", flush=True)
        detail = _wait_for_training(app, project_id, session_id, args.budget)
        if detail.get("status") != "finished":
            logs = (detail.get("training_logs") or "").strip().splitlines()
            raise RuntimeError(
                f"Training ended as {detail.get('status')}: {' | '.join(logs[-8:])}"
            )

        model_id = (detail.get("model") or {}).get("id")
        if not model_id:
            raise RuntimeError("Training finished without a registered model.")
        print("metrics:")
        print(json.dumps(detail.get("metric_logs"), indent=2, sort_keys=True))

        image_name, image = _image_from_archive(args.valid)
        inference = app.upload(
            f"/api/projects/{project_id}/models/{model_id}/inference",
            [("file", image_name, image)],
            timeout=600,
        ).ok("held-out inference")
        results = inference.get("results") or []
        visible = sum(
            point.get("visible", False)
            for instance in results
            for point in instance.get("keypoints", [])
        )
        print(
            f"model {model_id}: {len(results)} instances and {visible} visible "
            f"landmarks on {image_name}"
        )
        if args.preview:
            _save_preview(inference, args.preview)
            print(f"preview: {args.preview}")
    except BaseException:
        print("application log tail:")
        print(feature_test.tail(log_path))
        raise
    finally:
        if not args.keep:
            app.drop_projects()
        feature_test.stop_binary(process)
        if args.keep:
            print(f"kept data root: {data_root}")
        else:
            shutil.rmtree(data_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("train", type=pathlib.Path)
    parser.add_argument("valid", type=pathlib.Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--preview", type=pathlib.Path)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--keypoint-names",
        nargs=4,
        default=("top_left", "top_right", "bottom_left", "bottom_right"),
    )
    args = parser.parse_args()
    for path in (args.binary, args.train, args.valid):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    run(args)


if __name__ == "__main__":
    main()
