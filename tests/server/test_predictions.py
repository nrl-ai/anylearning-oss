import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from anylearning.inference import InferenceRequest
from anylearning.inference.backends.yolo_onnx import YoloOnnxBackend
from anylearning.server import (
    ServerModelDefinition,
    ServerSettings,
    create_server_app,
    encode_request_header,
    hash_password,
    load_server_model_manifest,
)
from anylearning.server.transport import encoded_image_source_id

PASSWORD = "correct horse battery staple"


def _constant_yolo(path):
    predictions = np.asarray(
        [[[16], [16], [8], [8], [0.9], [0.1]]],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["stored_predictions"], ["predictions"])],
        "server-real-onnx",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info(
                "predictions",
                TensorProto.FLOAT,
                list(predictions.shape),
            )
        ],
        initializer=[numpy_helper.from_array(predictions, name="stored_predictions")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save_model(model, path)


@pytest.fixture
def real_server(tmp_path):
    model_path = tmp_path / "detector.onnx"
    _constant_yolo(model_path)
    config = {
        "name": "server-real-onnx",
        "model_path": model_path,
        "model_revision": "fixture-1",
        "format": "yolov8",
        "class_names": ["cat", "dog"],
    }
    definition = ServerModelDefinition(backend="yolo_onnx", config=config)
    settings = ServerSettings(
        password_hash=hash_password(PASSWORD),
        token_secret=b"s" * 32,
        token_ttl_seconds=60,
        max_prediction_body_bytes=1024 * 1024,
        max_image_pixels=1024 * 1024,
        max_decoded_image_bytes=3 * 1024 * 1024,
        max_pending_image_bytes_per_model=6 * 1024 * 1024,
        prediction_timeout_seconds=10,
        prediction_result_ttl_seconds=60,
    )
    app = create_server_app(settings, model_definitions=(definition,))
    return app, YoloOnnxBackend().capabilities(config)


def _login(client):
    response = client.post("/v1/auth/token", json={"password": PASSWORD})
    assert response.status_code == 200
    return response.json()["access_token"]


def _png_image(tmp_path):
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[4:28, 4:28] = (20, 40, 60)
    path = tmp_path / "image.png"
    Image.fromarray(array, mode="RGB").save(path)
    return path.read_bytes(), array


def test_authenticated_prediction_runs_actual_onnx_and_isolates_token_jobs(
    real_server, tmp_path
):
    app, capabilities = real_server
    encoded, _image = _png_image(tmp_path)
    request = InferenceRequest(
        request_id="client-request-1",
        source_id=encoded_image_source_id(encoded),
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
    )
    with TestClient(app) as client:
        token = _login(client)
        response = client.post(
            "/v1/predictions",
            content=encoded,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "image/png",
                "X-AnyLearning-Request": encode_request_header(request),
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        assert submitted["request_id"] == request.request_id
        job_id = submitted["job_id"]
        assert request.request_id not in job_id

        other_token = _login(client)
        assert (
            client.get(
                f"/v1/predictions/{job_id}",
                headers={"Authorization": f"Bearer {other_token}"},
            ).status_code
            == 404
        )

        deadline = time.monotonic() + 3
        while True:
            completed = client.get(
                f"/v1/predictions/{job_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert completed.status_code == 200
            payload = completed.json()
            if payload["state"] == "succeeded":
                break
            assert payload["state"] in {"queued", "running"}
            assert time.monotonic() < deadline
            time.sleep(0.01)

        result = payload["result"]
        assert result["request_id"] == request.request_id
        assert result["source_id"] == request.source_id
        assert [shape["label"] for shape in result["shapes"]] == ["cat"]
        assert (
            client.delete(
                f"/v1/predictions/{job_id}",
                headers={"Authorization": f"Bearer {token}"},
            ).status_code
            == 204
        )


def test_prediction_authentication_precedes_body_read_and_source_must_match(
    real_server, tmp_path
):
    app, capabilities = real_server
    encoded, _image = _png_image(tmp_path)
    request = InferenceRequest(
        request_id="wrong-source",
        source_id="image-sha256:" + "0" * 64,
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
    )
    metadata = encode_request_header(request)
    with TestClient(app) as client:
        unauthenticated = client.post(
            "/v1/predictions",
            content=b"not-an-image" * 4096,
            headers={
                "Content-Type": "image/png",
                "X-AnyLearning-Request": metadata,
            },
        )
        assert unauthenticated.status_code == 401

        token = _login(client)
        mismatch = client.post(
            "/v1/predictions",
            content=encoded,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "image/png",
                "X-AnyLearning-Request": metadata,
            },
        )
        assert mismatch.status_code == 409
        assert mismatch.json()["detail"] == (
            "source_id does not match the encoded image"
        )


def test_prediction_upload_admission_is_bounded_before_body_read(
    real_server, tmp_path, monkeypatch
):
    app, capabilities = real_server
    encoded, _image = _png_image(tmp_path)
    settings = ServerSettings(
        password_hash=hash_password(PASSWORD),
        token_secret=b"s" * 32,
        token_ttl_seconds=60,
        max_prediction_body_bytes=1024 * 1024,
        max_image_pixels=1024 * 1024,
        max_decoded_image_bytes=3 * 1024 * 1024,
        max_pending_image_bytes_per_model=6 * 1024 * 1024,
        max_concurrent_prediction_requests=1,
        prediction_timeout_seconds=10,
        prediction_result_ttl_seconds=60,
    )
    definition = ServerModelDefinition(
        backend="yolo_onnx",
        config={
            "name": capabilities.model_id,
            "model_path": tmp_path / "detector.onnx",
            "model_revision": capabilities.model_revision,
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        },
    )
    _constant_yolo(definition.config["model_path"])
    app = create_server_app(settings, model_definitions=(definition,))

    first_body_started = threading.Event()
    release_first_body = threading.Event()
    body_reads = 0

    async def controlled_body_read(_request, _maximum):
        nonlocal body_reads
        body_reads += 1
        first_body_started.set()
        await asyncio.to_thread(release_first_body.wait, 3)
        return encoded

    monkeypatch.setattr(
        "anylearning.server.app._read_prediction_body",
        controlled_body_read,
    )
    first_request = InferenceRequest(
        request_id="admission-first",
        source_id=encoded_image_source_id(encoded),
        model_id=capabilities.model_id,
        model_revision=capabilities.model_revision,
    )
    second_request = first_request.model_copy(update={"request_id": "admission-second"})
    with TestClient(app) as client:
        token = _login(client)
        base_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "image/png",
        }
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                client.post,
                "/v1/predictions",
                content=encoded,
                headers={
                    **base_headers,
                    "X-AnyLearning-Request": encode_request_header(first_request),
                },
            )
            assert first_body_started.wait(3)
            second = client.post(
                "/v1/predictions",
                content=encoded,
                headers={
                    **base_headers,
                    "X-AnyLearning-Request": encode_request_header(second_request),
                },
            )
            assert second.status_code == 429
            assert second.json()["detail"] == "Prediction request capacity reached"
            assert body_reads == 1
            release_first_body.set()
            assert first.result(timeout=3).status_code == 202


def test_server_model_manifest_is_bounded_onnx_only_and_rejects_credentials(
    tmp_path,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "backend": "yolo_onnx",
                        "config": {
                            "name": "detector",
                            "model_path": "model.onnx",
                            "class_names": ["object"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    definitions = load_server_model_manifest(manifest)
    assert definitions[0].backend == "yolo_onnx"
    assert definitions[0].config["config_file"] == str(manifest.resolve())

    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "backend": "segment_anything",
                        "config": {
                            "name": "promptable",
                            "encoder_model_path": "encoder.onnx",
                            "decoder_model_path": "decoder.onnx",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    definitions = load_server_model_manifest(manifest)
    assert definitions[0].backend == "segment_anything"
    assert (
        ServerModelDefinition(
            backend="efficient_sam",
            config={
                "name": "efficient-promptable",
                "encoder_model_path": "encoder.onnx",
                "decoder_model_path": "decoder.onnx",
            },
        ).backend
        == "efficient_sam"
    )

    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {"backend": "torch_model", "config": {"name": "not-approved"}}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ONNX"):
        load_server_model_manifest(manifest)

    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "backend": "yolo_onnx",
                        "config": {"name": "detector", "api_token": "secret"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="credentials"):
        load_server_model_manifest(manifest)

    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "backend": "yolo_onnx",
                        "config": {"name": "detector", "confidence": float("inf")},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        load_server_model_manifest(manifest)

    link = tmp_path / "models-link.json"
    link.symlink_to(manifest)
    with pytest.raises(ValueError, match="non-link"):
        load_server_model_manifest(link)
