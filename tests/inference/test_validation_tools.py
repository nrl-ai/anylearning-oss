import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from anylearning.inference import InferenceRequest, TextPrompt
from anylearning.inference.backends.yolo_onnx import YoloOnnxBackend
from anylearning.inference.validation import (
    ValidationTextPrompt,
    _model_artifact_details,
    _request_prompts,
)

_ROOT = Path(__file__).resolve().parents[2]


def _script(name):
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _single_file_prediction_model(path):
    predictions = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["stored_predictions"], ["predictions"])],
        "conversion-fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info(
                "predictions", TensorProto.FLOAT, list(predictions.shape)
            )
        ],
        initializer=[numpy_helper.from_array(predictions, name="stored_predictions")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save_model(model, path)


def test_verified_downloader_uses_bounded_curl_and_atomic_digest_gate(
    tmp_path, monkeypatch
):
    module = _script("download_verified_file.py")
    content = b"verified model bytes"
    digest = hashlib.sha256(content).hexdigest()

    def fake_run(command, *, check, timeout):
        assert check
        assert timeout == 42
        assert command[command.index("--proto") + 1] == "=https"
        assert command[command.index("--max-filesize") + 1] == "1024"
        Path(command[command.index("--output") + 1]).write_bytes(content)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "model.onnx"
    module.download_verified_file(
        "https://models.example/model.onnx",
        output,
        expected_sha256=digest,
        max_bytes=1024,
        timeout_seconds=12,
    )
    assert output.read_bytes() == content
    assert not list(tmp_path.glob("*.part"))

    with pytest.raises(ValueError, match="HTTPS"):
        module.download_verified_file(
            "http://models.example/model.onnx",
            output,
            expected_sha256=digest,
            max_bytes=1024,
        )


def test_external_validation_converter_produces_loadable_real_onnx_bundle(tmp_path):
    module = _script("prepare_external_onnx_validation.py")
    source_model = tmp_path / "source.onnx"
    _single_file_prediction_model(source_model)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image path fixture")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "name": "conversion-fixture",
                "backend": "yolo_onnx",
                "provenance": {
                    "source_url": "https://example.com/source",
                    "artifact_url": "https://example.com/model.onnx",
                    "source_revision": "fixture",
                    "code_license": "Apache-2.0",
                    "artifact_license": "Apache-2.0",
                    "license_url": "https://example.com/license",
                },
                "config": {
                    "name": "conversion-fixture",
                    "model_path": source_model.name,
                    "sha256": hashlib.sha256(source_model.read_bytes()).hexdigest(),
                    "format": "yolov8",
                    "class_names": ["cat", "dog"],
                },
                "runs": 2,
                "images": [{"path": image.name}],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = module.prepare_external_validation_bundle(
        source_model, source_manifest, tmp_path / "external"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload["config"]["external_data_sha256"]) == {"weights.bin"}
    graph = onnx.load_model(
        manifest_path.parent / "model.onnx", load_external_data=False
    )
    assert any(
        tensor.data_location == TensorProto.EXTERNAL
        for tensor in graph.graph.initializer
    )

    config = dict(payload["config"])
    config["config_file"] = manifest_path
    session = YoloOnnxBackend().create_session(config)
    session.load()
    request = InferenceRequest(
        request_id="conversion-test",
        source_id="image-sha256:fixture",
        model_id=session.capabilities.model_id,
        model_revision=session.capabilities.model_revision,
    )
    result = session.predict(request, np.zeros((32, 32, 3), dtype=np.uint8))
    session.unload()
    assert [shape.label for shape in result.shapes] == ["cat"]


def test_validation_text_prompt_converts_to_shared_contract():
    prompts = _request_prompts((ValidationTextPrompt(type="text", text="dog"),))

    assert prompts == (TextPrompt(text="dog"),)


def test_validation_evidence_hashes_sam3_graph_triplet_and_external_data(tmp_path):
    config = {}
    expected_total = 0
    for role in ("image_encoder", "language_encoder", "decoder"):
        graph = tmp_path / f"{role}.onnx"
        graph.write_bytes(f"{role}-graph".encode())
        digest = hashlib.sha256(graph.read_bytes()).hexdigest()
        config[f"{role}_model_path"] = graph.name
        config[f"{role}_sha256"] = digest
        expected_total += graph.stat().st_size
        if role != "decoder":
            external = tmp_path / f"{role}.onnx.data"
            external.write_bytes(f"{role}-weights".encode())
            external_digest = hashlib.sha256(external.read_bytes()).hexdigest()
            config[f"{role}_external_data_sha256"] = {external.name: external_digest}
            expected_total += external.stat().st_size

    details = _model_artifact_details(config, tmp_path)

    assert [item["role"] for item in details["graphs"]] == [
        "image_encoder",
        "language_encoder",
        "decoder",
    ]
    assert details["bytes"] == expected_total
    assert details["graphs"][0]["external_files"][0]["location"] == (
        "image_encoder.onnx.data"
    )
