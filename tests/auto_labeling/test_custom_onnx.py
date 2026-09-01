import hashlib

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from anylearning.auto_labeling.custom_onnx import install_custom_yolo_onnx
from anylearning.auto_labeling.inference_model import InferenceModel


def _constant_yolo(path):
    predictions = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["predictions"],
                value=numpy_helper.from_array(predictions),
            )
        ],
        "custom-yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info(
                "predictions", TensorProto.FLOAT, list(predictions.shape)
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save_model(model, path)


def test_custom_onnx_is_copied_hashed_and_runs_through_desktop_adapter(tmp_path):
    source = tmp_path / "source.onnx"
    _constant_yolo(source)
    models = tmp_path / "installed"

    config = install_custom_yolo_onnx(
        source,
        {
            "display_name": "My YOLOv8 detector",
            "format": "yolov8",
            "task": "detection",
            "class_names": ["cat", "dog"],
        },
        models_root=models,
    )

    installed = models / config["name"] / "model.onnx"
    assert installed.read_bytes() == source.read_bytes()
    assert (
        config["inference_config"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    persisted = yaml.safe_load(
        (models / config["name"] / "config.yaml").read_text(encoding="utf-8")
    )
    assert persisted["backend"] == "yolo_onnx"
    assert persisted["inference_config"]["format"] == "yolov8"

    model = InferenceModel(config, lambda _message: None)
    result = model.predict_shapes(np.zeros((32, 32, 3), dtype=np.uint8))
    model.unload()
    assert [shape.label for shape in result.shapes] == ["cat"]

    renamed = install_custom_yolo_onnx(
        source,
        {
            "display_name": "Renamed detector",
            "format": "yolov8",
            "task": "detection",
            "class_names": ["cat", "dog"],
        },
        models_root=models,
    )
    assert renamed["name"] == config["name"]
    assert renamed["display_name"] == "Renamed detector"
    assert (
        yaml.safe_load(
            (models / config["name"] / "config.yaml").read_text(encoding="utf-8")
        )["display_name"]
        == "Renamed detector"
    )


def test_custom_onnx_rejects_a_redirected_install_directory(tmp_path):
    source = tmp_path / "source.onnx"
    _constant_yolo(source)
    models = tmp_path / "installed"
    first = install_custom_yolo_onnx(
        source,
        {"display_name": "First", "class_names": ["cat", "dog"]},
        models_root=models,
    )
    installed = models / first["name"]
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    for child in installed.iterdir():
        child.unlink()
    installed.rmdir()
    try:
        installed.symlink_to(redirected, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this host")

    with pytest.raises(ValueError, match="conflicting custom model"):
        install_custom_yolo_onnx(
            source,
            {"display_name": "Second", "class_names": ["cat", "dog"]},
            models_root=models,
        )
    assert list(redirected.iterdir()) == []


def test_custom_onnx_requires_an_unambiguous_label_source(tmp_path):
    source = tmp_path / "source.onnx"
    _constant_yolo(source)

    with pytest.raises(ValueError, match="label preset or an explicit class list"):
        install_custom_yolo_onnx(
            source,
            {
                "display_name": "Ambiguous",
                "label_space": "coco80",
                "class_names": ["cat"],
            },
            models_root=tmp_path / "installed",
        )


def test_custom_onnx_rejects_non_onnx_and_external_data(tmp_path):
    wrong = tmp_path / "model.bin"
    wrong.write_bytes(b"not an ONNX graph")
    with pytest.raises(ValueError, match=r"\.onnx"):
        install_custom_yolo_onnx(
            wrong,
            {"display_name": "Wrong", "class_names": ["cat"]},
            models_root=tmp_path / "installed",
        )

    external = tmp_path / "external.onnx"
    predictions = np.zeros((1, 6, 1), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["stored"], ["predictions"])],
        "external-yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 6, 1])],
        initializer=[numpy_helper.from_array(predictions, name="stored")],
    )
    onnx.save_model(
        helper.make_model(graph),
        external,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    with pytest.raises(ValueError, match="External-data ONNX models"):
        install_custom_yolo_onnx(
            external,
            {"display_name": "External", "class_names": ["cat"]},
            models_root=tmp_path / "external-installed",
        )
