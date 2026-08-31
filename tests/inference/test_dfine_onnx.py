import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from anylearning.inference import (
    InferenceRequest,
    ModelTask,
    Point,
    PointPrompt,
    ShapeType,
)
from anylearning.inference.backends.dfine_onnx import (
    DFineOnnxBackend,
    DFineOnnxConfig,
    _prepare_image,
)
from anylearning.inference.defaults import create_default_registry
from anylearning.inference.runtime import SessionState


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_graph(
    path: Path,
    *,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    sizes_shape: tuple[int, int] = (1, 2),
    opset: int = 16,
    labels_shape: tuple[int | str, ...] | None = None,
    node_domain: str = "",
) -> None:
    initializers = [
        numpy_helper.from_array(labels.astype(np.int64), name="stored_labels"),
        numpy_helper.from_array(boxes.astype(np.float32), name="stored_boxes"),
        numpy_helper.from_array(scores.astype(np.float32), name="stored_scores"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["stored_labels"], ["labels"], domain=node_domain
            ),
            helper.make_node(
                "Identity", ["stored_boxes"], ["boxes"], domain=node_domain
            ),
            helper.make_node(
                "Identity", ["stored_scores"], ["scores"], domain=node_domain
            ),
        ],
        "dfine-contract-fixture",
        [
            helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info(
                "orig_target_sizes", TensorProto.INT64, sizes_shape
            ),
        ],
        [
            helper.make_tensor_value_info(
                "labels", TensorProto.INT64, labels_shape or labels.shape
            ),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes.shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores.shape),
        ],
        initializer=initializers,
    )
    opsets = [helper.make_opsetid("", opset)]
    if node_domain:
        opsets.append(helper.make_opsetid(node_domain, 1))
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def _detection_graph(path: Path, *, labels: np.ndarray | None = None) -> None:
    _write_graph(
        path,
        labels=(
            labels if labels is not None else np.asarray([[1, 0, 2]], dtype=np.int64)
        ),
        boxes=np.asarray(
            [[[50, 20, 150, 80], [10, 10, 40, 40], [-5, 30, 250, 120]]],
            dtype=np.float32,
        ),
        scores=np.asarray([[0.8, 0.95, 0.2]], dtype=np.float32),
    )


def _config(path: Path, **updates):
    config = {
        "name": "dfine-fixture",
        "model_path": path,
        "sha256": _sha256(path),
        "class_names": ["cat", "dog", "car"],
        "max_detections": 3,
        "release_cpu_memory_on_unload": False,
    }
    config.update(updates)
    return config


def _request(session, **updates):
    values = {
        "request_id": "request",
        "source_id": "image-sha256:fixture",
        "model_id": session.capabilities.model_id,
        "model_revision": session.capabilities.model_revision,
    }
    values.update(updates)
    return InferenceRequest(**values)


def test_real_onnx_graph_runs_through_lifecycle_and_preserves_identity(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    session = DFineOnnxBackend().create_session(_config(graph))

    session.load()
    result = session.predict(
        _request(session),
        np.zeros((100, 200, 3), dtype=np.uint8),
    )

    assert session.state is SessionState.READY
    assert result.request_id == "request"
    assert result.source_id == "image-sha256:fixture"
    assert result.model_revision == session.capabilities.model_revision
    assert [shape.label for shape in result.shapes] == ["cat", "dog"]
    assert [shape.attributes["class_id"] for shape in result.shapes] == [0, 1]
    assert [shape.score for shape in result.shapes] == [0.95, 0.8]
    assert [(point.x, point.y) for point in result.shapes[0].points] == [
        (10, 10),
        (40, 40),
    ]
    assert set(result.timings_ms) == {"preprocess", "inference", "postprocess", "total"}
    session.unload()
    assert session.state is SessionState.CLOSED


def test_prediction_filters_classes_clips_boxes_and_recovers_after_bad_request(
    tmp_path,
):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    session = DFineOnnxBackend().create_session(_config(graph, confidence=0.1))
    session.load()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported"):
        session.predict(_request(session, parameters={"iou": 0.5}), image)

    result = session.predict(
        _request(
            session,
            request_id="recovered",
            parameters={"confidence": 0.1, "class_ids": [2]},
        ),
        image,
    )
    assert result.request_id == "recovered"
    assert [shape.label for shape in result.shapes] == ["car"]
    assert [(point.x, point.y) for point in result.shapes[0].points] == [
        (0, 30),
        (200, 100),
    ]
    session.unload()


def test_preprocessing_matches_native_rgb_stretch_and_original_size_order():
    image = np.asarray(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    tensor, sizes, height, width = _prepare_image(
        image,
        input_height=3,
        input_width=3,
        max_image_pixels=100,
    )

    assert tensor.shape == (1, 3, 3, 3)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert sizes.dtype == np.int64
    assert sizes.tolist() == [[2, 2]]
    assert (height, width) == (2, 2)
    assert tensor[0, :, 1, 1] == pytest.approx(
        np.asarray([45, 55, 65], dtype=np.float32) / 255,
        abs=1e-6,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"supported_opsets": [17]}, "opset"),
        ({"max_queries": 2}, "query count"),
        ({"max_output_elements": 17}, "output elements"),
    ],
)
def test_load_rejects_graphs_outside_explicit_bounds(tmp_path, updates, message):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    session = DFineOnnxBackend().create_session(_config(graph, **updates))

    with pytest.raises(ValueError, match=message):
        session.load()
    assert session.state is SessionState.FAILED


def test_load_rejects_symbolic_output_shapes_before_session_creation(tmp_path):
    graph = tmp_path / "dynamic.onnx"
    _write_graph(
        graph,
        labels=np.asarray([[0]], dtype=np.int64),
        boxes=np.asarray([[[1, 1, 2, 2]]], dtype=np.float32),
        scores=np.asarray([[0.9]], dtype=np.float32),
        labels_shape=(1, "queries"),
    )
    session = DFineOnnxBackend().create_session(_config(graph))

    with pytest.raises(ValueError, match="positive static shape"):
        session.load()


def test_load_rejects_custom_operator_domains_before_session_creation(tmp_path):
    graph = tmp_path / "custom.onnx"
    _write_graph(
        graph,
        labels=np.asarray([[0]], dtype=np.int64),
        boxes=np.asarray([[[1, 1, 2, 2]]], dtype=np.float32),
        scores=np.asarray([[0.9]], dtype=np.float32),
        node_domain="untrusted.custom",
    )
    session = DFineOnnxBackend().create_session(_config(graph))

    with pytest.raises(ValueError, match="custom operator domains"):
        session.load()


def test_predict_rejects_prompts_output_types_and_invalid_images(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    session = DFineOnnxBackend().create_session(_config(graph))
    session.load()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not accept"):
        session.predict(
            _request(session, prompts=(PointPrompt(point=Point(x=1, y=1)),)),
            image,
        )
    with pytest.raises(ValueError, match="rectangle"):
        session.predict(_request(session, output_shape=ShapeType.POLYGON), image)
    with pytest.raises(ValueError, match="uint8 RGB"):
        session.predict(_request(session), image.astype(np.float32))
    session.unload()


def test_runtime_rejects_unknown_label_slots(tmp_path):
    graph = tmp_path / "bad-label.onnx"
    _detection_graph(graph, labels=np.asarray([[3, 0, 1]], dtype=np.int64))
    session = DFineOnnxBackend().create_session(_config(graph))
    session.load()

    with pytest.raises(ValueError, match="unknown class index"):
        session.predict(
            _request(session),
            np.zeros((100, 200, 3), dtype=np.uint8),
        )
    session.unload()


def test_runtime_rejects_non_finite_outputs(tmp_path):
    graph = tmp_path / "non-finite.onnx"
    _write_graph(
        graph,
        labels=np.asarray([[0]], dtype=np.int64),
        boxes=np.asarray([[[1, 1, 2, 2]]], dtype=np.float32),
        scores=np.asarray([[np.nan]], dtype=np.float32),
    )
    session = DFineOnnxBackend().create_session(_config(graph))
    session.load()

    with pytest.raises(ValueError, match="NaN or infinity"):
        session.predict(
            _request(session),
            np.zeros((100, 200, 3), dtype=np.uint8),
        )
    session.unload()


def test_image_and_class_filters_fail_within_configured_bounds(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    session = DFineOnnxBackend().create_session(_config(graph, max_image_pixels=100))
    session.load()

    with pytest.raises(ValueError, match="configured limit"):
        session.predict(
            _request(session),
            np.zeros((11, 10, 3), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="unknown class"):
        session.predict(
            _request(session, parameters={"class_ids": [3]}),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
    session.unload()


def test_max_shapes_is_enforced_independently_of_requested_maximum(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    session = DFineOnnxBackend().create_session(
        _config(graph, max_shapes=1, confidence=0.1)
    )
    session.load()

    with pytest.raises(ValueError, match="max_shapes"):
        session.predict(
            _request(session),
            np.zeros((100, 200, 3), dtype=np.uint8),
        )
    session.unload()


def test_provider_is_explicitly_cpu_only_until_accelerator_validation(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)

    with pytest.raises(ValueError, match="CPUExecutionProvider"):
        DFineOnnxConfig.model_validate(
            _config(graph, providers=["CUDAExecutionProvider"])
        )


def test_default_registry_exposes_backend_lazily(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    registry = create_default_registry()

    capabilities = registry.get("dfine_onnx").capabilities(_config(graph))

    assert capabilities.metadata["backend"] == "dfine_onnx"
    assert capabilities.metadata["preprocessing"] == "rgb-stretch"
    assert capabilities.tasks == (ModelTask.DETECTION,)


def test_backend_import_does_not_require_training_frameworks():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "sys.modules['torchvision'] = None; "
            "import anylearning.inference.backends.dfine_onnx",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_config_is_frozen_and_integrity_map_is_immutable(tmp_path):
    graph = tmp_path / "dfine.onnx"
    _detection_graph(graph)
    config = DFineOnnxConfig.model_validate(
        _config(graph, external_data_sha256={"weights.bin": "0" * 64})
    )

    with pytest.raises(TypeError, match="immutable"):
        config.external_data_sha256["weights.bin"] = "1" * 64
