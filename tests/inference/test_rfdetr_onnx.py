import hashlib
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
from anylearning.inference.backends.rfdetr_onnx import (
    RfDetrOnnxBackend,
    RfDetrOnnxConfig,
    _prepare_image,
    _top_indices,
)
from anylearning.inference.defaults import create_default_registry
from anylearning.inference.runtime import SessionState


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_graph(
    path: Path,
    *,
    boxes: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray | None = None,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    opset: int = 17,
) -> None:
    initializers = [
        numpy_helper.from_array(boxes.astype(np.float32), name="stored_boxes"),
        numpy_helper.from_array(logits.astype(np.float32), name="stored_logits"),
    ]
    nodes = [
        helper.make_node("Identity", ["stored_boxes"], ["dets"]),
        helper.make_node("Identity", ["stored_logits"], ["labels"]),
    ]
    outputs = [
        helper.make_tensor_value_info("dets", TensorProto.FLOAT, boxes.shape),
        helper.make_tensor_value_info("labels", TensorProto.FLOAT, logits.shape),
    ]
    if masks is not None:
        initializers.append(
            numpy_helper.from_array(masks.astype(np.float32), name="stored_masks")
        )
        nodes.append(helper.make_node("Identity", ["stored_masks"], ["masks"]))
        outputs.append(
            helper.make_tensor_value_info("masks", TensorProto.FLOAT, masks.shape)
        )
    graph = helper.make_graph(
        nodes,
        "rfdetr-contract-fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def _detection_graph(path: Path) -> None:
    _write_graph(
        path,
        boxes=np.asarray(
            [[[0.5, 0.5, 0.5, 0.4], [0.25, 0.25, 0.2, 0.2]]],
            dtype=np.float32,
        ),
        logits=np.asarray(
            [[[5.0, 2.0, -5.0], [-2.0, 4.0, -5.0]]],
            dtype=np.float32,
        ),
    )


def _config(path: Path, **updates):
    config = {
        "name": "rfdetr-fixture",
        "model_path": path,
        "sha256": _sha256(path),
        "class_names": ["cat", "dog"],
        "background_class_id": -1,
        "max_detections": 10,
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


def test_real_onnx_detection_graph_runs_through_lifecycle(tmp_path):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    session = RfDetrOnnxBackend().create_session(_config(graph))

    session.load()
    result = session.predict(
        _request(session),
        np.zeros((100, 200, 3), dtype=np.uint8),
    )

    assert session.state is SessionState.READY
    assert [shape.label for shape in result.shapes] == ["cat", "dog"]
    assert [shape.attributes["class_id"] for shape in result.shapes] == [0, 1]
    assert result.shapes[0].score == pytest.approx(0.993307, abs=1e-6)
    assert np.asarray(
        [(point.x, point.y) for point in result.shapes[0].points]
    ) == pytest.approx(np.asarray([(50, 30), (150, 70)]), abs=1e-5)
    assert np.asarray(
        [(point.x, point.y) for point in result.shapes[1].points]
    ) == pytest.approx(np.asarray([(30, 15), (70, 35)]), abs=1e-5)
    assert set(result.timings_ms) == {"preprocess", "inference", "postprocess", "total"}
    session.unload()
    assert session.state is SessionState.CLOSED


def test_multiclass_selection_matches_query_count_before_threshold(tmp_path):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    session = RfDetrOnnxBackend().create_session(_config(graph))
    session.load()

    result = session.predict(
        _request(session, parameters={"confidence": 0.8, "class_ids": [1]}),
        np.zeros((100, 200, 3), dtype=np.uint8),
    )

    assert [shape.label for shape in result.shapes] == ["dog", "dog"]
    assert result.shapes[0].score > result.shapes[1].score
    session.unload()


def test_real_onnx_segmentation_graph_returns_editable_instance_polygons(tmp_path):
    graph = tmp_path / "rfdetr-seg.onnx"
    masks = np.full((1, 1, 8, 8), -10, dtype=np.float32)
    masks[0, 0, 2:7, 1:4] = 10
    masks[0, 0, 3:6, 5:7] = 10
    _write_graph(
        graph,
        boxes=np.asarray([[[0.5, 0.5, 0.8, 0.8]]], dtype=np.float32),
        logits=np.asarray([[[6.0, -6.0]]], dtype=np.float32),
        masks=masks,
    )
    session = RfDetrOnnxBackend().create_session(
        _config(graph, task="instance_segmentation", class_names=["dog"])
    )
    session.load()

    result = session.predict(
        _request(session),
        np.zeros((80, 120, 3), dtype=np.uint8),
    )

    assert len(result.shapes) == 2
    assert {shape.type for shape in result.shapes} == {ShapeType.POLYGON}
    assert {shape.label for shape in result.shapes} == {"dog"}
    assert {shape.group_id for shape in result.shapes} == {0}
    assert all(len(shape.points) >= 3 for shape in result.shapes)
    assert all(
        0 <= point.x <= 120 and 0 <= point.y <= 80
        for shape in result.shapes
        for point in shape.points
    )

    rectangle = session.predict(
        _request(session, request_id="rectangle", output_shape=ShapeType.RECTANGLE),
        np.zeros((80, 120, 3), dtype=np.uint8),
    )
    assert len(rectangle.shapes) == 1
    assert rectangle.shapes[0].type is ShapeType.RECTANGLE
    session.unload()


def test_config_supports_sparse_exported_class_slots(tmp_path):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    config = _config(
        graph,
        class_names=["cat", None, "dog"],
        background_class_id=None,
    )
    session = RfDetrOnnxBackend().create_session(config)
    session.load()
    result = session.predict(
        _request(session),
        np.zeros((100, 200, 3), dtype=np.uint8),
    )
    assert [shape.label for shape in result.shapes] == ["cat"]
    session.unload()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"supported_opsets": [18]}, "opset"),
        ({"class_names": ["cat", "dog", "background"]}, "background_class_id"),
        ({"max_queries": 1}, "query count"),
        ({"max_classes": 2}, "class count"),
        ({"max_output_elements": 10}, "output elements"),
    ],
)
def test_load_rejects_graphs_outside_explicit_contract(tmp_path, updates, message):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    session = RfDetrOnnxBackend().create_session(_config(graph, **updates))

    with pytest.raises(ValueError, match=message):
        session.load()
    assert session.state is SessionState.FAILED


def test_predict_rejects_prompts_unknown_parameters_and_invalid_images(tmp_path):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    session = RfDetrOnnxBackend().create_session(_config(graph))
    session.load()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not accept"):
        session.predict(
            _request(session, prompts=(PointPrompt(point=Point(x=1, y=1)),)),
            image,
        )
    with pytest.raises(ValueError, match="Unsupported"):
        session.predict(_request(session, parameters={"iou": 0.5}), image)
    with pytest.raises(ValueError, match="uint8 RGB"):
        session.predict(_request(session), image.astype(np.float32))
    session.unload()


def test_preprocessing_uses_float_resize_without_uint8_quantization():
    image = np.asarray(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    tensor, height, width = _prepare_image(
        image,
        input_height=3,
        input_width=3,
        max_image_pixels=100,
    )

    assert tensor.shape == (1, 3, 3, 3)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert (height, width) == (2, 2)
    # The center is the exact bilinear average before normalization. A uint8
    # resize would round 55/255 and shift all three normalized channels.
    expected_center = (
        np.asarray([45, 55, 65], dtype=np.float32) / 255
        - np.asarray([0.485, 0.456, 0.406])
    ) / np.asarray([0.229, 0.224, 0.225])
    assert tensor[0, :, 1, 1] == pytest.approx(expected_center, abs=1e-6)


def test_top_selection_is_deterministic_at_tied_cutoff():
    scores = np.asarray([0.9, 0.8, 0.8, 0.8, 0.1], dtype=np.float32)
    assert _top_indices(scores, 3).tolist() == [0, 1, 2]


def test_default_registry_exposes_backend_without_eager_training_import(tmp_path):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    registry = create_default_registry()

    capabilities = registry.get("rfdetr_onnx").capabilities(_config(graph))

    assert capabilities.metadata["backend"] == "rfdetr_onnx"
    assert capabilities.tasks == (ModelTask.DETECTION,)


def test_config_is_frozen_and_integrity_map_is_immutable(tmp_path):
    graph = tmp_path / "rfdetr.onnx"
    _detection_graph(graph)
    config = RfDetrOnnxConfig.model_validate(
        _config(
            graph,
            external_data_sha256={"weights.bin": "0" * 64},
        )
    )

    with pytest.raises(TypeError, match="immutable"):
        config.external_data_sha256["weights.bin"] = "1" * 64
