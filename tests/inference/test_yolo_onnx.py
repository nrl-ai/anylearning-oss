import hashlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from anylearning.inference import InferenceRequest, ModelTask, SessionState
from anylearning.inference.backends.onnx_safety import (
    OnnxArtifactError,
    _iter_tensor_protos,
    select_providers,
    validate_onnx_artifact,
)
from anylearning.inference.backends.yolo_onnx import (
    DecodedDetection,
    YoloOnnxBackend,
    YoloOnnxConfig,
    _prepare_yolox_image,
    _request_options,
    decode_end_to_end_yolo_tensor,
    decode_yolo_tensor,
    decode_yolox_tensor,
    non_maximum_suppression,
    normalize_yolo_tensor,
)


def _constant_model(path, outputs, *, input_shape=(1, 3, 32, 32)):
    graph_outputs = []
    nodes = []
    for name, value in outputs.items():
        value = np.asarray(value, dtype=np.float32)
        graph_outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
        )
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[name],
                value=numpy_helper.from_array(value),
            )
        )
    graph = helper.make_graph(
        nodes,
        "bounded-yolo-fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)],
        graph_outputs,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.save_model(model, path)


def _request(session, **parameters):
    return InferenceRequest(
        request_id="request-1",
        source_id="image-sha256:fixture",
        model_id=session.capabilities.model_id,
        model_revision=session.capabilities.model_revision,
        parameters=parameters,
    )


def test_v5_and_v8_layouts_decode_their_distinct_confidence_rules():
    # v5: xywh, objectness, two class probabilities
    v5 = np.asarray([[[10, 10, 4, 4, 0.5, 0.8, 0.2]]], dtype=np.float32)
    decoded_v5 = decode_yolo_tensor(
        v5,
        class_count=2,
        confidence=0.39,
        layout="yolov5",
    )
    assert len(decoded_v5) == 1
    assert decoded_v5[0].confidence == pytest.approx(0.4)
    assert decoded_v5[0].box == pytest.approx((8, 8, 12, 12))

    # v8/11 commonly exports channels first and has no objectness channel.
    v8 = np.asarray([[[10], [10], [4], [4], [0.8], [0.2]]], dtype=np.float32)
    decoded_v8 = decode_yolo_tensor(
        v8,
        class_count=2,
        confidence=0.5,
        layout="yolo11",
    )
    assert len(decoded_v8) == 1
    assert decoded_v8[0].confidence == pytest.approx(0.8)
    assert decoded_v8[0].class_id == 0


@pytest.mark.parametrize("layout", ["yolov9", "yolov10", "yolo11", "yolo12", "yolo26"])
def test_later_raw_yolo_exports_share_the_v8_plus_layout(layout):
    output = np.asarray([[[10], [10], [4], [4], [0.8], [0.2]]], dtype=np.float32)

    decoded = decode_yolo_tensor(
        output,
        class_count=2,
        confidence=0.5,
        layout=layout,
    )

    assert len(decoded) == 1
    assert decoded[0].box == pytest.approx((8, 8, 12, 12))
    assert decoded[0].class_id == 0


def test_end_to_end_yolo_output_decodes_boxes_classes_and_mask_coefficients():
    output = np.asarray(
        [
            [0, 1, 10, 11, 0.8, 1, 2.0, 3.0],
            [5, 5, 9, 9, 0.9, 0, 4.0, 5.0],
            [1, 1, 2, 2, 0.1, 1, 6.0, 7.0],
        ],
        dtype=np.float32,
    )[None]

    decoded = decode_end_to_end_yolo_tensor(
        output,
        class_count=2,
        confidence=0.5,
        class_ids=frozenset({1}),
        mask_dim=2,
    )

    assert len(decoded) == 1
    assert decoded[0].box == pytest.approx((0, 1, 10, 11))
    assert decoded[0].class_id == 1
    assert decoded[0].mask_coefficients == pytest.approx((2, 3))


def test_end_to_end_yolo_output_rejects_fractional_or_out_of_range_classes():
    fractional = np.asarray([[[0, 0, 10, 10, 0.9, 0.5]]], dtype=np.float32)
    with pytest.raises(ValueError, match="class IDs must be integers"):
        decode_end_to_end_yolo_tensor(fractional, class_count=2, confidence=0.25)

    outside = np.asarray([[[0, 0, 10, 10, 0.9, 2]]], dtype=np.float32)
    with pytest.raises(ValueError, match="outside configured classes"):
        decode_end_to_end_yolo_tensor(outside, class_count=2, confidence=0.25)


def test_yolox_grid_output_decodes_objectness_and_class_scores():
    # A 32x32 P5 profile has 4x4 + 2x2 + 1x1 = 21 prediction cells.
    output = np.zeros((1, 21, 7), dtype=np.float32)
    output[0, 0] = [0.5, 0.5, 0, 0, 0.8, 0.25, 0.75]

    decoded = decode_yolox_tensor(
        output,
        class_count=2,
        input_height=32,
        input_width=32,
        confidence=0.5,
    )

    assert len(decoded) == 1
    assert decoded[0].box == pytest.approx((0, 0, 8, 8))
    assert decoded[0].confidence == pytest.approx(0.6)
    assert decoded[0].class_id == 1


def test_yolox_decoder_rejects_grid_mismatch_and_unsafe_dimension_logits():
    with pytest.raises(ValueError, match="requires 21"):
        decode_yolox_tensor(
            np.zeros((1, 20, 6), dtype=np.float32),
            class_count=1,
            input_height=32,
            input_width=32,
            confidence=0.25,
        )

    unsafe = np.zeros((1, 21, 6), dtype=np.float32)
    unsafe[0, 0, 2] = 100
    with pytest.raises(ValueError, match="box dimensions exceed"):
        decode_yolox_tensor(
            unsafe,
            class_count=1,
            input_height=32,
            input_width=32,
            confidence=0.25,
        )


def test_yolox_preprocessing_is_top_left_bgr_and_unnormalized():
    rgb = np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    tensor, transform = _prepare_yolox_image(
        rgb,
        input_height=16,
        input_width=16,
        max_image_pixels=100,
        dtype=np.dtype(np.float32),
    )

    assert tensor.shape == (1, 3, 16, 16)
    assert tensor[0, :, 0, 0] == pytest.approx((30, 20, 10))
    assert transform.pad_x == 0
    assert transform.pad_y == 0


def test_yolox_defaults_to_reference_class_agnostic_nms_and_allows_override():
    config = YoloOnnxConfig(
        name="fixture",
        model_path="unused.onnx",
        format="yolox",
        class_names=("a", "b"),
    )
    request = InferenceRequest(
        request_id="request-1",
        source_id="image-sha256:fixture",
        model_id="fixture",
        model_revision="fixture",
    )

    assert _request_options(request, config)[3] is True
    assert (
        _request_options(
            request.model_copy(update={"parameters": {"agnostic_nms": False}}), config
        )[3]
        is False
    )


def test_layout_diagnostics_reject_ambiguous_malformed_and_oversized_outputs():
    with pytest.raises(ValueError, match=r"no matching layout.*shape=\(4, 4\)"):
        normalize_yolo_tensor(np.zeros((4, 4)), class_count=2)
    with pytest.raises(ValueError, match="ambiguous orientation"):
        normalize_yolo_tensor(np.zeros((6, 6)), class_count=2, layout="yolov8")
    with pytest.raises(ValueError, match="limit is 10"):
        normalize_yolo_tensor(
            np.zeros((1, 6, 2)), class_count=2, max_output_elements=10
        )
    malformed = np.asarray([[[1], [1], [1], [1], [np.nan], [0.2]]])
    with pytest.raises(ValueError, match="NaN or infinity"):
        normalize_yolo_tensor(malformed, class_count=2)


def test_decoder_caps_candidates_before_nms_work():
    rows = np.asarray(
        [
            [10, 10, 4, 4, 0.40, 0.10],
            [20, 20, 4, 4, 0.90, 0.10],
            [30, 30, 4, 4, 0.80, 0.10],
            [40, 40, 4, 4, 0.70, 0.10],
        ],
        dtype=np.float32,
    )

    decoded = decode_yolo_tensor(
        rows,
        class_count=2,
        confidence=0,
        layout="yolov8",
        max_candidates=2,
    )

    assert [item.source_index for item in decoded] == [1, 2]


def test_nms_is_class_aware_bounded_and_deterministic():
    detections = [
        DecodedDetection((0, 0, 10, 10), 0.9, 0, 0),
        DecodedDetection((1, 1, 11, 11), 0.8, 0, 1),
        DecodedDetection((1, 1, 11, 11), 0.8, 1, 2),
    ]

    class_aware = non_maximum_suppression(
        detections, iou_threshold=0.5, max_detections=10
    )
    assert [(item.class_id, item.source_index) for item in class_aware] == [
        (0, 0),
        (1, 2),
    ]
    class_agnostic = non_maximum_suppression(
        detections,
        iou_threshold=0.5,
        max_detections=1,
        class_agnostic=True,
    )
    assert class_agnostic == [detections[0]]


def test_provider_selection_reports_fallback_and_can_fail_closed():
    selected, warnings = select_providers(
        ("CUDAExecutionProvider",),
        ("CPUExecutionProvider",),
        allow_cpu_fallback=True,
    )
    assert selected == ("CPUExecutionProvider",)
    assert warnings == (
        "Unavailable ONNX Runtime providers were skipped: CUDAExecutionProvider",
        "Fell back to CPUExecutionProvider",
    )

    with pytest.raises(RuntimeError, match="None of the requested"):
        select_providers(
            ("CUDAExecutionProvider",),
            ("CPUExecutionProvider",),
            allow_cpu_fallback=False,
        )
    with pytest.raises(ValueError, match="Invalid ONNX Runtime provider"):
        select_providers(
            ("../../plugin",),
            ("CPUExecutionProvider",),
            allow_cpu_fallback=True,
        )


def test_external_data_is_rejected_without_resolving_its_path(tmp_path):
    external = numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="weight")
    external.ClearField("raw_data")
    external.data_location = TensorProto.EXTERNAL
    entry = external.external_data.add()
    entry.key = "location"
    entry.value = "../../should-never-be-read.bin"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "external-data",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=[external],
    )
    model = helper.make_model(graph)
    path = tmp_path / "external.onnx"
    path.write_bytes(model.SerializeToString())

    with pytest.raises(OnnxArtifactError, match="External-data ONNX models"):
        validate_onnx_artifact(path, max_bytes=1024 * 1024)


def test_external_data_in_a_tensor_attribute_is_also_rejected(tmp_path):
    external = numpy_helper.from_array(np.ones((1,), dtype=np.float32))
    external.ClearField("raw_data")
    external.data_location = TensorProto.EXTERNAL
    entry = external.external_data.add()
    entry.key = "location"
    entry.value = "nested/weights.bin"
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=external)],
        "external-attribute",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "external-attribute.onnx"
    path.write_bytes(helper.make_model(graph).SerializeToString())

    with pytest.raises(OnnxArtifactError, match="External-data ONNX models") as error:
        validate_onnx_artifact(path, max_bytes=1024 * 1024)
    assert "nested/weights.bin" not in str(error.value)


def test_nested_graph_walk_has_a_depth_budget():
    model = helper.make_model(
        helper.make_graph(
            [],
            "depth-fixture",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
    )

    with pytest.raises(OnnxArtifactError, match="nesting-depth limit"):
        list(_iter_tensor_protos(model, max_depth=0))


def test_detection_session_preserves_identity_filters_and_letterbox_geometry(tmp_path):
    predictions = np.asarray(
        [
            [16, 17, 4],
            [16, 17, 12],
            [16, 16, 4],
            [16, 16, 4],
            [0.90, 0.80, 0.10],
            [0.10, 0.20, 0.85],
        ],
        dtype=np.float32,
    )[None]
    path = tmp_path / "detector.onnx"
    _constant_model(path, {"predictions": predictions})
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    session = YoloOnnxBackend().create_session(
        {
            "name": "fixture-detector",
            "model_path": path,
            "sha256": sha256,
            "format": "yolov8",
            "class_names": ["cat", "dog"],
            "providers": ["MissingExecutionProvider"],
        }
    )
    assert session.capabilities.tasks == (ModelTask.DETECTION,)
    assert session.capabilities.model_revision == f"sha256:{sha256}"
    session.load()

    image = np.zeros((16, 32, 3), dtype=np.uint8)
    result = session.predict(_request(session), image)
    assert result.request_id == "request-1"
    assert [shape.label for shape in result.shapes] == ["cat", "dog"]
    assert result.shapes[0].points[0].x == pytest.approx(8)
    assert result.shapes[0].points[0].y == pytest.approx(0)
    assert result.shapes[0].points[1].x == pytest.approx(24)
    assert result.shapes[0].points[1].y == pytest.approx(16)
    assert "Fell back to CPUExecutionProvider" in result.warnings
    assert all(value >= 0 for value in result.timings_ms.values())

    filtered = session.predict(
        _request(session, class_names=("dog",), confidence=0.5), image
    )
    assert [shape.label for shape in filtered.shapes] == ["dog"]
    session.unload()
    assert session.state is SessionState.CLOSED
    assert session._session is None


def test_yolo26_end_to_end_session_skips_host_nms_and_preserves_geometry(tmp_path):
    predictions = np.asarray(
        [
            [4, 12, 28, 20, 0.9, 0],
            [8, 8, 24, 24, 0.8, 1],
            [0, 0, 2, 2, 0.1, 1],
        ],
        dtype=np.float32,
    )[None]
    path = tmp_path / "yolo26-end-to-end.onnx"
    _constant_model(path, {"predictions": predictions})
    session = YoloOnnxBackend().create_session(
        {
            "name": "yolo26-end-to-end",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolo26",
            "class_names": ["cat", "dog"],
        }
    )
    assert session.capabilities.metadata["end_to_end"] is True
    session.load()

    result = session.predict(
        _request(session, confidence=0.5, iou=0.2, agnostic_nms=True),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )

    assert [shape.label for shape in result.shapes] == ["cat", "dog"]
    assert result.shapes[0].points[0].x == pytest.approx(4)
    assert result.shapes[0].points[0].y == pytest.approx(12)
    assert result.shapes[0].points[1].x == pytest.approx(28)
    assert result.shapes[0].points[1].y == pytest.approx(20)
    assert any("Ignored request NMS settings" in item for item in result.warnings)


def test_later_end_to_end_default_can_be_disabled_for_raw_export(tmp_path):
    predictions = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    path = tmp_path / "yolo26-raw.onnx"
    _constant_model(path, {"predictions": predictions})
    session = YoloOnnxBackend().create_session(
        {
            "name": "yolo26-raw",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolo26",
            "end_to_end": False,
            "class_names": ["cat", "dog"],
        }
    )
    assert session.capabilities.metadata["end_to_end"] is False
    session.load()

    result = session.predict(_request(session), np.zeros((32, 32, 3), dtype=np.uint8))
    assert [shape.label for shape in result.shapes] == ["cat"]


def test_segmentation_session_decodes_editable_polygon(tmp_path):
    predictions = np.asarray(
        [[[16], [16], [16], [16], [0.9], [10.0], [0.0]]],
        dtype=np.float32,
    )
    prototypes = np.zeros((1, 2, 8, 8), dtype=np.float32)
    prototypes[:, 0, 1:7, 1:7] = 1
    path = tmp_path / "segmenter.onnx"
    _constant_model(
        path,
        {"predictions": predictions, "prototypes": prototypes},
    )
    session = YoloOnnxBackend().create_session(
        {
            "name": "fixture-segmenter",
            "model_path": path,
            "model_revision": "fixture-1",
            "task": "instance_segmentation",
            "format": "yolov8",
            "class_names": ["object"],
        }
    )
    assert session.capabilities.tasks == (ModelTask.INSTANCE_SEGMENTATION,)
    session.load()

    result = session.predict(_request(session), np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(result.shapes) == 1
    assert result.shapes[0].type.value == "polygon"
    assert result.shapes[0].label == "object"
    assert result.shapes[0].attributes == {"class_id": 0}
    assert len(result.shapes[0].points) >= 3


def test_dynamic_input_requires_explicit_bounded_size(tmp_path):
    predictions = np.zeros((1, 6, 1), dtype=np.float32)
    path = tmp_path / "dynamic.onnx"
    _constant_model(path, {"predictions": predictions}, input_shape=(1, 3, "h", "w"))
    session = YoloOnnxBackend().create_session(
        {
            "name": "dynamic",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        }
    )

    with pytest.raises(ValueError, match="Dynamic YOLO inputs require"):
        session.load()
    assert session.state is SessionState.FAILED


def test_static_output_limit_is_enforced_before_first_inference(tmp_path):
    path = tmp_path / "oversized-output.onnx"
    _constant_model(path, {"predictions": np.zeros((1, 6, 1))})
    session = YoloOnnxBackend().create_session(
        {
            "name": "oversized-output",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolov8",
            "class_names": ["cat", "dog"],
            "max_output_elements": 5,
        }
    )

    with pytest.raises(ValueError, match="Declared ONNX outputs contain 6"):
        session.load()
    assert session.state is SessionState.FAILED


def test_sha256_mismatch_fails_before_runtime_load(tmp_path):
    path = tmp_path / "model.onnx"
    _constant_model(path, {"predictions": np.zeros((1, 6, 1))})
    session = YoloOnnxBackend().create_session(
        {
            "name": "mismatch",
            "model_path": path,
            "sha256": "0" * 64,
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        }
    )

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        session.load()
    assert session.state is SessionState.FAILED


def test_runtime_load_uses_the_same_artifact_that_was_verified(tmp_path, monkeypatch):
    original = tmp_path / "model.onnx"
    replacement = tmp_path / "replacement.onnx"
    cat = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    dog = np.asarray([[[16], [16], [8], [8], [0.1], [0.9]]], dtype=np.float32)
    _constant_model(original, {"predictions": cat})
    _constant_model(replacement, {"predictions": dog})
    sha256 = hashlib.sha256(original.read_bytes()).hexdigest()

    import anylearning.inference.backends.yolo_onnx as backend_module

    real_validate = backend_module.validate_onnx_artifact

    def replace_after_validation(artifact, *, max_bytes):
        model = real_validate(artifact, max_bytes=max_bytes)
        replacement.replace(original)
        return model

    monkeypatch.setattr(
        backend_module, "validate_onnx_artifact", replace_after_validation
    )
    session = YoloOnnxBackend().create_session(
        {
            "name": "stable-artifact",
            "model_path": original,
            "sha256": sha256,
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        }
    )
    session.load()

    result = session.predict(_request(session), np.zeros((32, 32, 3), dtype=np.uint8))
    assert [shape.label for shape in result.shapes] == ["cat"]


def test_in_place_model_mutation_fails_the_load(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    replacement = tmp_path / "replacement.onnx"
    predictions = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    _constant_model(path, {"predictions": predictions})
    _constant_model(replacement, {"predictions": predictions})

    import anylearning.inference.backends.yolo_onnx as backend_module

    real_validate = backend_module.validate_onnx_artifact

    def mutate_after_validation(artifact, *, max_bytes):
        model = real_validate(artifact, max_bytes=max_bytes)
        path.write_bytes(replacement.read_bytes())
        return model

    monkeypatch.setattr(
        backend_module, "validate_onnx_artifact", mutate_after_validation
    )
    session = YoloOnnxBackend().create_session(
        {
            "name": "mutated-artifact",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        }
    )

    with pytest.raises(OnnxArtifactError, match="changed while it was being loaded"):
        session.load()


def test_symbolic_outputs_are_rejected_before_inference_by_default(tmp_path):
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 32, 32]
    )
    shape = numpy_helper.from_array(
        np.asarray([1, 6, -1], dtype=np.int64), name="shape"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["images", "shape"], ["predictions"]),
        ],
        "dynamic-output",
        [input_info],
        [
            helper.make_tensor_value_info(
                "predictions", TensorProto.FLOAT, [1, 6, "predictions"]
            )
        ],
        initializer=[shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = tmp_path / "dynamic-output.onnx"
    onnx.save_model(model, path)
    session = YoloOnnxBackend().create_session(
        {
            "name": "dynamic-output",
            "model_path": path,
            "model_revision": "fixture-1",
            "format": "yolov8",
            "class_names": ["cat", "dog"],
        }
    )

    with pytest.raises(ValueError, match="outputs must have static dimensions"):
        session.load()


def test_extreme_coordinates_are_rejected_before_nms():
    predictions = np.asarray(
        [[[1_000_001], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32
    )

    with pytest.raises(ValueError, match="coordinates exceed"):
        decode_yolo_tensor(
            predictions,
            class_count=2,
            confidence=0.25,
            layout="yolov8",
        )


def test_repeated_yolo_lifecycle_releases_each_runtime_session(tmp_path):
    predictions = np.asarray([[[16], [16], [8], [8], [0.9], [0.1]]], dtype=np.float32)
    path = tmp_path / "lifecycle.onnx"
    _constant_model(path, {"predictions": predictions})
    config = {
        "name": "lifecycle",
        "model_path": path,
        "model_revision": "fixture-1",
        "format": "yolov8",
        "class_names": ["cat", "dog"],
    }
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    for index in range(25):
        session = YoloOnnxBackend().create_session(config)
        session.load()
        result = session.predict(
            InferenceRequest(
                request_id=f"lifecycle-{index}",
                source_id=f"image-sha256:lifecycle-{index}",
                model_id=session.capabilities.model_id,
                model_revision=session.capabilities.model_revision,
            ),
            image,
        )
        assert [shape.label for shape in result.shapes] == ["cat"]
        session.unload()
        assert session.state is SessionState.CLOSED
        assert session._session is None
