import hashlib
import importlib.util
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from anylearning.inference import (
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    Point,
    ShapeType,
    TextPrompt,
)
from anylearning.inference.backends.yolo_onnx import YoloOnnxBackend
from anylearning.inference.validation import (
    ValidationTextPrompt,
    _caption_shape_indexes,
    _lifecycle_rss_metrics,
    _model_artifact_details,
    _prediction_digest,
    _request_prompts,
    load_validation_manifest,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_lifecycle_rss_metrics_separate_allocator_warmup_from_growth():
    metrics = _lifecycle_rss_metrics([100, 420, 421])

    assert metrics == {
        "warmup_retained_rss_growth_bytes": 320,
        "steady_state_rss_baseline_bytes": 420,
        "steady_state_rss_baseline_cycle": 2,
        "steady_state_rss_growth_bytes": 1,
    }


def test_two_cycle_lifecycle_rss_metrics_keep_conservative_cold_baseline():
    metrics = _lifecycle_rss_metrics([100, 420])

    assert metrics["steady_state_rss_baseline_cycle"] == 1
    assert metrics["steady_state_rss_growth_bytes"] == 320


@pytest.mark.parametrize("samples", [[], [1], [1, -1], [1, True]])
def test_lifecycle_rss_metrics_reject_invalid_samples(samples):
    with pytest.raises(ValueError, match="Lifecycle RSS"):
        _lifecycle_rss_metrics(samples)


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


def _efficientvit_decoder_fixture(path, *, raw_mask_operation="Reshape", opset=17):
    mask_source = numpy_helper.from_array(
        np.zeros((1, 4, 256, 256), dtype=np.float32), name="mask_source"
    )
    mask_shape = numpy_helper.from_array(
        np.asarray([1, 4, 256, 256], dtype=np.int64), name="mask_shape"
    )
    score_shape = numpy_helper.from_array(
        np.asarray([-1, 2], dtype=np.int64), name="score_shape"
    )
    score_weights = numpy_helper.from_array(
        np.arange(8, dtype=np.float32).reshape(2, 4), name="score_weights"
    )
    selected_index = numpy_helper.from_array(
        np.asarray(0, dtype=np.int64), name="selected_index"
    )
    mask_axis = numpy_helper.from_array(
        np.asarray([1], dtype=np.int64), name="mask_axis"
    )
    score_axis = numpy_helper.from_array(
        np.asarray([1], dtype=np.int64), name="score_axis"
    )
    raw_mask_inputs = (
        ["mask_source", "mask_shape"]
        if raw_mask_operation == "Reshape"
        else ["mask_source"]
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                raw_mask_operation,
                raw_mask_inputs,
                ["/Reshape_5_output_0"],
            ),
            helper.make_node(
                "Reshape", ["point_coords", "score_shape"], ["score_inputs"]
            ),
            helper.make_node(
                "Gemm",
                ["score_inputs", "score_weights"],
                ["/iou_prediction_head/layers.2/Gemm_output_0"],
            ),
            helper.make_node(
                "Gather",
                ["/Reshape_5_output_0", "selected_index"],
                ["selected_mask"],
                axis=1,
            ),
            helper.make_node("Unsqueeze", ["selected_mask", "mask_axis"], ["masks"]),
            helper.make_node(
                "Gather",
                [
                    "/iou_prediction_head/layers.2/Gemm_output_0",
                    "selected_index",
                ],
                ["selected_score"],
                axis=1,
            ),
            helper.make_node(
                "Unsqueeze",
                ["selected_score", "score_axis"],
                ["iou_predictions"],
            ),
        ],
        "efficientvit-decoder-fixture",
        [
            helper.make_tensor_value_info(
                "image_embeddings", TensorProto.FLOAT, [1, 256, 64, 64]
            ),
            helper.make_tensor_value_info(
                "point_coords", TensorProto.FLOAT, ["batch_size", 1, 2]
            ),
            helper.make_tensor_value_info(
                "point_labels", TensorProto.FLOAT, ["batch_size", 1]
            ),
        ],
        [
            helper.make_tensor_value_info("masks", TensorProto.FLOAT, [1, 1, 256, 256]),
            helper.make_tensor_value_info(
                "iou_predictions", TensorProto.FLOAT, ["batch_size", 1]
            ),
        ],
        initializer=[
            mask_source,
            mask_shape,
            score_shape,
            score_weights,
            selected_index,
            mask_axis,
            score_axis,
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def _sam2_encoder_fixture(path, *, family="sam2", changed_contract=False):
    image = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, 1024, 1024]
    )
    outputs = [
        helper.make_tensor_value_info(
            "high_res_feats_0", TensorProto.FLOAT, [1, 32, 1024, 1024]
        ),
        helper.make_tensor_value_info(
            "high_res_feats_1", TensorProto.FLOAT, [1, 64, 1024, 1024]
        ),
        helper.make_tensor_value_info(
            "image_embed", TensorProto.FLOAT, [1, 3, 1024, 1024]
        ),
    ]
    nodes = []
    initializers = [
        numpy_helper.from_array(np.asarray(9, dtype=np.int64), name="unused")
    ]
    value_info = []
    if family == "sam2":
        first = "/conv_s0/Conv_output_0"
        second = "/conv_s1/Conv_output_0"
        nodes.extend(
            (
                helper.make_node(
                    "Conv",
                    ["image", "first_weight"],
                    [first],
                    name="/conv_s0/Conv",
                ),
                helper.make_node(
                    "Conv",
                    ["image", "second_weight"],
                    [second],
                    name="/conv_s1/Conv",
                ),
                helper.make_node("Identity", [first], ["high_res_feats_0"]),
                helper.make_node("Identity", [second], ["high_res_feats_1"]),
                helper.make_node("Identity", ["image"], ["image_embed"]),
            )
        )
        initializers.extend(
            (
                numpy_helper.from_array(
                    np.zeros((32, 3, 1, 1), dtype=np.float32),
                    name="first_weight",
                ),
                numpy_helper.from_array(
                    np.zeros((64, 3, 1, 1), dtype=np.float32),
                    name="second_weight",
                ),
            )
        )
        stale_shape = [1] if changed_contract else []
        value_info.extend(
            (
                helper.make_tensor_value_info(first, TensorProto.FLOAT, stale_shape),
                helper.make_tensor_value_info(second, TensorProto.FLOAT, stale_shape),
            )
        )
        opset = 17
        producer_version = "2.4.0"
    else:
        outputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, 1024, 1024])
            for name in ("high_res_feats_0", "high_res_feats_1", "image_embed")
        ]
        captured = numpy_helper.from_array(
            np.asarray([1, 2], dtype=np.int64), name="captured"
        )
        initializers.extend(
            (
                numpy_helper.from_array(np.asarray(True), name="condition"),
                captured,
            )
        )
        then_output_shape = [4] if changed_contract else [5]
        then_graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["branch_values"],
                    value=numpy_helper.from_array(np.asarray([3, 4], dtype=np.int64)),
                ),
                helper.make_node(
                    "Concat",
                    ["captured", "branch_values"],
                    ["/image_encoder/trunk/Concat_3_output_0"],
                    name="/image_encoder/trunk/Concat_3",
                    axis=0,
                ),
            ],
            "then_branch",
            [],
            [
                helper.make_tensor_value_info(
                    "/image_encoder/trunk/Concat_3_output_0",
                    TensorProto.INT64,
                    then_output_shape,
                )
            ],
        )
        else_graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["else_values"],
                    value=numpy_helper.from_array(
                        np.asarray([1, 2, 3, 4], dtype=np.int64)
                    ),
                ),
                helper.make_node(
                    "Identity",
                    ["else_values"],
                    ["/image_encoder/trunk/Identity_output_0"],
                    name="/image_encoder/trunk/Identity",
                ),
            ],
            "else_branch",
            [],
            [
                helper.make_tensor_value_info(
                    "/image_encoder/trunk/Identity_output_0",
                    TensorProto.INT64,
                    [4],
                )
            ],
        )
        nodes.extend(
            (
                helper.make_node(
                    "If",
                    ["condition"],
                    ["selected_shape"],
                    name="/image_encoder/trunk/If",
                    then_branch=then_graph,
                    else_branch=else_graph,
                ),
                helper.make_node("Identity", ["image"], ["high_res_feats_0"]),
                helper.make_node("Identity", ["image"], ["high_res_feats_1"]),
                helper.make_node("Identity", ["image"], ["image_embed"]),
            )
        )
        opset = 18
        producer_version = "2.11.0"

    graph = helper.make_graph(
        nodes,
        f"{family}-encoder-fixture",
        [image],
        outputs,
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    model.producer_name = "pytorch"
    model.producer_version = producer_version
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def _real_matrix_fixture(root, *, variant="l0"):
    for platform in ("Linux", "Windows", "macOS"):
        artifact = root / (
            f"efficientvit-sam-{variant}-real-model-validation-{platform}"
        )
        direct = artifact / platform / f"efficientvit-sam-{variant}" / "direct"
        server = artifact / platform / f"server-efficientvit-sam-{variant}" / "server"
        direct.mkdir(parents=True)
        server.mkdir(parents=True)
        direct_images = []
        server_images = []
        for index in range(2):
            direct_name = f"{index:03d}-direct.png"
            server_name = f"{index:03d}-server.png"
            Image.new("RGB", (4, 3), (index * 20, 40, 80)).save(direct / direct_name)
            Image.new("RGB", (4, 3), (index * 20, 40, 80)).save(server / server_name)
            digest = hashlib.sha256(f"prediction-{index}".encode()).hexdigest()
            direct_images.append(
                {
                    "annotated_image": direct_name,
                    "consistent_runs": True,
                    "failures": [],
                    "passed": True,
                    "prediction_digest": digest,
                }
            )
            server_images.append(
                {
                    "annotated_image": server_name,
                    "consistent_runs": True,
                    "failures": [],
                    "prediction_digest": digest,
                }
            )
        (direct / "summary.json").write_text(
            json.dumps(
                {
                    "backend": "efficientvit_sam",
                    "failures": [],
                    "images": direct_images,
                    "passed": True,
                    "peak_observed_rss_bytes": 100,
                    "steady_state_rss_growth_bytes": 0,
                }
            )
        )
        (server / "summary.json").write_text(
            json.dumps(
                {
                    "failures": [],
                    "images": server_images,
                    "manifest": f"efficientvit_sam_{variant}_official.json",
                    "passed": True,
                    "peak_observed_rss_bytes": 120,
                }
            )
        )


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


def test_real_matrix_verifier_proves_transport_and_platform_pixel_identity(tmp_path):
    module = _script("verify_real_model_matrix.py")
    _real_matrix_fixture(tmp_path)

    report = module.verify_matrix(
        tmp_path,
        artifact_prefix="efficientvit-sam",
        variants=("l0",),
        platforms=("Linux", "Windows", "macOS"),
        expected_cases=2,
    )

    assert report["passed"] is True
    assert list(report["variants"]) == ["l0"]
    assert len(report["variants"]["l0"]["prediction_digests"]) == 2


def test_real_matrix_verifier_rejects_transport_and_platform_drift(tmp_path):
    module = _script("verify_real_model_matrix.py")
    _real_matrix_fixture(tmp_path)
    windows_server = next(
        tmp_path.glob(
            "efficientvit-sam-l0-real-model-validation-Windows/"
            "Windows/server-efficientvit-sam-l0/*/summary.json"
        )
    )
    report = json.loads(windows_server.read_text())
    report["images"][0]["prediction_digest"] = "f" * 64
    windows_server.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Transport prediction mismatch"):
        module.verify_matrix(
            tmp_path,
            artifact_prefix="efficientvit-sam",
            variants=("l0",),
            platforms=("Linux", "Windows", "macOS"),
            expected_cases=2,
        )

    _real_matrix_fixture(tmp_path / "pixel-drift")
    mac_direct = next(
        (tmp_path / "pixel-drift").glob(
            "efficientvit-sam-l0-real-model-validation-macOS/"
            "macOS/efficientvit-sam-l0/*/000-direct.png"
        )
    )
    mac_server = next(
        (tmp_path / "pixel-drift").glob(
            "efficientvit-sam-l0-real-model-validation-macOS/"
            "macOS/server-efficientvit-sam-l0/*/000-server.png"
        )
    )
    Image.new("RGB", (4, 3), (255, 0, 0)).save(mac_direct)
    Image.new("RGB", (4, 3), (255, 0, 0)).save(mac_server)
    with pytest.raises(ValueError, match="Cross-platform pixel mismatch"):
        module.verify_matrix(
            tmp_path / "pixel-drift",
            artifact_prefix="efficientvit-sam",
            variants=("l0",),
            platforms=("Linux", "Windows", "macOS"),
            expected_cases=2,
        )


def test_real_matrix_verifier_records_strictly_bounded_renderer_drift(tmp_path):
    module = _script("verify_real_model_matrix.py")
    _real_matrix_fixture(tmp_path)
    for role in ("efficientvit-sam-l0", "server-efficientvit-sam-l0"):
        image_path = next(
            tmp_path.glob(
                "efficientvit-sam-l0-real-model-validation-macOS/"
                f"macOS/{role}/*/000-*.png"
            )
        )
        image = Image.open(image_path).convert("RGB")
        image.putpixel((0, 0), (0, 40, 81))
        image.save(image_path)

    report = module.verify_matrix(
        tmp_path,
        artifact_prefix="efficientvit-sam",
        variants=("l0",),
        platforms=("Linux", "Windows", "macOS"),
        expected_cases=2,
        max_cross_platform_differing_pixels=1,
        max_cross_platform_channel_delta=1,
    )

    drift = report["variants"]["l0"]["platforms"]["macOS"]["cross_platform_pixel_drift"]
    assert drift == [
        {"case": 0, "differing_pixels": 1, "maximum_channel_delta": 1},
        {"case": 1, "differing_pixels": 0, "maximum_channel_delta": 0},
    ]


def test_efficientvit_decoder_transform_is_deterministic_and_runnable(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    module = _script("prepare_efficientvit_sam_decoder.py")
    source = tmp_path / "decoder.onnx"
    _efficientvit_decoder_fixture(source)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    first = tmp_path / "prepared-one.onnx"
    second = tmp_path / "prepared-two.onnx"

    first_report = module.prepare_decoder(
        source, first, source_sha256=source_digest.upper()
    )
    second_report = module.prepare_decoder(
        source,
        second,
        source_sha256=source_digest,
        expected_output_sha256=first_report["output_sha256"],
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_report["output_sha256"] == second_report["output_sha256"]
    assert first_report["source_sha256"] == source_digest
    assert first.stat().st_mode & 0o777 == 0o644
    prepared = onnx.load_model(first, load_external_data=False)
    assert [
        (item.name, item.type.tensor_type.elem_type) for item in prepared.graph.output
    ] == [
        ("masks", TensorProto.FLOAT),
        ("iou_predictions", TensorProto.FLOAT),
    ]
    assert [
        [
            dimension.dim_value
            if dimension.HasField("dim_value")
            else dimension.dim_param
            for dimension in item.type.tensor_type.shape.dim
        ]
        for item in prepared.graph.output
    ] == [["batch_size", 4, 256, 256], ["batch_size", 4]]
    assert {item.key: item.value for item in prepared.metadata_props} == {
        "anylearning.source_sha256": source_digest,
        "anylearning.transform": "efficientvit-sam-multimask-v1",
    }

    session = onnxruntime.InferenceSession(
        str(first), providers=["CPUExecutionProvider"]
    )
    masks, scores = session.run(
        None,
        {
            "image_embeddings": np.zeros((1, 256, 64, 64), dtype=np.float32),
            "point_coords": np.asarray([[[2.0, 3.0]]], dtype=np.float32),
            "point_labels": np.ones((1, 1), dtype=np.float32),
        },
    )
    assert masks.shape == (1, 4, 256, 256)
    assert scores.shape == (1, 4)
    assert np.isfinite(masks).all() and np.isfinite(scores).all()


@pytest.mark.parametrize(
    ("source_digest", "expected_digest", "message"),
    [
        ("invalid", None, "64 hexadecimal"),
        ("0" * 64, None, "mismatch"),
        (None, "invalid", "64 hexadecimal"),
        (None, "0" * 64, "Prepared decoder SHA-256 mismatch"),
    ],
)
def test_efficientvit_decoder_transform_rejects_unverified_artifacts(
    tmp_path, source_digest, expected_digest, message
):
    module = _script("prepare_efficientvit_sam_decoder.py")
    source = tmp_path / "decoder.onnx"
    output = tmp_path / "prepared.onnx"
    _efficientvit_decoder_fixture(source)
    actual_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        module.prepare_decoder(
            source,
            output,
            source_sha256=source_digest or actual_digest,
            expected_output_sha256=expected_digest,
        )
    assert not output.exists()


def test_efficientvit_decoder_transform_rejects_changed_contract_and_overwrite(
    tmp_path,
):
    module = _script("prepare_efficientvit_sam_decoder.py")
    changed = tmp_path / "changed.onnx"
    _efficientvit_decoder_fixture(changed, raw_mask_operation="Identity")
    digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    output = tmp_path / "prepared.onnx"

    with pytest.raises(ValueError, match="not produced by Reshape"):
        module.prepare_decoder(changed, output, source_sha256=digest)

    source = tmp_path / "decoder.onnx"
    _efficientvit_decoder_fixture(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output.write_bytes(b"keep existing artifact")
    with pytest.raises(FileExistsError, match="already exists"):
        module.prepare_decoder(source, output, source_sha256=digest)
    assert output.read_bytes() == b"keep existing artifact"


def test_efficientvit_decoder_transform_rejects_symlink_paths(tmp_path):
    module = _script("prepare_efficientvit_sam_decoder.py")
    source = tmp_path / "decoder.onnx"
    _efficientvit_decoder_fixture(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    try:
        linked_directory.symlink_to(real_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="may not traverse a symlink"):
        module.prepare_decoder(
            source, linked_directory / "prepared.onnx", source_sha256=digest
        )
    assert not (real_directory / "prepared.onnx").exists()


@pytest.mark.parametrize("family", ["sam2", "sam2_1"])
def test_sam2_encoder_transform_is_deterministic_strict_and_runnable(tmp_path, family):
    onnxruntime = pytest.importorskip("onnxruntime")
    module = _script("prepare_sam2_encoder.py")
    source = tmp_path / f"{family}.encoder.onnx"
    _sam2_encoder_fixture(source, family=family)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    first = tmp_path / f"{family}-prepared-one.onnx"
    second = tmp_path / f"{family}-prepared-two.onnx"

    first_report = module.prepare_encoder(
        source,
        first,
        family=family,
        source_sha256=source_digest.upper(),
    )
    second_report = module.prepare_encoder(
        source,
        second,
        family=family,
        source_sha256=source_digest,
        expected_output_sha256=first_report["output_sha256"],
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_report["output_sha256"] == second_report["output_sha256"]
    assert first_report["source_sha256"] == source_digest
    assert first_report["removed_initializers"] == ["unused"]
    assert first.stat().st_mode & 0o777 == 0o644
    prepared = onnx.load_model(first, load_external_data=False)
    assert {item.key: item.value for item in prepared.metadata_props} == {
        "anylearning.family": family,
        "anylearning.source_producer": "pytorch",
        "anylearning.source_producer_version": (
            "2.4.0" if family == "sam2" else "2.11.0"
        ),
        "anylearning.source_sha256": source_digest,
        "anylearning.transform": "sam2-encoder-metadata-v1",
    }
    if family == "sam2_1":
        assert "captured" in {item.name for item in prepared.graph.initializer}
    onnx.shape_inference.infer_shapes(
        prepared, check_type=True, strict_mode=True, data_prop=True
    )
    session = onnxruntime.InferenceSession(
        str(first), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_outputs()] == [
        "high_res_feats_0",
        "high_res_feats_1",
        "image_embed",
    ]

    if family == "sam2":
        repaired = set(first_report["repaired_values"])
        assert repaired == {
            "/conv_s0/Conv_output_0",
            "/conv_s1/Conv_output_0",
        }
        values = {item.name: item for item in prepared.graph.value_info}
        assert len(values["/conv_s0/Conv_output_0"].type.tensor_type.shape.dim) == 4
        assert len(values["/conv_s1/Conv_output_0"].type.tensor_type.shape.dim) == 4
    else:
        if_node = next(
            item
            for item in prepared.graph.node
            if item.name == "/image_encoder/trunk/If"
        )
        then_graph = next(
            item.g for item in if_node.attribute if item.name == "then_branch"
        )
        assert [
            item.dim_value for item in then_graph.output[0].type.tensor_type.shape.dim
        ] == [4]


@pytest.mark.parametrize(
    ("source_digest", "expected_digest", "message"),
    [
        ("invalid", None, "64 hexadecimal"),
        ("0" * 64, None, "mismatch"),
        (None, "invalid", "64 hexadecimal"),
        (None, "0" * 64, "Prepared encoder SHA-256 mismatch"),
    ],
)
def test_sam2_encoder_transform_rejects_unverified_artifacts(
    tmp_path, source_digest, expected_digest, message
):
    module = _script("prepare_sam2_encoder.py")
    source = tmp_path / "sam2.encoder.onnx"
    output = tmp_path / "prepared.onnx"
    _sam2_encoder_fixture(source)
    actual_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        module.prepare_encoder(
            source,
            output,
            family="sam2",
            source_sha256=source_digest or actual_digest,
            expected_output_sha256=expected_digest,
        )
    assert not output.exists()


@pytest.mark.parametrize("family", ["sam2", "sam2_1"])
def test_sam2_encoder_transform_rejects_changed_contract_and_overwrite(
    tmp_path, family
):
    module = _script("prepare_sam2_encoder.py")
    changed = tmp_path / f"changed-{family}.onnx"
    _sam2_encoder_fixture(changed, family=family, changed_contract=True)
    digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    output = tmp_path / "prepared.onnx"
    expected_message = "stale SAM2 tensor" if family == "sam2" else "stale SAM2.1"
    with pytest.raises(ValueError, match=expected_message):
        module.prepare_encoder(
            changed,
            output,
            family=family,
            source_sha256=digest,
        )

    source = tmp_path / f"{family}.onnx"
    _sam2_encoder_fixture(source, family=family)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output.write_bytes(b"keep existing artifact")
    with pytest.raises(FileExistsError, match="already exists"):
        module.prepare_encoder(
            source,
            output,
            family=family,
            source_sha256=digest,
        )
    assert output.read_bytes() == b"keep existing artifact"


def test_sam2_encoder_transform_rejects_symlink_paths(tmp_path):
    module = _script("prepare_sam2_encoder.py")
    source = tmp_path / "sam2.encoder.onnx"
    _sam2_encoder_fixture(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    try:
        linked_directory.symlink_to(real_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="may not traverse a symlink"):
        module.prepare_encoder(
            source,
            linked_directory / "prepared.onnx",
            family="sam2",
            source_sha256=digest,
        )
    assert not (real_directory / "prepared.onnx").exists()


def test_exact_zip_extractor_accepts_only_manifested_regular_files(tmp_path):
    module = _script("extract_verified_zip.py")
    archive = tmp_path / "models.zip"
    payloads = {"encoder.onnx": b"encoder", "decoder.onnx": b"decoder"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in payloads.items():
            bundle.writestr(name, payload)

    output = tmp_path / "models"
    output.mkdir()
    extracted = module.extract_exact_zip(
        archive,
        output,
        {name: len(payload) for name, payload in payloads.items()},
        remove_archive=True,
    )

    assert {path.name for path in extracted} == set(payloads)
    assert {path.name: path.read_bytes() for path in extracted} == payloads
    assert not archive.exists()


@pytest.mark.parametrize("failure", ["extra", "size", "duplicate", "link"])
def test_exact_zip_extractor_rejects_changed_or_unsafe_archives(tmp_path, failure):
    module = _script("extract_verified_zip.py")
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("encoder.onnx", b"encoder")
        if failure == "extra":
            bundle.writestr("unexpected.txt", b"unexpected")
        elif failure == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr("encoder.onnx", b"duplicate")
        elif failure == "link":
            link = zipfile.ZipInfo("decoder.onnx")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(link, b"encoder.onnx")

    output = tmp_path / "models"
    output.mkdir()
    expected = {"encoder.onnx": 8 if failure == "size" else 7}
    if failure == "link":
        expected["decoder.onnx"] = len(b"encoder.onnx")

    with pytest.raises(ValueError):
        module.extract_exact_zip(archive, output, expected)
    assert not list(output.iterdir())


def test_all_committed_real_model_manifests_are_schema_valid():
    manifest_root = _ROOT / "tests/fixtures/inference/real_models"
    manifests = sorted(manifest_root.glob("*.json"))

    assert manifests
    for path in manifests:
        manifest = load_validation_manifest(path)
        assert manifest.provenance.source_revision
        assert all(item.source_revision for item in manifest.component_provenance)
        assert manifest.runs >= 2
        assert manifest.lifecycle_cycles >= 2


def test_prediction_digest_ignores_transport_identity_and_timings():
    result = InferenceResult(
        request_id="request-one",
        source_id="content-sha256:one",
        model_id="model",
        model_revision="revision",
        warnings=("review output",),
        timings_ms={"total": 10.0},
    )

    equivalent = result.model_copy(
        update={
            "request_id": "request-two",
            "source_id": "content-sha256:two",
            "timings_ms": {"total": 20.0},
        }
    )
    changed = result.model_copy(update={"warnings": ("different output",)})

    assert _prediction_digest(equivalent) == _prediction_digest(result)
    assert _prediction_digest(changed) != _prediction_digest(result)


def test_visual_report_captions_only_largest_shape_in_each_instance_group():
    shapes = (
        InferenceShape(
            type=ShapeType.POLYGON,
            points=(Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1)),
            group_id=7,
        ),
        InferenceShape(
            type=ShapeType.POLYGON,
            points=(Point(x=0, y=0), Point(x=5, y=0), Point(x=5, y=5)),
            group_id=7,
        ),
        InferenceShape(
            type=ShapeType.RECTANGLE,
            points=(Point(x=10, y=10), Point(x=12, y=12)),
        ),
    )

    assert _caption_shape_indexes(shapes) == {1, 2}


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


def test_validation_evidence_hashes_composite_child_artifacts(tmp_path):
    detector = tmp_path / "detector.onnx"
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    detector.write_bytes(b"detector")
    encoder.write_bytes(b"encoder")
    decoder.write_bytes(b"decoder")
    config = {
        "detector": {
            "backend": "yolo_onnx",
            "config": {
                "model_path": detector.name,
                "sha256": hashlib.sha256(detector.read_bytes()).hexdigest(),
            },
        },
        "segmenter": {
            "backend": "segment_anything",
            "config": {
                "encoder_model_path": encoder.name,
                "encoder_sha256": hashlib.sha256(encoder.read_bytes()).hexdigest(),
                "decoder_model_path": decoder.name,
                "decoder_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
            },
        },
    }

    details = _model_artifact_details(config, tmp_path)

    assert details["bytes"] == sum(
        path.stat().st_size for path in (detector, encoder, decoder)
    )
    assert [component["role"] for component in details["components"]] == [
        "detector",
        "segmenter",
    ]
    assert details["components"][0]["filename"] == detector.name
    assert [graph["role"] for graph in details["components"][1]["graphs"]] == [
        "encoder",
        "decoder",
    ]
