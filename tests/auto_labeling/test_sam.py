from importlib import resources
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from anylearning.auto_labeling.model_manager import _complete_bundled_config
from anylearning.auto_labeling.sam2_onnx import SAM2ImageEncoder
from anylearning.auto_labeling.sam_onnx import SegmentAnythingONNX
from anylearning.configs import auto_labeling as auto_labeling_configs


@pytest.fixture
def mock_encoder_session():
    mock = Mock()
    mock.get_inputs.return_value = [
        SimpleNamespace(
            name="image", shape=("height", "width", 3), type="tensor(float)"
        )
    ]
    mock.run.return_value = [np.zeros((1, 256, 64, 64))]
    return mock


@pytest.fixture
def mock_decoder_session():
    mock = Mock()
    mock.get_inputs.return_value = [
        SimpleNamespace(name=name)
        for name in (
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        )
    ]
    mock.get_outputs.return_value = [
        SimpleNamespace(name="masks"),
        SimpleNamespace(name="iou_predictions"),
        SimpleNamespace(name="low_res_masks"),
    ]
    mask = np.zeros((1, 1, 256, 256), dtype=np.float32)
    mock.run.return_value = [mask, np.ones((1, 1), dtype=np.float32), mask]
    return mock


@pytest.fixture
def sam_model(mock_encoder_session, mock_decoder_session):
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.side_effect = [mock_encoder_session, mock_decoder_session]
        model = SegmentAnythingONNX("encoder.onnx", "decoder.onnx")
        return model


def test_init(sam_model):
    assert sam_model.target_size == 1024
    assert sam_model.encoder_input_rank == 3
    assert sam_model.encoder_session is not None
    assert sam_model.decoder_session is not None


def test_get_input_points(sam_model):
    prompt = [
        {"type": "point", "data": [100, 200], "label": 1},
        {"type": "rectangle", "data": [10, 20, 30, 40]},
    ]

    points, labels = sam_model.get_input_points(prompt)

    expected_points = np.array([[100, 200], [10, 20], [30, 40]])
    expected_labels = np.array([1, 2, 3])

    np.testing.assert_array_equal(points, expected_points)
    np.testing.assert_array_equal(labels, expected_labels)


def test_get_input_points_rejects_invalid_prompts(sam_model):
    with pytest.raises(ValueError, match="Unsupported prompt type"):
        sam_model.get_input_points([{"type": "scribble", "data": [1, 2]}])
    with pytest.raises(ValueError, match="At least one prompt"):
        sam_model.get_input_points([])


def test_get_preprocess_shape():
    result = SegmentAnythingONNX.get_preprocess_shape(100, 200, 400)
    assert result == (200, 400)

    result = SegmentAnythingONNX.get_preprocess_shape(300, 100, 400)
    assert result == (400, 133)


def test_apply_coords(sam_model):
    coords = np.array([[[10, 20], [30, 40]]])
    original_size = (100, 200)
    target_length = 400

    result = sam_model.apply_coords(coords, original_size, target_length)

    assert result.shape == coords.shape


def test_postprocess_masks(sam_model):
    masks = np.zeros((1, 2, 256, 256))
    original_size = (100, 200)
    resized_size = (512, 1024)

    result = sam_model.postprocess_masks(masks, original_size, resized_size)

    assert result.shape == (1, 2, 100, 200)


def test_encode(sam_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = sam_model.encode(image)

    assert "image_embedding" in result
    assert "original_size" in result
    assert "resized_size" in result
    assert result["original_size"] == (100, 200)
    assert result["resized_size"] == (512, 1024)


def test_predict_masks(sam_model):
    embedding = {
        "image_embedding": np.zeros((1, 256, 64, 64)),
        "original_size": (100, 200),
        "resized_size": (512, 1024),
    }
    prompt = [{"type": "point", "data": [10, 20], "label": 1}]

    masks = sam_model.predict_masks(embedding, prompt)
    assert isinstance(masks, np.ndarray)


def test_sam2_encoder_preserves_pil_rgb_channel_order():
    """The labeling path gives SAM 2 an RGB array, not an OpenCV BGR array."""
    encoder = object.__new__(SAM2ImageEncoder)
    encoder.input_width = 1
    encoder.input_height = 1

    red, green, blue = 250, 100, 10
    tensor = encoder.prepare_input(np.array([[[red, green, blue]]], dtype=np.uint8))

    expected = (
        np.array([red, green, blue], dtype=np.float64) / 255.0
        - np.array([0.485, 0.456, 0.406])
    ) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6)


def test_sam2_small_is_the_default_bundled_auto_labeling_model():
    model_file = resources.files(auto_labeling_configs).joinpath("models.yaml")
    models = yaml.safe_load(model_file.read_text())

    assert models[0]["name"] == "sam2_hiera_small_20240803"
    assert all(model["type"] == "segment_anything" for model in models)


def test_bundled_config_recovers_metadata_from_legacy_stub(tmp_path):
    model_dir = tmp_path / "sam2"
    model_dir.mkdir()
    (model_dir / "small.encoder.onnx").touch()
    (model_dir / "small.decoder.onnx").touch()

    config = _complete_bundled_config(
        {"name": "sam2", "type": "segment_anything"},
        {"display_name": "SAM 2", "has_downloaded": True},
        str(model_dir / "config.yaml"),
    )

    assert config["type"] == "segment_anything"
    assert config["encoder_model_path"] == "small.encoder.onnx"
    assert config["decoder_model_path"] == "small.decoder.onnx"
