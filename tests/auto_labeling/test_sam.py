from importlib import resources
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from anylearning.auto_labeling.sam_onnx import SegmentAnythingONNX
from anylearning.auto_labeling.sam2_onnx import SAM2ImageEncoder
from anylearning.configs import auto_labeling as auto_labeling_configs


@pytest.fixture
def mock_encoder_session():
    mock = Mock()
    mock.get_inputs.return_value = [Mock(name="input")]
    mock.run.return_value = [np.zeros((1, 256, 64, 64))]
    return mock


@pytest.fixture
def mock_decoder_session():
    mock = Mock()
    mock.run.return_value = [np.zeros((1, 1, 256, 256)), None, None]
    return mock


@pytest.fixture
def sam_model(mock_encoder_session, mock_decoder_session):
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.side_effect = [mock_encoder_session, mock_decoder_session]
        model = SegmentAnythingONNX("encoder.onnx", "decoder.onnx")
        return model


def test_init(sam_model):
    assert sam_model.target_size == 1024
    assert sam_model.input_size == (684, 1024)
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


def test_transform_masks(sam_model):
    masks = np.zeros((1, 2, 256, 256))
    original_size = (100, 200)
    transform_matrix = np.eye(3)

    result = sam_model.transform_masks(masks, original_size, transform_matrix)

    assert result.shape == (1, 2, 100, 200)


def test_encode(sam_model):
    image = np.zeros((100, 200, 3))
    result = sam_model.encode(image)

    assert "image_embedding" in result
    assert "original_size" in result
    assert "transform_matrix" in result
    assert result["original_size"] == (100, 200)
    assert result["transform_matrix"].shape == (3, 3)


def test_predict_masks(sam_model):
    embedding = {
        "image_embedding": np.zeros((1, 256, 64, 64)),
        "original_size": (100, 200),
        "transform_matrix": np.eye(3),
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
