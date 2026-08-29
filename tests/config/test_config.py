from unittest.mock import mock_open, patch

import pytest

from anylearning.config import (
    MODEL_VARIANTS,
    get_config,
    get_default_config,
    get_model_variant_name,
    update_dict,
    validate_config_item,
)


def test_get_default_config():
    config = get_default_config()
    assert isinstance(config, dict)


def test_update_dict():
    target = {"a": 1, "b": {"c": 2}}
    new = {"b": {"c": 3}, "d": 4}
    update_dict(target, new)
    assert target == {"a": 1, "b": {"c": 3}}


def test_validate_config_item():
    # Valid cases
    validate_config_item("validate_label", None)
    validate_config_item("validate_label", "exact")
    validate_config_item("shape_color", None)
    validate_config_item("shape_color", "auto")
    validate_config_item("shape_color", "manual")
    validate_config_item("labels", ["a", "b", "c"])

    # Invalid cases
    with pytest.raises(ValueError):
        validate_config_item("validate_label", "invalid")
    with pytest.raises(ValueError):
        validate_config_item("shape_color", "invalid")
    with pytest.raises(ValueError):
        validate_config_item("labels", ["a", "a", "b"])


def test_get_config_with_yaml_string():
    yaml_str = """
    key1: value1
    key2: value2
    """
    with patch("anylearning.config.get_default_config") as mock_default:
        mock_default.return_value = {"key1": "default", "key2": "default"}
        config = get_config(yaml_str)
        assert config["key1"] == "value1"
        assert config["key2"] == "value2"


def test_get_config_with_file():
    yaml_content = """
    key1: value1
    key2: value2
    """
    mock_file = mock_open(read_data=yaml_content)
    with patch("builtins.open", mock_file):
        with patch("anylearning.config.get_default_config") as mock_default:
            mock_default.return_value = {"key1": "default", "key2": "default"}
            config = get_config("config.yaml")
            assert config["key1"] == "value1"
            assert config["key2"] == "value2"


def test_get_config_with_args():
    args = {"key1": "arg_value"}
    with patch("anylearning.config.get_default_config") as mock_default:
        mock_default.return_value = {"key1": "default"}
        config = get_config(config_from_args=args)
        assert config["key1"] == "arg_value"


def test_get_model_variant_name():
    variants = MODEL_VARIANTS["Image Classification"]

    # Test existing variant
    name = get_model_variant_name(variants, "resnet18", "lightweight")
    assert name == "ResNet18-Lightweight"

    # Test non-existing variant
    name = get_model_variant_name(variants, "invalid", "invalid")
    assert name == "Unknown"


def test_model_variants_structure():
    # Test basic structure of MODEL_VARIANTS
    assert "Image Classification" in MODEL_VARIANTS
    assert "Object Detection" in MODEL_VARIANTS
    assert "Image Segmentation" in MODEL_VARIANTS
    assert "Handpose Classification" in MODEL_VARIANTS
    assert "Instance Segmentation" in MODEL_VARIANTS
    assert "Keypoint Detection" in MODEL_VARIANTS

    # Test variant structure
    for _category, variants in MODEL_VARIANTS.items():
        for variant in variants:
            assert "name" in variant
            assert "model_architecture" in variant
            assert "model_size" in variant
