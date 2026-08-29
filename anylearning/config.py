import os
import os.path as osp
import pathlib
from typing import List

import yaml
from loguru import logger

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from anylearning import configs as anylearning_configs

# ANYLEARNING_DATA_ROOT moves the whole store -- database, projects, images.
# The self-test sets it to a temporary directory so that running it on someone's
# machine cannot touch the projects they care about. Read at import time,
# because everything below is derived from it and nothing re-reads it later.
DATA_ROOT = os.path.abspath(
    os.environ.get("ANYLEARNING_DATA_ROOT")
    or os.path.join(os.path.expanduser("~"), "anylearning-data")
)
pathlib.Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)

DATABASE_PATH = os.path.abspath(os.path.join(DATA_ROOT, "anylearning.db"))


PROJECTS_ROOT = os.path.abspath(os.path.join(DATA_ROOT, "projects"))
pathlib.Path(PROJECTS_ROOT).mkdir(parents=True, exist_ok=True)


# Save current config file
current_config_file = None


def update_dict(target_dict, new_dict, validate_item=None):
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            logger.warning("Skipping unexpected key in config: %s", key)
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


def save_config(config):
    # Local config file
    user_config_file = osp.join(osp.expanduser("~"), ".anylearningrc")
    try:
        with open(user_config_file, "w") as f:
            yaml.safe_dump(config, f)
    except Exception:  # noqa
        logger.warning("Failed to save config: %s", user_config_file)


def get_default_config():
    config_file = "anylearning_config.yaml"
    with pkg_resources.open_text(anylearning_configs, config_file) as f:
        config = yaml.safe_load(f)

    # Save default config to ~/.anylearningrc
    if not osp.exists(osp.join(osp.expanduser("~"), ".anylearningrc")):
        save_config(config)

    return config


def validate_config_item(key, value):
    if key == "validate_label" and value not in [None, "exact"]:
        raise ValueError(f"Unexpected value for config key 'validate_label': {value}")
    if key == "shape_color" and value not in [None, "auto", "manual"]:
        raise ValueError(f"Unexpected value for config key 'shape_color': {value}")
    if key == "labels" and value is not None and len(value) != len(set(value)):
        raise ValueError(f"Duplicates are detected for config key 'labels': {value}")


def get_config(config_file_or_yaml=None, config_from_args=None):
    # 1. default config
    config = get_default_config()

    # 2. specified as file or yaml
    if config_file_or_yaml is None:
        config_file_or_yaml = current_config_file
    if config_file_or_yaml is not None:
        config_from_yaml = yaml.safe_load(config_file_or_yaml)
        if not isinstance(config_from_yaml, dict):
            with open(config_from_yaml) as f:
                logger.info("Loading config file from: %s", config_from_yaml)
                config_from_yaml = yaml.safe_load(f)
        update_dict(config, config_from_yaml, validate_item=validate_config_item)

    # 3. command line argument or specified config file
    if config_from_args is not None:
        update_dict(config, config_from_args, validate_item=validate_config_item)

    return config


_TEXT_AI_MODEL_VARIANTS = [
    {
        "name": "Word + character TF-IDF classifier",
        "model_architecture": "tfidf-logreg",
        "model_size": "lightweight",
    },
]

MODEL_VARIANTS = {
    "Tabular AI": [
        {
            "name": "CatBoost Balanced",
            "model_architecture": "catboost",
            "model_size": "balanced",
        },
        {
            "name": "CatBoost Accurate",
            "model_architecture": "catboost",
            "model_size": "accurate",
        },
    ],
    "Text AI": _TEXT_AI_MODEL_VARIANTS,
    # Existing preview projects used these names. They open and train with the
    # same engine; only new projects use the corrected product terminology.
    "Text AI & LLM Evaluation": _TEXT_AI_MODEL_VARIANTS,
    "Text & LLM": _TEXT_AI_MODEL_VARIANTS,
    "Sentiment Analysis": _TEXT_AI_MODEL_VARIANTS,
    "Image Classification": [
        {
            "name": "ResNet18-Lightweight",
            "model_architecture": "resnet18",
            "model_size": "lightweight",
        },
        {
            "name": "ResNet34-Medium",
            "model_architecture": "resnet34",
            "model_size": "medium",
        },
    ],
    "Object Detection": [
        {
            "name": "NanoDet-Lightweight",
            "model_architecture": "nanodet",
            "model_size": "lightweight",
        },
        {
            "name": "NanoDet-Medium",
            "model_architecture": "nanodet",
            "model_size": "medium",
        },
        {
            "name": "NanoDet-Large",
            "model_architecture": "nanodet",
            "model_size": "large",
        },
        # RF-DETR is the accurate end of the same project type: a DETR over a
        # DINOv2 backbone, which needs a GPU to be pleasant but reaches a usable
        # model from far fewer labelled images. The sizes keep RF-DETR's own
        # names rather than being mapped onto lightweight/medium/large, because
        # the pairing is what identifies a model row forever and "RF-DETR-Nano"
        # is the name every other tool uses for these weights.
        {
            "name": "RF-DETR-Nano",
            "model_architecture": "rfdetr",
            "model_size": "nano",
        },
        {
            "name": "RF-DETR-Small",
            "model_architecture": "rfdetr",
            "model_size": "small",
        },
    ],
    "Image Segmentation": [
        {
            "name": "Deeplabv3-Lightweight",
            "model_architecture": "resnet18",
            "model_size": "lightweight",
        },
        {
            "name": "Deeplabv3-Medium",
            "model_architecture": "resnet34",
            "model_size": "medium",
        },
        {
            "name": "Deeplabv3-Large",
            "model_architecture": "resnet50",
            "model_size": "large",
        },
    ],
    "Handpose Classification": [
        {
            "name": "MLP-Small",
            "model_architecture": "mlp",
            "model_size": "lightweight",
        },
        {
            "name": "MLP-Medium",
            "model_architecture": "mlp",
            "model_size": "medium",
        },
        {
            "name": "MLP-Large",
            "model_architecture": "mlp",
            "model_size": "large",
        },
    ],
    "Instance Segmentation": [
        {
            "name": "Mask R-CNN Medium",
            "model_architecture": "maskrcnn-resnet50",
            "model_size": "medium",
        },
        {
            "name": "Mask R-CNN Large",
            "model_architecture": "maskrcnn-resnet101",
            "model_size": "large",
        },
        {
            "name": "RF-DETR-Seg-Nano",
            "model_architecture": "rfdetr-seg",
            "model_size": "nano",
        },
        {
            "name": "RF-DETR-Seg-Small",
            "model_architecture": "rfdetr-seg",
            "model_size": "small",
        },
    ],
    "Keypoint Detection": [
        {
            # RF-DETR currently publishes one keypoint model. "Preview" is
            # part of the upstream API contract; keeping it in the visible name
            # is more honest than presenting an experimental checkpoint as a
            # settled family.
            "name": "RF-DETR-Keypoint-Preview",
            "model_architecture": "rfdetr-keypoint",
            "model_size": "preview",
        },
    ],
}


def get_model_variant_name(
    model_variants: List[dict], model_architecture: str, model_size: str
):
    for variant in model_variants:
        if (
            variant["model_architecture"] == model_architecture
            and variant["model_size"] == model_size
        ):
            return variant["name"]
    return "Unknown"
