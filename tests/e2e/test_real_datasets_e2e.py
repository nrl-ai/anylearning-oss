"""End-to-end training on the real, licence-cleared datasets.

These use a local snapshot of https://huggingface.co/datasets/nrl-ai/anylearning-data next to
this repository (or ``ANYLEARNING_DATA_DIR``). That checkout is ~500 MB, so every
test here skips cleanly when it is absent rather than failing.

Only datasets listed in ``REAL_DATASETS`` are used. ``neu_surface_defect`` and
``dental_segment`` are excluded on purpose because neither records a
redistribution licence. See ``anylearning-data/LICENSES.md``.

Each test trains on a small slice -- the goal is that the real data flows through
the real pipeline, not that the model converges.
"""

import pytest
import yaml

from tests.fixtures.datasets import (
    REAL_DATASETS,
    anylearning_data_root,
    convert_anylabeling_dir,
    extract_real_dataset,
    subsample_image_folder,
    subsample_segmentation_dir,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    anylearning_data_root() is None,
    reason="anylearning-data checkout not found (set ANYLEARNING_DATA_DIR)",
)


def test_only_licence_cleared_datasets_are_exposed():
    """Guard: the unlicensed datasets must never be added to REAL_DATASETS."""
    assert "neu_surface_defect" not in REAL_DATASETS
    assert "dental_segment" not in REAL_DATASETS

    for name, (task, licence, subsets) in REAL_DATASETS.items():
        assert licence, f"{name} has no recorded licence"
        assert "NC" not in licence and "non-commercial" not in licence.lower()
        assert subsets, f"{name} declares no subsets"


def test_chest_xray_classification_trains(tmp_path):
    """Real classification data (CC BY 4.0) through the real training path."""
    from anylearning.training.models.classification.train import train_fn

    source = extract_real_dataset("zhanglabdata_chest_xray", "train")
    if source is None:
        pytest.skip("chest x-ray archive missing")

    train_dir = tmp_path / "train"
    classes = subsample_image_folder(source, train_dir, per_class=4)
    assert classes, "no class directories found in the real dataset"

    save_dir = tmp_path / "output"
    config = {
        "save_dir": str(save_dir),
        "seed": 67,
        "data": {
            "train_dir": str(train_dir),
            "val_dir": str(train_dir),  # tiny slice: reuse for validation
            "test_dir": str(train_dir),
            "class_names": classes,
            "img_size": 64,
            "num_workers": 0,
        },
        "model": {"arch": "resnet18", "pretrained": None, "num_classes": len(classes)},
        "training": {
            "gradient_checkpointing": False,
            "scheduler": "cosine",
            "resume": False,
            "epochs": 1,
            "optim": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "min_lr": 1e-6,
            "eps": 1e-6,
            "batch_size": 2,
            "fp16": False,
            "clip_grad_norm": 10,
            "accumulation_steps": 1,
        },
    }
    config_path = tmp_path / "cfg_classification.yaml"
    config_path.write_text(yaml.safe_dump(config))

    train_fn(str(config_path))

    written = list(save_dir.glob("*.pth")) + list(save_dir.glob("*.pt"))
    assert written, f"no checkpoint in {save_dir}: {list(save_dir.iterdir())}"


def test_particle_segmentation_trains(tmp_path):
    """Real segmentation data (CC BY 4.0), including the AnyLabeling conversion."""
    from anylearning.training.models.semantic_segmentation.train import train_fn

    source = extract_real_dataset(
        "electron_microscopy_particle_segmentation", "train"
    )
    if source is None:
        pytest.skip("particle segmentation archive missing")

    # The dataset ships LabelMe/AnyLabeling sidecars; convert via the production
    # converter so this exercises the same path the importer uses.
    labels = convert_anylabeling_dir(source)
    assert labels, "conversion produced no labels"

    train_dir = tmp_path / "train"
    copied = subsample_segmentation_dir(source, train_dir, limit=4)
    assert copied, "no image/sidecar pairs found"

    label_set = [{"name": "background", "id": 0}] + [
        {"name": name, "id": i + 1} for i, name in enumerate(labels)
    ]

    save_dir = tmp_path / "output"
    config = {
        "save_dir": str(save_dir),
        "seed": 67,
        "data": {
            "train_dir": str(train_dir),
            "val_dir": str(train_dir),
            "test_dir": str(train_dir),
            "label_set": label_set,
            "img_size": 64,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "num_workers": 0,
            "ignore_index": 255,
        },
        "model": {
            "arch": "resnet18",
            "pretrained": None,
            "num_classes": len(label_set),
            "output_stride": 16,
        },
        "training": {
            "gradient_checkpointing": False,
            "scheduler": "cosine",
            "resume": False,
            "epochs": 1,
            "optim": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "min_lr": 1e-6,
            "eps": 1e-6,
            "batch_size": 2,
            "fp16": False,
            "clip_grad_norm": 10,
            "accumulation_steps": 1,
            "verbose_steps": 1,
        },
    }
    config_path = tmp_path / "cfg_semantic_segmentation.yml"
    config_path.write_text(yaml.safe_dump(config))

    train_fn(str(config_path))

    written = list(save_dir.glob("*.pth")) + list(save_dir.glob("*.pt"))
    assert written, f"no checkpoint in {save_dir}: {list(save_dir.iterdir())}"


def test_helmet_jacket_detection_data_loads():
    """The detection dataset extracts and carries AnyLabeling annotations."""
    source = extract_real_dataset("helmet_jacket", "train")
    if source is None:
        pytest.skip("helmet/jacket archive missing")

    images = [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sidecars = list(source.glob("*.json"))
    assert images, "no images extracted"
    assert sidecars, "no annotations extracted"
