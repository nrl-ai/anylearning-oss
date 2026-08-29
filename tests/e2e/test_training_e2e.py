"""End-to-end training / export smoke tests on openly licensed data.

These run the real ``train_fn`` entry points against the procedurally generated
shapes dataset from ``tests.fixtures.datasets`` -- no network, no licence
encumbrance, and small enough to stay in the normal test run.

The point is not accuracy. It is that the whole path still works after a
dependency upgrade: data loading -> model construction -> optimiser step ->
checkpoint -> ONNX export -> onnxruntime inference.

Every model here is constructed with ``pretrained: None`` so the tests never
reach for the network.
"""

from pathlib import Path

import pytest
import yaml

from tests.fixtures.datasets import (
    build_classification_dataset,
    build_detection_coco,
    build_segmentation_dataset,
    fetch_oxford_pet,
)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def shapes_classification(tmp_path_factory):
    root = tmp_path_factory.mktemp("cls_data")
    classes = build_classification_dataset(root, per_class=4, size=32, seed=1)
    return root, classes


@pytest.fixture(scope="module")
def shapes_segmentation(tmp_path_factory):
    root = tmp_path_factory.mktemp("seg_data")
    labels = build_segmentation_dataset(root, per_class=3, size=32, seed=2)
    return root, labels


def test_classification_trains_and_exports_onnx(shapes_classification, tmp_path):
    """Full classification path: train -> checkpoint -> ONNX -> onnxruntime."""
    from anylearning.training.models.classification.train import train_fn

    data_root, classes = shapes_classification
    save_dir = tmp_path / "output"

    config = {
        "save_dir": str(save_dir),
        "seed": 67,
        "data": {
            "train_dir": str(data_root / "train"),
            "val_dir": str(data_root / "val"),
            "test_dir": str(data_root / "test"),
            "class_names": classes,
            "img_size": 32,
            "num_workers": 0,
        },
        "model": {"arch": "resnet18", "pretrained": None, "num_classes": len(classes)},
        "training": {
            "gradient_checkpointing": False,
            "scheduler": "cosine",
            "resume": False,
            "epochs": 2,
            "optim": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "min_lr": 1e-6,
            "eps": 1e-6,
            "batch_size": 2,
            "fp16": False,  # CPU autocast for fp16 is not supported
            "clip_grad_norm": 10,
            "accumulation_steps": 1,
        },
    }
    config_path = tmp_path / "cfg_classification.yaml"
    config_path.write_text(yaml.safe_dump(config))

    train_fn(str(config_path))

    checkpoints = list(save_dir.glob("*.pth")) + list(save_dir.glob("*.pt"))
    assert checkpoints, f"no checkpoint written to {save_dir}: {list(save_dir.iterdir())}"

    # The exported graph must actually load and run.
    from anylearning.training.device_utils import load_torch_model_for_export

    model = load_torch_model_for_export(str(checkpoints[0]))
    onnx_path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
    )
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    import numpy as np

    outputs = session.run(None, {"input": np.random.rand(1, 3, 32, 32).astype("float32")})
    assert outputs[0].shape == (1, len(classes))


def test_semantic_segmentation_trains(shapes_segmentation, tmp_path):
    """Full semantic segmentation path: annotated fixtures -> train -> checkpoint."""
    from anylearning.training.models.semantic_segmentation.train import train_fn

    data_root, labels = shapes_segmentation
    save_dir = tmp_path / "output"

    # The trainer expects background at id 0, so shift the shape classes up.
    label_set = [{"name": "background", "id": 0}] + [
        {"name": item["name"], "id": item["id"] + 1} for item in labels
    ]

    config = {
        "save_dir": str(save_dir),
        "seed": 67,
        "data": {
            "train_dir": str(data_root / "train"),
            "val_dir": str(data_root / "val"),
            "test_dir": str(data_root / "test"),
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
    assert written, f"no checkpoint written to {save_dir}: {list(save_dir.iterdir())}"


def test_detection_fixture_is_valid_coco(tmp_path):
    """The detection/instance-seg fixture must be loadable by pycocotools."""
    pycocotools = pytest.importorskip("pycocotools.coco")

    root = tmp_path / "det"
    names = build_detection_coco(root, per_class=3, size=32, seed=3)

    coco = pycocotools.COCO(str(root / "train.json"))
    assert len(coco.getCatIds()) == len(names)
    assert len(coco.getImgIds()) == 3 * len(names)

    for ann in coco.loadAnns(coco.getAnnIds()):
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0, "degenerate box in fixture"
        # Polygons are flat [x1, y1, x2, y2, ...] and need >= 3 points.
        assert len(ann["segmentation"][0]) >= 6


def test_oxford_pet_is_opt_in():
    """The CC BY-SA dataset must never download implicitly.

    Keeping this opt-in is what makes the default test run offline, and keeps
    ShareAlike content out of the working tree.
    """
    from tests.fixtures.datasets import oxford_pet_cache_dir

    cache = oxford_pet_cache_dir()
    repo_root = Path(__file__).resolve().parents[2]
    assert repo_root not in cache.resolve().parents, (
        f"Oxford-IIIT Pet would be cached inside the repo at {cache}"
    )

    # download=False must not touch the network.
    result = fetch_oxford_pet(download=False)
    assert result is None or (result / "images").is_dir()


def test_fixture_data_is_deterministic(tmp_path):
    """Same seed must produce identical bytes, or 'reproducible' runs are a lie."""
    a = build_classification_dataset(tmp_path / "a", per_class=2, size=32, seed=42)
    b = build_classification_dataset(tmp_path / "b", per_class=2, size=32, seed=42)
    assert a == b

    first = (tmp_path / "a" / "train" / "0" / "circle_000.jpg").read_bytes()
    second = (tmp_path / "b" / "train" / "0" / "circle_000.jpg").read_bytes()
    assert first == second
