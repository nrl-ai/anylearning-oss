"""Every project type, every model variant, end to end.

`anylearning/config.py` advertises a fixed set of MODEL_VARIANTS in the UI. This
module drives each of them through the real training entry point on generated
data, so a dependency bump cannot quietly break one task type while the others
stay green. The variant lists here are asserted against `config.MODEL_VARIANTS`,
so adding a variant to the UI without adding it here fails the suite.

One exception, and it is deliberate: detection and instance segmentation each
offer variants from a second trainer, RF-DETR, whose runs need checkpoints that
ship with the application rather than with the source. Those live in
`test_rfdetr_e2e.py`, which skips when the checkpoints are absent, and the
assertions here filter to the architecture each test goes on to train -- a list
naming every variant while training half of them would be the worse lie.

These are smoke tests: one epoch, tiny images. They prove the flow still runs,
not that the model is any good.

Unlike `test_training_e2e.py`, some of these need the network — NanoDet and
Mask R-CNN configs load pretrained backbones, which is what real training does.
"""

import pytest
import yaml

from anylearning import config as anylearning_config
from tests.fixtures.datasets import (
    build_classification_dataset,
    build_detection_coco,
    build_detection_yolo,
    build_handpose_dataset,
    build_segmentation_dataset,
)

torch = pytest.importorskip("torch")


class RecordingLogger:
    """Stands in for TrainingLogsWriter without a database."""

    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(str(message))

    def write_metrics(self, metrics):
        self.messages.append(str(metrics))


def _variants(project_type):
    return anylearning_config.MODEL_VARIANTS[project_type]


@pytest.fixture
def pristine_nanodet_cfg():
    """Restore NanoDet's module-level ``cfg`` between runs.

    ``load_config`` *merges* into a global CfgNode, so training one variant then
    another in the same process leaks keys across them -- the PAN-based
    lightweight config leaves ``num_outs`` behind, which the GhostPAN-based
    medium/large configs then reject with a TypeError.

    Production is unaffected: `routers/training.py` runs each job in its own
    `multiprocessing.Process`, so the global starts clean every time. This only
    bites a test process that trains several variants in a row.
    """
    import copy

    from nanodet.util import cfg

    snapshot = copy.deepcopy(cfg)
    yield cfg
    cfg.defrost()
    cfg.clear()
    cfg.update(copy.deepcopy(snapshot))


@pytest.fixture
def clean_detectron2_catalog():
    """Drop detectron2's global dataset registrations between runs.

    `maskrcnn/train.py` calls `register_coco_instances("train_ds", ...)` with a
    fixed name. Registering the same name twice with a different `json_file`
    raises "Attribute 'json_file' ... cannot be set to a different value", so a
    second variant in the same process fails.

    As with NanoDet's global cfg, production sidesteps this by running each
    training job in its own process.
    """
    detectron2_data = pytest.importorskip("detectron2.data")
    names = ("train_ds", "val_ds")

    def clear():
        for name in names:
            detectron2_data.DatasetCatalog.pop(name, None)
            detectron2_data.MetadataCatalog.pop(name, None)

    clear()
    yield
    clear()


def _checkpoints(save_dir):
    return (
        list(save_dir.glob("**/*.pth"))
        + list(save_dir.glob("**/*.pt"))
        + list(save_dir.glob("**/*.ckpt"))
    )


# --------------------------------------------------------------------------
# Image Classification -- ResNet18 / ResNet34
# --------------------------------------------------------------------------


def test_classification_variants_match_config():
    assert [v["model_architecture"] for v in _variants("Image Classification")] == [
        "resnet18",
        "resnet34",
    ]


@pytest.mark.parametrize("arch", ["resnet18", "resnet34"])
def test_classification_flow(arch, tmp_path):
    from anylearning.training.models.classification.train import train_fn

    data_root = tmp_path / "data"
    classes = build_classification_dataset(data_root, per_class=4, size=32, seed=1)
    save_dir = tmp_path / "out"

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
        "model": {"arch": arch, "pretrained": None, "num_classes": len(classes)},
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
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))

    train_fn(str(path), RecordingLogger())
    checkpoints = _checkpoints(save_dir)
    assert checkpoints, f"{arch}: no checkpoint in {save_dir}"

    # The app's "test this model on an image" path.
    import numpy as np

    from anylearning.training.trainers.classification_trainer import (
        ClassificationTrainer,
    )

    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result, _ = ClassificationTrainer.run_inference(
        yaml.safe_dump(config), str(checkpoints[0]), image
    )
    probabilities = next(iter(result.values()))
    assert set(probabilities) <= set(classes)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-3, "softmax should sum to 1"


# --------------------------------------------------------------------------
# Image Segmentation -- DeepLabV3+ over ResNet18 / 34 / 50
# --------------------------------------------------------------------------


def test_semseg_variants_match_config():
    assert [v["model_architecture"] for v in _variants("Image Segmentation")] == [
        "resnet18",
        "resnet34",
        "resnet50",
    ]


@pytest.mark.parametrize("arch", ["resnet18", "resnet34", "resnet50"])
def test_semseg_flow(arch, tmp_path):
    from anylearning.training.models.semantic_segmentation.train import train_fn

    data_root = tmp_path / "data"
    labels = build_segmentation_dataset(data_root, per_class=2, size=32, seed=2)
    # run_inference reads a "color" per label to build the visualisation, so the
    # label set carries one here as it does in a real project.
    palette = ["#000000", "#FF0000", "#00FF00", "#0000FF"]
    label_set = [{"name": "background", "id": 0, "color": palette[0]}] + [
        {"name": item["name"], "id": item["id"] + 1, "color": palette[item["id"] + 1]}
        for item in labels
    ]
    save_dir = tmp_path / "out"

    config = {
        "save_dir": str(save_dir),
        "seed": 67,
        "data": {
            "train_dir": str(data_root / "train"),
            "val_dir": str(data_root / "val"),
            "test_dir": str(data_root / "test"),
            "label_set": label_set,
            "img_size": 64,
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "num_workers": 0,
            "ignore_index": 255,
        },
        "model": {
            "arch": arch,
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
    path = tmp_path / "cfg.yml"
    path.write_text(yaml.safe_dump(config))

    train_fn(str(path), RecordingLogger())
    checkpoints = _checkpoints(save_dir)
    assert checkpoints, f"{arch}: no checkpoint in {save_dir}"

    import numpy as np

    from anylearning.training.trainers.semseg_trainer import SemSegTrainer

    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result, visualisation = SemSegTrainer.run_inference(
        yaml.safe_dump(config), str(checkpoints[0]), image
    )
    assert result is not None
    assert visualisation is not None


# --------------------------------------------------------------------------
# Object Detection -- NanoDet lightweight / medium / large
# --------------------------------------------------------------------------


def test_detection_variants_match_config():
    """The NanoDet sizes this file trains, in the order the dialog offers them.

    RF-DETR's variants share the project type and are checked in
    tests/e2e/test_rfdetr_e2e.py, which needs the bundled checkpoints and skips
    without them -- so this filters rather than listing everything, and stays a
    statement about the trainer it goes on to exercise.
    """
    nanodet = [
        v for v in _variants("Object Detection") if v["model_architecture"] == "nanodet"
    ]
    assert [v["model_size"] for v in nanodet] == [
        "lightweight",
        "medium",
        "large",
    ]


# A model trained for one epoch on a dozen 64px images predicts no boxes, so
# NanoDet's evaluator warns that the detection result is empty. That is the
# correct thing for it to say here -- scoped to this test rather than ignored
# globally, where it would hide a genuine "training produced nothing" regression.
@pytest.mark.filterwarnings("ignore:Detection result is empty")
@pytest.mark.parametrize("model_size", ["lightweight", "medium", "large"])
def test_detection_flow(model_size, tmp_path, pristine_nanodet_cfg):
    """Builds the config exactly as NanoDetTrainer.prepare_config does."""
    from anylearning.training.logging import NanoDetLogger
    from anylearning.training.models.nanodet.tools.train import main as nanodet_train
    from anylearning.training.trainers.nanodet_trainer import (
        CONFIG_TEMPLATES,
        TrainArgs,
    )

    data_root = tmp_path / "data"
    labels = build_detection_yolo(data_root, per_class=4, size=64, seed=3)
    class_names = [item["name"] for item in labels]
    save_dir = tmp_path / "out"

    with open(CONFIG_TEMPLATES[model_size]) as f:
        config = yaml.safe_load(f)

    config["save_dir"] = str(save_dir)
    config["class_names"] = class_names
    for subset in ("train", "val"):
        config["data"][subset]["class_names"] = class_names
        config["data"][subset]["img_path"] = str(data_root / subset)
        config["data"][subset]["ann_path"] = str(data_root / subset)
    config["device"]["batchsize_per_gpu"] = 2
    config["device"]["workers_per_gpu"] = 0
    config["device"]["gpu_ids"] = [0] if torch.cuda.is_available() else "-1"
    config["schedule"]["total_epochs"] = 1
    config["schedule"]["val_intervals"] = 1
    # The fixture is far smaller than a real dataset, so drop the logging
    # interval below the batch count -- otherwise Lightning (correctly) warns
    # that it will never log, and warnings are errors here.
    config["log"]["interval"] = 1
    if "aux_head" in config["model"]["arch"]:
        config["model"]["arch"]["aux_head"]["num_classes"] = len(class_names)
    config["model"]["arch"]["head"]["num_classes"] = len(class_names)

    path = tmp_path / "nanodet.yml"
    path.write_text(yaml.safe_dump(config))

    # main() calls load_config itself; doing it here as well is redundant.
    args = TrainArgs()
    args.config = str(path)
    nanodet_train(
        args, logger=NanoDetLogger(writer=RecordingLogger(), save_dir=str(save_dir))
    )

    checkpoints = _checkpoints(save_dir)
    assert checkpoints, f"{model_size}: no checkpoint in {save_dir}"

    # ONNX export is the last step of every real training job, so the flow is not
    # verified until the graph exists and actually loads.
    from nanodet.export_onnx import convert_onnx

    onnx_path = save_dir / "exported_model.onnx"
    convert_onnx(str(path), str(checkpoints[0]), str(onnx_path))
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0

    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs(), f"{model_size}: exported graph has no inputs"

    import numpy as np

    from anylearning.training.trainers.nanodet_trainer import NanoDetTrainer

    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    detections, visualisation = NanoDetTrainer.run_inference(
        yaml.safe_dump(config), str(checkpoints[0]), image
    )
    # An undertrained detector legitimately returns nothing; the contract is that
    # it returns without raising and hands back a visualisation.
    assert detections is not None
    assert visualisation is not None


# --------------------------------------------------------------------------
# Instance Segmentation -- Mask R-CNN ResNet50 / ResNet101
# --------------------------------------------------------------------------


def test_instseg_variants_match_config():
    """The Mask R-CNN backbones this file trains. See the detection note above:
    RF-DETR-Seg shares this project type and is exercised elsewhere."""
    maskrcnn = [
        v
        for v in _variants("Instance Segmentation")
        if v["model_architecture"].startswith("maskrcnn")
    ]
    assert [v["model_architecture"] for v in maskrcnn] == [
        "maskrcnn-resnet50",
        "maskrcnn-resnet101",
    ]


def test_keypoint_variant_matches_config():
    assert _variants("Keypoint Detection") == [
        {
            "name": "RF-DETR-Keypoint-Preview",
            "model_architecture": "rfdetr-keypoint",
            "model_size": "preview",
        }
    ]


@pytest.mark.parametrize("backbone", ["resnet50", "resnet101"])
def test_instseg_flow(backbone, tmp_path, clean_detectron2_catalog):
    pytest.importorskip("detectron2")
    from anylearning.training.models.instance_segmentation.maskrcnn.train import (
        train_fn,
    )

    data_root = tmp_path / "data"
    class_names = build_detection_coco(data_root, per_class=2, size=64, seed=4)
    save_dir = tmp_path / "out"

    config = {
        "save_dir": str(save_dir),
        "detectron2_cfg_file": "detectron2_cfg.pkl",
        "seed": 67,
        "data": {
            "train_dir": str(data_root / "train"),
            "val_dir": str(data_root / "val"),
            "train_ann_file": str(data_root / "train.json"),
            "val_ann_file": str(data_root / "val.json"),
            "img_size": 64,
            "num_workers": 0,
            "label_set": [
                {"id": i + 1, "name": n, "color": "#FF0000"}
                for i, n in enumerate(class_names)
            ],
        },
        "model": {
            "arch": "maskrcnn",
            "backbone": backbone,
            "pretrained": "coco_lsj",
            "num_classes": len(class_names),
        },
        "training": {
            "scheduler": "WarmupCosineLR",
            "epochs": 1,
            "batch_size": 2,
            "verbose_steps": 1,
            "optim": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.001,
                "betas": [0.9, 0.999],
            },
            "min_lr": 1e-7,
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))

    train_fn(str(path), RecordingLogger())
    checkpoints = _checkpoints(save_dir)
    assert checkpoints, f"{backbone}: no checkpoint in {save_dir}"

    # Inference reads the detectron2 config pickled during training, so this also
    # checks that training wrote it where run_inference expects to find it.
    import numpy as np

    from anylearning.training.trainers.instseg_trainer import InstSegTrainer

    assert (save_dir / "detectron2_cfg.pkl").is_file(), (
        "training did not pickle the d2 cfg"
    )

    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    predictions, visualisation = InstSegTrainer.run_inference(
        yaml.safe_dump(config), str(checkpoints[0]), image
    )
    assert predictions is not None
    assert visualisation is not None


# --------------------------------------------------------------------------
# Handpose Classification -- MLP small / medium / large
# --------------------------------------------------------------------------


def test_handpose_variants_match_config():
    assert [v["model_size"] for v in _variants("Handpose Classification")] == [
        "lightweight",
        "medium",
        "large",
    ]


@pytest.mark.parametrize("model_size", ["lightweight", "medium", "large"])
def test_handpose_flow(model_size, tmp_path):
    from anylearning.training.logging import MLPLogger
    from anylearning.training.models.handpose.handpose.tools.train import train
    from anylearning.training.trainers.handpose_classification_trainer import (
        HANDPOSE_CONFIG_TEMPLATE,
    )

    data_root = tmp_path / "data"
    labels = build_handpose_dataset(data_root, per_class=6, seed=5)
    class_names = [item["name"] for item in labels]
    save_dir = tmp_path / "out"

    with open(HANDPOSE_CONFIG_TEMPLATE[model_size]) as f:
        config = yaml.safe_load(f)

    config["save_dir"] = str(save_dir)
    config["class_names"] = class_names
    for subset in ("train", "val", "test"):
        config["data"][subset]["annotation_path"] = str(data_root / subset)
        config["data"][subset]["class_names"] = class_names
        config["data"][subset]["batch_size"] = 2
        config["data"][subset]["num_workers"] = 0
    config["models"]["arch"]["head"]["output_units"] = len(class_names)
    config["schedule"]["epochs"] = 1
    config["schedule"]["warmup"]["steps"] = 1

    path = tmp_path / "mlp.yml"
    path.write_text(yaml.safe_dump(config))

    train(str(path), logger=MLPLogger(writer=RecordingLogger(), save_dir=str(save_dir)))
    checkpoints = _checkpoints(save_dir)
    assert checkpoints, f"{model_size}: no checkpoint in {save_dir}"

    # Same as the other flows: export is part of the job, so exercise it.
    import onnx

    from anylearning.training.models.handpose.handpose.models.mlp import MLP

    state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    model = MLP(config)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    onnx_path = save_dir / "exported_model.onnx"
    # dynamo=False, because that is what every trainer passes: the dynamo
    # exporter routes through onnxscript, which reads function source, and a
    # packaged binary has none. Exporting here by a route the app never takes
    # would test a graph nobody ships.
    torch.onnx.export(model, torch.randn(1, 63), str(onnx_path), dynamo=False)
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)

    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    import numpy as np

    sample = np.random.rand(1, 63).astype("float32")
    outputs = session.run(None, {session.get_inputs()[0].name: sample})
    assert outputs[0].shape == (1, len(class_names))

    # The batch norm layers used to be skipped for a batch of one in *every*
    # mode, so an export -- which traces exactly one sample -- silently dropped
    # them and stopped matching the network that was trained. Nothing failed;
    # the predictions were simply wrong.
    #
    # Compared numerically rather than by looking for a BatchNormalization node
    # in the graph. A correct export may fold batch norm into the preceding
    # Gemm -- the dynamo exporter always does -- and a structural check cannot
    # tell folding from dropping. The numbers can: dropped layers move the
    # output by ~1e-2, folded ones by ~1e-8.
    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()
    difference = float(np.max(np.abs(expected - outputs[0])))
    assert difference < 1e-4, (
        f"{model_size}: exported model disagrees with the trained one by "
        f"{difference:.2e} -- batch norm was dropped rather than folded"
    )
