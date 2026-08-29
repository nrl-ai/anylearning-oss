#! /usr/bin/env python
"""Download every pretrained weight the app can ask for, into `weights/`.

Run once before building. `build_app.sh` ships the directory, and
`anylearning/weights.py` points each library's cache at it, so a user's first
training run finds its backbone on disk instead of on the network.

The weights are fetched **by running the same loaders the trainers run**, not by
downloading URLs into a layout of our own. That matters: each library decides
its own cache path and file name, and a hand-built layout is a guess that stops
being true the first time one of them changes. Whatever these calls leave
behind is by definition what the loaders will look for.

    python fetch_weights.py            # everything, skipping what is present
    python fetch_weights.py --list     # what would be fetched, and from where

About 1.7 GB in total: the two Mask R-CNN checkpoints and the four RF-DETR ones
between them are most of it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
TARGET = ROOT / "weights"


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit != "GB" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def fetch_torchvision_classifiers() -> None:
    """The classification backbones, and NanoDet's ShuffleNetV2.

    Both go through torch.hub's checkpoint cache, so one download populates it
    for whichever trainer asks next.
    """
    import torchvision.models as models

    for name, weights in (
        ("resnet18", "ResNet18_Weights.DEFAULT"),
        ("resnet34", "ResNet34_Weights.DEFAULT"),
        ("resnet50", "ResNet50_Weights.DEFAULT"),
    ):
        print(f"  torchvision {name}")
        models.get_model(name, weights=weights)


def fetch_nanodet_backbones() -> None:
    from torch.hub import load_state_dict_from_url

    from nanodet.model.backbone.shufflenetv2 import model_urls

    for name, url in model_urls.items():
        if url is None:
            continue
        print(f"  nanodet {name}")
        load_state_dict_from_url(url, progress=False)


def fetch_segmentation_encoders() -> None:
    """The smp encoders named in the deeplabv3 configs."""
    from segmentation_models_pytorch.encoders import get_encoder

    for name in ("resnet18", "resnet34", "resnet50"):
        print(f"  smp encoder {name}")
        get_encoder(name, weights="imagenet")


def fetch_detectron2_checkpoints() -> None:
    """Mask R-CNN R50 and R101 -- the two largest downloads in the product."""
    from detectron2.checkpoint import DetectionCheckpointer
    from iopath.common.file_io import PathManager

    urls = [
        "https://dl.fbaipublicfiles.com/detectron2/new_baselines/"
        "mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl",
        "https://dl.fbaipublicfiles.com/detectron2/new_baselines/"
        "mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl",
    ]
    # Through detectron2's own PathManager, so the file lands in the cache its
    # checkpointer will consult rather than somewhere we chose.
    manager = DetectionCheckpointer(None).path_manager or PathManager()
    for url in urls:
        print(f"  detectron2 {url.rsplit('/', 1)[-1]}")
        manager.get_local_path(url)


def fetch_rfdetr_checkpoints() -> None:
    """RF-DETR's COCO checkpoints, fetched through RF-DETR and then slimmed.

    Two steps rather than one, and the split is the point.

    The download goes through ``download_pretrain_weights``, which knows the
    URLs and verifies each file's MD5 -- the same code the application would run
    if it ever reached the network, so what lands here is what RF-DETR itself
    would have fetched.

    Then each file is stripped of its optimizer state and written under a name
    of ours. The detection checkpoints are training snapshots:
    ``rf-detr-nano.pth`` is 349 MB, of which 233 MB is optimizer state that
    fine-tuning never reads. Renaming matters as much as shrinking -- a
    basename RF-DETR recognises is one it will checksum against a file that no
    longer matches. See anylearning/training/rfdetr_weights.py.
    """
    import shutil

    import torch
    from rfdetr.assets.model_weights import (
        download_pretrain_weights,
        get_model_cache_dir,
    )

    from anylearning.training import rfdetr_weights

    # RF_HOME, which weights.use_bundled has already pointed at the directory
    # being filled -- asked of RF-DETR rather than rebuilt here, for the same
    # reason the other steps run the real loaders: the layout is whatever the
    # library says it is.
    folder = pathlib.Path(get_model_cache_dir())
    cache = folder / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)

    for our_name, upstream_name in rfdetr_weights.CHECKPOINTS.items():
        destination = folder / our_name
        if destination.is_file():
            print(f"  {our_name} (already here)")
            continue

        print(f"  {upstream_name}")
        original = cache / upstream_name
        download_pretrain_weights(str(original))
        if not original.is_file():
            raise RuntimeError(f"RF-DETR did not download {upstream_name}")

        checkpoint = torch.load(original, map_location="cpu", weights_only=False)
        slimmed = rfdetr_weights.strip_training_state(checkpoint)
        if "model" not in slimmed:
            raise RuntimeError(
                f"{upstream_name} has no 'model' weights after stripping; "
                "RF-DETR's checkpoint format has changed."
            )
        torch.save(slimmed, destination)
        print(
            f"    -> {our_name} "
            f"({human(original.stat().st_size)} -> {human(destination.stat().st_size)})"
        )

    # The originals are three times the size of what ships and are needed only
    # to produce it. Leaving them behind would put them in the installer.
    shutil.rmtree(cache, ignore_errors=True)


#: The auto-labelling models bundled by default.
#:
#: Not all five. Together they are 1.6 GB on top of an installer that is
#: already over 3 GB, and the two largest -- SAM2 Base+ and Large -- are 1.2 GB
#: of that for accuracy most people labelling a few hundred images will not
#: notice. These three make auto-labelling work offline out of the box; the
#: other two still download on demand, as every model did before.
DEFAULT_AUTO_LABELING = (
    "mobile_sam_20230629",
    "sam2_hiera_tiny_20240803",
    "sam2_hiera_small_20240803",
)


def fetch_auto_labeling(target: pathlib.Path, wanted) -> None:
    """Download and unpack the auto-labelling models.

    Stored unpacked, in the shape ModelManager expects to find in the data
    root, because that is what gets copied there on first run -- shipping the
    zip would mean every user unpacks it again for no reason.
    """
    import io
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    import yaml

    catalogue = yaml.safe_load(
        (ROOT / "anylearning/configs/auto_labeling/models.yaml").read_text()
    )
    by_name = {model["name"]: model for model in catalogue}

    for name in wanted:
        model = by_name.get(name)
        if model is None:
            print(f"  unknown auto-labelling model {name!r}", file=sys.stderr)
            continue
        destination = target / "auto_labeling" / name
        existing = destination / "config.yaml"
        if existing.is_file():
            # "Already here" has to mean usable. An earlier version of this
            # script wrote a config with no `type`, which ModelManager cannot
            # load, and a plain existence check would keep that file forever on
            # every machine that had already run the old script -- including
            # the build machines.
            record = yaml.safe_load(existing.read_text()) or {}
            if record.get("type"):
                print(f"  {model['display_name']} (already here)")
                continue
            print(f"  {model['display_name']} (re-fetching: config has no type)")
            shutil.rmtree(destination)

        print(f"  {model['display_name']}")
        destination.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(model["download_url"], timeout=120) as response:
            archive = io.BytesIO(response.read())

        # Unpack the way ModelManager does: the archive carries its own
        # config.yaml, sometimes inside a folder, and that config is the only
        # place `type`, `encoder_model_path`, `decoder_model_path` and
        # `input_size` exist. models.yaml has none of them -- it is a catalogue
        # of names and download urls.
        with tempfile.TemporaryDirectory() as staging:
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(staging)
            folder = next(
                (
                    pathlib.Path(root)
                    for root, _, files in os.walk(staging)
                    if "config.yaml" in files
                ),
                None,
            )
            if folder is None:
                print(f"  no config.yaml inside {name}, skipping", file=sys.stderr)
                continue
            shutil.copytree(folder, destination, dirs_exist_ok=True)

        # Then the two markers, merged into the archive's config rather than
        # replacing it. Overwriting it shipped a config with no `type`, and
        # loading such a model died in a worker thread with KeyError: 'type' --
        # so the bundled models, the whole point of shipping them, could not be
        # loaded at all on a machine that had never downloaded them.
        record = yaml.safe_load((destination / "config.yaml").read_text())
        record["has_downloaded"] = True
        record["is_custom_model"] = False
        # config_file is an absolute path on whichever machine wrote it. The
        # application sets it for the user's own data root; a build machine's
        # path has no meaning there.
        record.pop("config_file", None)
        (destination / "config.yaml").write_text(yaml.dump(record))


def settle(target: pathlib.Path) -> None:
    """Make the directory safe to copy into a package: no symlinks, no junk.

    The Hugging Face cache stores each file once under `blobs/` and links to it
    from `snapshots/`. Nuitka's macOS build silently dropped every one of those
    links *and* the blobs behind them -- 236 MB of segmentation encoders, from a
    step whose log said "Included 41 data files". The Linux build kept them, so
    this is not something to discover per platform on release day: replacing
    each link with the file it points at leaves nothing to lose. The cache is
    read exactly the same way afterwards.

    Also drops what should never have been shipped: the lock directory, xet's
    session logs, and any stray dotfile the fetching machine dropped in.
    """
    import shutil

    for link in sorted(target.rglob("*")):
        if link.is_symlink():
            resolved = link.resolve()
            if not resolved.is_file():
                continue
            link.unlink()
            shutil.copy2(resolved, link)

    for junk in (
        target / "huggingface" / "hub" / ".locks",
        target / "huggingface" / "xet" / "logs",
    ):
        if junk.is_dir():
            shutil.rmtree(junk)
    for stray in target.rglob(".agent_harnesses.json"):
        stray.unlink()


STEPS = (
    ("torchvision classification backbones", fetch_torchvision_classifiers),
    ("NanoDet ShuffleNetV2 backbones", fetch_nanodet_backbones),
    ("segmentation encoders", fetch_segmentation_encoders),
    ("Mask R-CNN checkpoints", fetch_detectron2_checkpoints),
    ("RF-DETR checkpoints", fetch_rfdetr_checkpoints),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="say what would be fetched and stop"
    )
    parser.add_argument(
        "--target", type=pathlib.Path, default=TARGET, help="where to put them"
    )
    parser.add_argument(
        "--auto-labeling",
        nargs="*",
        default=list(DEFAULT_AUTO_LABELING),
        help="auto-labelling models to bundle; pass none to skip them",
    )
    arguments = parser.parse_args()

    if arguments.list:
        for label, _ in STEPS:
            print(f"  {label}")
        return 0

    arguments.target.mkdir(parents=True, exist_ok=True)

    # Set before importing torch: the cache variables are read at import time,
    # and use_bundled() deliberately does not overwrite what is already set --
    # which is what lets this script aim the same machinery at a directory it is
    # in the middle of filling.
    sys.path.insert(0, str(ROOT))
    from anylearning import weights

    weights.use_bundled(arguments.target)
    # This one has to fetch, so the offline default gets out of the way.
    os.environ["HF_HUB_OFFLINE"] = "0"
    print(f"Fetching into {arguments.target}")

    for label, step in STEPS:
        print(label)
        try:
            step()
        except Exception as error:  # noqa: BLE001
            print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            return 1

    if arguments.auto_labeling:
        print("auto-labelling models")
        try:
            fetch_auto_labeling(arguments.target, arguments.auto_labeling)
        except Exception as error:  # noqa: BLE001
            print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            return 1

    settle(arguments.target)

    described = weights.describe()
    print()
    print(f"{described['files']} files, {human(described['bytes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
