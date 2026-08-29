"""Where pretrained weights come from, so a machine with no network can train.

The app promises offline operation, but the first training run of
each architecture downloaded its backbone. NanoDet fetched ShuffleNetV2 from
download.pytorch.org, classification fetched a torchvision ResNet, segmentation
fetched an encoder from the Hugging Face hub, and instance segmentation fetched
350-500 MB of Mask R-CNN from dl.fbaipublicfiles.com. On a machine with no
internet the run failed *after* the dataset had been prepared, inside the
training subprocess, where the user saw only a run that stopped.

Rather than intercept each library's download, this points all three caches at
a directory shipped with the app. Each library then finds its file exactly where
it expects to and never asks the network -- no patching, no URL rewriting, and
nothing to keep in step when a library changes how it downloads.

`fetch_weights.py` fills that directory by running the same loaders, so the
layout is whatever the loaders themselves produce rather than a guess.
"""

from __future__ import annotations

import os
import pathlib

import anylearning

#: The packaged copy first: `build_app.sh` includes this directory next to the
#: package, and that copy is the one a user has. The repository root is the
#: development fallback.
_CANDIDATES = (
    pathlib.Path(anylearning.__file__).parent / "weights",
    pathlib.Path(anylearning.__file__).parent.parent / "weights",
)

#: The environment variables each library reads for its cache. Set together
#: because a run touches more than one: detection uses torch hub, segmentation
#: uses the Hugging Face hub, instance segmentation uses iopath.
CACHE_VARIABLES = {
    # torchvision and torch.hub: <TORCH_HOME>/hub/checkpoints/<file>
    "TORCH_HOME": ".",
    # segmentation_models_pytorch, through huggingface_hub
    "HF_HOME": "huggingface",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    # detectron2, through iopath/fvcore
    "FVCORE_CACHE": "fvcore",
    # RF-DETR's own checkpoint cache, which otherwise defaults to
    # ~/.roboflow/models. The trainer hands RF-DETR an absolute path to a
    # bundled file and so never reaches the downloader -- this is here for the
    # case that stops being true, so that anything RF-DETR does decide to fetch
    # lands beside the rest of the weights rather than in a hidden folder in
    # the user's home directory.
    "RF_HOME": "rfdetr",
}


def bundled_dir() -> pathlib.Path | None:
    """The shipped weights directory, or None if this build has none."""
    for candidate in _CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def _is_writable(directory: pathlib.Path) -> bool:
    """Whether a file can be created in `directory`."""
    probe = directory / ".anylearning-write-probe"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def _would_download(fvcore: pathlib.Path) -> bool:
    """Whether anything is likely to need *writing* into this cache.

    The mirror exists so iopath has somewhere writable. Since the trainers now
    hand detectron2 a resolved local path rather than a URL (see
    `maskrcnn/train.py`), nothing writes here at all when the checkpoints are
    present -- no download, and no lock, because a local path goes through
    `NativePathHandler`.

    So the redirect is worth doing only when they are *absent*, which means a
    build without bundled weights that is about to fetch them. Checking for
    that rather than redirecting unconditionally is what keeps 426 MB out of a
    Windows user's data root: os.link from Program Files is denied to a
    standard user (CreateHardLinkW needs write access on the source), so the
    mirror there is always a full copy, of files nothing was going to read
    through it.
    """
    return not any(fvcore.rglob("*.pkl"))


def _writable_mirror(source: pathlib.Path) -> pathlib.Path:
    """A writable view of `source` under the data root, hardlinked where it can be.

    For the one cache that needs to *write* into its own directory in order to
    read from it. detectron2 goes through iopath, which takes a portalocker
    lock on "<weight>.pkl.lock" beside the file before opening it, so a
    read-only weights directory fails with a PermissionError on the lock -- and
    instance segmentation cannot train at all from an installed location unless
    the application runs as administrator. Windows found it, because
    Program Files is read-only for a standard user, but the same holds for any
    installation the user does not own.

    Hardlinks, falling back to copies: the Mask R-CNN checkpoints here are
    350-500 MB each, and on the same volume a link costs nothing. The mirror is
    only the fvcore subtree, not the whole weights directory -- torch.hub and
    huggingface_hub read without locking and keep pointing at the bundle.
    """
    import shutil

    from anylearning import config

    target = pathlib.Path(config.DATA_ROOT) / "weights-cache" / source.name
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        # Never mirror the lock files, and this is the whole point of the
        # function rather than a detail. A hardlink shares an inode, and mode
        # travels with the inode -- so linking `model_final_x.pkl.lock` out of
        # a read-only installation produces a read-only lock in the mirror, and
        # iopath fails to open it for writing exactly as it did before, one
        # directory further along. Measured on macOS: the redirect worked, the
        # checkpoints hardlinked, and the run still died with EACCES on the
        # lock.
        #
        # Skipping them is enough. iopath creates the lock it wants, and the
        # mirror directory is writable, which was the only thing missing. The
        # locks are zero bytes and ship only because `fetch_weights.py` runs
        # the real loaders and iopath leaves them behind.
        if item.suffix == ".lock":
            continue
        destination = target / item.relative_to(source)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, destination)
        except OSError:
            shutil.copy2(item, destination)
    return target


def use_bundled(target: pathlib.Path | None = None) -> pathlib.Path | None:
    """Point every download cache at the bundled weights.

    Called before torch, smp or detectron2 are asked for a model -- once a
    library has read its cache variable, changing it does nothing.

    An existing value is left alone: someone who has set TORCH_HOME on purpose,
    or `fetch_weights.py` pointing it at a directory being filled, means it more
    than this default does.
    """
    root = target or bundled_dir()
    if root is None:
        return None

    for variable, relative in CACHE_VARIABLES.items():
        if os.environ.get(variable):
            continue
        path = root if relative == "." else root / relative
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # An installed tree the user cannot write to. Only fvcore needs to,
            # and it is handled below; the rest read happily from a read-only
            # directory that already contains what they want.
            pass
        if (
            variable == "FVCORE_CACHE"
            and path.is_dir()
            and not _is_writable(path)
            and _would_download(path)
        ):
            path = _writable_mirror(path)
        os.environ[variable] = str(path)

    # Belt and braces for the segmentation encoders: with the files present,
    # this stops huggingface_hub from reaching out to check for a newer
    # revision, which on a machine with no network is a timeout rather than a
    # failure -- minutes of waiting before the run continues.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return root


def seed_auto_labeling_models() -> list[str]:
    """Copy bundled auto-labelling models into the data root, once.

    Unlike the training backbones, these are not read from a library cache:
    `ModelManager` looks for them under `~/anylearning-data/models/<name>/` and
    downloads anything missing. So the bundled copies are placed there on first
    run, with the `has_downloaded` marker their config carries, and the download
    never happens.

    Copied rather than symlinked or read in place: the data root is the user's,
    they may delete a model to reclaim space, and that must not reach into the
    installation. Skipped when a model is already there -- a user who replaced
    one keeps their copy.

    "Already there" means the weights are there, not that a config.yaml is:
    `ModelManager.load_model_configs()` writes a stub config for every model it
    knows about, marked `has_downloaded: false`, and it runs at import time --
    before this does. Skipping on the config alone therefore skipped every
    model on a fresh install, and the first click on MobileSAM downloaded 40 MB
    that was sitting inside the application the whole time. On a machine with
    no network, which is the machine this product is sold for, it simply failed.
    """
    root = bundled_dir()
    if root is None:
        return []
    source = root / "auto_labeling"
    if not source.is_dir():
        return []

    import shutil

    from anylearning import config

    def has_weights(directory: pathlib.Path) -> bool:
        return any(
            path.is_file() and path.name != "config.yaml"
            for path in directory.glob("*")
        )

    seeded = []
    destination_root = pathlib.Path(config.DATA_ROOT) / "models"
    for model in sorted(source.iterdir()):
        if not (model / "config.yaml").is_file() or not has_weights(model):
            continue
        destination = destination_root / model.name
        if has_weights(destination):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # dirs_exist_ok, and the config last: the destination usually exists
        # already, holding the stub written at import time, and that stub says
        # the model is not downloaded.
        shutil.copytree(model, destination, dirs_exist_ok=True)
        # And writable afterwards, whatever the installation's modes were.
        # copytree copies permission bits, so seeding out of a read-only
        # installation -- /opt, Program Files, an app bundle -- left the user
        # unable to delete a model to reclaim space or to replace one by
        # downloading it again, in a directory that is supposed to be theirs.
        # Same defect that stopped a read-only install starting twice; this is
        # the other place it copies out of the bundle.
        from anylearning.utils.file import make_writable

        make_writable(destination)
        seeded.append(model.name)
    return seeded


def describe() -> dict:
    """What is bundled, for the health endpoint and for the self-test."""
    root = bundled_dir()
    if root is None:
        return {"bundled": False, "path": None, "files": 0, "bytes": 0}

    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "bundled": True,
        "path": str(root),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }
