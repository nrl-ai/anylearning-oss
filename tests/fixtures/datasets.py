"""Openly licensed datasets for AnyLearning's training and inference tests.

Two sources, both chosen for licence safety (see ``docs/model_license_policy.md``):

* The ``build_*`` helpers generate images procedurally. Generated data carries no
  licence at all, needs no network, and is deterministic given a seed -- so the
  default test run stays fully offline, which is also how AnyLearning itself is
  meant to run.
* :func:`fetch_oxford_pet` downloads the Oxford-IIIT Pet dataset (CC BY-SA 4.0)
  for opt-in realistic runs. It is cached **outside** the repository so the
  ShareAlike term never reaches our source tree, and it is never committed.

The shapes are drawn so that a small model can actually separate them: each class
gets a distinct shape *and* a distinct colour. A smoke test that cannot reduce its
loss only proves the code did not crash, which is a much weaker claim.
"""

import json
import os
import pathlib
import random
import tarfile
import urllib.request
import zipfile

from PIL import Image, ImageDraw

# class name -> (shape, RGB fill)
DEFAULT_CLASSES = {
    "circle": ("circle", (220, 60, 60)),
    "square": ("square", (60, 120, 220)),
    "triangle": ("triangle", (60, 180, 90)),
}

SUBSETS = ("train", "val", "test")


def _draw_shape(draw, shape, box, fill):
    x0, y0, x1, y1 = box
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    elif shape == "triangle":
        draw.polygon([(x0, y1), ((x0 + x1) / 2, y0), (x1, y1)], fill=fill)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _polygon_for(shape, box):
    """Return the polygon outline used as a segmentation annotation."""
    x0, y0, x1, y1 = box
    if shape == "triangle":
        return [[x0, y1], [(x0 + x1) / 2, y0], [x1, y1]]
    if shape == "circle":
        # Approximate the ellipse with a coarse polygon; enough for a mask.
        import math

        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        return [
            [
                cx + rx * math.cos(2 * math.pi * i / 12),
                cy + ry * math.sin(2 * math.pi * i / 12),
            ]
            for i in range(12)
        ]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _render(rng, shape, fill, size):
    """Draw one shape on a noisy background and return (image, box)."""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    # A little background clutter keeps the task from being trivially separable
    # by mean pixel value alone.
    for _ in range(6):
        x, y = rng.randrange(size), rng.randrange(size)
        g = rng.randrange(200, 240)
        draw.point((x, y), fill=(g, g, g))

    span = size // 2
    x0 = rng.randrange(size // 8, size - span - size // 8 + 1)
    y0 = rng.randrange(size // 8, size - span - size // 8 + 1)
    box = (x0, y0, x0 + span, y0 + span)
    _draw_shape(draw, shape, box, fill)
    return img, box


def build_classification_dataset(
    root, classes=None, per_class=6, size=64, seed=0, subsets=SUBSETS
):
    """Write the layout ``ClassificationTrainer`` consumes.

    ``<root>/<subset>/<class_id>/<name>.jpg`` where ``class_id`` is the integer
    index of the class, matching ``DataItem.class_id``.

    Returns the list of class names, ordered by class id.
    """
    classes = classes or DEFAULT_CLASSES
    root = pathlib.Path(root)
    rng = random.Random(seed)

    for subset in subsets:
        for class_id, (name, (shape, fill)) in enumerate(classes.items()):
            out = root / subset / str(class_id)
            out.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img, _ = _render(rng, shape, fill, size)
                img.save(out / f"{name}_{i:03d}.jpg", quality=95)

    return list(classes)


def build_segmentation_dataset(
    root, classes=None, per_class=6, size=64, seed=0, subsets=SUBSETS
):
    """Write the layout ``SemSegTrainer`` consumes.

    Images sit directly in ``<root>/<subset>/`` with a same-named ``.json``
    sidecar holding AnyLearning's native annotation payload::

        [{"categories": ["circle"], "points": [[x, y], ...]}]

    Also writes ``<root>/labels.json``, which the trainer reads for the label set.
    """
    classes = classes or DEFAULT_CLASSES
    root = pathlib.Path(root)
    rng = random.Random(seed)

    for subset in subsets:
        out = root / subset
        out.mkdir(parents=True, exist_ok=True)
        for name, (shape, fill) in classes.items():
            for i in range(per_class):
                img, box = _render(rng, shape, fill, size)
                stem = f"{name}_{i:03d}"
                img.save(out / f"{stem}.jpg", quality=95)
                annotation = [
                    {"categories": [name], "points": _polygon_for(shape, box)}
                ]
                (out / f"{stem}.json").write_text(json.dumps(annotation))

    labels = [{"name": name, "id": i} for i, name in enumerate(classes)]
    (root / "labels.json").write_text(json.dumps(labels))
    return labels


def build_detection_coco(
    root, classes=None, per_class=6, size=64, seed=0, subsets=SUBSETS
):
    """Write a COCO-format detection dataset (NanoDet / instance-seg trainers).

    ``<root>/<subset>/`` holds the images and ``<root>/<subset>.json`` the COCO
    annotations, with both boxes and polygon segmentations so the same fixture
    serves detection and instance segmentation.
    """
    classes = classes or DEFAULT_CLASSES
    root = pathlib.Path(root)
    rng = random.Random(seed)
    class_names = list(classes)

    for subset in subsets:
        out = root / subset
        out.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        image_id = ann_id = 1

        for name, (shape, fill) in classes.items():
            for i in range(per_class):
                img, box = _render(rng, shape, fill, size)
                filename = f"{name}_{i:03d}.jpg"
                img.save(out / filename, quality=95)

                x0, y0, x1, y1 = box
                polygon = _polygon_for(shape, box)
                images.append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": size,
                        "height": size,
                    }
                )
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_names.index(name) + 1,  # COCO is 1-based
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": float((x1 - x0) * (y1 - y0)),
                        "iscrowd": 0,
                        "segmentation": [[c for point in polygon for c in point]],
                    }
                )
                image_id += 1
                ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": i + 1, "name": n, "supercategory": "shape"}
                for i, n in enumerate(class_names)
            ],
        }
        (root / f"{subset}.json").write_text(json.dumps(coco))

    return class_names


# --------------------------------------------------------------------------
# Real, openly licensed dataset
# --------------------------------------------------------------------------

OXFORD_PET_IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
OXFORD_PET_ANNOTATIONS_URL = (
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
)

OXFORD_PET_ATTRIBUTION = (
    "Oxford-IIIT Pet Dataset -- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "
    "'Cats and Dogs', CVPR 2012. Licensed CC BY-SA 4.0. "
    "https://www.robots.ox.ac.uk/~vgg/data/pets/"
)


def oxford_pet_cache_dir():
    """Cache location, deliberately outside the repository.

    Override with ``ANYLEARNING_TEST_DATA``. Keeping the download out of the
    working tree is what stops CC BY-SA content being committed by accident.
    """
    base = os.environ.get("ANYLEARNING_TEST_DATA")
    if base:
        return pathlib.Path(base) / "oxford-iiit-pet"
    return pathlib.Path.home() / ".cache" / "anylearning-test-data" / "oxford-iiit-pet"


def fetch_oxford_pet(download=False):
    """Return the cached Oxford-IIIT Pet root, downloading it only if asked.

    Args:
        download: when False (the default) this never touches the network and
            returns None if the dataset is absent, so tests can skip cleanly.

    The dataset is CC BY-SA 4.0. Attribution is written next to the data in
    ``ATTRIBUTION.txt`` -- see :data:`OXFORD_PET_ATTRIBUTION`.
    """
    root = oxford_pet_cache_dir()
    if (root / "images").is_dir() and (root / "annotations").is_dir():
        return root
    if not download:
        return None

    root.mkdir(parents=True, exist_ok=True)
    (root / "ATTRIBUTION.txt").write_text(OXFORD_PET_ATTRIBUTION + "\n")

    for url in (OXFORD_PET_IMAGES_URL, OXFORD_PET_ANNOTATIONS_URL):
        archive = root / os.path.basename(url)
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            # filter="data" refuses absolute paths and traversal entries.
            try:
                tar.extractall(root, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(root)

    return root


# --------------------------------------------------------------------------
# The AnyLearning-data repository
# --------------------------------------------------------------------------
#
# https://huggingface.co/datasets/nrl-ai/anylearning-data holds real datasets, shipped as
# per-subset zips. Only the licence-cleared ones are listed here -- see
# LICENSES.md in that repository. `neu_surface_defect` and `dental_segment` are
# deliberately absent: neither records a licence, so tests must not depend on
# them until that is resolved.

# name -> (task, licence, {subset: zip name})
REAL_DATASETS = {
    "zhanglabdata_chest_xray": (
        "Image Classification",
        "CC BY 4.0",
        {"train": "train.zip", "test": "test.zip"},
    ),
    "helmet_jacket": (
        "Object Detection",
        "Apache 2.0",
        {"train": "train.zip", "val": "val.zip", "test": "test.zip"},
    ),
    "electron_microscopy_particle_segmentation": (
        "Image Segmentation",
        "CC BY 4.0",
        {"train": "train.zip", "val": "val.zip"},
    ),
    "american_sign_language": (
        "Handpose Classification",
        "Public Domain",
        {"train": "train.zip", "valid": "valid.zip", "test": "test.zip"},
    ),
}


def anylearning_data_root():
    """Locate the anylearning-data checkout, or None if it is not present.

    Looks at ``ANYLEARNING_DATA_DIR`` first, then the conventional sibling
    checkout next to this repository. Returning None lets tests skip rather than
    fail on a machine that has not cloned the (multi-hundred-megabyte) data repo.
    """
    env = os.environ.get("ANYLEARNING_DATA_DIR")
    if env:
        root = pathlib.Path(env)
        return root if root.is_dir() else None

    sibling = pathlib.Path(__file__).resolve().parents[2].parent / "anylearning-data"
    return sibling if sibling.is_dir() else None


def real_dataset_cache_dir():
    """Where extracted datasets land. Never inside either repository."""
    base = os.environ.get("ANYLEARNING_TEST_DATA")
    if base:
        return pathlib.Path(base) / "extracted"
    return pathlib.Path.home() / ".cache" / "anylearning-test-data" / "extracted"


def extract_real_dataset(name, subset):
    """Extract one subset of a licence-cleared dataset and return its directory.

    Returns None when the data repository or the requested zip is missing, so
    callers can skip. Extraction is cached: the zip is only unpacked once.
    """
    if name not in REAL_DATASETS:
        raise KeyError(
            f"{name!r} is not licence-cleared for tests. "
            f"Known: {sorted(REAL_DATASETS)}. See anylearning-data/LICENSES.md."
        )

    root = anylearning_data_root()
    if root is None:
        return None

    _, _, subsets = REAL_DATASETS[name]
    if subset not in subsets:
        raise KeyError(f"{name!r} has no {subset!r} subset; has {sorted(subsets)}")

    archive = root / name / subsets[subset]
    if not archive.is_file():
        return None

    dest = real_dataset_cache_dir() / name
    marker = dest / f".{subset}.extracted"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                # Skip macOS resource-fork noise, and refuse absolute paths or
                # traversal entries rather than trusting the archive.
                base = pathlib.PurePosixPath(member).name
                if member.startswith("__MACOSX/") or base.startswith("._"):
                    continue
                if base == ".DS_Store":
                    continue
                if os.path.isabs(member) or ".." in pathlib.PurePosixPath(member).parts:
                    continue
                zf.extract(member, dest)
        marker.write_text("")

    # Zips contain a top-level directory named after the subset.
    inner = dest / subset
    return inner if inner.is_dir() else dest


def convert_anylabeling_dir(directory):
    """Rewrite AnyLabeling sidecars in place into AnyLearning's native format.

    The segmentation datasets ship LabelMe/AnyLabeling JSON (a dict with
    ``shapes``), while ``SegmentationDataset`` expects a flat list of
    ``{"categories": [...], "points": [...]}``. Reuses the production converter
    so the tests exercise the same code path the importer does.

    Returns the sorted set of label names encountered. This is idempotent: the
    extraction cache persists between runs, so a directory converted by an
    earlier run must still report its labels rather than an empty list.
    """
    from anylearning.utils.converters import convert_anylabeling_to_anylearning

    labels = set()
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "shapes" in payload:
            payload = convert_anylabeling_to_anylearning(payload)
            path.write_text(json.dumps(payload))
        for obj in payload:
            labels.update(obj.get("categories", []))
    return sorted(labels)


def subsample_image_folder(source, dest, per_class=4):
    """Copy at most ``per_class`` images from each class directory.

    The real datasets run to thousands of images. Tests want to prove the
    pipeline works, not to train to convergence, so they train on a slice.
    Selection is sorted (not random) to keep runs reproducible.
    """
    import shutil

    source, dest = pathlib.Path(source), pathlib.Path(dest)
    classes = []
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = [p for p in sorted(class_dir.iterdir()) if is_image_file(p.name)]
        if not images:
            continue
        out = dest / class_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for image in images[:per_class]:
            shutil.copy2(image, out / image.name)
        classes.append(class_dir.name)
    return classes


def subsample_segmentation_dir(source, dest, limit=6):
    """Copy at most ``limit`` image + sidecar pairs from a segmentation folder."""
    import shutil

    source, dest = pathlib.Path(source), pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    images = [p for p in sorted(source.iterdir()) if is_image_file(p.name)]
    copied = 0
    for image in images:
        sidecar = image.with_suffix(".json")
        if not sidecar.exists():
            continue
        shutil.copy2(image, dest / image.name)
        shutil.copy2(sidecar, dest / sidecar.name)
        copied += 1
        if copied >= limit:
            break
    return copied


def is_image_file(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def build_detection_yolo(
    root, classes=None, per_class=6, size=64, seed=0, subsets=SUBSETS
):
    """Write the layout ``NanoDetTrainer`` produces.

    NanoDet's configs use ``YoloDataset``, so each image gets a same-named
    ``.txt`` of ``class cx cy w h`` in normalised coordinates, in the same
    directory. ``labels.json`` sits at the root, as ``prepare_data`` writes it.

    Reuses the production converter so the fixture exercises the same code path
    the trainer does.
    """
    from anylearning.utils.converters import convert_anylearning_to_yolo

    classes = classes or DEFAULT_CLASSES
    root = pathlib.Path(root)
    rng = random.Random(seed)
    labels = [{"name": name, "id": i} for i, name in enumerate(classes)]

    for subset in subsets:
        out = root / subset
        out.mkdir(parents=True, exist_ok=True)
        for name, (shape, fill) in classes.items():
            for i in range(per_class):
                img, box = _render(rng, shape, fill, size)
                stem = f"{name}_{i:03d}"
                img.save(out / f"{stem}.jpg", quality=95)
                annotation = [
                    {
                        "id": 1,
                        "categories": [name],
                        "points": _polygon_for(shape, box),
                        "type": "polygon",
                        "phi": 0,
                    }
                ]
                (out / f"{stem}.txt").write_text(
                    convert_anylearning_to_yolo(annotation, labels, (size, size))
                )

    (root / "labels.json").write_text(json.dumps(labels))
    return labels


def build_handpose_dataset(
    root, classes=("open", "fist"), per_class=8, seed=0, subsets=SUBSETS
):
    """Write the landmark JSONs ``HandPoseDataset`` reads.

    Each file is ``{"data": {"landmarks": {"0".."20": {x, y, z}}, "label": int}}``
    -- 21 MediaPipe hand landmarks. Each class gets a distinct landmark cloud
    (offset plus jitter) so the MLP has a real signal to fit rather than noise.
    """
    root = pathlib.Path(root)
    rng = random.Random(seed)
    labels = [{"name": name, "id": i} for i, name in enumerate(classes)]

    for subset in subsets:
        out = root / subset
        out.mkdir(parents=True, exist_ok=True)
        for class_id, name in enumerate(classes):
            # Separate the classes in landmark space by a per-class offset.
            offset = class_id * 0.4
            for i in range(per_class):
                landmarks = {
                    str(k): {
                        "x": offset + k * 0.01 + rng.uniform(-0.02, 0.02),
                        "y": offset + k * 0.01 + rng.uniform(-0.02, 0.02),
                        "z": rng.uniform(-0.02, 0.02),
                    }
                    for k in range(21)
                }
                payload = {"data": {"landmarks": landmarks, "label": class_id}}
                (out / f"{name}_{i:03d}.json").write_text(json.dumps(payload))

    (root / "labels.json").write_text(json.dumps(labels))
    return labels
