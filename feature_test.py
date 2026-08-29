#!/usr/bin/env python3
"""Everything the app does *except* training, exercised against a packaged build.

`smoke_test_build.sh` proves a build starts and that a handful of routes answer.
`--self-test` and `smoke_test_training.py` prove it can train. Between them sits
the rest of the product -- projects, labels, uploads, imports and exports in four
formats, annotation, auto-labelling, settings, the legal
notices -- and none of that was ever exercised in the artefact we publish.

Black-box and over HTTP on purpose. Every release failure so far has been a
packaging failure that pytest could not see: a migrations folder that shipped
without its .py files, a frontend-dist that was stale, a module dropped from the
compiled set that segfaulted on startup. So each check asks the *running application* a question and reads
its answer, and nothing here imports anylearning.

Usage:
    python feature_test.py ./app.dist/app.bin                       # Linux
    python feature_test.py "AnyLearning.App/AnyLearning.exe"        # Windows
    python feature_test.py /Applications/AnyLearning.app/Contents/MacOS/AnyLearning
    python feature_test.py --base-url http://127.0.0.1:5678         # a server you started

--base-url is for debugging a check against a development server, and that
server has to have been started with ANYLEARNING_DEVELOPMENT=1 (no bearer token).
Accepting a release means running it against the binary, where this script owns
the data root.

Options:
    --port N          port for the binary we start (default: a free one)
    --keep            keep the temporary data root, and the projects created
    --only a,b        run only these checks
    --skip a,b        run everything but these
    --base-url URL    talk to an already-running server instead of starting one

Checks, in the order they run:
    health frontend migrations projects labels dataset_upload dataset_import
    dataset_export annotations class_distribution copy_subset project_archive
    structured_workflows settings legal auto_labeling data_isolation
    training_terminate training update_check

Standard library only. It has to run from whatever Python is on the machine
where the build is being accepted, which is not necessarily the environment the
app was built in -- and on Windows that is frequently a bare python.org install.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import re
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib

# ---------------------------------------------------------------------------
# What the app is supposed to offer. Kept here rather than read from the
# repository: this script is run against a binary, on a machine that may have no
# checkout at all, and a constant that disagrees with the build is exactly the
# drift worth catching.
# ---------------------------------------------------------------------------

#: The project types the creation form offers -- frontend/src/components/
#: project-creation-form.tsx. Every type in the build's own MODEL_VARIANTS has
#: to be one of these, or the API supports a project the UI cannot create.
#: Sentiment Analysis is a legacy project name kept for old archives. New
#: structured projects use Tabular AI or Text AI.
FRONTEND_PROJECT_TYPES = (
    "Object Detection",
    "Image Classification",
    "Image Segmentation",
    "Handpose Classification",
    "Sentiment Analysis",
    "Tabular AI",
    "Text AI",
    "Text AI & LLM Evaluation",
    "Text & LLM",
    "Instance Segmentation",
    "Keypoint Detection",
)

STRUCTURED_PROJECT_TYPES = {
    "Tabular AI",
    "Text AI",
    "Text AI & LLM Evaluation",
    "Text & LLM",
    "Sentiment Analysis",
}

#: The routes the app links to. Next.js is exported statically and the backend's
#: catch-all falls back to index.html, so each of these must come back as HTML --
#: an empty or half-extracted frontend-dist answers `{"error": "File not found"}`.
SPA_ROUTES = (
    "/",
    "/projects",
    "/projects/overview",
    "/projects/dataset",
    "/projects/models",
    "/projects/training",
    "/settings",
)

#: The auto-labelling models `weights/auto_labeling/` ships, which `create_app`
#: seeds into the data root so a machine with no network can still label.
BUNDLED_AUTO_LABELING_MODELS = (
    "mobile_sam_20230629",
    "sam2_hiera_tiny_20240803",
    "sam2_hiera_small_20240803",
)

#: Small enough that a full sweep is minutes rather than an afternoon, large
#: enough that a batch of 2 is not larger than the training subset -- a batch
#: bigger than the dataset drops every partial batch, runs zero iterations and
#: fails with "No model found in training output".
IMAGE_SIZE = 96
PER_SUBSET = 6
TRAIN_EPOCHS = 2
TRAIN_BATCH = 2
TRAIN_BUDGET = 1800  # seconds for one run, including ONNX export

#: Three classes, each a flat colour on a light ground. Distinct in every
#: channel, so a run that separates them has learned something -- the same
#: choice anylearning/selftest/synthetic.py makes, for the same reason.
CLASSES = (
    ("red", (220, 60, 60)),
    ("blue", (60, 120, 220)),
    ("green", (60, 180, 90)),
)
CLASS_NAMES = tuple(name for name, _ in CLASSES)


# ---------------------------------------------------------------------------
# Talking to the app
# ---------------------------------------------------------------------------


class Failed(Exception):
    """A check found the app doing the wrong thing."""


class Skipped(Exception):
    """A check cannot run here, and says why rather than passing quietly."""


def expect(condition, message: str) -> None:
    if not condition:
        raise Failed(message)


class Reply:
    """One HTTP response, non-2xx included -- checks assert on status codes."""

    def __init__(self, status: int, body: bytes, headers: dict):
        self.status = status
        self.body = body
        self.headers = headers

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self):
        try:
            return json.loads(self.body)
        except ValueError:
            raise Failed(f"expected JSON, got {self.text[:200]!r}")

    def ok(self, what: str):
        """The body, having insisted on a 2xx. Most checks want exactly this."""
        expect(
            200 <= self.status < 300,
            f"{what}: HTTP {self.status} {self.text[:300]}",
        )
        return self.json()

    def __repr__(self) -> str:
        return f"<Reply {self.status} {self.text[:80]!r}>"


class App:
    """The running application, and the bookkeeping a check should not repeat."""

    def __init__(self, base: str, data_root: str | None, log_path: str | None):
        self.base = base.rstrip("/")
        #: Known only when we started the binary ourselves. Checks that read the
        #: databases or the seeded models skip without it rather than guessing.
        self.data_root = data_root
        self.log_path = log_path
        self.projects: list[int] = []

    # -- HTTP ---------------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        payload=None,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: int = 60,
    ) -> Reply:
        url = f"{self.base}{path}"
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        elif content_type:
            headers["Content-Type"] = content_type
        # No Authorization header: the server is started with
        # ANYLEARNING_DEVELOPMENT, which makes verify_token return early. A
        # packaged binary cannot be run with --development instead -- that flag
        # puts uvicorn in reload mode, which re-imports "anylearning.app:
        # create_app" in a subprocess and cannot work inside a frozen binary.
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return Reply(response.status, response.read(), dict(response.headers))
        except urllib.error.HTTPError as error:
            return Reply(error.code, error.read(), dict(error.headers))
        except urllib.error.URLError as error:
            raise Failed(f"{method} {path}: {error.reason}")

    def get(self, path: str, timeout: int = 60) -> Reply:
        return self.request("GET", path, timeout=timeout)

    def post(self, path: str, payload=None, timeout: int = 60) -> Reply:
        return self.request("POST", path, payload=payload, timeout=timeout)

    def patch(self, path: str, payload) -> Reply:
        return self.request("PATCH", path, payload=payload)

    def put(self, path: str, payload) -> Reply:
        return self.request("PUT", path, payload=payload)

    def delete(self, path: str, payload=None) -> Reply:
        return self.request("DELETE", path, payload=payload)

    def upload(self, path: str, files, timeout: int = 300) -> Reply:
        """multipart/form-data, the way the browser sends it.

        `files` is [(field, filename, bytes)]. The dataset form posts every
        image under the same field name, which is why this takes a list.
        """
        boundary = uuid.uuid4().hex
        parts = []
        for field, filename, content in files:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(
                f'Content-Disposition: form-data; name="{field}"; '
                f'filename="{filename}"\r\n'.encode()
            )
            parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
            parts.append(content)
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return self.request(
            "POST",
            path,
            body=b"".join(parts),
            content_type=f"multipart/form-data; boundary={boundary}",
            timeout=timeout,
        )

    # -- projects -----------------------------------------------------------

    def project(self, name: str, project_type: str, labels=None) -> int:
        """Create a project and remember it, so the runner can clean up."""
        created = self.post(
            "/api/projects",
            {
                "name": name,
                "type": project_type,
                "description": "Created by feature_test.py; safe to delete.",
                "labels": labels if labels is not None else [],
            },
        ).ok(f"create {project_type} project")
        project_id = created["id"]
        self.projects.append(project_id)
        return project_id

    def drop_projects(self) -> None:
        for project_id in reversed(self.projects):
            try:
                self.delete(f"/api/projects/{project_id}")
            except Failed:
                pass
        self.projects.clear()

    # -- dataset ------------------------------------------------------------

    def upload_images(
        self,
        project_id: int,
        files,
        subset: int = 0,
        auto_create_categories: bool = False,
    ) -> dict:
        """Post images (or one zip) and wait for the background task to finish.

        upload_data returns as soon as the task is queued, so every caller has
        to poll upload_status -- and a failed upload reports itself there rather
        than in the response, which is where "my images did not appear" lives.
        """
        query = urllib.parse.urlencode(
            {
                "subset": subset,
                "auto_create_categories": str(auto_create_categories).lower(),
            }
        )
        reply = self.upload(
            f"/api/projects/{project_id}/upload_data?{query}",
            [("file", name, content) for name, content in files],
        )
        reply.ok(f"upload to project {project_id}")

        deadline = time.time() + 600
        while time.time() < deadline:
            status = self.get(f"/api/projects/{project_id}/upload_status").ok(
                "upload status"
            )
            if status["status"] == "completed":
                return status
            if status["status"] == "failed":
                raise Failed(f"upload failed: {status.get('error_message')}")
            time.sleep(1)
        raise Failed("upload never finished")

    def items(self, project_id: int, subset: int | None = None, limit: int = 500):
        query = f"?limit={limit}" + ("" if subset is None else f"&subset={subset}")
        page = self.get(f"/api/projects/{project_id}/data_items{query}").ok(
            "list data items"
        )
        return page["data_items"]


# ---------------------------------------------------------------------------
# Images and archives, from the standard library
#
# No PIL. It is not a dependency of this script's *interpreter* -- the machine
# accepting a build has the binary and nothing else -- and a hand-written PNG
# encoder is twenty lines, produces identical bytes everywhere, and cannot be
# the reason a release check does not run.
# ---------------------------------------------------------------------------


def _png(width: int, height: int, rows) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(row) for row in rows)  # filter type 0 per row
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def draw(size: int, boxes, background=(245, 245, 245)) -> bytes:
    """An RGB PNG with filled rectangles. `boxes` is [(l, t, r, b, colour)]."""
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            colour = background
            for left, top, right, bottom, fill in boxes:
                if left <= x < right and top <= y < bottom:
                    colour = fill
            row += bytes(colour)
        rows.append(row)
    return _png(size, size, rows)


def sample(index: int, size: int = IMAGE_SIZE):
    """One image of one class. Returns (class name, png bytes, box).

    Deterministic: a failing check should be reproducible by re-running it, and
    a random dataset makes "it failed once" impossible to act on.
    """
    name, colour = CLASSES[index % len(CLASSES)]
    span = size // 3
    left = 8 + (index * 7) % (size - span - 16)
    top = 8 + (index * 11) % (size - span - 16)
    box = (left, top, left + span, top + span)
    return name, draw(size, [(*box, colour)]), box


def rectangle(box, category: str, shape_id: int = 1) -> dict:
    """A box in AnyLearning's own annotation format -- four corners, clockwise."""
    left, top, right, bottom = box
    return {
        "id": shape_id,
        "points": [[left, top], [right, top], [right, bottom], [left, bottom]],
        "phi": 0,
        "categories": [category],
        "type": "rectangle",
    }


def polygon(box, category: str, shape_id: int = 1) -> dict:
    left, top, right, bottom = box
    middle = (left + right) // 2
    return {
        "id": shape_id,
        "points": [[left, bottom], [middle, top], [right, bottom]],
        "phi": 0,
        "categories": [category],
        "type": "polygon",
    }


def labels_for(names) -> list[dict]:
    colours = ("#dc3c3c", "#3c78dc", "#3cb45a")
    return [
        {"name": name, "color": colours[index % len(colours)], "id": index}
        for index, name in enumerate(names)
    ]


def zip_of(members) -> bytes:
    """A zip from [(arcname, bytes)] -- what upload_data reads."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return buffer.getvalue()


def open_zip(content: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(content))


# ---------------------------------------------------------------------------
# The check registry
# ---------------------------------------------------------------------------

CHECKS: list = []


def check(name: str):
    """Register a check. Order of declaration is the order they run in.

    That order carries meaning in one place: `settings` asks the API which
    devices exist, and `training` -- which runs afterwards -- is what proves
    that asking did not initialise CUDA in the server process. See both.
    """

    def register(function):
        CHECKS.append((name, function))
        return function

    return register


# ---------------------------------------------------------------------------
# 1. It is alive, and it is the whole application
# ---------------------------------------------------------------------------


@check("health")
def check_health(app: App) -> str:
    """Liveness, the heavy imports, and the API surface itself.

    /api/health/imports imports torch, torchvision.ops and every trainer *in the
    running process*, which is the only place a packaging mistake shows up: the
    build that excluded torch._dynamo compiled, linked and then died because
    torchvision.ops imports it.
    """
    health = app.get("/api/health").ok("GET /api/health")
    expect(health.get("status") == "ok", f"health said {health}")

    imports = app.get("/api/health/imports", timeout=300).ok("GET /api/health/imports")
    expect(imports.get("ok"), f"broken imports: {imports.get('broken')}")

    marker = app.get("/api/is_anylearning").ok("GET /api/is_anylearning")
    expect(marker.get("is_anylearning") is True, f"is_anylearning said {marker}")

    schema = app.get("/openapi.json").ok("GET /openapi.json")
    paths = schema.get("paths", {})
    for required in (
        "/api/projects",
        "/api/projects/{project_id}/upload_data",
        "/api/projects/{project_id}/training_sessions",
        "/api/projects/{project_id}/models/{model_id}/inference",
    ):
        expect(required in paths, f"{required} is missing from the OpenAPI schema")

    return f"{len(paths)} routes, python {health.get('python')}"


@check("frontend")
def check_frontend(app: App) -> str:
    """The frontend the backend extracts and serves, not just index.html.

    The catch-all answers index.html for anything it cannot find, so serving
    "/" proves very little on its own. Fetching an asset that index.html itself
    references is what catches a frontend-dist that was extracted stale or
    half-copied -- and a build once shipped exactly that.
    """
    index = app.get("/")
    expect(index.status == 200, f"GET / returned {index.status}")
    body = index.text
    expect(
        "<html" in body.lower() or "<!doctype html" in body.lower(),
        f"GET / did not return HTML: {body[:120]!r}",
    )

    for route in SPA_ROUTES:
        reply = app.get(route)
        expect(reply.status == 200, f"GET {route} returned {reply.status}")
        expect(
            "<html" in reply.text.lower(),
            f"GET {route} did not return the app shell: {reply.text[:120]!r}",
        )

    assets = re.findall(r'"(/_next/[^"]+\.(?:js|css))"', body)
    expect(assets, "index.html references no /_next asset -- is it the real export?")
    asset = app.get(assets[0])
    expect(asset.status == 200, f"GET {assets[0]} returned {asset.status}")
    expect(
        "<html" not in asset.text[:200].lower(),
        f"{assets[0]} fell through to index.html -- that asset is not in the build",
    )

    return f"{len(SPA_ROUTES)} routes, {len(assets)} assets referenced"


@check("migrations")
def check_migrations(app: App) -> str:
    """Every database this build creates carries an alembic stamp.

    Alembic reads env.py and its revisions from disk by path, and Nuitka's
    --include-data-dir silently skips .py files -- so a build once shipped a
    migrations folder holding only a README. Nothing failed: MigrationManager
    catches its own errors and logs them, so every database went unstamped and
    the next schema change would have broken every install. Checked for the main
    registry *and* for a project database created by this run, because they are
    stamped by different code paths.
    """
    if not app.data_root:
        raise Skipped("--base-url: the server's data root is not ours to read")

    def read_stamp(path: pathlib.Path) -> str:
        expect(path.is_file(), f"no database at {path}")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "select version_num from alembic_version"
            ).fetchone()
        except sqlite3.Error as error:
            raise Failed(
                f"{path} has no alembic_version table ({error}) -- "
                "the migration revisions are missing from this build"
            )
        finally:
            connection.close()
        expect(row and row[0], f"{path} has an empty alembic_version table")
        return row[0]

    root = pathlib.Path(app.data_root)
    main = read_stamp(root / "anylearning.db")

    project_id = app.project("feature-test migrations", "Object Detection")
    project = read_stamp(root / "projects" / str(project_id) / "database.db")
    expect(
        project == main,
        f"the main database is at {main} but a new project database is at {project}",
    )
    return f"stamped {main}"


# ---------------------------------------------------------------------------
# 2. Projects and labels
# ---------------------------------------------------------------------------


@check("projects")
def check_projects(app: App) -> str:
    """Create, read, update, list and delete -- once per project type.

    The type string is not decoration: TrainerBuilder maps it to a trainer class
    and the frontend's creation form sends it verbatim, so a type the API
    accepts but the form cannot offer (or the reverse) is a broken product with
    no failing test anywhere else.
    """
    variants = app.get("/api/model-variants").ok("GET /api/model-variants")
    expect(variants, "the build offers no model variants at all")

    unknown = [t for t in variants if t not in FRONTEND_PROJECT_TYPES]
    expect(
        not unknown,
        f"MODEL_VARIANTS offers {unknown}, which the creation form cannot send",
    )

    for project_type in variants:
        project_id = app.project(f"feature-test {project_type}", project_type)
        fetched = app.get(f"/api/projects/{project_id}").ok("read back the project")
        expect(
            fetched["type"] == project_type,
            f"created a {project_type} project and got back a {fetched['type']} one",
        )

        renamed = app.patch(
            f"/api/projects/{project_id}",
            {"name": "feature-test renamed", "description": "edited"},
        ).ok("patch the project")
        expect(renamed["name"] == "feature-test renamed", "the rename did not stick")

        listed = app.get("/api/projects?limit=100").ok("list projects")
        expect(
            any(item["id"] == project_id for item in listed),
            f"project {project_id} is missing from GET /api/projects",
        )

        app.delete(f"/api/projects/{project_id}").ok("delete the project")
        app.projects.remove(project_id)
        gone = app.get(f"/api/projects/{project_id}")
        expect(gone.status == 404, f"a deleted project still answers {gone.status}")

    return f"{len(variants)} types: {', '.join(sorted(variants))}"


@check("labels")
def check_labels(app: App) -> str:
    """Adding, renaming and deleting labels -- and that ids do not collide.

    The labels array is stored whole, and the UI sends it as a JSON *string*
    (ProjectUpdate.labels is a pydantic Json field), so an array is rejected.

    The last part is the one that matters. `next_label_id` used to be
    `len(project.labels)`: with ids 0, 1, 2, delete the middle one and the next
    upload creates another id 2. For classification that is not cosmetic --
    class_id points at a label id, so every image of one class silently reads as
    the other.
    """
    project_id = app.project(
        "feature-test labels", "Image Classification", labels_for(CLASS_NAMES)
    )

    def set_labels(labels):
        return app.patch(
            f"/api/projects/{project_id}", {"labels": json.dumps(labels)}
        ).ok("patch labels")

    labels = labels_for(CLASS_NAMES)
    rejected = app.patch(f"/api/projects/{project_id}", {"labels": labels})
    expect(
        rejected.status == 422,
        f"labels as an array returned {rejected.status}; the UI sends a string",
    )

    renamed = list(labels)
    renamed[1] = {**renamed[1], "name": "azure"}
    stored = set_labels(renamed)
    expect(
        [entry["name"] for entry in stored["labels"]] == ["red", "azure", "green"],
        f"rename produced {stored['labels']}",
    )

    kept = [entry for entry in renamed if entry["id"] != 1]
    stored = set_labels(kept)
    expect(
        [entry["id"] for entry in stored["labels"]] == [0, 2],
        f"delete produced {stored['labels']}",
    )

    # A folder name the project has never seen, so the upload has to mint an id.
    _, image, _ = sample(0)
    status = app.upload_images(
        project_id,
        [("data.zip", zip_of([("cyan/one.png", image)]))],
        auto_create_categories=True,
    )
    expect(
        status["created_labels"] == ["cyan"],
        f"upload reported created_labels={status['created_labels']}",
    )

    after = app.get(f"/api/projects/{project_id}").ok("read labels back")["labels"]
    minted = [entry for entry in after if entry["name"] == "cyan"]
    expect(minted, f"the new label is not on the project: {after}")
    expect(
        minted[0]["id"] == 3,
        f"the new label got id {minted[0]['id']}, colliding with an existing id "
        f"({[entry['id'] for entry in after]}) -- next_label_id is counting labels",
    )
    return "add, rename, delete, and ids survive a delete"


# ---------------------------------------------------------------------------
# 3. Datasets: upload, import, export
# ---------------------------------------------------------------------------


@check("dataset_upload")
def check_dataset_upload(app: App) -> str:
    """The three ways images arrive, plus listing, paging, splits and delete.

    Dropping forty images on the window and choosing a zip are one code path --
    loose files are repacked at the door -- so both are worth posting, and the
    per-subset counts are what the dataset screen is made of.
    """
    project_id = app.project(
        "feature-test upload", "Object Detection", labels_for(CLASS_NAMES)
    )

    loose = []
    for index in range(3):
        _, image, _ = sample(index)
        loose.append((f"loose_{index}.png", image))
    app.upload_images(project_id, loose, subset=0)
    expect(len(app.items(project_id, subset=0)) == 3, "three loose images did not land")

    archive = zip_of(
        [(f"zipped_{index}.png", sample(index + 3)[1]) for index in range(2)]
    )
    app.upload_images(project_id, [("data.zip", archive)], subset=1)
    expect(len(app.items(project_id, subset=1)) == 2, "the zip did not land in val")

    # Mixing the two is refused rather than half-applied.
    mixed = app.upload(
        f"/api/projects/{project_id}/upload_data",
        [("file", "a.zip", archive), ("file", "b.png", loose[0][1])],
    )
    expect(mixed.status == 400, f"a zip plus an image returned {mixed.status}")

    # Paging and the subset filter, which the dataset table depends on.
    page = app.get(f"/api/projects/{project_id}/data_items?offset=1&limit=2").ok("page")
    expect(page["total_count"] == 5, f"total_count was {page['total_count']}, not 5")
    expect(len(page["data_items"]) == 2, "limit=2 returned a different number of rows")
    bad = app.get(f"/api/projects/{project_id}/data_items?subset=7")
    expect(bad.status == 400, f"subset=7 returned {bad.status}")

    splits = app.get(f"/api/projects/{project_id}/datasets").ok("GET datasets")
    counts = {row["type"]: row["num_total"] for row in splits}
    expect(
        counts == {"train": 3, "validation": 2, "test": 0},
        f"the split summary says {counts}",
    )

    # Class from folder names, on the project type that uses it.
    classified = app.project(
        "feature-test folders", "Image Classification", labels_for(CLASS_NAMES)
    )
    folders = zip_of(
        [
            (f"{name}/{name}_{index}.png", sample(position)[1])
            for position, (name, _) in enumerate(CLASSES)
            for index in range(2)
        ]
    )
    status = app.upload_images(
        classified, [("classes.zip", folders)], auto_create_categories=True
    )
    expect(
        status["total_files"] == 6 and status["processed_files"] == 6,
        f"upload processed {status['processed_files']}/{status['total_files']}",
    )
    # The names already exist on the project, so nothing new is created --
    # created_labels reporting them would mean duplicate labels were made.
    expect(
        status["created_labels"] == [],
        f"folders matching existing labels created {status['created_labels']}",
    )
    by_class = {item["class_id"] for item in app.items(classified)}
    expect(by_class == {0, 1, 2}, f"folder names produced class ids {by_class}")

    doomed = app.items(project_id, subset=0)[0]
    removed = app.delete(f"/api/projects/{project_id}/data_items", [doomed["id"]]).ok(
        "delete a data item"
    )
    expect("1" in str(removed), f"delete reported {removed}")
    expect(len(app.items(project_id, subset=0)) == 2, "the item was not removed")
    gone = app.get(f"/api/projects/{project_id}/data_items/{doomed['id']}/download")
    expect(gone.status == 404, f"a deleted item still downloads ({gone.status})")

    return "loose files, zip, folder classes, paging, splits, delete"


@check("dataset_import")
def check_dataset_import(app: App) -> str:
    """An annotated archive in each format the importer claims to read.

    COCO, YOLO and the AnyLabeling/LabelMe sidecar are found by
    `scan_archive_formats` in that order, and each one is a separate reader with
    its own way of going wrong -- YOLO's coordinates are normalised, COCO keeps
    one file for the whole dataset, LabelMe keeps one beside each image. All
    three have to name the categories they found, or the import silently
    produces an unlabelled dataset.
    """
    _, image, box = sample(0)
    left, top, right, bottom = box
    results = []

    # -- COCO: one dataset-wide annotations file.
    coco = {
        "images": [
            {"id": 1, "file_name": "one.png", "width": IMAGE_SIZE, "height": IMAGE_SIZE}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [left, top, right - left, bottom - top],
                "area": (right - left) * (bottom - top),
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 7, "name": "coco_thing", "supercategory": "none"}],
    }
    project_id = app.project("feature-test import coco", "Object Detection")
    status = app.upload_images(
        project_id,
        [
            (
                "coco.zip",
                zip_of(
                    [
                        ("images/one.png", image),
                        ("annotations/instances.json", json.dumps(coco).encode()),
                    ]
                ),
            )
        ],
    )
    results.append(("coco", _imported_categories(app, project_id, status)))
    expect(
        results[-1][1] == {"coco_thing"},
        f"COCO import produced categories {results[-1][1]}",
    )

    # -- YOLO: normalised coordinates, class names in a separate file.
    centre_x = (left + right) / 2 / IMAGE_SIZE
    centre_y = (top + bottom) / 2 / IMAGE_SIZE
    width = (right - left) / IMAGE_SIZE
    height = (bottom - top) / IMAGE_SIZE
    project_id = app.project("feature-test import yolo", "Object Detection")
    status = app.upload_images(
        project_id,
        [
            (
                "yolo.zip",
                zip_of(
                    [
                        ("images/one.png", image),
                        (
                            "labels/one.txt",
                            f"0 {centre_x} {centre_y} {width} {height}\n".encode(),
                        ),
                        ("data.yaml", b"names:\n  - yolo_thing\n"),
                    ]
                ),
            )
        ],
    )
    results.append(("yolo", _imported_categories(app, project_id, status)))
    expect(
        results[-1][1] == {"yolo_thing"},
        f"YOLO import produced categories {results[-1][1]}",
    )

    # -- LabelMe / AnyLabeling: one sidecar beside each image.
    sidecar = {
        "version": "1.0",
        "imagePath": "one.png",
        "imageWidth": IMAGE_SIZE,
        "imageHeight": IMAGE_SIZE,
        "shapes": [
            {
                "label": "labelme_thing",
                "shape_type": "rectangle",
                "points": [[left, top], [right, bottom]],
                "group_id": None,
                "flags": {},
            }
        ],
    }
    project_id = app.project("feature-test import labelme", "Object Detection")
    status = app.upload_images(
        project_id,
        [
            (
                "labelme.zip",
                zip_of(
                    [("one.png", image), ("one.json", json.dumps(sidecar).encode())]
                ),
            )
        ],
    )
    results.append(("labelme", _imported_categories(app, project_id, status)))
    expect(
        results[-1][1] == {"labelme_thing"},
        f"LabelMe import produced categories {results[-1][1]}",
    )

    return ", ".join(f"{fmt}: {sorted(found)[0]}" for fmt, found in results)


def _imported_categories(app: App, project_id: int, status: dict) -> set:
    """What an import actually attached: the categories on the stored shapes.

    Read from the annotation rather than from created_labels alone. A label can
    be created and no shape attached -- which is how an import looks when the
    coordinates could not be read.
    """
    items = app.items(project_id)
    expect(items, "the import stored no images at all")
    expect(
        all(item["labeled"] for item in items),
        f"imported images are not marked as labelled: {items}",
    )
    expect(
        status["created_labels"],
        "the import created no labels, so its categories were not read",
    )

    found = set()
    for item in items:
        shapes = app.get(
            f"/api/projects/{project_id}/data_items/{item['id']}/get_annotation"
        ).ok("get_annotation")
        expect(shapes, f"item {item['id']} was marked labelled with no shapes")
        for shape in shapes:
            expect(
                len(shape.get("points") or []) >= 3,
                f"a shape came back with {shape.get('points')}",
            )
            found.update(shape.get("categories") or [])
    expect(
        found == set(status["created_labels"]),
        f"created_labels {status['created_labels']} but the shapes say {found}",
    )
    return found


@check("dataset_export")
def check_dataset_export(app: App) -> str:
    """Export in each format, and look inside the zip that comes back.

    A completed export status is not evidence: the exporter counts an item as
    processed before it writes the annotation, so a format whose converter threw
    still reports 100%. The archive's members are the evidence.
    """
    project_id = app.project(
        "feature-test export", "Object Detection", labels_for(CLASS_NAMES)
    )
    for subset in (0, 1):
        name, image, box = sample(subset)
        app.upload_images(project_id, [(f"{name}_{subset}.png", image)], subset=subset)
        item = app.items(project_id, subset=subset)[0]
        app.post(
            f"/api/projects/{project_id}/data_items/{item['id']}/set_annotation",
            [rectangle(box, name)],
        ).ok("annotate for export")

    expected_members = {
        "yolo": ("yolo/dataset.yaml", ".txt"),
        "coco": ("coco/annotations/instances.json", ".png"),
        "labelme": (None, ".json"),
        "anylabeling": (None, ".json"),
    }
    written = {}
    for fmt, (required, extension) in expected_members.items():
        app.post(f"/api/projects/{project_id}/export_data", {"format": fmt}).ok(
            f"start {fmt} export"
        )

        deadline = time.time() + 300
        while time.time() < deadline:
            status = app.get(f"/api/projects/{project_id}/export_status").ok(
                "export status"
            )
            if status["status"] == "completed":
                break
            expect(
                status["status"] != "failed",
                f"{fmt} export failed: {status.get('error_message')}",
            )
            time.sleep(1)
        else:
            raise Failed(f"{fmt} export never finished")

        download = app.get(f"/api/projects/{project_id}/download_export", timeout=300)
        expect(
            download.status == 200,
            f"downloading the {fmt} export returned {download.status}",
        )
        names = open_zip(download.body).namelist()
        expect(names, f"the {fmt} export is an empty archive")
        if required:
            expect(required in names, f"{required} is missing from the {fmt} export")
        expect(
            any(name.endswith(extension) for name in names),
            f"the {fmt} export has no {extension} file: {names[:8]}",
        )
        if fmt == "coco":
            payload = json.loads(
                open_zip(download.body).read("coco/annotations/instances.json")
            )
            expect(
                payload["annotations"],
                "the COCO export wrote no annotations for two labelled images",
            )
        if fmt == "yolo":
            labels = [
                n for n in names if n.startswith("yolo/labels/") and n.endswith(".txt")
            ]
            expect(labels, f"the YOLO export wrote no label files: {names[:8]}")
        written[fmt] = len(names)

    app.delete(f"/api/projects/{project_id}/cleanup_export").ok("clean up the export")
    return ", ".join(f"{fmt}: {count} files" for fmt, count in written.items())


# ---------------------------------------------------------------------------
# 4. Annotation
# ---------------------------------------------------------------------------


@check("annotations")
def check_annotations(app: App) -> str:
    """What the labelling canvas saves, read back exactly as it was sent.

    set_annotation takes a bare JSON array. It used to be declared as a typed
    parameter, which current FastAPI expects wrapped as {"annotation": [...]},
    and every auto-save answered 422 and lost the work -- so the array shape is
    asserted here as well as the round trip.

    Handpose is the one type this cannot round-trip: its annotation is a dict of
    21 mediapipe landmarks, written by the upload path from a real hand, and
    set_annotation only accepts a list. What is checked instead is that the
    upload path itself works in the build -- the landmark model loads and the
    upload completes -- because that model failing to load in a frozen binary is
    the packaging failure this would otherwise miss.
    """
    project_id = app.project(
        "feature-test annotations", "Object Detection", labels_for(CLASS_NAMES)
    )
    name, image, box = sample(0)
    app.upload_images(project_id, [("box.png", image), ("poly.png", sample(1)[1])])
    items = app.items(project_id)
    expect(len(items) == 2, f"expected two images, got {len(items)}")

    for item, shape in (
        (items[0], rectangle(box, name)),
        (items[1], polygon(box, "blue")),
    ):
        app.post(
            f"/api/projects/{project_id}/data_items/{item['id']}/set_annotation",
            [shape],
        ).ok("save an annotation")
        stored = app.get(
            f"/api/projects/{project_id}/data_items/{item['id']}/get_annotation"
        ).ok("load the annotation")
        expect(stored == [shape], f"saved {shape} and got back {stored}")

    wrapped = app.post(
        f"/api/projects/{project_id}/data_items/{items[0]['id']}/set_annotation",
        {"annotation": [rectangle(box, name)]},
    )
    expect(
        wrapped.status == 422,
        f"a wrapped annotation body returned {wrapped.status}, not a clear 422",
    )
    expect(
        all(item["labeled"] for item in app.items(project_id)),
        "annotated items are not reported as labelled",
    )

    # Classification labels a whole image rather than a region.
    classified = app.project(
        "feature-test class labels", "Image Classification", labels_for(CLASS_NAMES)
    )
    app.upload_images(classified, [("one.png", image)])
    item = app.items(classified)[0]
    app.post(
        f"/api/projects/{classified}/data_items/{item['id']}/class_id", {"class_id": 2}
    ).ok("set class_id")
    stored = app.items(classified)[0]
    expect(stored["class_id"] == 2, f"class_id came back as {stored['class_id']}")
    expect(stored["labeled"], "setting a class did not mark the image as labelled")

    # Handpose: the upload runs mediapipe over each image and drops the ones
    # with no hand in them. A drawn square has no hand, so zero items is the
    # correct outcome -- and getting there proves the landmark model loaded.
    handpose = app.project(
        "feature-test handpose", "Handpose Classification", labels_for(("open", "fist"))
    )
    status = app.upload_images(
        handpose, [("hand.png", image), ("hand2.png", sample(1)[1])]
    )
    expect(
        status["total_files"] == 2,
        f"the handpose upload saw {status['total_files']} files, not 2",
    )
    kept = app.items(handpose)
    expect(
        kept == [],
        f"mediapipe found hand landmarks in a drawn square ({len(kept)} items kept)",
    )

    return "box, polygon, class_id; handpose upload runs the landmark model"


@check("class_distribution")
def check_class_distribution(app: App) -> str:
    """Per-class, per-subset counts -- for a detection project and a handpose one.

    Handpose is named explicitly because this endpoint used to 500 on it: the
    annotation there is a dict of landmarks rather than a list of shapes, and
    the counter iterated it as shapes. A project screen that cannot be opened
    for one project type is a release blocker, and nothing else asks this
    endpoint for a handpose project.
    """
    project_id = app.project(
        "feature-test distribution", "Object Detection", labels_for(CLASS_NAMES[:2])
    )
    for subset, count in ((0, 2), (1, 1)):
        for index in range(count):
            _, image, _ = sample(index)
            app.upload_images(
                project_id, [(f"{subset}_{index}.png", image)], subset=subset
            )
        for position, item in enumerate(app.items(project_id, subset=subset)):
            # One shape of a known class, plus -- on the first item -- a shape
            # naming a class the project does not list. Those are real
            # annotations that training will see, so they are reported as
            # unknown rather than dropped. The geometry is irrelevant here;
            # what is counted is the category on each shape.
            box = (4, 4, 20, 20)
            shapes = [rectangle(box, CLASS_NAMES[position % 2])]
            if position == 0:
                shapes.append(rectangle(box, "retired_class", shape_id=2))
            app.post(
                f"/api/projects/{project_id}/data_items/{item['id']}/set_annotation",
                shapes,
            ).ok("annotate for the distribution")

    distribution = app.get(f"/api/projects/{project_id}/class_distribution").ok(
        "GET class_distribution"
    )
    rows = {row["name"]: row for row in distribution["classes"]}
    expect(set(rows) == {"red", "blue", "retired_class"}, f"classes are {sorted(rows)}")
    expect(rows["red"]["train"] == 1, f"red/train is {rows['red']['train']}")
    expect(rows["blue"]["train"] == 1, f"blue/train is {rows['blue']['train']}")
    expect(rows["red"]["validation"] == 1, f"red/val is {rows['red']['validation']}")
    expect(
        rows["retired_class"]["known"] is False,
        "a class the project no longer lists is reported as known",
    )
    expect(
        distribution["unlabeled"]["total"] == 0,
        f"unlabelled total is {distribution['unlabeled']['total']}, not 0",
    )

    handpose = app.project(
        "feature-test handpose distribution",
        "Handpose Classification",
        labels_for(("open", "fist")),
    )
    reply = app.get(f"/api/projects/{handpose}/class_distribution")
    expect(
        reply.status == 200,
        f"class_distribution on a handpose project returned {reply.status}: "
        f"{reply.text[:200]}",
    )
    names = {row["name"] for row in reply.json()["classes"]}
    expect(names == {"open", "fist"}, f"handpose classes came back as {names}")

    return "detection counts exact, handpose answers 200"


@check("copy_subset")
def check_copy_subset(app: App) -> str:
    """Cloning one subset into another, which the UI recommends for small sets.

    The files are copied rather than referenced twice, and that is the part
    worth checking: two rows pointing at one image means deleting either item
    deletes the image out from under the other.
    """
    project_id = app.project(
        "feature-test copy", "Image Classification", labels_for(CLASS_NAMES)
    )
    for index in range(3):
        app.upload_images(
            project_id, [(f"val_{index}.png", sample(index)[1])], subset=1
        )

    same = app.post(
        f"/api/projects/{project_id}/data_items/copy_subset",
        {"from_subset": 1, "to_subset": 1},
    )
    expect(same.status == 400, f"copying a subset onto itself returned {same.status}")
    empty = app.post(
        f"/api/projects/{project_id}/data_items/copy_subset",
        {"from_subset": 2, "to_subset": 0},
    )
    expect(empty.status == 400, f"copying an empty subset returned {empty.status}")

    copied = app.post(
        f"/api/projects/{project_id}/data_items/copy_subset",
        {"from_subset": 1, "to_subset": 2},
    ).ok("copy validation into test")
    expect(copied["copied"] == 3, f"copy_subset reported {copied}")

    source = app.items(project_id, subset=1)
    clones = app.items(project_id, subset=2)
    expect(len(clones) == 3, f"the test subset has {len(clones)} items")

    # The model screen picks a test image at random to try a model on, and a
    # test subset that exists only as copies is exactly when it is first used.
    picked = app.get(f"/api/projects/{project_id}/data_items/random_test_sample")
    expect(picked.status == 200, f"random_test_sample returned {picked.status}")
    expect(picked.body[:8] == b"\x89PNG\r\n\x1a\n", "random_test_sample sent no image")
    app.delete(f"/api/projects/{project_id}/data_items", [clones[0]["id"]]).ok(
        "delete one clone"
    )
    for item in source:
        still_there = app.get(
            f"/api/projects/{project_id}/data_items/{item['id']}/download"
        )
        expect(
            still_there.status == 200,
            f"deleting a copy took the original's image with it ({still_there.status})",
        )
    return "3 copied, originals survive deleting a copy"


@check("project_archive")
def check_project_archive(app: App) -> str:
    """Export a whole project and import it back.

    This is the app's own backup format -- a tar.gz of the project folder,
    including its database -- and it is the only feature here whose work happens
    in a FastAPI background task holding a database session created for the
    request.
    """
    project_id = app.project(
        "feature-test archive", "Object Detection", labels_for(CLASS_NAMES)
    )
    app.upload_images(project_id, [("one.png", sample(0)[1])])

    # Exported without reading /api/projects/{id} first, on purpose. That route
    # is what computes and writes `size` onto the row, so a project that has not
    # been opened exports "size": null -- and the importer's float() has no
    # default for a key that is present and null. Doing the GET here would hide
    # that, and it is the archive of a freshly created project that a user is
    # most likely to be carrying to another machine.
    app.post(f"/api/projects/{project_id}/export").ok("start the project export")
    deadline = time.time() + 300
    while time.time() < deadline:
        status = app.get(f"/api/projects/{project_id}/export/status").ok(
            "project export status"
        )
        if status["status"] == "completed":
            break
        expect(
            status["status"] != "failed",
            f"project export failed: {status.get('error')}",
        )
        time.sleep(1)
    else:
        raise Failed("the project export never completed")

    download = app.get(f"/api/projects/{project_id}/export/download", timeout=300)
    expect(download.status == 200, f"downloading the export returned {download.status}")
    expect(download.body[:2] == b"\x1f\x8b", "the export is not a gzip archive")

    started = app.upload(
        "/api/projects/import",
        [("import_file", "project.tar.gz", download.body)],
        timeout=300,
    ).ok("start the import")
    import_id = started["import_id"]

    deadline = time.time() + 300
    while time.time() < deadline:
        status = app.get(f"/api/projects/import/{import_id}/status").ok("import status")
        if status["status"] == "completed":
            break
        expect(
            status["status"] != "failed",
            f"project import failed: {status.get('error')}",
        )
        time.sleep(1)
    else:
        raise Failed("the project import never completed")

    imported = status["project_id"]
    app.projects.append(imported)
    restored = app.get(f"/api/projects/{imported}").ok("read the imported project")
    expect(
        restored["type"] == "Object Detection",
        f"the imported project is a {restored['type']}",
    )
    expect(
        [entry["name"] for entry in restored["labels"]] == list(CLASS_NAMES),
        f"the imported project's labels are {restored['labels']}",
    )
    expect(
        len(app.items(imported)) == 1,
        "the imported project has no data items -- its database did not come across",
    )
    return f"exported {len(download.body) // 1024} KB and imported it back"


@check("structured_workflows")
def check_structured_workflows(app: App) -> str:
    """Tables, text AI, and response evaluation through the packaged HTTP surface.

    This deliberately imports the lazy structured stack from the running
    binary. A source test cannot prove Nuitka included CatBoost's shared
    libraries, PyArrow, the sparse scikit-learn pipeline or Excel support.
    """

    def train(project_id: int, params: dict) -> tuple[int, dict]:
        started = app.post(
            f"/api/projects/{project_id}/training_sessions", params, timeout=180
        ).ok("start structured training")
        session_id = started["session_id"]
        deadline = time.time() + 300
        while time.time() < deadline:
            detail = app.get(
                f"/api/projects/{project_id}/training_sessions/{session_id}"
            ).ok("structured training status")
            if detail["status"] in {"finished", "error", "terminated"}:
                break
            time.sleep(1)
        else:
            raise Failed("structured training did not finish in 300 seconds")
        expect(
            detail["status"] == "finished",
            f"structured training ended as {detail['status']}: {_tail(detail)}",
        )
        model_id = (detail.get("model") or {}).get("id")
        expect(model_id, f"structured training registered no model: {_tail(detail)}")
        return model_id, detail

    tabular = app.project("feature-test tabular", "Tabular AI")
    table = ["age,job,balance,target"]
    for index in range(90):
        target = "yes" if index % 3 == 0 else "no"
        job = "engineer" if index % 2 else "teacher"
        table.append(f"{20 + index % 40},{job},{100 + index * 17},{target}")
    uploaded = app.upload(
        f"/api/projects/{tabular}/structured/upload",
        [("file", "people.csv", ("\n".join(table) + "\n").encode())],
    ).ok("upload a tabular CSV")
    expect(uploaded["source"]["rows"] == 90, f"profile says {uploaded['source']}")
    expect(
        uploaded.get("performance", {}).get("storage_engine") == "DuckDB + Parquet",
        f"out-of-core storage contract is missing: {uploaded.get('performance')}",
    )
    page = app.get(
        f"/api/projects/{tabular}/structured/rows?offset=75&limit=5&columns=age&columns=target"
    ).ok("load a projected table page")
    expect(
        page.get("paged") is True and page.get("dataset_total") == 90,
        f"table page was not partial: {page}",
    )
    expect(
        set(page["rows"][0]) == {"_row_id", "age", "target"},
        f"table projection returned extra columns: {page['rows'][0]}",
    )
    filtered = app.get(
        f"/api/projects/{tabular}/structured/rows?query=engineer&limit=5&columns=job"
    ).ok("filter table rows")
    expect(filtered["total"] == 45, f"table filter returned {filtered['total']} rows")
    edited = app.patch(
        f"/api/projects/{tabular}/structured/rows/89",
        {"values": {"job": "reviewed-job"}},
    ).ok("edit a partially loaded row")
    expect(edited["job"] == "reviewed-job", f"row edit returned {edited}")
    exported = app.get(f"/api/projects/{tabular}/structured/export")
    expect(
        exported.status == 200 and b"reviewed-job" in exported.body,
        "streamed table export omitted the review override",
    )
    configured = app.put(
        f"/api/projects/{tabular}/structured/config",
        {
            "type": "classification",
            "target": "target",
            "ignored_columns": [],
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
        },
    ).ok("configure tabular classification")
    expect(configured["configured"] is True, "the tabular project is not configured")
    tabular_model, _ = train(
        tabular,
        {
            "model_architecture": "catboost",
            "model_size": "balanced",
            "model_variant": "catboost_balanced",
            "batch_size": 16,
            "epochs": 30,
            "learning_rate": 0.08,
            "pretrained_model": "default",
            "device": "cpu",
        },
    )
    report = app.get(
        f"/api/projects/{tabular}/structured/models/{tabular_model}/report"
    ).ok("read the CatBoost report")
    expect("Macro F1" in report["metrics"], f"tabular metrics are {report['metrics']}")
    expect(report["review_queue"], "the tabular Smart Review queue is empty")
    prediction = app.post(
        f"/api/projects/{tabular}/structured/models/{tabular_model}/predict",
        {"rows": [{"age": 44, "job": "teacher", "balance": 900}]},
    ).ok("predict a table row")
    expect(len(prediction["predictions"]) == 1, f"prediction was {prediction}")
    raw = app.get(f"/api/projects/{tabular}/models/{tabular_model}/download")
    expect(
        raw.status == 200 and len(raw.body) > 1_000,
        "native CatBoost model did not download",
    )

    text_project = app.project("feature-test text", "Text AI")
    text_rows = ["text,intent"]
    examples = {
        "card_arrival": ("where is my card", "card delivery status"),
        "cash_withdrawal": ("cash machine failed", "atm did not pay"),
        "transfer_pending": ("transfer still pending", "where is my transfer"),
    }
    for repeat in range(10):
        for label, samples in examples.items():
            for sample_text in samples:
                text_rows.append(f'"{sample_text} {repeat}",{label}')
    app.upload(
        f"/api/projects/{text_project}/structured/upload",
        [("file", "intents.csv", ("\n".join(text_rows) + "\n").encode())],
    ).ok("upload text classification data")
    app.put(
        f"/api/projects/{text_project}/structured/config",
        {
            "type": "text_classification",
            "target": "intent",
            "text_column": "text",
            "ignored_columns": [],
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
        },
    ).ok("configure text classification")
    text_model, _ = train(
        text_project,
        {
            "model_architecture": "tfidf-logreg",
            "model_size": "lightweight",
            "model_variant": "tfidf_logreg",
            "batch_size": 16,
            "epochs": 200,
            "learning_rate": 0.05,
            "pretrained_model": "default",
            "device": "cpu",
        },
    )
    text_prediction = app.post(
        f"/api/projects/{text_project}/structured/models/{text_model}/predict",
        {"rows": [{"text": "my card has not arrived"}]},
    ).ok("predict a text intent")
    expect(
        text_prediction["predictions"][0]["prediction"] == "card_arrival",
        f"text prediction was {text_prediction}",
    )

    evaluation = app.project("feature-test response evaluation", "Text AI")
    llm_csv = (
        "prompt,response,reference\n"
        '"What is two plus two?","Four","four"\n'
        '"Where is my transfer?","","Check transfer status"\n'
    ).encode()
    app.upload(
        f"/api/projects/{evaluation}/structured/upload",
        [("file", "responses.csv", llm_csv)],
    ).ok("upload LLM responses")
    app.put(
        f"/api/projects/{evaluation}/structured/config",
        {
            "type": "llm_evaluation",
            "prompt_column": "prompt",
            "response_column": "response",
            "reference_column": "reference",
            "ignored_columns": [],
        },
    ).ok("configure response evaluation")
    evaluated = app.post(f"/api/projects/{evaluation}/structured/evaluate").ok(
        "evaluate LLM responses"
    )
    expect(
        evaluated["metrics"]["completion_rate"] == 0.5,
        f"response evaluation metrics were {evaluated['metrics']}",
    )
    app.put(
        f"/api/projects/{evaluation}/structured/config",
        {
            "type": "lexical_search",
            "text_column": "prompt",
            "ignored_columns": [],
        },
    ).ok("configure streaming search")
    searched = app.post(
        f"/api/projects/{evaluation}/structured/search",
        {"query": "transfer", "limit": 2},
    ).ok("search Parquet batches")
    expect(
        searched["engine"] == "streaming character hashing"
        and searched["rows_scanned"] == 2
        and searched["results"],
        f"streaming search returned {searched}",
    )
    return (
        "paged DuckDB table + streamed export/search/evaluation + CatBoost/text "
        "training, prediction, Smart Review and native download"
    )


# ---------------------------------------------------------------------------
# 5. Settings, legal, licence
# ---------------------------------------------------------------------------


@check("settings")
def check_settings(app: App) -> str:
    """Read, write every performance mode, read back -- and the devices endpoint.

    Settings live on the backend because the *training process* reads them, and
    it has no browser; a mode that does not survive the round trip is a mode
    that never reaches a run.

    /api/settings/devices is the one endpoint that must not initialise CUDA in
    the API process: training is started with multiprocessing.Process, which
    forks on Linux, and a process that has initialised CUDA cannot fork a child
    that uses it. Asserting that from outside is not possible directly, so this
    checks the answer's shape and the `training` check -- which runs after this
    one and asks for the device reported here -- is the proof.
    """
    original = app.get("/api/settings").ok("GET /api/settings")
    expect("performance_mode" in original, f"settings look like {original}")
    for field in ("cpu_count", "num_workers_gpu", "num_workers_cpu"):
        expect(field in original.get("resolved", {}), f"resolved has no {field}")

    modes = app.get("/api/settings/performance-modes").ok("GET performance-modes")
    expect(
        set(modes["modes"]) == {"maximum", "balanced", "power_saving"},
        f"the build offers modes {modes['modes']}",
    )

    try:
        for mode in modes["modes"]:
            app.put("/api/settings", {"performance_mode": mode}).ok(f"set {mode}")
            back = app.get("/api/settings").ok("read the mode back")
            expect(
                back["performance_mode"] == mode,
                f"set {mode} and read back {back['performance_mode']}",
            )
            expect(
                isinstance(back["resolved"]["num_workers_gpu"], int),
                f"{mode} resolved to {back['resolved']}",
            )
        rejected = app.put("/api/settings", {"performance_mode": "ludicrous"})
        expect(rejected.status == 422, f"an unknown mode returned {rejected.status}")
    finally:
        # Put it back. Under --base-url this is somebody's real machine.
        app.put("/api/settings", {"performance_mode": original["performance_mode"]})

    devices = app.get("/api/settings/devices").ok("GET /api/settings/devices")
    expect(isinstance(devices.get("cuda"), bool), f"devices said {devices}")
    expect(
        not devices.get("error"),
        f"the devices endpoint could not ask torch: {devices.get('error')}",
    )

    # `accelerators`, not `cuda`. Reading only the CUDA flag made this report
    # "no GPU on this machine" on an M1 that was about to train on Metal --
    # wrong in exactly the release that added Metal. `cuda` is still checked
    # above because the shipped frontend reads it, and it is honestly false on
    # a Mac.
    accelerators = devices.get("accelerators", [])
    expect(isinstance(accelerators, list), f"devices said {devices}")
    for accelerator in accelerators:
        expect(
            accelerator.get("id") and accelerator.get("label"),
            f"an accelerator has no id or label: {accelerator}",
        )
    if devices["cuda"]:
        expect(devices.get("name"), "cuda is available but the GPU has no name")

    if accelerators:
        found = ", ".join(a["label"] for a in accelerators)
        excluded = sorted(
            {
                project_type
                for a in accelerators
                for project_type in a.get("excluded_project_types", [])
            }
        )
        summary = f"accelerator: {found}"
        if excluded:
            summary += f" (not for {len(excluded)} project types)"
    else:
        summary = "no accelerator on this machine, CPU only"

    return f"{len(modes['modes'])} modes round-trip; {summary}"


@check("legal")
def check_legal(app: App) -> str:
    """The notices, the terms and the model policy, as shipped in the build.

    These are obligations, not features: the permissive licences we redistribute
    under all require the notice to travel with the binary, and a build that
    dropped LICENSES.md answers 404 here while looking perfectly healthy
    everywhere else.
    """
    notices = app.get("/api/legal/notices").ok("GET /api/legal/notices")
    components = notices.get("components") or []
    expect(len(components) > 10, f"only {len(components)} components in the notices")
    expect(
        all(entry.get("name") and entry.get("text") for entry in components),
        "some notice components came back with no name or no text",
    )

    license_text = app.get("/api/legal/license").ok("GET /api/legal/license")
    expect(
        "Apache License" in license_text.get("text", ""),
        "the project license is missing or unexpected",
    )

    policy = app.get("/api/legal/model-policy").ok("GET /api/legal/model-policy")
    expect(len(policy.get("text", "")) > 200, "the model policy is suspiciously short")

    return f"{len(components)} components, project license and model policy present"


@check("auto_labeling")
def check_auto_labeling(app: App) -> str:
    """The bundled SAM models: installed, loadable, and able to segment offline.

    The app is sold on working with no network, and auto-labelling is the part
    that used to download 28 MB the first time somebody clicked it. The models
    are shipped in the build and seeded into the data root at startup, so all
    three must report as installed *without* a download -- which is observable,
    because ModelManager reports "Downloading <url>: n%" through its status
    while it fetches one.
    """
    project_id = app.project(
        "feature-test auto labeling", "Object Detection", labels_for(CLASS_NAMES)
    )
    app.upload_images(project_id, [("one.png", sample(0)[1])])
    item = app.items(project_id)[0]

    listed = app.get(f"/api/projects/{project_id}/auto_labeling/models").ok(
        "GET auto_labeling/models"
    )
    by_name = {entry["name"]: entry for entry in listed}
    missing = [name for name in BUNDLED_AUTO_LABELING_MODELS if name not in by_name]
    expect(not missing, f"the build does not offer bundled models {missing}")

    # Load and segment first, then assert the whole bundle is installed. The
    # other order stops at the first model that reports itself missing and
    # leaves the part that matters -- that a model in this build can actually
    # produce a mask -- untested.
    target = BUNDLED_AUTO_LABELING_MODELS[0]
    app.post(
        f"/api/projects/{project_id}/auto_labeling/load_model", {"model_name": target}
    ).ok(f"load {target}")

    # Loading happens on a thread; the status is how the UI knows it finished.
    deadline = time.time() + 300
    seen = ""
    while time.time() < deadline:
        seen = app.get(f"/api/projects/{project_id}/auto_labeling/status").ok(
            "auto_labeling status"
        )["status"]
        expect(
            "Downloading" not in seen,
            f"loading a bundled model started a download: {seen}",
        )
        expect("Error" not in seen, f"loading {target} failed: {seen}")
        if "Ready" in seen or "loaded" in seen.lower():
            break
        time.sleep(2)
    else:
        raise Failed(f"{target} never finished loading (last status: {seen})")

    marks = [{"data": [IMAGE_SIZE // 2, IMAGE_SIZE // 2], "label": 1, "type": "point"}]
    result = app.post(
        f"/api/projects/{project_id}/auto_labeling/inference",
        {"model_name": target, "data_item_id": item["id"], "marks": marks},
        timeout=600,
    ).ok("auto-labelling inference")
    expect(result.get("status") == "success", f"inference said {result}")
    shapes = (result.get("result") or {}).get("shapes")
    expect(shapes, f"the model returned no shapes: {str(result)[:300]}")
    expect(
        any(shape.get("points") for shape in shapes),
        f"the returned shapes have no points: {str(shapes)[:300]}",
    )

    uninstalled = [
        name
        for name in BUNDLED_AUTO_LABELING_MODELS
        if not by_name[name].get("has_downloaded")
    ]
    if uninstalled and app.data_root:
        # Where the copies actually are, so a failure names the disagreement
        # rather than only the symptom: create_app seeds them into DATA_ROOT.
        seeded = [
            name
            for name in uninstalled
            if (pathlib.Path(app.data_root) / "models" / name / "config.yaml").is_file()
        ]
        expect(
            not seeded,
            f"{seeded} are seeded under {app.data_root}/models and the app still "
            "reports them as not installed -- it is reading a different models "
            "directory, and on a machine without one it would download them",
        )
    expect(
        not uninstalled,
        f"bundled models {uninstalled} report as not installed, so clicking them "
        "in the labelling screen would download them",
    )

    return f"{len(by_name)} models offered, {target} segmented offline"


@check("data_isolation")
def check_data_isolation(app: App) -> str:
    """The run touched nothing in the default data root.

    ANYLEARNING_DATA_ROOT is supposed to move the whole store. Anything the app
    writes under ~/anylearning-data while pointed somewhere else is a component
    that hardcoded the default path -- which means a customer who moved their
    data root has a feature reading the wrong directory, and it means this
    script is quietly editing the tester's own projects.
    """
    if not app.data_root:
        raise Skipped("--base-url: this run did not choose the data root")

    default_root = pathlib.Path(os.path.expanduser("~")) / "anylearning-data"
    if str(default_root) == str(pathlib.Path(app.data_root)):
        raise Skipped("the temporary data root is the default one")

    after = _snapshot(default_root)
    created = sorted(after - DEFAULT_ROOT_BEFORE)
    expect(
        not created,
        f"the app created {len(created)} path(s) under {default_root} while "
        f"ANYLEARNING_DATA_ROOT pointed elsewhere: {created[:5]}",
    )
    return f"{default_root} unchanged ({len(after)} paths)"


# ---------------------------------------------------------------------------
# 6. Training, end to end
# ---------------------------------------------------------------------------


def _labelled_classification_project(app: App, name: str) -> int:
    """A small, real, labelled project -- built the way the UI builds one."""
    project_id = app.project(name, "Image Classification", labels_for(CLASS_NAMES))
    for subset in (0, 1, 2):
        images = []
        for index in range(PER_SUBSET):
            class_name, image, _ = sample(index + subset)
            images.append((f"{class_name}_{subset}_{index}.png", image))
        app.upload_images(project_id, images, subset=subset)
        for position, item in enumerate(app.items(project_id, subset=subset)):
            app.post(
                f"/api/projects/{project_id}/data_items/{item['id']}/class_id",
                {"class_id": (position + subset) % len(CLASS_NAMES)},
            ).ok("label for training")
    return project_id


def _training_params(app: App, project_type: str, epochs: int, device: str) -> dict:
    variants = app.get("/api/model-variants").ok("GET /api/model-variants")
    options = variants.get(project_type) or []
    expect(options, f"no model variant is configured for {project_type}")
    variant = options[0]
    return {
        "model_architecture": variant["model_architecture"],
        "model_size": variant["model_size"],
        "model_variant": variant["name"],
        "batch_size": TRAIN_BATCH,
        "epochs": epochs,
        "learning_rate": 0.001,
        "pretrained_model": "default",
        "device": device,
        "image_size": None,
    }


def _session(app: App, project_id: int, session_id: int) -> dict:
    # A long timeout on purpose: the server shares the machine with the training
    # process, and a slow answer is the run being busy, not the app being broken.
    return app.get(
        f"/api/projects/{project_id}/training_sessions/{session_id}", timeout=180
    ).ok("training session detail")


@check("training_terminate")
def check_training_terminate(app: App) -> str:
    """Starting a run and stopping it again.

    Stop is a shipped button, and it is the only thing standing between a user
    and a run they cannot get rid of. The run is given far more epochs than it
    needs so that it cannot finish before it is asked to stop.
    """
    project_id = _labelled_classification_project(app, "feature-test terminate")
    params = _training_params(app, "Image Classification", epochs=200, device="cpu")
    started = app.post(
        f"/api/projects/{project_id}/training_sessions", params, timeout=180
    ).ok("start a training session")
    session_id = started["session_id"]

    deadline = time.time() + 600
    status = "unknown"
    while time.time() < deadline:
        status = _session(app, project_id, session_id)["status"]
        if status in {"training", "evaluating"}:
            break
        expect(
            status not in {"error", "finished", "terminated"},
            f"the run ended as '{status}' before it could be stopped",
        )
        time.sleep(3)
    else:
        raise Failed(f"the run never started (last status: {status})")

    app.post(
        f"/api/projects/{project_id}/training_sessions/{session_id}/terminate",
        timeout=180,
    ).ok("terminate the run")

    ended = _session(app, project_id, session_id)
    expect(
        ended["status"] == "terminated",
        f"after terminate the session reports '{ended['status']}'",
    )

    again = app.post(
        f"/api/projects/{project_id}/training_sessions/{session_id}/terminate"
    )
    expect(
        again.status == 400,
        f"terminating a stopped run returned {again.status}, not a refusal",
    )
    return "started, reported training, stopped on request"


@check("training")
def check_training(app: App) -> str:
    """One run, end to end, and everything that hangs off the model it produces.

    The training matrix -- every type, both devices, every OS -- lives in
    smoke_test_training.py and `--self-test`. What this proves is the rest of the
    chain in the package: that a run started over HTTP registers a model (which
    only happens after ONNX export succeeds), that the model router can then run
    inference through it, and that both artefacts download.

    The device is whatever /api/settings/devices reported, and the run asserts it
    trained there. On Linux that is also the check that the API process had not
    initialised CUDA when it answered: a process that has cannot fork a child
    that uses it, so a GPU run started afterwards would fail.
    """
    devices = app.get("/api/settings/devices").ok("GET /api/settings/devices")
    device = "gpu" if devices.get("cuda") else "cpu"

    augmentations = app.get("/api/augmentations").ok("GET /api/augmentations")
    variants = app.get("/api/model-variants").ok("GET /api/model-variants")
    missing = [
        name
        for name in variants
        if name not in augmentations and name not in STRUCTURED_PROJECT_TYPES
    ]
    expect(not missing, f"the training dialog has no augmentation list for {missing}")

    project_id = _labelled_classification_project(app, "feature-test training")
    params = _training_params(app, "Image Classification", TRAIN_EPOCHS, device)
    started = app.post(
        f"/api/projects/{project_id}/training_sessions", params, timeout=180
    ).ok("start training")
    session_id = started["session_id"]

    deadline = time.time() + TRAIN_BUDGET
    detail: dict = {}
    status = "unknown"
    while time.time() < deadline:
        try:
            detail = _session(app, project_id, session_id)
        except Failed:
            # No error budget: the deadline is the only timeout. A busy server
            # answering slowly is a run in progress, not a run that failed.
            time.sleep(5)
            continue
        status = detail.get("status", "unknown")
        if status in {"finished", "error", "terminated"}:
            break
        time.sleep(5)
    else:
        raise Failed(f"still '{status}' after {TRAIN_BUDGET}s: {_tail(detail)}")

    expect(status == "finished", f"the run ended as '{status}': {_tail(detail)}")
    model_id = (detail.get("model") or {}).get("id")
    expect(
        model_id,
        "the run finished but registered no model -- the ONNX export failed, and "
        f"the run was discarded: {_tail(detail)}",
    )

    trained_on = _device_from_logs(detail.get("training_logs"))
    if trained_on:
        expect(
            device.upper() in trained_on.upper(),
            f"asked to train on the {device} and it trained on {trained_on}",
        )

    listed = app.get(f"/api/projects/{project_id}/models").ok("list models")
    expect(listed["total_count"] >= 1, "the models list is empty after a finished run")
    model = app.get(f"/api/projects/{project_id}/models/{model_id}").ok(
        "read the model"
    )
    expect(model.get("exported_path"), f"the model has no exported ONNX: {model}")
    expect(
        model.get("model_variant") not in (None, "Unknown"),
        f"the model's variant reads as {model.get('model_variant')}",
    )

    # The last training session endpoint is what the project screen polls.
    last = app.get(f"/api/projects/{project_id}/last_training_session").ok(
        "last training session"
    )
    expect(last and last["id"] == session_id, f"last_training_session is {last}")

    inference = app.upload(
        f"/api/projects/{project_id}/models/{model_id}/inference",
        [("file", "test.png", sample(0)[1])],
        timeout=600,
    ).ok("run inference through the trained model")
    expect("results" in inference, f"inference returned {str(inference)[:200]}")
    expect(
        str(inference.get("visualization_image", "")).startswith("data:image/"),
        "inference returned no visualisation image",
    )

    raw = app.get(f"/api/projects/{project_id}/models/{model_id}/download", timeout=600)
    expect(raw.status == 200, f"downloading the checkpoint returned {raw.status}")
    expect(len(raw.body) > 10_000, f"the checkpoint is {len(raw.body)} bytes")

    exported = app.get(
        f"/api/projects/{project_id}/models/{model_id}/download_exported", timeout=600
    )
    expect(exported.status == 200, f"downloading the ONNX returned {exported.status}")
    names = open_zip(exported.body).namelist()
    expect(
        any(name.endswith(".onnx") for name in names),
        f"the exported archive holds {names}",
    )
    expect("config.yml" in names, f"the exported archive has no config: {names}")

    return (
        f"trained on {trained_on or device}, model {model_id}, "
        f"{len(raw.body) // 1024} KB checkpoint + {len(exported.body) // 1024} KB ONNX"
    )


def _tail(detail: dict, lines: int = 2) -> str:
    logs = (detail.get("training_logs") or "").strip().splitlines()
    return " | ".join(logs[-lines:])[:300] if logs else "(no logs)"


def _device_from_logs(logs) -> str | None:
    for line in reversed((logs or "").splitlines()):
        if "Training device:" in line:
            return line.split("Training device:", 1)[1].strip()
    return None


@check("update_check")
def check_update_check(app: App) -> str:
    """Whether this build can tell the user a new version exists.

    Discovered from the build's own OpenAPI schema rather than by guessing URLs:
    if a route for it is ever added, this check starts exercising it instead of
    reporting that there is none.
    """
    schema = app.get("/openapi.json").ok("GET /openapi.json")
    candidates = [
        path
        for path in schema.get("paths", {})
        if re.search(r"update|upgrade|new[-_]?version|releases?$", path, re.I)
    ]
    if not candidates:
        raise Skipped(
            "no update-check endpoint, and there should not be one: the check "
            "lives in the frontend, which fetches the website's "
            "check-for-update.json directly (see layout/sidebar.tsx). Nothing "
            "here can exercise it -- it is a browser fetch to a remote host."
        )
    for path in candidates:
        reply = app.get(path)
        expect(reply.status == 200, f"GET {path} returned {reply.status}")
    return f"answered: {', '.join(candidates)}"


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------

#: Filled in before the server starts, and read by the data_isolation check.
DEFAULT_ROOT_BEFORE: set = set()


def _snapshot(root: pathlib.Path, depth: int = 3) -> set:
    """The paths under `root`, down to `depth`. An absent root is an empty set.

    Bounded rather than a full walk: a real data root holds every image of every
    project, and reading all of it twice would take longer than some of the
    checks. Three levels is enough to see anything a component with a hardcoded
    path would create -- `models/<name>/config.yaml`, `projects/<id>/database.db`
    -- and, unlike a truncated walk, it produces the same set every time, so the
    comparison cannot report a difference that is only iteration order.
    """
    if not root.is_dir():
        return set()

    found = set()

    def walk(directory: pathlib.Path, level: int) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            found.add(str(entry.relative_to(root)))
            if entry.is_dir() and level < depth:
                walk(entry, level + 1)

    walk(root, 1)
    return found


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start_binary(binary: str, port: int, data_root: str, log_path: str):
    """Run the packaged app as a server, with its output going to a file.

    Never to a pipe. A pipe nobody drains holds 64KB and then blocks whoever is
    writing, and this app writes far more than that -- nanodet's ONNX export
    prints its whole graph, and on a spawn platform every dataloader worker
    re-imports the application and repeats its warnings. Training then stops
    mid-write, which reads as a hang and has cost this project a day twice.
    """
    env = dict(os.environ)
    env["ANYLEARNING_DATA_ROOT"] = data_root
    # Relaxes the per-window bearer token so a script can call the API. Not
    # --development: that flag puts uvicorn in reload mode, which re-imports the
    # entry point in a subprocess and cannot work inside a frozen binary.
    env["ANYLEARNING_DEVELOPMENT"] = "1"
    # A package in ~/.local shadows the environment and produces failures that
    # reproduce nowhere else. Harmless for a frozen binary, essential when this
    # is pointed at `python anylearning/app.py`.
    env["PYTHONNOUSERSITE"] = "1"
    # Actively *removed*: TORCH_HOME, HF_HOME, HUGGINGFACE_HUB_CACHE and
    # FVCORE_CACHE. This used to say "deliberately not set", which was true and
    # not enough -- `dict(os.environ)` inherits them, and anything that imports
    # `anylearning` in this process has already pointed them at the checkout's
    # weights. `weights.use_bundled()` then skips every variable that has a
    # value, so the binary reads the developer's directory instead of its own
    # and the whole sweep passes against a configuration nobody ships.
    #
    # It hid a real bug: instance segmentation could not train from a read-only
    # installation on macOS, and no harness could see it.
    from anylearning.selftest.driver import scrub_cache_variables

    dropped = scrub_cache_variables(env)
    if dropped:
        print(f"  env       dropped inherited {', '.join(dropped)}")
    # HF_HUB_OFFLINE is the one worth forcing: it turns "quietly reached the
    # network" into a visible failure.
    env["HF_HUB_OFFLINE"] = "1"

    # Closed as soon as the child has inherited it: the child writes through its
    # own descriptor, and leaving a handle open here would stop Windows deleting
    # the data root the log sits in.
    with open(log_path, "ab", buffering=0) as handle:
        return subprocess.Popen(
            [binary, "--server", "--port", str(port)],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def wait_for_server(base: str, process, seconds: int = 300) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise SystemExit(f"the app exited with {process.returncode} before serving")
        try:
            with urllib.request.urlopen(f"{base}/openapi.json", timeout=5):
                return
        except Exception:  # noqa: BLE001 -- not up yet is the normal case here
            time.sleep(2)
    raise SystemExit(
        f"nothing answered on {base} within {seconds}s. When the app is started "
        "with a port that is already taken it silently moves to a random one, so "
        "the log is worth reading before assuming the build is broken."
    )


def stop_binary(process) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=60)


def tail(path: str | None, lines: int = 40) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "binary",
        nargs="?",
        help="the packaged executable to test (omit with --base-url)",
    )
    parser.add_argument("--port", type=int, default=0, help="default: a free one")
    parser.add_argument(
        "--base-url", help="talk to a server already running here, and start nothing"
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the temporary data root and projects"
    )
    parser.add_argument("--only", default="", help="comma-separated check names")
    parser.add_argument("--skip", default="", help="comma-separated check names")
    arguments = parser.parse_args()

    names = [name for name, _ in CHECKS]
    only = {n.strip() for n in arguments.only.split(",") if n.strip()}
    skip = {n.strip() for n in arguments.skip.split(",") if n.strip()}
    unknown = (only | skip) - set(names)
    if unknown:
        print(f"unknown check(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        print(f"known checks: {', '.join(names)}", file=sys.stderr)
        return 2
    selected = [
        (name, function)
        for name, function in CHECKS
        if (not only or name in only) and name not in skip
    ]
    if not selected:
        print("no checks selected", file=sys.stderr)
        return 2

    if not arguments.base_url and not arguments.binary:
        parser.error("give a binary to test, or --base-url for a server you started")
    if arguments.binary and not os.path.exists(arguments.binary):
        print(f"no binary at {arguments.binary}", file=sys.stderr)
        return 2

    process = None
    data_root = None
    log_path = None

    if arguments.base_url:
        base = arguments.base_url.rstrip("/")
        print(f"Testing the server at {base} (started by someone else)")
        wait_for_server(base, None, seconds=30)
    else:
        # Its own data root, always. A sweep that creates and deletes projects
        # must not be able to do that to ~/anylearning-data.
        data_root = tempfile.mkdtemp(prefix="anylearning-feature-test-")
        log_path = os.path.join(data_root, "app.log")
        global DEFAULT_ROOT_BEFORE
        DEFAULT_ROOT_BEFORE = _snapshot(
            pathlib.Path(os.path.expanduser("~")) / "anylearning-data"
        )
        port = arguments.port or free_port()
        base = f"http://127.0.0.1:{port}"
        print(f"Testing {arguments.binary}")
        print(f"  data root {data_root}")
        print(f"  serving   {base}")
        process = start_binary(arguments.binary, port, data_root, log_path)
        try:
            wait_for_server(base, process)
        except SystemExit as never_started:
            # Say what the app said, and take the server and the temporary root
            # with us -- an aborted start used to leave both behind, and the
            # next run then talked to the previous server.
            print(never_started, file=sys.stderr)
            print(tail(log_path), file=sys.stderr)
            stop_binary(process)
            if not arguments.keep:
                shutil.rmtree(data_root, ignore_errors=True)
            return 1

    app = App(base, data_root, log_path)
    results = []
    started = time.time()

    try:
        for name, function in selected:
            print(f"  {name} ...", end=" ", flush=True)
            begin = time.time()
            try:
                detail = function(app) or ""
                outcome = "PASS"
            except Skipped as reason:
                detail, outcome = str(reason), "SKIP"
            except Failed as failure:
                detail, outcome = str(failure), "FAIL"
            except Exception as error:  # noqa: BLE001 -- one broken check, not a broken run
                detail = f"{type(error).__name__}: {error}"
                outcome = "FAIL"
            finally:
                if not arguments.keep:
                    app.drop_projects()
                else:
                    app.projects.clear()
            elapsed = time.time() - begin
            results.append((name, outcome, detail, elapsed))
            print(f"{outcome} ({elapsed:.0f}s)", flush=True)
            if outcome != "PASS":
                print(f"      {detail}", flush=True)
    finally:
        stop_binary(process)

    width = max(len(name) for name, *_ in results)
    print("\n" + "=" * 78, flush=True)
    for name, outcome, detail, elapsed in results:
        print(
            f"{name:<{width}}  {outcome:<4}  {elapsed:>5.0f}s  {detail[:200]}",
            flush=True,
        )
    print("=" * 78, flush=True)

    failed = [row for row in results if row[1] == "FAIL"]
    skipped = [row for row in results if row[1] == "SKIP"]
    minutes = (time.time() - started) / 60
    summary = (
        f"{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped in {minutes:.1f} min"
    )

    # Read the log before the data root goes: it lives inside it, and what the
    # app said while a check was failing is usually the whole diagnosis.
    if failed and log_path:
        print(f"\nlast lines of {log_path}:", flush=True)
        print(tail(log_path), flush=True)

    if data_root and not arguments.keep:
        shutil.rmtree(data_root, ignore_errors=True)
    elif data_root:
        print(f"kept the data root at {data_root}", flush=True)

    if failed:
        print(f"FAIL: {summary}", file=sys.stderr)
        return 1
    print(f"PASS: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Not reachable from here
#
# Everything below is part of the product and has no HTTP surface, so a release
# still needs a human to look at it. Listed rather than faked: a check that
# cannot fail is worse than no check.
#
# * The desktop window itself. /window/close, /maximize, /restore and /minimize
#   exist, but they act on a live pywebview window and this script starts the
#   app with --server, which has none -- they answer "No window to ...". The
#   frameless title bar, the drag regions, the rounded corners and the platform
#   chrome are covered by smoke_test_window_chrome.py and its PowerShell
#   counterpart, on a machine with a display.
# * File dialogs. Downloading a model or an export from the UI goes through
#   pywebview's SAVE_DIALOG via window.expose(download_file); over HTTP this
#   script gets the bytes instead, which proves the route but not the dialog.
# * Auto-labelling by rectangle marks and the multi-image preload path. The
#   point-mark path is exercised; the rest is a canvas interaction.
# * The handpose annotation itself -- a dict of 21 mediapipe landmarks. It is
#   only ever written by the upload path from an image with a real hand in it,
#   and no drawn shape produces landmarks, so a round trip needs real data:
#   `python smoke_test_training.py <binary>` against a machine that has some.
# * Custom auto-labelling models (ModelManager.load_custom_model). Loaded from a
#   config file the user picks in a file dialog; there is no route for it.
# * Cancelling an export (/projects/{id}/export/cancel, and DELETE .../export).
#   Both need the cancel to land while the background task is still running, and
#   the exports this script makes finish in under a second -- a check that races
#   would fail for reasons that have nothing to do with the build.
# * An update check. This build has no endpoint for one -- see the update_check
#   check, which reports that rather than inventing a URL.
# ---------------------------------------------------------------------------
