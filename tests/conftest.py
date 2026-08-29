"""Shared test fixtures.

The main job here is stopping SQLite connections leaking between tests.
``DatabaseManager`` caches one engine per project and only ever released them
individually, so a test that touched a project database left its connection open.
Python 3.13 surfaces those as unraisable ResourceWarnings whenever the object is
finally collected -- which lands on whichever unrelated test happens to be
running at the time, making failures look random and order-dependent.
"""

import gc
import os
import pathlib

import pytest


@pytest.fixture(autouse=True)
def ignore_ambient_overrides(monkeypatch):
    """Run tests independently of the caller's data-root override."""
    monkeypatch.delenv("ANYLEARNING_DATA_ROOT", raising=False)


@pytest.fixture(autouse=True)
def dispose_database_engines():
    """Close every cached engine after each test.

    Autouse rather than opt-in: the leak is caused by any test that touches the
    database, and attributing the resulting warning to the right test is exactly
    the problem this avoids.
    """
    yield

    from anylearning.database import db_manager

    db_manager.dispose_all()
    # Force collection while the warning can still be attributed to this test
    # rather than to whatever runs next.
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def keep_downloads_out_of_the_weights_directory():
    """Point the weight caches somewhere disposable for the whole session.

    `anylearning/weights.py` aims TORCH_HOME, HF_HOME, HUGGINGFACE_HUB_CACHE and
    FVCORE_CACHE at the repository's `weights/` directory, which is a *build
    input*: `build_app.sh` ships whatever is in it. The vendored NanoDet tests
    construct backbones with `pretrain=True`, so running the suite downloaded
    resnet18, efficientnet_lite0 and a NanoDet state dict straight into the
    payload -- 86 MB, three files, and a build afterwards shipped them.

    The payload audit cannot catch this on its own, because it compares the
    source directory against the artefact and both contain the extra files. A
    Linux build on a clean machine counted 33 and a Windows build on a machine
    that had run pytest counted 36; that difference was the only visible sign.

    `use_bundled()` leaves any variable that already has a value alone, so
    setting them here is enough to redirect every download in the session.

    A **stable** directory, not `tmp_path_factory`. A fresh one per session was
    the first version and it re-downloaded those 86 MB on every run, which
    eventually got GitHub to answer the GhostNet state dict with `HTTP 429: Too
    Many Requests` -- a green suite turning red because of how often it had been
    run. Caching across sessions is also what makes the suite work offline once
    it has passed once, which matters for a product whose whole claim is that it
    needs no network.
    """
    cache = (
        pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache")
        / "anylearning-test-weights"
    )
    cache.mkdir(parents=True, exist_ok=True)
    previous = {}
    # Derived, so a cache variable added to weights.py is redirected here without
    # anyone remembering to. `anylearning.selftest.driver` keeps its own copy for
    # a documented reason and has a test pinning it to this same set.
    from anylearning.weights import CACHE_VARIABLES

    for variable in CACHE_VARIABLES:
        previous[variable] = os.environ.get(variable)
        os.environ[variable] = str(cache)
    yield cache
    for variable, value in previous.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value
