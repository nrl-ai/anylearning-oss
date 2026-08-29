"""Repairs for a Nuitka-compiled binary.

These exist because a frozen build has no source files on disk, so a library
that derives a value from its own ``__file__`` gets a different answer or none
at all. That is invisible in development, which is exactly why it needs tests:
nothing else here would notice the repair being removed.
"""

import os

import pytest


def test_lightning_version_is_restored_when_missing(monkeypatch):
    """Lightning's own checkpoint writer reads pl.__version__.

    It is set at import behind os.path.isfile checks against __file__, so in a
    compiled binary the attribute never exists and every NanoDet run died at
    the first checkpoint save -- after training had already succeeded.
    """
    pl = pytest.importorskip("pytorch_lightning")
    from anylearning import frozen_compat

    monkeypatch.delattr(pl, "__version__", raising=False)
    assert not hasattr(pl, "__version__")

    frozen_compat.apply()

    assert getattr(pl, "__version__", None)


def test_an_existing_version_is_left_alone(monkeypatch):
    pl = pytest.importorskip("pytorch_lightning")
    from anylearning import frozen_compat

    monkeypatch.setattr(pl, "__version__", "9.9.9", raising=False)
    frozen_compat.apply()
    assert pl.__version__ == "9.9.9"


def test_falls_back_when_metadata_is_unavailable(monkeypatch):
    """Distribution metadata is not guaranteed to be bundled either."""
    pl = pytest.importorskip("pytorch_lightning")
    from anylearning import frozen_compat

    monkeypatch.delattr(pl, "__version__", raising=False)

    import importlib.metadata

    def no_metadata(_name):
        raise importlib.metadata.PackageNotFoundError("not bundled")

    monkeypatch.setattr(importlib.metadata, "version", no_metadata)

    frozen_compat.apply()

    assert pl.__version__ == "unknown"


def test_apply_never_raises(monkeypatch):
    """It runs before training; a failure here must not become the failure."""
    from anylearning import frozen_compat

    def explode():
        raise RuntimeError("repair itself is broken")

    monkeypatch.setattr(frozen_compat, "_ensure_lightning_version", explode)
    frozen_compat.apply()  # must not raise


def test_apply_is_idempotent():
    from anylearning import frozen_compat

    frozen_compat.apply()
    frozen_compat.apply()


def test_spawned_training_processes_are_given_utf8_io(monkeypatch):
    from anylearning import frozen_compat

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    frozen_compat._ensure_utf8_child_io()
    assert os.environ["PYTHONIOENCODING"] == "utf-8"


def test_an_explicit_io_encoding_is_preserved(monkeypatch):
    from anylearning import frozen_compat

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8:backslashreplace")
    frozen_compat._ensure_utf8_child_io()
    assert os.environ["PYTHONIOENCODING"] == "utf-8:backslashreplace"
