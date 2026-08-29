"""Locate files that ship inside the ``anylearning`` package.

This replaces ``pkg_resources.resource_filename``. ``pkg_resources`` is part of
setuptools, which is no longer installed by default on Python 3.12+ and which
removed the module outright in setuptools 81. Importing it therefore breaks on a
freshly created environment, before any AnyLearning code gets a chance to run.

``importlib.resources`` is the stdlib replacement and needs no dependency at all.
``anylearning/config.py`` and ``anylearning/auto_labeling/model_manager.py``
already used it; this helper covers the ``resource_filename`` call sites, which
need a real filesystem path rather than an open file handle.
"""

import os
from importlib import resources


def resource_path(package: str, resource: str) -> str:
    """Return the absolute filesystem path of a resource inside ``package``.

    Behaves like ``pkg_resources.resource_filename(package, resource)`` for the
    regular (unzipped) installs AnyLearning ships, including editable installs
    and the Nuitka-built desktop app.

    Args:
        package: importable package name, e.g. ``"anylearning"``.
        resource: path relative to that package, e.g. ``"training/configs/x.yml"``.
    """
    root = resources.files(package)
    # joinpath() takes one segment at a time on Traversable implementations.
    for part in resource.replace(os.sep, "/").split("/"):
        if part:
            root = root.joinpath(part)
    return str(root)
