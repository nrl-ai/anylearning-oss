"""Repairs for things that only break in a Nuitka-compiled binary.

A frozen build has no source files on disk, so any library that works out a
value by looking at its own ``__file__`` quietly gets a different answer -- or
none at all. That is invisible in development and fatal in the packaged app, so
the fixes live here rather than scattered through the trainers.

Call :func:`apply` early in every process that runs library code. That means the
server *and* the training child: macOS and Windows spawn rather than fork, so
the child re-imports everything from scratch and inherits none of this.
"""

from __future__ import annotations

import os

from loguru import logger


def _ensure_utf8_child_io() -> None:
    """Make spawned Windows training processes able to log Unicode.

    The server starts each training run with ``multiprocessing.Process``. On
    Windows that launches a new interpreter, whose standard-stream encoding is
    otherwise the machine's legacy ANSI code page. RF-DETR's progress output
    contains Unicode box-drawing characters, so a healthy keypoint run died
    while printing its first epoch with ``charmap can't encode``.

    Setting this in the server process is early enough for every subsequently
    spawned training process to inherit it. Preserve an explicit operator
    choice if one is already present.
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _ensure_lightning_version() -> None:
    """Give ``pytorch_lightning.__version__`` a value.

    Lightning sets it at import time behind ``os.path.isfile`` checks against
    its own ``__file__``. Those files are not on disk in a compiled binary, so
    neither branch runs and the attribute never exists -- and Lightning's own
    checkpoint writer reads it:

        pytorch_lightning/trainer/connectors/checkpoint_connector.py, in
        dump_checkpoint
        AttributeError: module 'pytorch_lightning' has no attribute '__version__'

    Every NanoDet run died there, at the first checkpoint save, after training
    had already succeeded. The value is only stamped into the checkpoint for
    provenance, so any honest string will do.
    """
    try:
        import pytorch_lightning as pl
    except Exception:  # noqa: BLE001 -- not every process needs Lightning
        return

    if getattr(pl, "__version__", None):
        return

    version = None
    try:
        from importlib.metadata import version as distribution_version

        version = distribution_version("pytorch-lightning")
    except Exception:  # noqa: BLE001 -- metadata may not be bundled either
        pass

    pl.__version__ = version or "unknown"
    logger.info(
        f"Set pytorch_lightning.__version__ to {pl.__version__} for the frozen build"
    )


def _ensure_transformers_docstrings() -> None:
    """Stop transformers reading source that a compiled module does not have.

    ``transformers.utils.doc.get_docstring_indentation_level`` calls
    ``inspect.getsource`` on the function whose docstring it is about to rewrite.
    RF-DETR decorates ``WindowedDinov2WithRegistersModel.forward`` with
    ``add_start_docstrings_to_model_forward``, so in a compiled build the
    decorator runs at class-definition time, finds no source, and ``import
    rfdetr`` raises

        OSError: could not get source code

    before a single batch. It takes out training *and* inference, on CPU and GPU
    alike, for detection and instance segmentation both, and it is invisible
    outside a build: every test and every development run has source on disk.

    The fallback is only used when the real answer is unavailable, and it is the
    same 4 that transformers already returns for a class without reading
    anything. The cost is a cosmetically mis-indented docstring in a build where
    nobody reads docstrings.
    """
    try:
        from transformers.utils import doc
    except Exception:  # noqa: BLE001 -- a build without transformers has no RF-DETR
        return

    original = getattr(doc, "get_docstring_indentation_level", None)
    if original is None or getattr(original, "_anylearning_frozen_safe", False):
        return

    def get_docstring_indentation_level(func):
        try:
            return original(func)
        except OSError:
            return 4

    get_docstring_indentation_level._anylearning_frozen_safe = True
    doc.get_docstring_indentation_level = get_docstring_indentation_level


def _alias_torch_onnx_init() -> None:
    """Let torch find ``torch.onnx`` under the name a compiled build gives it.

    Nuitka names a compiled package initialiser ``<package>.__init__``, so a
    function defined in ``torch/onnx/__init__.py`` reports
    ``__module__ == "torch.onnx.__init__"``. When ``torch.onnx.export`` traces a
    model containing a custom ``torch.autograd.Function`` it resolves that name
    back to a module, and there is no such module:

        File ".../torch/autograd/function.py", line 596, in apply
        ModuleNotFoundError: No module named 'torch.onnx.__init__'

    RF-DETR's segmentation head has exactly one such Function --
    ``_DepthwiseConvWithoutCuDNN`` in ``rfdetr/models/heads/segmentation.py`` --
    so RF-DETR-Seg trains in a packaged build and then dies in
    ``export_onnx``. Registration is gated on the export, so the run produces no
    model at all. Detection has no custom Function and is unaffected, which is
    why only half of RF-DETR looked broken.

    An alias rather than a patch: the two names should mean the same module, and
    ``import_module`` consults ``sys.modules`` before the finders.
    """
    import sys as _sys

    module = _sys.modules.get("torch.onnx")
    if module is None:
        try:
            import torch.onnx as module  # noqa: PLC0415
        except Exception:  # noqa: BLE001 -- a build without onnx has no export
            return
    _sys.modules.setdefault("torch.onnx.__init__", module)


def apply() -> None:
    """Apply every repair. Safe to call more than once, and never raises."""
    for repair in (
        _ensure_utf8_child_io,
        _ensure_lightning_version,
        _ensure_transformers_docstrings,
        _alias_torch_onnx_init,
    ):
        try:
            repair()
        except Exception as exc:  # noqa: BLE001 -- a repair must never be the failure
            logger.warning(f"Frozen-build repair {repair.__name__} failed: {exc}")
