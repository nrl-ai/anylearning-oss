"""Every module must import on its own.

A module that only imports because something else was imported first is a latent
failure: it breaks the moment import order changes -- a different entry point, a
new test, or a Nuitka-frozen build, where the order is not the interpreter's.

`instance_segmentation/factory.py` had exactly this problem. It pulled
`TrainingLogsWriter` in via `instseg_trainer`, which imports the factory back, so
importing the factory first raised ImportError. Nothing noticed, because the
trainer was always imported first in practice.
"""

import importlib

import pytest

# Modules that must be importable with nothing else loaded first.
STANDALONE_MODULES = [
    "anylearning.config",
    "anylearning.database",
    "anylearning.utils.resources",
    "anylearning.utils.converters",
    "anylearning.training.logging",
    "anylearning.training.device_utils",
    "anylearning.training.trainers.base_trainer",
    "anylearning.training.trainers.trainer_builder",
    "anylearning.training.models.instance_segmentation.factory",
    "anylearning.routers.dataset",
    "anylearning.routers.labeling",
    "anylearning.routers.model",
    "anylearning.routers.project",
    "anylearning.routers.training",
]


@pytest.mark.parametrize("module_name", STANDALONE_MODULES)
def test_module_imports_standalone(module_name, subprocess_import):
    """Import in a fresh interpreter, so nothing is already in sys.modules."""
    subprocess_import(module_name)


@pytest.fixture(scope="session")
def subprocess_import():
    import subprocess
    import sys

    def _import(module_name):
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"`import {module_name}` failed in a fresh interpreter:\n{result.stderr}"
        )

    return _import


def test_trainer_builder_resolves_every_project_type():
    """Each advertised project type must map to an importable trainer."""
    from anylearning import config
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    for project_type in config.MODEL_VARIANTS:
        trainer = TrainerBuilder.get_trainer_class(project_type)
        assert trainer is not None, f"no trainer for {project_type}"

    with pytest.raises(ValueError):
        TrainerBuilder.get_trainer_class("Not A Real Project Type")


def test_every_project_type_has_variants():
    """A project type with no variants would render an empty dropdown."""
    from anylearning import config

    for project_type, variants in config.MODEL_VARIANTS.items():
        assert variants, f"{project_type} has no model variants"
        for variant in variants:
            assert variant["name"]
            assert variant["model_architecture"]
            assert variant["model_size"]


def test_importlib_is_used_not_pkg_resources():
    """pkg_resources is gone from modern setuptools; nothing may import it."""
    import pathlib

    root = pathlib.Path(importlib.import_module("anylearning").__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import pkg_resources") or stripped.startswith(
                "from pkg_resources"
            ):
                offenders.append(f"{path}: {stripped}")
    assert not offenders, "use anylearning.utils.resources instead:\n" + "\n".join(
        offenders
    )
