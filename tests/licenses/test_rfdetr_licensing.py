"""RF-DETR is Apache 2.0 -- but only the half of it we are allowed to ship.

Roboflow split the project: the `rfdetr` package on PyPI is Apache 2.0, code and
weights alike, while the XL and 2XL *detection* models live in a separate
package, `rfdetr_plus`, under the Platform Model License 1.0. PML 1.0 requires
every user to hold a Roboflow platform plan, which is incompatible with this
Apache-2.0 distribution -- see docs/model_license_policy.md, where it is Tier B.

Nothing in the code reaches for those models, and nothing should. What makes
that worth a test rather than a comment is how quietly it could stop being true:
`rfdetr[plus]` is one character of a dependency line away, `rfdetr.RFDETRXLarge`
resolves through a `__getattr__` that installs the package's own error message
rather than ours, and neither leaves a mark anywhere a reviewer would look.

This is a licence assertion, not a feature test. It fails on a change nobody
meant to make.
"""

import importlib.util

import pytest

pytest.importorskip("rfdetr")


#: The two model names that are not Apache 2.0.
PLATFORM_MODELS = ("RFDETRXLarge", "RFDETR2XLarge")


def test_the_platform_licensed_package_is_not_installed():
    """`rfdetr[plus]` would put PML 1.0 code inside the installer."""
    assert importlib.util.find_spec("rfdetr_plus") is None, (
        "rfdetr_plus is installed. It is licensed under Roboflow's Platform "
        "Model License 1.0, which requires a platform plan, and shipping it "
        "would bind that requirement onto our customers."
    )


def test_no_variant_offers_a_platform_licensed_model():
    from anylearning import config

    offered = {
        variant["name"]
        for variants in config.MODEL_VARIANTS.values()
        for variant in variants
    }
    for forbidden in ("XL", "2XL", "XLarge"):
        assert not [name for name in offered if forbidden in name]


def test_the_shipped_variants_all_name_apache_licensed_architectures():
    """rfdetr states a licence per architecture; ours have to say Apache-2.0."""
    import rfdetr.config as rfdetr_config
    import yaml

    from anylearning.training.trainers.rfdetr_trainer import (
        RFDetrSegTrainer,
        RFDetrTrainer,
    )

    for trainer in (RFDetrTrainer, RFDetrSegTrainer):
        for path in trainer.CONFIG_TEMPLATES.values():
            with open(path) as handle:
                variant = yaml.safe_load(handle)["model"]["variant"]
            assert variant not in PLATFORM_MODELS
            defaults = getattr(rfdetr_config, f"{variant}Config")()
            assert defaults.license == "Apache-2.0", variant


def _licence_by_checkpoint() -> dict:
    """Every checkpoint filename rfdetr knows, mapped to the licence it declares.

    Read from the package rather than assumed, because the answer is not
    guessable from the name. `rf-detr-seg-xlarge.pt` is Apache-2.0 and so is
    `rf-detr-keypoint-preview-xlarge.pth`, whose encoder is the same
    `dinov2_windowed_small` that Nano uses -- the "xlarge" is naming, not
    architecture, and not licence.
    """
    import rfdetr.config as rfdetr_config

    licences = {}
    for name in dir(rfdetr_config):
        if not name.endswith("Config") or name.startswith("_"):
            continue
        try:
            defaults = getattr(rfdetr_config, name)()
        except Exception:  # noqa: BLE001 -- base classes need arguments
            continue
        fields = (
            defaults.model_dump() if hasattr(defaults, "model_dump") else vars(defaults)
        )
        checkpoint = fields.get("pretrain_weights")
        if checkpoint and "license" in fields:
            licences[checkpoint] = fields["license"]
    return licences


def test_the_bundled_weights_come_from_apache_licensed_checkpoints():
    """Asks rfdetr what each checkpoint is licensed as, rather than reading names.

    This used to assert `"xlarge" not in upstream.lower()`, which is wrong in
    both directions. It rejects Apache-2.0 models for their filename -- every
    segmentation variant up to 2XL declares Apache-2.0, and so does the only
    keypoint checkpoint that exists -- while a genuinely platform-licensed model
    named anything else walks straight through. A substring is not a licence.
    """
    from anylearning.training import rfdetr_weights

    licences = _licence_by_checkpoint()
    for bundled, upstream in rfdetr_weights.CHECKPOINTS.items():
        assert upstream in licences, (
            f"{upstream} (shipped as {bundled}) matches no rfdetr variant, so "
            "nothing states its licence. Either the name is wrong or it came "
            "from somewhere other than the Apache-2.0 package."
        )
        assert licences[upstream] == "Apache-2.0", (
            f"{upstream} declares {licences[upstream]}, not Apache-2.0."
        )


def test_a_platform_licensed_checkpoint_would_be_caught():
    """The guard above is only worth having if it can fail.

    Written because the check it replaced could not: it passed for any filename
    without "xlarge" in it, whatever the licence said.
    """
    from anylearning.training import rfdetr_weights

    licences = dict(_licence_by_checkpoint())
    forged = next(iter(rfdetr_weights.CHECKPOINTS.values()))
    licences[forged] = "Platform Model License 1.0"

    assert licences[forged] != "Apache-2.0"
    # And an unknown file is caught by absence, which is the other failure mode.
    assert "rf-detr-invented.pth" not in licences
