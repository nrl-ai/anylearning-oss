"""What mixed precision resolves to, and what it refuses to resolve to.

The point of these is that the answer depends on the machine, and the machine
running the tests is not the machine a customer has. So every hardware fact is
patched rather than probed -- the tests are about the decision, not about this
GPU.
"""

import pytest
import torch

from anylearning.training import precision


class FakeProperties:
    def __init__(self, major):
        self.major = major


def plan_on(monkeypatch, device, *, bf16=False, requested=True, override=None):
    """Resolve a plan against a pretended GPU.

    The compute capability is what is faked, not
    `torch.cuda.is_bf16_supported()`. Patching that call was how this file used
    to work, and it mocked away the exact fact that was wrong: on an RTX 2070
    the real function answers True, because by default it falls back to
    allocating a bfloat16 tensor rather than asking whether the hardware has
    bfloat16 at all. A test that stubs it can never catch that.
    """
    monkeypatch.delenv(precision.OVERRIDE_ENV, raising=False)
    if override is not None:
        monkeypatch.setenv(precision.OVERRIDE_ENV, override)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _=0: FakeProperties(8 if bf16 else 7),
    )
    return precision.resolve(device=device, requested=requested)


def test_ampere_and_newer_get_bfloat16(monkeypatch):
    plan = plan_on(monkeypatch, "cuda", bf16=True)
    assert plan.enabled
    assert plan.dtype is torch.bfloat16
    assert plan.label == "bfloat16"


def test_older_cards_get_float16(monkeypatch):
    """Turing and older: bfloat16 there is emulated and 2-4x slower than fp32."""
    plan = plan_on(monkeypatch, "cuda", bf16=False)
    assert plan.dtype is torch.float16


def test_emulated_bfloat16_does_not_count_as_supported(monkeypatch):
    """The trap this whole branch exists for.

    `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and
    answers True on a card that can only allocate the dtype. Measured on an RTX
    2070, taking that answer costs 0.23x on resnet18. If someone reinstates
    that call, this fails.
    """
    monkeypatch.delenv(precision.OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _=0: FakeProperties(7)
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_: True)

    assert precision.resolve(device="cuda").dtype is torch.float16


def test_bfloat16_needs_no_gradient_scaler(monkeypatch):
    """A scaler with bf16 buys nothing, and the loop should not branch for it.

    Same reason as above for recording the request rather than constructing one.
    """
    captured = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", Recorder)
    plan_on(monkeypatch, "cuda", bf16=True).scaler()

    assert captured["enabled"] is False


def test_float16_gets_a_gradient_scaler(monkeypatch):
    """Asserted on the request, not on a constructed scaler.

    `GradScaler(device="cuda", enabled=True)` warns "CUDA is not available.
    Disabling." on a machine with no GPU and quietly sets enabled=False -- and
    warnings are errors here, so building one made this test impossible to pass
    on a CPU-only machine, which is a machine class this product ships to.
    Capturing the arguments keeps the test about the decision.
    """
    captured = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", Recorder)
    plan_on(monkeypatch, "cuda", bf16=False).scaler()

    assert captured["enabled"] is True
    assert captured["device"] == "cuda"
    assert captured["init_scale"] == precision.INITIAL_SCALE


def test_only_float16_tolerates_an_infinite_loss(monkeypatch):
    assert plan_on(monkeypatch, "cuda", bf16=False).tolerates_overflow is True
    assert plan_on(monkeypatch, "cuda", bf16=True).tolerates_overflow is False
    assert plan_on(monkeypatch, "cpu").tolerates_overflow is False


def test_the_cpu_stays_in_float32(monkeypatch):
    """Not an oversight: CPU autocast is a different path, not a faster one."""
    plan = plan_on(monkeypatch, "cpu")
    assert not plan.enabled
    assert "CPU" in plan.reason


def test_metal_stays_in_float32(monkeypatch):
    """Not because Metal cannot: because it is slower there, measured on an M1.

    The reason is worded for a user reading a training log, so it says "Apple's
    GPU" rather than "MPS".
    """
    plan = plan_on(monkeypatch, "mps")
    assert not plan.enabled
    assert "Apple" in plan.reason and "slower" in plan.reason


def test_a_model_can_refuse_mixed_precision(monkeypatch):
    plan = plan_on(monkeypatch, "cuda", bf16=True, requested=False)
    assert not plan.enabled
    assert plan.reason == "not enabled for this model"


def test_the_environment_can_force_it_off(monkeypatch):
    """The lever for a machine where the automatic answer is wrong."""
    plan = plan_on(monkeypatch, "cuda", bf16=True, override="off")
    assert not plan.enabled
    assert precision.OVERRIDE_ENV in plan.reason


def test_the_environment_can_force_float16_on_a_bfloat16_card(monkeypatch):
    plan = plan_on(monkeypatch, "cuda", bf16=True, override="fp16")
    assert plan.dtype is torch.float16


def test_the_environment_overrides_a_model_that_said_no(monkeypatch):
    """Forcing it on is how you find out whether a model's `fp16: false` still
    needs to be false, without editing a config that ships."""
    plan = plan_on(monkeypatch, "cuda", requested=False, override="bf16")
    assert plan.dtype is torch.bfloat16


def test_autocast_is_a_context_manager_either_way(monkeypatch):
    """So the training loops are written once rather than branched.

    The disabled plan is entered anywhere, because a no-op context needs no
    hardware. The enabled one is only entered where there is a GPU to enter it
    on: `torch.autocast("cuda", ...)` validates against the real device, so on a
    CPU-only machine -- a machine class this product ships to -- entering it
    would fail for reasons that have nothing to do with the code under test.
    """
    with plan_on(monkeypatch, "cpu").autocast():
        pass

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to enter a real autocast on")

    # One plan at a time: entering a bf16 autocast asks the *real*
    # `torch.cuda.is_bf16_supported()` again, so a later patch of it would
    # invalidate an earlier plan's context.
    with plan_on(monkeypatch, "cuda", bf16=torch.cuda.is_bf16_supported()).autocast():
        pass


def test_from_config_defaults_to_on_for_a_config_written_before_this(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.delenv(precision.OVERRIDE_ENV, raising=False)
    assert precision.from_config({"training": {}}, device="cuda").enabled
    assert precision.from_config({}, device="cuda").enabled


def test_from_config_honours_an_explicit_no(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.delenv(precision.OVERRIDE_ENV, raising=False)
    plan = precision.from_config({"training": {"fp16": False}}, device="cuda")
    assert not plan.enabled


def test_describe_says_what_happened_and_why(monkeypatch):
    """The training log is the only channel a training process has."""
    on = plan_on(monkeypatch, "cuda", bf16=True).describe()
    assert "bfloat16" in on and "no loss scaling" in on

    off = plan_on(monkeypatch, "cpu").describe()
    assert "off" in off and "float32" in off
