"""What numeric precision this machine can actually train in.

Mixed precision runs the forward and backward passes in 16-bit and keeps the
weights and the optimiser step in 32-bit. On hardware that has tensor cores it
is close to free speed -- roughly 1.5-2x on the models here -- and it halves
activation memory, which is what decides whether a batch size fits.

The point of this module is that **asking for it is not the same as getting
it**, and the difference has to be decided per machine rather than written into
a config file that ships to everyone:

* fp16 autocast is a GPU feature. On a CPU it is not merely slower, it is a
  different code path that torch will happily run at a loss.
* bfloat16 has the same exponent range as fp32, so it needs no gradient
  scaling and cannot overflow to inf the way fp16 can -- but it needs Ampere
  or newer (compute capability 8.0+).
* Apple's MPS backend supports **both** 16-bit dtypes, which is worth stating
  because the opposite is widely repeated. In torch 2.11
  (`torch/amp/autocast_mode.py`) the supported set is the default
  ``[bfloat16, float16]`` for every backend, and the only MPS-specific rule is
  a guard that disables bfloat16 below macOS 14. The "BFloat16 is not supported
  on MPS" errors people hit come from Intel Macs with AMD GPUs, or from macOS
  13 and earlier, not from Apple silicon on a current OS.

  ``GradScaler`` works there too, which is also worth stating: measured on an
  M1, it scales by 65536, spots a forced overflow, skips the step and halves
  the scale. Loss scaling matters on Metal for the same reason it does on
  CUDA -- unscaled float16 zeroed 86.5% of the gradient elements against
  float32's 63.0%.

  Metal is nonetheless left in float32, because 16-bit there is *slower*
  rather than unsafe. See the MPS branch below for the numbers.

So a trainer asks for mixed precision, and this decides what that means here.
Every caller gets a :class:`Plan` that says what was chosen and, in a sentence
fit to show a user, why -- the training log is the only channel a training
process has, and "fp16 requested, running in fp32" with no reason reads as a
bug.

Two properties the callers rely on:

* ``Plan.autocast()`` is a context manager either way, so the loop is written
  once rather than branched.
* ``Plan.scaler()`` returns a ``GradScaler`` that is disabled unless fp16 is
  actually in use -- a disabled scaler passes gradients through untouched, so
  ``scaler.scale(loss).backward()`` stays correct in every mode, including
  bf16 and plain fp32.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch

from anylearning.training.device_utils import get_device

#: Overrides everything below, for the machine where the automatic answer is
#: wrong: "off", "fp16", "bf16", or "auto".
#:
#: There is no setting for this and it is not in the UI on purpose -- it exists
#: because mixed precision is a hardware-compatibility feature, and the failure
#: it guards against is one nobody can predict from here: a driver that
#: miscompiles a kernel, a card that reports bf16 and is slower for it, a model
#: that will not converge in 16 bits on someone else's data. Reaching for an
#: environment variable is a fair price for a lever that is needed once.
#:
#: In the environment rather than in a config file for the same reason
#: DEVICE_PREFERENCE_ENV is: training runs in its own process, and an
#: environment variable survives both fork and spawn.
OVERRIDE_ENV = "ANYLEARNING_MIXED_PRECISION"

#: What the gradient scaler starts at, against torch's default of 65536.
#:
#: The scaler finds its own working scale by overflowing and backing off, and
#: each back-off **discards that optimiser step**. Starting high costs a handful
#: of updates, which is nothing in a run of thousands and a great deal in the
#: runs this product actually does: measured on an RTX 2070, Mask R-CNN on 49
#: images skipped 4 of its first 7 updates and 5 of 42 overall, and finished at
#: mAP@0.5 0.13 against float32's 0.31. The same run at 1024 skipped **none**,
#: scored 0.31, and was still faster than float32 (132.8s against 142.8s).
#:
#: 1024 is not a magic number, it is "low enough that a short run does not
#: spend its first epoch calibrating". The scaler still grows it back on a long
#: run, so nothing is lost where the default would have been fine.
INITIAL_SCALE = 1024.0

#: The config key a trainer's YAML uses to ask for mixed precision.
#:
#: Named `fp16` because that is what the classification and segmentation
#: templates already shipped, and a config key is a compatibility surface: it
#: is read back from the text stored on models that are already trained.
#: Asking for `fp16` now means "use mixed precision", and this module decides
#: whether that ends up being bfloat16 or float16.
CONFIG_KEY = "fp16"


@dataclass(frozen=True)
class Plan:
    """What one training run will actually do about precision."""

    dtype: torch.dtype | None
    device_type: str
    reason: str

    @property
    def enabled(self) -> bool:
        return self.dtype is not None

    @property
    def label(self) -> str:
        if self.dtype is torch.bfloat16:
            return "bfloat16"
        if self.dtype is torch.float16:
            return "float16"
        return "float32"

    def describe(self) -> str:
        """One line for the training log, which is all a user ever sees."""
        if self.enabled:
            return f"Mixed precision: {self.label} ({self.reason})."
        return f"Mixed precision: off, training in float32 ({self.reason})."

    @property
    def tolerates_overflow(self) -> bool:
        """Whether a non-finite loss is expected rather than a broken run.

        float16 holds numbers up to 65504, and an activation or a loss term can
        pass that in a forward pass that is otherwise healthy. The gradient
        scaler's answer is to skip the step and try again with a smaller scale,
        so a loop that treats the first inf as a fatal error throws away a run
        that fp16 was designed to survive.

        False for bfloat16 and fp32, where the same value is a real divergence
        and should stop the run.
        """
        return self.dtype is torch.float16

    def autocast(self):
        """The autocast context for this plan, or a no-op when it is off."""
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def scaler(self, init_scale: float = INITIAL_SCALE):
        """A gradient scaler, enabled only where fp16 makes one necessary.

        bfloat16 keeps fp32's exponent range, so gradients do not underflow to
        zero and scaling them buys nothing. A disabled scaler is a passthrough,
        which lets a loop call ``scaler.scale(loss).backward()`` and
        ``scaler.step(optimizer)`` unconditionally.
        """
        return torch.amp.GradScaler(
            device=self.device_type,
            enabled=self.dtype is torch.float16,
            init_scale=init_scale,
        )


def _bf16_is_native() -> bool:
    """Whether this GPU runs bfloat16 in hardware rather than emulating it.

    **Not** ``torch.cuda.is_bf16_supported()``, which is the obvious call and
    the wrong one. In torch 2.11 it takes ``including_emulation=True`` by
    default, and with that it falls back to allocating a bfloat16 tensor and
    reporting whether that worked. An RTX 2070 can allocate one, so it answers
    True -- and then runs the arithmetic emulated:

        resnet18, 224px, batch 32    fp32 56.7 ms   bf16 245.9 (0.23x)
        resnet50, 512px, batch 8     fp32 243.8 ms  bf16 543.1 (0.45x)

    That is four times slower than the fp32 it replaced, on the users least
    able to afford it, and nothing raises. Compute capability 8.0 is the real
    line (Ampere), which is the same test torch itself applies before it
    considers emulation.
    """
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8
    except Exception:  # noqa: BLE001 - a driver that cannot answer is a "no"
        return False


def _cuda_plan() -> Plan:
    prefer_bf16 = _bf16_is_native()

    if prefer_bf16:
        return Plan(
            dtype=torch.bfloat16,
            device_type="cuda",
            reason="this GPU supports bfloat16, which needs no loss scaling",
        )
    return Plan(
        dtype=torch.float16,
        device_type="cuda",
        reason="this GPU predates bfloat16, so float16 with loss scaling",
    )


def resolve(device=None, requested: bool = True) -> Plan:
    """Decide what precision to train in.

    Args:
        device: the device training will run on. Defaults to what
            :func:`~anylearning.training.device_utils.get_device` picks, which
            already honours the user's CPU/GPU choice for this run.
        requested: whether the model's config asked for mixed precision. A
            model that has to train in fp32 -- because it diverges otherwise,
            or exports badly -- says so by setting `fp16: false`.
    """
    device = torch.device(device) if device is not None else get_device()
    device_type = device.type

    override = os.environ.get(OVERRIDE_ENV, "").strip().lower()
    if override in {"off", "no", "0", "false", "fp32", "float32"}:
        return Plan(None, device_type, f"turned off by {OVERRIDE_ENV}")
    if override in {"fp16", "float16"}:
        return Plan(torch.float16, device_type, f"float16 asked for by {OVERRIDE_ENV}")
    if override in {"bf16", "bfloat16"}:
        return Plan(
            torch.bfloat16, device_type, f"bfloat16 asked for by {OVERRIDE_ENV}"
        )

    if not requested:
        return Plan(None, device_type, "not enabled for this model")

    if device_type == "cuda":
        return _cuda_plan()

    if device_type == "mps":
        # Measured on an M1 (macOS 15.6.1, torch 2.11), and the answer is "no"
        # for a different reason than anyone assumes. Autocast on Metal is
        # *correct*: fp16 and bf16 both run, the scaler works, and every op
        # these trainers use lands within the error the same dtype produces on
        # the CPU. It is simply slower, in all three types that train on Metal:
        #
        #   image classification    6.36 s/epoch fp32   6.76 fp16   7.98 bf16
        #   image segmentation      6.86                7.16        8.64
        #   object detection       12.09               13.36       13.33
        #
        # The kernels do gain -- fp16 matmul 1.26-1.37x, a 7x7 conv 1.53x --
        # but autocast's per-op casts cost more than that on a GPU with no
        # tensor cores, and bfloat16 is not native on M1 at all: a bf16 matmul
        # runs at 0.62x of fp32's, slower than the arithmetic it replaces.
        # Memory is no consolation either: live tensors fall 10-21% while
        # Metal's reservation for the classification run rose 2.19 to 3.10 GB,
        # and on unified memory the reservation is the number that matters.
        #
        # M3 and later added bfloat16 to the GPU. Re-measure there rather than
        # assuming this still holds; ANYLEARNING_MIXED_PRECISION=bf16 is the
        # lever, and both override paths are verified working on Metal.
        return Plan(
            None,
            device_type,
            "measured on Apple's GPU: 16-bit is slower there, not faster",
        )

    # CPU autocast is bfloat16 and is only a win on cores with AVX512-BF16 or
    # AMX. Everywhere else it adds conversions to a path that is already the
    # slow one, and the machines that train on the CPU here are the ones with
    # no GPU at all -- the last place to gamble throughput on a guess.
    return Plan(None, device_type, "mixed precision has no benefit on the CPU")


def from_config(config: dict, device=None, default: bool = True) -> Plan:
    """Resolve a plan from a trainer's config dict.

    Reads ``training.fp16``, missing being the default, so a config written
    before this existed keeps working.
    """
    training = config.get("training", {}) if isinstance(config, dict) else {}
    requested = bool(training.get(CONFIG_KEY, default))
    return resolve(device=device, requested=requested)
