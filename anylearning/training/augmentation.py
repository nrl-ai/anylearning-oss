"""What each trainer can be asked to do to its training images.

Augmentation is not one feature with one set of switches. NanoDet's pipeline
takes rotation, shear and per-channel colour ranges; torchvision classification
gets a flip and a jitter; detectron2 decides most of it inside its own dataset
mapper and only really exposes flipping. Offering the union of those everywhere
would mean silently ignoring most of what the user set, which is worse than not
offering it -- someone turns off flipping to protect a LEFT/RIGHT distinction,
sees no error, and the run flips the images anyway.

So each trainer declares its own options, and the UI renders exactly those.
Every option carries the value it takes when nobody chooses -- which is what
that trainer already did, so a project trained before any of this existed keeps
training the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Option:
    """One augmentation control, described well enough for a UI to draw it."""

    key: str
    label: str
    #: "bool" renders a switch; "int" and "float" render a numeric input with
    #: the range below.
    type: str
    #: What this trainer does when the user chooses nothing. Never None: an
    #: option with no honest default is one this trainer should not declare.
    default: object
    help: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None

    def coerce(self, value):
        """The value as this option's type, clamped, or None if unusable."""
        if value is None:
            return None
        if self.type == "bool":
            return bool(value)
        try:
            number = int(value) if self.type == "int" else float(value)
        except (TypeError, ValueError):
            return None
        if self.minimum is not None:
            number = max(self.minimum, number)
        if self.maximum is not None:
            number = min(self.maximum, number)
        return int(number) if self.type == "int" else float(number)

    def as_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


# The options shared by the trainers built on torchvision transforms. Kept here
# rather than duplicated: the wording a user reads should not depend on which
# model they picked.
HORIZONTAL_FLIP = Option(
    key="horizontal_flip",
    label="Horizontal flip",
    type="bool",
    default=True,
    help="Mirrors images left to right. Turn this off when left and right mean "
    "different things -- a LEFT and a RIGHT traffic sign are the same image "
    "flipped.",
)

VERTICAL_FLIP = Option(
    key="vertical_flip",
    label="Vertical flip",
    type="bool",
    default=False,
    help="Mirrors images top to bottom. Useful for overhead or microscope "
    "images; wrong for anything with a ground and a sky.",
)

ROTATION = Option(
    key="rotation_degrees",
    label="Rotation",
    type="int",
    default=0,
    minimum=0,
    maximum=180,
    step=5,
    help="Rotates by up to this many degrees either way.",
)

COLOR_JITTER = Option(
    key="color_jitter",
    label="Colour jitter",
    type="bool",
    default=True,
    help="Varies brightness, contrast and saturation, so the model does not "
    "depend on one camera's colours.",
)


def resolve(options, requested, log=None) -> dict:
    """Merge what the user asked for into what this trainer supports.

    Unknown keys and unusable values are dropped with a line in the training
    log rather than raising: the request comes from a dialog that has already
    been dismissed, and failing here would end the run after the whole dataset
    had been exported.
    """
    requested = requested or {}
    by_key = {option.key: option for option in options}
    resolved = {option.key: option.default for option in options}

    for key, value in requested.items():
        option = by_key.get(key)
        if option is None:
            if log and value is not None:
                log.write(
                    f"This model has no '{key}' augmentation; it was not applied."
                )
            continue
        if value is None:
            continue
        coerced = option.coerce(value)
        if coerced is None:
            if log:
                log.write(f"Ignoring {key}={value!r}: not a {option.type}.")
            continue
        if log and coerced != value:
            log.write(f"{option.label} adjusted from {value} to {coerced}.")
        resolved[key] = coerced

    return resolved
