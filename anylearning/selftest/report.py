"""What the machine is, and what happened on it.

The environment block is not decoration. Cross-platform failures are often
identifiable here before anyone reads a stack trace, such as a torch build
without CUDA on a machine with a GPU.
"""

from __future__ import annotations

import platform
import sys


def _torch_facts() -> list[tuple[str, str]]:
    """Import torch late: it costs seconds, and only this needs it."""
    try:
        import torch
    except Exception as error:  # a build that cannot import torch cannot train
        return [("torch", f"unavailable: {type(error).__name__}: {error}")]

    facts = [("torch", torch.__version__)]
    if torch.cuda.is_available():
        facts.append(
            ("gpu", f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        )
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        facts.append(("gpu", f"Apple Metal (MPS), {platform.processor()}"))
    else:
        facts.append(("gpu", "none"))
    return facts


def has_gpu() -> bool:
    """Whether the GPU half of the matrix has anything to run on.

    Any accelerator, not CUDA specifically -- on a Mac the GPU column is Metal,
    and skipping it there is how the platform went untested.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True
        return bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def environment(binary: str, data_root) -> None:
    from anylearning.app_info import __version__

    facts = [
        ("anylearning", __version__),
        ("platform", f"{platform.platform()} ({platform.machine()})"),
        ("python", sys.version.split()[0]),
        *_torch_facts(),
        ("binary", binary),
        ("data", f"{data_root} (temporary)"),
    ]

    width = max(len(name) for name, _ in facts)
    print()
    for name, value in facts:
        print(f"  {name:<{width}}  {value}")
    # Flushed, and so is every line below. Redirected to a file, Python holds
    # output in a block buffer, so a matrix that takes half an hour shows
    # nothing at all until it ends -- and progress is most wanted exactly when
    # a run is slow enough to worry about.
    print(flush=True)


def progress(project_type: str, device: str, passed, detail: str) -> None:
    """One line as each run lands, so a long matrix is watchable."""
    mark = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    print(f"  {mark}  {project_type} on {device}: {detail}", flush=True)


def results(rows, elapsed: float) -> int:
    """Print the table and return how many runs failed."""
    if not rows:
        print("  nothing ran")
        return 1

    type_width = max(len(str(row[0])) for row in rows)
    print(f"  {'':4} {'project type':<{type_width}}  {'dev':<4} detail")
    print(f"  {'-' * 4} {'-' * type_width}  {'-' * 4} {'-' * 40}")

    failures = 0
    for project_type, device, passed, detail in rows:
        if passed is None:
            mark = "SKIP"
        elif passed:
            mark = "PASS"
        else:
            mark = "FAIL"
            failures += 1
        print(f"  {mark:4} {project_type:<{type_width}}  {device:<4} {detail}")

    print()
    print(
        f"  {len(rows)} runs in {elapsed / 60:.1f} min, {failures} failed", flush=True
    )
    if failures:
        print(
            "  A failure here is this build on this machine, not the code in general."
        )
    return failures
