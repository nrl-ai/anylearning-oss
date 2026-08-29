"""Machine-level settings that outlive any one project.

These live on the backend rather than in the frontend's localStorage because the
training process reads them, and that runs in its own
``multiprocessing.Process`` with no access to the browser.

The setting that matters is how hard to work the machine. The shipped
classification config asked for ``num_workers: 0``, which decodes and resizes
every image on the training process's own thread. Measured on 1,072 images at
224px, one process per configuration so each pays CUDA start-up once:

| workers | GPU img/s | CPU img/s |
|---------|-----------|-----------|
| 0       | 37        | 17        |
| 2       | -         | 20.4      |
| 4       | 115       | 20.5      |
| 8       | 125       | 21.5      |

The two columns are why the automatic figure is device-aware rather than a
single number. With a GPU the loader is the bottleneck and workers are nearly
free (3.3x). On CPU they compete with training for the same cores: two workers
already take 17 to 20.4 img/s, and seven more buy only another 1.1 -- so the
CPU budget is deliberately a quarter of the cores rather than most of them.
"""

from __future__ import annotations

import json
import os
import threading

from loguru import logger

from anylearning import config

_LOCK = threading.Lock()

#: How hard to work the machine.
#:
#: - ``maximum``: finish sooner; use the cores. The default, because a user who
#:   pressed "train" is waiting for the result.
#: - ``balanced``: leave room to keep using the machine while a run is going.
#: - ``power_saving``: for laptops on battery, or when training in the
#:   background matters more than finishing quickly.
PERFORMANCE_MODES = ("maximum", "balanced", "power_saving")

DEFAULTS: dict = {
    "performance_mode": "maximum",
    # Advanced overrides. "auto" means "derive from performance_mode"; an
    # explicit value wins, for the machine where our heuristic is wrong.
    "training_num_workers": "auto",
    "training_pin_memory": "auto",
    "training_persistent_workers": "auto",
}


def _path() -> str:
    return os.path.join(config.DATA_ROOT, "settings.json")


def load() -> dict:
    """Stored settings merged over the defaults. Never raises."""
    values = dict(DEFAULTS)
    try:
        with open(_path()) as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            values.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 -- a corrupt file must not stop training
        logger.warning(f"Ignoring unreadable settings file: {exc}")
    if values.get("performance_mode") not in PERFORMANCE_MODES:
        values["performance_mode"] = DEFAULTS["performance_mode"]
    return values


def save(updates: dict) -> dict:
    """Merge `updates` into the stored settings and return the result."""
    with _LOCK:
        values = load()
        values.update({k: v for k, v in updates.items() if k in DEFAULTS})
        os.makedirs(config.DATA_ROOT, exist_ok=True)
        tmp = _path() + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(values, handle, indent=2)
        # Replace atomically: a half-written settings file read by a training
        # process would be worse than no file at all.
        os.replace(tmp, _path())
    return values


def _cgroup_cpu_limit() -> float | None:
    """CPU quota from cgroups, if this process is inside one.

    Neither os.cpu_count() nor affinity sees a cgroup quota, so a container
    limited to two cores still reports the host's total. Reading it wrong is
    worse than not reading it, so anything unexpected returns None.
    """
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as handle:
            quota, period = handle.read().split()
        if quota != "max":
            return int(quota) / int(period)
    except Exception:  # noqa: BLE001 -- absent, unreadable, or a shape we do not know
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as handle:
            quota = int(handle.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as handle:
            period = int(handle.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except Exception:  # noqa: BLE001
        pass
    return None


def _cpus() -> int:
    """Logical CPUs this process may actually use.

    os.cpu_count() reports the whole machine, which over-counts under taskset,
    inside a container, or on a shared box -- and spawning eight workers where
    two cores are permitted makes training slower, not faster.
    """
    counts = []

    # 3.13+; respects CPU affinity. Falls back to sched_getaffinity elsewhere.
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        counts.append(process_count() or 0)
    elif hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))

    counts.append(os.cpu_count() or 1)

    try:
        limit = _cgroup_cpu_limit()
    except Exception:  # noqa: BLE001 -- detection must never stop a training run
        limit = None
    if limit is not None:
        counts.append(int(limit) or 1)

    usable = [c for c in counts if c and c > 0]
    return max(1, min(usable)) if usable else 1


def _physical_cores() -> int:
    """Physical cores, falling back to logical when we cannot tell.

    Hyperthreads do not help work that is already saturating a core's execution
    units, and this machine reports 16 logical against 8 physical -- budgeting
    against the logical figure would leave the training process nothing.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return min(physical, _cpus())
    except Exception:  # noqa: BLE001 -- psutil absent or unable to tell
        pass
    # Assume SMT rather than assume its absence: halving is the safe direction.
    return max(1, _cpus() // 2)


def _memory_worker_cap() -> int:
    """Cap workers by free memory. Each holds its own prefetched batches.

    Returns a large number when memory cannot be determined, so this only ever
    lowers the figure it is combined with.
    """
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / (2**30)
        # Roughly a gigabyte of headroom per worker: decoded images, the batch
        # being assembled, and the worker's own copy of the interpreter.
        return max(0, int(available_gb))
    except Exception:  # noqa: BLE001
        return 1 << 30


#: Never ask for more than this many workers however large the machine. Past it
#: the measured gain has flattened while memory use keeps climbing.
MAX_WORKERS = 8


def auto_num_workers(on_gpu: bool, mode: str | None = None) -> int:
    """Loader workers for this machine, this device, this mode.

    Budgeted against *physical* cores, not logical ones: hyperthreads do not
    help work that already saturates a core, and on a 16-thread/8-core box
    budgeting against 16 would hand every real core to the loader and leave the
    training process nothing.

    Whatever the mode asks for is then clipped by what the machine can actually
    provide -- permitted CPUs, physical cores and free memory -- so a container
    with two cores, or a laptop with little memory left, does not get a figure
    picked for a workstation.
    """
    mode = mode or load()["performance_mode"]

    cores = _physical_cores()

    if mode == "power_saving":
        # Enough to overlap decode with compute, not enough to spin up the fans.
        wanted = 1 if cores > 2 else 0
    elif mode == "balanced":
        wanted = cores // 2 if on_gpu else cores // 4
    elif on_gpu:
        # The loader is the bottleneck when a GPU is doing the arithmetic, so
        # give it most of the cores but never the last one.
        wanted = cores - 1
    else:
        # On CPU the workers compete with training for the same cores; measured
        # gain is 1.3x rather than 3.3x, so take a quarter of them.
        wanted = cores // 4

    ceiling = min(
        MAX_WORKERS,
        max(0, _cpus() - 1),  # always leave a CPU for the training process
        _memory_worker_cap(),
    )
    return max(0, min(wanted, ceiling))


def resolve_num_workers(configured=None, on_gpu: bool = False) -> int:
    """Worker count to use, in precedence order.

    An explicit machine setting wins, then a positive value from the trainer's
    YAML, then the automatic figure. A configured 0 is treated as "unset": it is
    the old hard-coded default this setting exists to correct.
    """
    values = load()
    setting = values.get("training_num_workers", "auto")
    if isinstance(setting, int) and setting >= 0:
        return setting
    if isinstance(configured, int) and configured > 0:
        return configured
    return auto_num_workers(on_gpu, values["performance_mode"])


def resolve_pin_memory(device: str) -> bool:
    """Page-locked host memory: faster host-to-device copies, no effect on CPU.

    CUDA only, and Metal is excluded on purpose rather than by omission: on
    Apple silicon the GPU reads the same physical memory the CPU wrote, so there
    is no copy for pinning to accelerate -- torch warns and ignores it.
    """
    values = load()
    setting = values.get("training_pin_memory", "auto")
    if isinstance(setting, bool):
        return setting
    if values["performance_mode"] == "power_saving":
        # Pinned memory cannot be swapped out, which is the wrong trade when the
        # point is to stay out of the way.
        return False
    return str(getattr(device, "type", device)) == "cuda"


def resolve_persistent_workers(num_workers: int) -> bool:
    """Keep workers alive between epochs, so startup is paid once per run."""
    if num_workers <= 0:
        return False
    values = load()
    setting = values.get("training_persistent_workers", "auto")
    if isinstance(setting, bool):
        return setting
    return values["performance_mode"] != "power_saving"


def resolve_cudnn_benchmark() -> bool:
    """Let cudnn search for the fastest convolution for the input shape.

    Worth it when every image is resized to a fixed size, which they are here:
    the search runs once and is reused. Skipped in power saving, where the
    search itself is work the user did not ask for.
    """
    return load()["performance_mode"] != "power_saving"


def apply_torch_runtime(device) -> None:
    """Apply the mode's runtime knobs. Safe to call more than once."""
    import torch

    mode = load()["performance_mode"]

    # cudnn is CUDA's algorithm search. Metal has no equivalent knob, so an MPS
    # run takes the thread settings below and nothing else.
    if str(getattr(device, "type", device)) == "cuda" and resolve_cudnn_benchmark():
        torch.backends.cudnn.benchmark = True

    if mode == "power_saving":
        # torch defaults to one thread per core, which saturates the machine.
        torch.set_num_threads(max(1, _cpus() // 4))
    elif mode == "balanced":
        torch.set_num_threads(max(1, _cpus() // 2))


def describe() -> dict:
    """What the current settings resolve to, for the UI to show."""
    values = load()
    return {
        **values,
        "resolved": {
            "cpu_count": _cpus(),
            "physical_cores": _physical_cores(),
            "num_workers_gpu": resolve_num_workers(on_gpu=True),
            "num_workers_cpu": resolve_num_workers(on_gpu=False),
            "cudnn_benchmark": resolve_cudnn_benchmark(),
        },
    }
