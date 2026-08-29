"""`AnyLearning --self-test`: prove this build trains, on this machine.

The failures worth catching only exist in a packaged build on a particular
machine -- a module Nuitka dropped, a torch wheel with no CUDA in it -- and
until now finding them meant standing up a development environment on that
machine first: Python, a virtualenv, four gigabytes of wheels, MSVC for
detectron2, and a copy of somebody's projects. Hours before the first run.

So the driver moves inside the binary. It launches this same executable as a
server, builds a small labelled project of each type out of shapes it draws
itself, trains each one, and asserts that a model row appears -- which happens
only after ONNX export succeeds, so it stands for the whole chain surviving the
freeze. Then it prints what the machine is, because that line is where the
answers usually are.

Nothing it does touches the user's data: ANYLEARNING_DATA_ROOT points the whole
store at a temporary directory for the duration.

    AnyLearning --self-test
    AnyLearning --self-test --types "Object Detection" --devices cpu
    AnyLearning --self-test --architectures rfdetr,rfdetr-seg

The third line is worth knowing about. Detection and instance segmentation each
offer variants from two different trainers, and a type's *first* variant is what
this runs -- so without naming an architecture, half of two columns is never
touched.

What it cannot cover: the installer, the first run, and the window frame's
gestures. A binary that will not start cannot report that it did not start --
`smoke_test_build.sh` and `smoke_test_window_chrome.py` are for those.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import time

from anylearning.selftest import projects, report
from anylearning.selftest.driver import (
    api,
    pick_variant,
    start_server,
    stop,
    token_from_log,
    train,
    wait_for_server,
)

# Handpose is missing here and that is deliberate; see projects.KINDS.
DEFAULT_TYPES = tuple(projects.KINDS)


def _selftest_batch_size(project_type: str, per_subset: int) -> int:
    """A small, non-empty batch that is realistic for the tested trainer."""
    maximum = 2 if project_type == "Keypoint Detection" else 8
    return max(1, min(maximum, per_subset))


def launcher() -> list[str]:
    """How to start this application again.

    Packaged, that is the executable itself. From a checkout, sys.executable is
    the interpreter and it needs to be told what to run.
    """
    executable = pathlib.Path(sys.executable)
    if executable.stem.lower().startswith("python"):
        from anylearning import app

        return [str(executable), str(pathlib.Path(app.__file__).resolve())]
    return [str(executable)]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _run_device(
    binary,
    device: str,
    types,
    epochs: int,
    budget: int,
    per_subset: int,
    log_dir,
    development: bool = False,
    architectures=(),
) -> list[tuple[str, str, bool | None, str]]:
    """Build and train every requested type once, on one device."""
    results = []
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = log_dir / f"server-{device}.log"
    # The GPU stays visible even for the CPU row. Hiding it would test what a
    # machine without a GPU does, which is worth less than testing what the app
    # ships: the user asking for the CPU on a machine that has both. `train()`
    # then checks the run really happened there -- a broken picker otherwise
    # looks exactly like a working one, since the run finishes either way.
    server = start_server(binary, port, force_cpu=False, log_path=server_log)

    try:
        wait_for_server(base, server)

        # Production mode by default: the server mints a token and every route
        # but health is behind it, so the run exercises the same authentication
        # the shipped app uses rather than a relaxed one.
        token = None if development else token_from_log(server_log, server)
        if token is None and not development:
            results.append(
                (
                    "(server)",
                    device,
                    False,
                    "the server never printed an API token, so nothing could be called",
                )
            )
            return results

        variants = api(base, "/api/model-variants", token=token) or {}

        for project_type in types:
            try:
                variant = pick_variant(variants, project_type, architectures)
                if not variant:
                    results.append(
                        (project_type, device, None, "no model variant configured")
                    )
                    continue
                project = projects.build(
                    base, project_type, per_subset, 64, seed=7, token=token
                )
                passed, detail = train(
                    base,
                    project["id"],
                    variant,
                    epochs,
                    budget,
                    # Never larger than the training subset. The loader drops
                    # the last partial batch, so a batch bigger than the data
                    # runs zero iterations and writes no checkpoint -- which
                    # the app reports as "No model found in training output",
                    # indistinguishable from a broken trainer.
                    batch_size=_selftest_batch_size(project_type, per_subset),
                    token=token,
                    device=device,
                )
                outcome = (
                    project_type,
                    device,
                    passed,
                    f"{detail} ({project['labelled']} labelled)",
                )
                results.append(outcome)
                report.progress(*outcome)
            except Exception as error:  # one type failing must not end the run
                outcome = (
                    project_type,
                    device,
                    False,
                    f"{type(error).__name__}: {error}",
                )
                results.append(outcome)
                report.progress(*outcome)
    except Exception as error:
        results.append(("(server)", device, False, f"{type(error).__name__}: {error}"))
    finally:
        stop(server, port)

    return results


def run(
    types=DEFAULT_TYPES,
    devices=("gpu", "cpu"),
    epochs: int = 2,
    budget: int = 1800,
    per_subset: int = 6,
    keep: bool = False,
    workers: int | None = None,
    development: bool = False,
    architectures=(),
) -> int:
    binary = launcher()
    root = pathlib.Path(tempfile.mkdtemp(prefix="anylearning-selftest-"))

    # Inherited by the server this launches, and by the training process it
    # spawns in turn -- which is the point: nothing writes to the real store.
    os.environ["ANYLEARNING_DATA_ROOT"] = str(root)
    if development:
        os.environ["ANYLEARNING_DEVELOPMENT"] = "TRUE"
    else:
        os.environ.pop("ANYLEARNING_DEVELOPMENT", None)
    if workers is not None:
        # Written into the temporary store, so it applies to this run and
        # nothing else. Loader workers are the first thing to vary when a run
        # stalls rather than fails: they are the one setting that differs
        # between a GPU run and a CPU one, and on a spawn platform each worker
        # re-imports the whole application before it can hand over a batch.
        (root / "settings.json").write_text(
            json.dumps({"training_num_workers": workers}, indent=2)
        )

    report.environment(" ".join(binary), root)

    started = time.time()
    results = []
    for device in devices:
        if device == "gpu" and not report.has_gpu():
            results.append(("(all types)", device, None, "no GPU on this machine"))
            continue
        results.extend(
            _run_device(
                binary,
                device,
                types,
                epochs,
                budget,
                per_subset,
                root,
                development,
                architectures,
            )
        )

    failures = report.results(results, time.time() - started)

    if keep:
        print(f"data left in {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="AnyLearning --self-test", description=__doc__
    )
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--devices", default="gpu,cpu")
    parser.add_argument(
        "--architectures",
        default="",
        help=(
            "comma-separated model architectures to prefer, e.g. "
            "'rfdetr,rfdetr-seg'. Two project types offer variants from two "
            "different trainers, and the default -- the first variant -- would "
            "only ever exercise one of them"
        ),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--images", type=int, default=6, help="images per subset")
    parser.add_argument("--budget", type=int, default=1800, help="seconds per run")
    parser.add_argument("--keep", action="store_true", help="keep the temporary data")
    parser.add_argument(
        "--development",
        action="store_true",
        help="relax the token check (only for a build that cannot mint one)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="force the dataloader worker count instead of deriving it",
    )
    args = parser.parse_args(argv)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = [t for t in types if t not in projects.KINDS]
    if unknown:
        parser.error(f"cannot generate data for: {', '.join(unknown)}")

    return run(
        types=types,
        devices=[d.strip() for d in args.devices.split(",") if d.strip()],
        epochs=args.epochs,
        budget=args.budget,
        per_subset=args.images,
        keep=args.keep,
        workers=args.workers,
        development=args.development,
        architectures=[a.strip() for a in args.architectures.split(",") if a.strip()],
    )
