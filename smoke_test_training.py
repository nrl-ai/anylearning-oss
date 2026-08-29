#!/usr/bin/env python3
"""Train every project type with a *packaged* build, on GPU and on CPU.

Importing torch is not the same as training with it. Training runs in its own
``multiprocessing.Process``, reports progress through the project database, and
ends by exporting ONNX -- and ``run_training_job`` only registers a model once
that export succeeds. Each of those can break in a frozen binary while the app
still starts cleanly:

- ``triton`` excluded by Nuitka's own torch config broke *every GPU run*, while
  CPU runs were fine.
- ``onnxscript._framework_apis`` missing let training finish and then threw the
  result away.

Neither is visible to an import check, so the assertion here is that a model
row appears -- that proves the whole chain survived freezing.

Usage:
    python smoke_test_training.py <binary> [--epochs 2] [--devices gpu,cpu]
                                           [--types "Image Classification,..."]

Projects are discovered from the running app rather than hard-coded, so this
works against whatever ``~/anylearning-data`` holds. A type with no labelled
data is skipped and reported as such, not silently passed.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile

from anylearning.selftest.driver import (
    api,
    item_count,
    pick_variant,
    start_server,
    stop,
    token_from_log,
    train,
    wait_for_port_release,
    wait_for_server,
)

# The driver lives in the package so that this script and the binary's own
# `--self-test` are one implementation. What is left here is the part that only
# makes sense from outside: pointing at a binary somebody built, and training
# whatever projects already exist in ~/anylearning-data.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--port", type=int, default=5803)
    parser.add_argument("--devices", default="gpu,cpu")
    parser.add_argument(
        "--types", default="", help="comma-separated; default is every type found"
    )
    parser.add_argument("--budget", type=int, default=1800, help="seconds per run")
    parser.add_argument(
        "--architectures",
        default="",
        help=(
            "comma-separated model architectures to prefer, e.g. "
            "'rfdetr,rfdetr-seg'. Detection and instance segmentation each offer "
            "variants from two different trainers, and the default -- a type's "
            "first variant -- only ever exercises one of them"
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"no binary at {args.binary}", file=sys.stderr)
        return 2

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    wanted = {t.strip() for t in args.types.split(",") if t.strip()}
    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]

    results: list[tuple[str, str, bool, str]] = []

    for index, device in enumerate(devices):
        # A port each: the previous phase's server may still be shutting down,
        # and the app quietly picks a random port when its own is busy.
        port = args.port + index
        base = f"http://127.0.0.1:{port}"
        print(f"\n=== {device.upper()} (port {port}) ===", flush=True)
        wait_for_port_release(port)
        # To a log, so the API token it mints can be read back. Without one,
        # every call here answered 401: the app's per-window bearer token is
        # part of what ships, and this script asks the same questions a browser
        # would. It used to relax the check instead, which tested a
        # configuration nobody runs.
        server_log = (
            pathlib.Path(tempfile.gettempdir()) / f"anylearning-smoke-{port}.log"
        )
        server_log.unlink(missing_ok=True)
        server = start_server(
            args.binary, port, force_cpu=(device == "cpu"), log_path=server_log
        )
        try:
            wait_for_server(base, server)
            token = token_from_log(server_log, server)
            if token is None:
                raise RuntimeError(
                    f"the server never printed an API token; see {server_log}"
                )

            health = api(base, "/api/health/imports") or {}
            if not health.get("ok", False):
                print(f"  imports broken: {health.get('broken')}", flush=True)

            projects = api(base, "/api/projects", token=token) or []
            variants = api(base, "/api/model-variants", token=token) or {}

            # One project per type: the first that actually has data.
            by_type: dict[str, dict] = {}
            for project in projects:
                ptype = project.get("type")
                if not ptype or ptype in by_type:
                    continue
                if wanted and ptype not in wanted:
                    continue
                if item_count(base, project["id"], token=token) > 0:
                    by_type[ptype] = project

            if not by_type:
                print("  no projects with data -- nothing to train", flush=True)

            for ptype, project in sorted(by_type.items()):
                variant = pick_variant(variants, ptype, architectures)
                if not variant:
                    results.append(
                        (ptype, device, False, "no model variant configured")
                    )
                    print(f"  {ptype}: no variant", flush=True)
                    continue
                print(f"  {ptype} ({project['name']}) ...", end=" ", flush=True)
                ok, detail = train(
                    base,
                    project["id"],
                    variant,
                    args.epochs,
                    args.budget,
                    token=token,
                    device=device,
                )
                results.append((ptype, device, ok, detail))
                print("ok" if ok else f"FAIL {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- report, do not abort the matrix
            results.append(("(server)", device, False, str(exc)))
            print(f"  server error: {exc}", file=sys.stderr, flush=True)
        finally:
            stop(server, port)

    print("\n" + "=" * 72)
    width = max((len(r[0]) for r in results), default=20)
    for ptype, device, ok, detail in results:
        print(f"{ptype:<{width}}  {device:<4}  {'PASS' if ok else 'FAIL'}  {detail}")

    failed = [r for r in results if not r[2]]
    if failed:
        print(f"\nFAIL: {len(failed)} of {len(results)} runs failed", file=sys.stderr)
        return 1
    if not results:
        print("\nFAIL: nothing was trained", file=sys.stderr)
        return 1
    print(f"\nPASS: {len(results)} runs, every project type on {', '.join(devices)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
