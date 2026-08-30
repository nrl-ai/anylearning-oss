#!/usr/bin/env python3
"""Measure cold Python import latency and peak RSS without importing the target."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import psutil

MODULE_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


@dataclass(frozen=True, slots=True)
class Sample:
    duration_ms: float
    peak_rss_bytes: int


def percentile(values: list[float | int], percentage: float) -> float | int:
    """Return a nearest-rank percentile suitable for a small benchmark sample."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentage) - 1))
    return ordered[index]


def measure_import(module: str) -> Sample:
    started = time.perf_counter()
    child = subprocess.Popen(
        [sys.executable, "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    process = psutil.Process(child.pid)
    peak_rss = 0
    while child.poll() is None:
        try:
            peak_rss = max(peak_rss, process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            break
        time.sleep(0.001)
    _, stderr = child.communicate()
    duration_ms = (time.perf_counter() - started) * 1000
    if child.returncode:
        detail = stderr.strip() or f"exit status {child.returncode}"
        raise RuntimeError(f"Could not import {module}: {detail}")
    return Sample(duration_ms=duration_ms, peak_rss_bytes=peak_rss)


def summarize(samples: list[Sample]) -> dict[str, float | int | list[dict]]:
    durations = [sample.duration_ms for sample in samples]
    memory = [sample.peak_rss_bytes for sample in samples]
    return {
        "median_duration_ms": statistics.median(durations),
        "p95_duration_ms": percentile(durations, 0.95),
        "median_peak_rss_bytes": statistics.median(memory),
        "max_peak_rss_bytes": max(memory),
        "samples": [asdict(sample) for sample in samples],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Module to measure; repeat for multiple modules",
    )
    parser.add_argument("--runs", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modules = args.modules or ["anylearning", "anylearning.inference"]
    if not 1 <= args.runs <= 1_000:
        raise SystemExit("--runs must be between 1 and 1000")
    invalid = [module for module in modules if not MODULE_NAME.fullmatch(module)]
    if invalid:
        raise SystemExit(f"Invalid module name: {invalid[0]}")

    results = {
        module: summarize([measure_import(module) for _ in range(args.runs)])
        for module in modules
    }
    payload = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runs": args.runs,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
