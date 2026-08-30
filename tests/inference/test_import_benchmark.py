import json
import subprocess
import sys

from scripts.benchmark_inference_import import percentile


def test_percentile_uses_nearest_rank():
    assert percentile(list(range(1, 11)), 0.95) == 10


def test_import_benchmark_emits_machine_readable_bounded_results():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_inference_import.py",
            "--module",
            "anylearning.inference",
            "--runs",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    result = payload["results"]["anylearning.inference"]

    assert payload["schema_version"] == 1
    assert payload["runs"] == 2
    assert len(result["samples"]) == 2
    assert result["median_duration_ms"] > 0
    assert result["median_peak_rss_bytes"] > 0


def test_import_benchmark_rejects_code_as_a_module_name():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_inference_import.py",
            "--module",
            "os;raise SystemExit(0)",
            "--runs",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Invalid module name" in completed.stderr
