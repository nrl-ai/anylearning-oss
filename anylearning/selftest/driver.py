"""Driving a packaged AnyLearning over its own HTTP API.

Extracted from `smoke_test_training.py` so that the built-in `--self-test` and
the external script are the same code rather than two implementations that
agree today. Nothing here imports anything beyond the standard library: it runs
inside the frozen binary, and it also runs from a checkout against a binary
that was built somewhere else.

The assertion every caller ends up making is the same one -- a model row
appears -- because `run_training_job` registers a model only after ONNX export
succeeds, so a registered model proves the whole chain survived freezing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

TERMINAL = {"finished", "error", "terminated"}

# Colorama leaves an ANSI reset on redirected Loguru output on Windows. It is
# harmless in a terminal, but it used to become four extra characters on the
# API token parsed from the server log, making every packaged self-test request
# answer 401 on Windows while the server itself was healthy.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def api(
    base: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 30,
    token: str | None = None,
):
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if token:
        # Production mode: every route but /api/health is behind the per-window
        # token, and testing without it tests a configuration nobody ships.
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def wait_for_server(base: str, process: subprocess.Popen, seconds: int = 180) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with {process.returncode}")
        try:
            api(base, "/openapi.json", timeout=5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server never answered on {base}")


#: The cache variables `anylearning/weights.py` sets, which must never be
#: inherited by a packaged binary under test.
#:
#: Importing anything from `anylearning` runs `use_bundled()`, which points
#: these at the *checkout's* weights directory -- and `use_bundled()` leaves an
#: existing value alone, so the binary then reads the developer's weights
#: instead of its own. Everything downstream still passes, while testing a
#: configuration nobody ships.
#:
#: That is not hypothetical. It hid a real bug on macOS: instance segmentation
#: could not train from a read-only installation, and the harness could not see
#: it, because the installed app was quietly reading a writable directory in the
#: source tree. Only `<binary> --self-test` from a clean shell caught it.
# Deliberately a copy of `weights.CACHE_VARIABLES` rather than an import of it:
# importing `anylearning.weights` runs the package __init__, whose whole purpose
# is to *set* these variables in the importing process, and this module is
# stdlib-only by design. `test_weights_cache_writability.py` asserts the two
# lists are equal, so a variable added on one side fails a test rather than
# silently reopening the gap -- which is how RF-DETR's RF_HOME was caught.
INHERITED_CACHE_VARIABLES = (
    "TORCH_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "FVCORE_CACHE",
    "RF_HOME",
)


def scrub_cache_variables(env: dict) -> list:
    """Remove the inherited weight-cache paths, returning the ones dropped."""
    dropped = [name for name in INHERITED_CACHE_VARIABLES if name in env]
    for name in dropped:
        del env[name]
    return dropped


def start_server(binary, port: int, force_cpu: bool, log_path=None) -> subprocess.Popen:
    env = dict(os.environ)
    scrub_cache_variables(env)
    if force_cpu:
        # The trainers choose their device from torch.cuda, so this is what
        # exercises the path a user without a GPU takes.
        env["CUDA_VISIBLE_DEVICES"] = ""
    # `binary` is the packaged executable, or [python, app.py] from a checkout.
    command = list(binary) if isinstance(binary, (list, tuple)) else [binary]
    # To a file, never to a pipe. A pipe nobody reads holds 64KB and then
    # blocks whoever is writing -- and the app writes a lot: nanodet's ONNX
    # export prints the entire graph, and on a spawn platform every dataloader
    # worker re-imports the application and prints its warnings again. Training
    # then stops dead, in a way that reads as "the run hung" rather than "the
    # harness stopped listening".
    output = open(log_path, "ab", buffering=0) if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        [*command, "--server", "--port", str(port)],
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
    )


def token_from_log(log_path, process, seconds: int = 180) -> str | None:
    """The API token `--server` mints and prints, read back from its log.

    Without this the only way to talk to a packaged server is to relax the
    token check, and a run that relaxes it is not testing what ships.
    """
    marker = "API token for this server: "
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return None
        try:
            with open(log_path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if marker in line:
                        token = line.split(marker, 1)[1].strip()
                        return ANSI_ESCAPE.sub("", token).strip()
        except OSError:
            pass
        time.sleep(1)
    return None


def stop(process: subprocess.Popen, port: int | None = None) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    if port is not None:
        wait_for_port_release(port)


def wait_for_port_release(port: int, seconds: int = 60) -> None:
    """Block until nothing is listening on `port`.

    The app silently falls back to a random port when its own is taken, so a
    lingering server from the previous phase does not fail loudly -- the next
    phase just never finds anything at the address it is polling. That is
    exactly how the CPU half of this matrix reported "server never answered".
    """
    import socket

    deadline = time.time() + seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(1)


def item_count(base: str, project_id: int, token: str | None = None) -> int:
    total = 0
    for subset in (0, 1, 2):
        try:
            page = api(
                base,
                f"/api/projects/{project_id}/data_items?subset={subset}&limit=1",
                token=token,
            )
            total += (page or {}).get("total_count", 0)
        except Exception:
            pass
    return total


def pick_variant(variants: dict, project_type: str, architectures=()) -> dict | None:
    """Cheapest variant for the type -- these runs are about plumbing, not accuracy.

    Cheapest means first, which is how `config.MODEL_VARIANTS` is ordered. Two
    project types now offer variants from two entirely separate trainers --
    detection is NanoDet or RF-DETR, instance segmentation is Mask R-CNN or
    RF-DETR-Seg -- and taking the first would test one of them forever. Naming
    architectures picks the first variant belonging to one of them instead, so
    the other half of those columns can be driven on a customer's machine
    without a dataset.

    An architecture this project type does not offer is not an error: the caller
    passes one list for every type it runs, and "rfdetr" means nothing to
    classification. Falls back to the first variant, as before.
    """
    options = variants.get(project_type) or []
    if architectures:
        wanted = set(architectures)
        for option in options:
            if option.get("model_architecture") in wanted:
                return option
    return options[0] if options else None


def log_tail(
    base: str, detail_path: str, lines: int = 2, token: str | None = None
) -> str:
    """The last thing a training session said, for a failure message."""
    try:
        detail = api(base, detail_path, timeout=120, token=token) or {}
    except Exception:  # noqa: BLE001
        return "(the server stopped answering)"
    logs = (detail.get("training_logs") or "").strip().splitlines()
    return " | ".join(logs[-lines:])[:300] if logs else "(no logs)"


def train(
    base: str,
    project_id: int,
    variant: dict,
    epochs: int,
    budget: int,
    batch_size: int = 8,
    token: str | None = None,
    device: str = "auto",
    image_size: int | None = None,
) -> tuple[bool, str]:
    params = {
        "model_architecture": variant["model_architecture"],
        "model_size": variant["model_size"],
        "model_variant": variant.get("name", "variant"),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": 0.001,
        "pretrained_model": "default",
        "device": device,
        "image_size": image_size,
    }
    started = api(
        base, f"/api/projects/{project_id}/training_sessions", params, token=token
    )
    session_id = (started or {}).get("session_id")
    if not session_id:
        return False, "training did not start"

    # Poll tolerantly. The server shares the machine with the training process,
    # and while NanoDet holds the GPU and most of the cores a request can take
    # longer than the default 30s -- which is the run being slow, not failing.
    # A single timeout used to raise out of the whole device phase, abandoning
    # a run that was still going and reporting it as "server error: timed out".
    detail_path = f"/api/projects/{project_id}/training_sessions/{session_id}"
    deadline = time.time() + budget
    status = "unknown"
    while time.time() < deadline:
        try:
            detail = api(base, detail_path, timeout=120, token=token)
        except Exception:  # noqa: BLE001 -- a slow answer is not a failed run
            # No separate error budget: the deadline below is the only timeout.
            # An earlier version gave up after ten consecutive failures, about
            # fifty seconds, and NanoDet's finalise step leaves the server
            # unresponsive for longer than that -- so it reported a failure for
            # a run that went on to finish and register a model.
            time.sleep(5)
            continue
        status = (detail or {}).get("status", "unknown")
        if status in TERMINAL:
            break
        time.sleep(5)
    else:
        # Say what it was doing, not just that it was slow: a run that stalls
        # says so in its own log, and without this the report gives nobody
        # anything to act on.
        tail = log_tail(base, detail_path, token=token)
        return False, f"still '{status}' after {budget}s: {tail}"

    try:
        detail = api(base, detail_path, timeout=120, token=token) or {}
    except Exception:  # noqa: BLE001
        return False, f"ended as '{status}' but the server stopped answering"
    if status != "finished":
        logs = (detail.get("training_logs") or "").strip().splitlines()
        tail = " | ".join(logs[-2:])[:300] if logs else "(no logs)"
        # `run_training_job` writes "Error during training: <exception>" from
        # its own except block, so a failed run with no such line did not raise
        # -- the process died before it could, which is a different problem and
        # points somewhere else entirely (killed, or a native crash). Worth
        # saying, because the two-line tail then shows whatever the run had got
        # to and reads like a run that stopped for no reason. Seen once on a
        # loaded machine and undiagnosable afterwards, which is why this is here.
        if not any("Error during training" in line for line in logs):
            return False, (
                f"ended as '{status}' with no error written -- the training "
                f"process died before it could report one (killed, or a native "
                f"crash). Last it said: {tail}"
            )
        return False, f"ended as '{status}': {tail}"

    # The real assertion: registration happens only after ONNX export succeeds.
    model_id = (detail.get("model") or {}).get("id")
    if not model_id:
        return False, "finished but registered no model -- ONNX export failed"

    # And the run has to have happened on the hardware that was asked for.
    # Without this, a broken device picker looks exactly like a working one:
    # the run finishes either way, just on the wrong processor.
    used = _device_from_logs(detail.get("training_logs"))
    if device == "cpu" and used and not _ran_on_cpu(used):
        return False, f"asked for the CPU and trained on {used}"
    if device == "gpu" and used and _ran_on_cpu(used):
        # A type that cannot use this accelerator is downgraded on purpose, and
        # says so in the same line. That is a pass, not a wrong device.
        if " -- " not in used:
            return (
                False,
                f"asked for the GPU and trained on {used} with no reason given",
            )
    return True, f"model {model_id}" + (f", on {used}" if used else "")


def _ran_on_cpu(used: str) -> bool:
    """Whether the device line says the run happened on the CPU.

    Anchored to the start of the line, because the rest of it is prose. The
    check used to be `"GPU" not in used`, and the downgrade reasons contain
    "Apple's GPU" and "never fills a GPU" -- so instance segmentation and
    handpose passed the GPU rows by accident, and rewording either sentence
    would have turned a correct run red.
    """
    return used.split(" -- ", 1)[0].strip().upper().startswith("CPU")


def _device_from_logs(logs) -> str | None:
    """The device line the training job writes, if the run got that far."""
    for line in reversed((logs or "").splitlines()):
        if "Training device:" in line:
            return line.split("Training device:", 1)[1].strip()
    return None
