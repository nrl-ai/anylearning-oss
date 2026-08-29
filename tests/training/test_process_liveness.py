"""A run is over when its process is over -- not before, and not never.

Both halves of this were wrong at once, and each was found by watching a real
instance-segmentation run in the packaged build:

* the training child was killed mid-`torch.save`, became a zombie because
  nothing joins it, and `psutil.Process.is_running()` reports True for a
  zombie -- so the session stayed "training" forever and the project refused
  to start another run;
* a dataloader worker exiting during `children(recursive=True)` raised
  NoSuchProcess, which the caller treated as the *training* process dying, so
  a healthy run flashed "failed" in the UI several times before finishing.
"""

import multiprocessing
import os
import subprocess
import sys
import time

import psutil
import pytest

from anylearning.routers.training import training_process_ended


def test_no_pid_is_ended():
    assert training_process_ended(None) is True
    assert training_process_ended(0) is True


def test_a_live_process_has_not_ended():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert training_process_ended(process.pid) is False
    finally:
        process.kill()
        process.wait()


@pytest.mark.skipif(os.name == "nt", reason="no /proc to read pid_max from")
def test_a_pid_that_never_existed_has_ended():
    # A pid well above the machine's maximum, so it cannot be in use.
    with open("/proc/sys/kernel/pid_max") as handle:
        beyond = int(handle.read().strip()) + 1
    assert training_process_ended(beyond) is True


@pytest.mark.skipif(os.name == "nt", reason="Windows has no zombies")
def test_a_zombie_has_ended():
    """The failure that hung a project.

    A `multiprocessing.Process` that has exited and not been joined is a
    zombie, which is exactly the state the training child is left in when it
    dies on its own.
    """
    child = multiprocessing.Process(target=lambda: None)
    child.start()
    pid = child.pid

    # Wait for it to become a zombie, without joining it -- joining is what
    # reaps it, and the point is the state before anything reaps it.
    for _ in range(100):
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)

    assert training_process_ended(pid) is True
    child.join(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="Windows has no zombies")
def test_the_check_reaps_what_it_finds():
    """Asking the question should also clean up, or zombies accumulate: one per
    training run, for as long as the application is open."""
    child = multiprocessing.Process(target=lambda: None)
    child.start()
    pid = child.pid
    child.join(timeout=5)  # this one is reaped by us

    # A second, deliberately unjoined.
    other = multiprocessing.Process(target=lambda: None)
    other.start()
    time.sleep(0.3)
    assert training_process_ended(other.pid) is True
    # active_children() inside the check joins it, so the pid is gone entirely
    # rather than lingering as a zombie.
    assert not psutil.pid_exists(other.pid) or psutil.Process(
        other.pid
    ).status() != psutil.STATUS_ZOMBIE
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


def test_children_coming_and_going_do_not_end_the_run():
    """The false "failed" flash.

    The parent spawns and reaps short-lived children continuously, which is
    what a dataloader with workers looks like from outside. The check must
    answer "still running" every time.
    """
    script = (
        "import subprocess, sys, time\n"
        "end = time.time() + 6\n"
        "while time.time() < end:\n"
        "    subprocess.Popen([sys.executable, '-c', 'pass'])\n"
        "    time.sleep(0.01)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.time() + 3
        while time.time() < deadline:
            assert training_process_ended(parent.pid) is False
    finally:
        parent.kill()
        parent.wait()


@pytest.mark.skipif(
    os.name == "nt",
    reason="SIGTERM cannot be ignored on Windows: terminate() is TerminateProcess",
)
def test_terminating_kills_what_ignores_being_asked():
    """SIGTERM is not enough on its own.

    Python delivers SIGTERM by setting a flag the interpreter checks between
    bytecodes, so a process wedged in a C-level lock -- a deadlocked training
    job, which is how this was found -- never runs the handler. The Stop button
    has to end the run regardless.
    """
    import signal
    import subprocess
    import sys

    # Ignores SIGTERM, exactly like a process that cannot run its handler.
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        handle = psutil.Process(process.pid)

        handle.terminate()
        _, alive = psutil.wait_procs([handle], timeout=2)
        assert alive, "the test's premise is wrong: it did not ignore SIGTERM"

        for stubborn in alive:
            stubborn.kill()
        _, still_alive = psutil.wait_procs([handle], timeout=5)
        assert not still_alive
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
        process.wait(timeout=5)
        # Closed explicitly: left to the garbage collector it raises a
        # ResourceWarning, and this suite turns warnings into errors.
        if process.stdout is not None:
            process.stdout.close()
