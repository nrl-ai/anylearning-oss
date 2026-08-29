"""Machine-level performance settings.

The behaviour worth pinning is the resolution order and the device-awareness.
The numbers themselves come from measurement (see `anylearning/settings.py`),
but the *shape* of the heuristic is what a future change is likely to break:
more workers is nearly free with a GPU and much less so on CPU, and power
saving must actually ask for less work rather than merely a different number.
"""

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    from anylearning import config, settings

    monkeypatch.setattr(config, "DATA_ROOT", str(tmp_path))
    settings.save({"performance_mode": "maximum", "training_num_workers": "auto"})
    return settings


def test_defaults_to_maximum(store):
    assert store.load()["performance_mode"] == "maximum"


def test_unknown_mode_falls_back_rather_than_breaking_training(store, tmp_path):
    (tmp_path / "settings.json").write_text('{"performance_mode": "ludicrous"}')
    assert store.load()["performance_mode"] == "maximum"


def test_corrupt_file_is_ignored(store, tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert store.load()["performance_mode"] == "maximum"


def test_gpu_gets_more_workers_than_cpu(store):
    """Measured 3.3x from workers on GPU against 1.3x on CPU: on CPU they
    compete with training for the same cores."""
    assert store.resolve_num_workers(on_gpu=True) >= store.resolve_num_workers(on_gpu=False)


@pytest.mark.parametrize("mode", ["maximum", "balanced", "power_saving"])
def test_every_mode_resolves_to_something_usable(store, mode):
    store.save({"performance_mode": mode})
    for on_gpu in (True, False):
        workers = store.resolve_num_workers(on_gpu=on_gpu)
        assert isinstance(workers, int)
        assert 0 <= workers <= (store._cpus() or 1)


def test_power_saving_asks_for_less_than_maximum(store):
    store.save({"performance_mode": "maximum"})
    most = store.resolve_num_workers(on_gpu=True)
    assert store.resolve_cudnn_benchmark() is True

    store.save({"performance_mode": "power_saving"})
    least = store.resolve_num_workers(on_gpu=True)

    assert least < most, "power saving must actually reduce the work"
    assert store.resolve_cudnn_benchmark() is False
    assert store.resolve_pin_memory("cuda") is False


def test_an_explicit_setting_beats_the_mode(store):
    store.save({"performance_mode": "power_saving", "training_num_workers": 6})
    assert store.resolve_num_workers(on_gpu=True) == 6


def test_a_configured_zero_is_treated_as_unset(store):
    """0 is the old hard-coded default this setting exists to correct, so it
    must not pin the loader back to single-threaded decoding."""
    assert store.resolve_num_workers(0, on_gpu=True) > 0


def test_a_positive_config_value_is_respected(store):
    assert store.resolve_num_workers(3, on_gpu=True) == 3


def test_persistent_workers_needs_workers(store):
    assert store.resolve_persistent_workers(0) is False
    assert store.resolve_persistent_workers(4) is True


def test_pin_memory_is_a_cuda_concern(store):
    assert store.resolve_pin_memory("cuda") is True
    assert store.resolve_pin_memory("cpu") is False


def test_describe_reports_both_devices(store):
    resolved = store.describe()["resolved"]
    assert "num_workers_gpu" in resolved
    assert "num_workers_cpu" in resolved
    assert resolved["cpu_count"] >= 1


# --------------------------------------------------------------------------
# Machine detection
#
# os.cpu_count() reports the whole machine, not what this process may use, and
# counts hyperthreads as cores. Both over-count, and the consequence is asking
# for more workers than the machine can run -- which makes training slower.
# --------------------------------------------------------------------------


def test_worker_count_never_exceeds_the_cpus_we_may_use(store, monkeypatch):
    monkeypatch.setattr(store, "_cpus", lambda: 2)
    monkeypatch.setattr(store, "_physical_cores", lambda: 2)
    for mode in store.PERFORMANCE_MODES:
        for on_gpu in (True, False):
            # One CPU is always left for the training process itself.
            assert store.auto_num_workers(on_gpu, mode) <= 1


def test_a_single_cpu_machine_asks_for_no_workers(store, monkeypatch):
    """With one CPU a worker process can only take time away from training."""
    monkeypatch.setattr(store, "_cpus", lambda: 1)
    monkeypatch.setattr(store, "_physical_cores", lambda: 1)
    for mode in store.PERFORMANCE_MODES:
        assert store.auto_num_workers(True, mode) == 0


def test_low_memory_caps_the_worker_count(store, monkeypatch):
    """Each worker holds its own prefetched batches."""
    monkeypatch.setattr(store, "_cpus", lambda: 32)
    monkeypatch.setattr(store, "_physical_cores", lambda: 16)
    monkeypatch.setattr(store, "_memory_worker_cap", lambda: 2)
    assert store.auto_num_workers(True, "maximum") == 2


def test_budgets_against_physical_cores_not_hyperthreads(store, monkeypatch):
    """A 16-thread/8-core box must not hand all eight real cores to the loader."""
    monkeypatch.setattr(store, "_cpus", lambda: 16)
    monkeypatch.setattr(store, "_physical_cores", lambda: 8)
    monkeypatch.setattr(store, "_memory_worker_cap", lambda: 1 << 30)
    assert store.auto_num_workers(True, "maximum") < 8


def test_never_exceeds_the_absolute_cap(store, monkeypatch):
    monkeypatch.setattr(store, "_cpus", lambda: 256)
    monkeypatch.setattr(store, "_physical_cores", lambda: 128)
    monkeypatch.setattr(store, "_memory_worker_cap", lambda: 1 << 30)
    assert store.auto_num_workers(True, "maximum") == store.MAX_WORKERS


def test_cgroup_quota_is_respected(store, monkeypatch, tmp_path):
    """A container limited to two cores reports the host's total otherwise."""
    monkeypatch.setattr(store, "_cgroup_cpu_limit", lambda: 2.0)
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert store._cpus() <= 2


def test_a_failing_cgroup_read_does_not_stop_training(store, monkeypatch):
    """Detection runs on the training path, so it must degrade, not raise."""

    def explode():
        raise OSError("cgroup filesystem is not what we assumed")

    monkeypatch.setattr(store, "_cgroup_cpu_limit", explode)

    assert store._cpus() >= 1
    assert store.auto_num_workers(True, "maximum") >= 0


def test_detection_never_returns_zero(store):
    assert store._cpus() >= 1
    assert store._physical_cores() >= 1
