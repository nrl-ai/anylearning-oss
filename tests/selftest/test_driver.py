"""Regression checks for the packaged self-test driver."""


class RunningProcess:
    def poll(self):
        return None


def test_windows_ansi_reset_is_not_part_of_the_server_token(tmp_path):
    from anylearning.selftest.driver import token_from_log

    token = "0123456789abcdef0123456789abcdef"
    log = tmp_path / "server.log"
    log.write_text(
        f"INFO API token for this server: {token}\x1b[0m\n", encoding="utf-8"
    )

    assert token_from_log(log, RunningProcess(), seconds=1) == token


def test_selftest_uses_a_metal_safe_keypoint_batch():
    from anylearning.selftest import _selftest_batch_size

    assert _selftest_batch_size("Keypoint Detection", 6) == 2
    assert _selftest_batch_size("Object Detection", 6) == 6
    assert _selftest_batch_size("Object Detection", 20) == 8
