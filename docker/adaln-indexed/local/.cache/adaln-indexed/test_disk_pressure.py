"""CPU regressions for full disks; no GPU or large allocations."""

import errno
import json
import subprocess
import sys

import pytest

import run_ticket0607 as batch


def no_space(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_failed_child_is_not_masked_by_failure_timing_write(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "write", no_space)
    with pytest.raises(subprocess.CalledProcessError) as error:
        batch.execute([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "child.log")
    assert error.value.returncode == 7


def test_failure_record_enospc_does_not_crash_or_replace_last_checkpoint(
    tmp_path, monkeypatch, capsys
):
    original = dict(state="RUNNING", completed={}, sessions={})
    (tmp_path / "state.json").write_text(json.dumps(original))
    (tmp_path / "source-sha256.json").write_text("{}")
    (tmp_path / "candidate.json").write_text(json.dumps({"source_sha256": "wrong"}))
    monkeypatch.setattr(batch, "write", no_space)
    assert batch.run(tmp_path, "all", 1) == 1
    assert json.loads((tmp_path / "state.json").read_text()) == original
    captured = capsys.readouterr()
    assert "source manifest identity changed" in captured.err
    assert "No space left on device" in captured.err
