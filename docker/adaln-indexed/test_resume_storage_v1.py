"""CPU-only storage recovery tests, with no target GPU invocation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import resume_storage_v1 as recovery


def legacy_storage():
    path = Path(__file__).parent / "local/.cache/adaln-indexed/storage.py"
    spec = importlib.util.spec_from_file_location("storage_for_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, monkeypatch):
    storage = legacy_storage()
    monkeypatch.setattr(storage, "VOLUME", tmp_path)
    monkeypatch.setattr(storage, "mounted_volume", lambda path: dict(mount="/workspace"))
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=260004 * recovery.GIB)
    )
    plan = dict(evidence_limit_bytes=40 * recovery.GIB,
                archive_limit_bytes=41 * recovery.GIB, unpack_limit_bytes=43 * recovery.GIB)
    out = tmp_path / "batch"
    out.mkdir()
    return storage, plan, out


def test_declared_200gb_volume_ignores_shared_pool_baseline(tmp_path, monkeypatch):
    storage, plan, out = fixture(tmp_path, monkeypatch)
    check = recovery.corrected_check(storage.check_batch, storage, out, plan, 200)
    assert check["disk_floor_bytes"] == 85 * recovery.GIB
    assert check["free_bytes"] == 190_000_000_000
    assert check["filesystem_pool_free_bytes"] == 260004 * recovery.GIB
    assert check["declared_volume_bytes"] == 200_000_000_000


def test_declared_quota_can_refuse_despite_huge_filesystem_pool(tmp_path, monkeypatch):
    storage, plan, out = fixture(tmp_path, monkeypatch)
    with pytest.raises(storage.StoragePressure, match="BLOCKED_DECLARED_VOLUME"):
        recovery.corrected_check(storage.check_batch, storage, out, plan, 100)


def test_other_volume_contents_are_charged_to_declared_capacity(tmp_path, monkeypatch):
    storage, plan, out = fixture(tmp_path, monkeypatch)
    with (tmp_path / "other-batch").open("wb") as sparse:
        sparse.truncate(100 * recovery.GIB)
    with pytest.raises(storage.StoragePressure, match="BLOCKED_DECLARED_VOLUME"):
        recovery.corrected_check(storage.check_batch, storage, out, plan, 200)


def test_recovery_rejects_changed_legacy_helper_before_import(tmp_path):
    helpers = tmp_path / ".cache/adaln-indexed"
    helpers.mkdir(parents=True)
    (helpers / "storage.py").write_text("changed helper must not be loaded")
    with pytest.raises(RuntimeError, match="unmodified 88977ec"):
        recovery.load_legacy(tmp_path)


def test_fixed_floor_still_rejects_actual_low_filesystem_space(tmp_path, monkeypatch):
    storage, plan, out = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(storage.StoragePressure, match="BLOCKED_STORAGE"):
        recovery.corrected_check(storage.check_batch, storage, out, plan, 200)


def test_retained_controller_is_bound_to_executed_bytes_and_rejects_tamper(tmp_path):
    source = tmp_path / "controller.py"
    source.write_text("# CPU recovery controller fixture")
    audit = tmp_path / "audit"
    audit.mkdir()
    retained = recovery.retain_controller(source, audit)
    assert retained.read_bytes() == source.read_bytes()
    assert recovery.retain_controller(source, audit) == retained
    retained.write_text("# changed copy")
    with pytest.raises(RuntimeError, match="differs from executed source"):
        recovery.retain_controller(source, audit)


@pytest.mark.parametrize("state", [
    dict(state="PHASE_COMPLETE", error=None),
    dict(state="FAIL", error="accuracy mismatch"),
    dict(state="RUNNING", error="BLOCKED_STORAGE: old error"),
])
def test_recovery_refuses_unrelated_failures_or_unstarted_performance(state):
    with pytest.raises(RuntimeError, match="only accepts a recorded BLOCKED_STORAGE"):
        recovery.verify_storage_failure(state)


def test_recovery_accepts_the_recorded_storage_supervisor_failure():
    recovery.verify_storage_failure(dict(
        state="FAIL", error="BLOCKED_STORAGE: need 260003.79 GiB free, have 260003.54 GiB"
    ))


@pytest.mark.parametrize("pressure", [False, True])
def test_post_run_storage_check_prevents_false_completion(tmp_path, pressure):
    import json

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(dict(
        state="EXECUTION_COMPLETE_REVIEW_PENDING", sessions={"0": {}}
    )))
    calls = []

    def record_storage(out, state, label, plan):
        calls.append(label)
        if pressure:
            raise RuntimeError("evidence exceeds its locked storage budget")
        state["final_storage_checked"] = True

    batch = SimpleNamespace(
        record_storage=record_storage,
        record_failure=lambda *args: calls.append("failure retained"),
        write=lambda path, value: path.write_text(json.dumps(value)),
    )
    assert recovery.finish_recovery(batch, tmp_path, {}) == int(pressure)
    saved = json.loads(state_file.read_text())
    assert saved["sessions"] == {"0": {}}
    assert saved["state"] == ("FAIL" if pressure else "EXECUTION_COMPLETE_REVIEW_PENDING")
    assert calls[0] == "recovery-complete"
