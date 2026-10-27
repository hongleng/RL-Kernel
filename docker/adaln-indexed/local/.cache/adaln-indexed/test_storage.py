"""Storage planning and disk-pressure checks use CPU fixtures only."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import benchmark_adaln_indexed as bench
import storage


MOUNTS = "1 0 0:1 / / rw - overlay overlay rw\n2 1 8:1 / /workspace rw - ext4 /dev/volume rw\n"


def test_volume_guard_rejects_container_tmpfs_and_nested_mounts():
    assert storage.mounted_volume(Path("/workspace/evidence/run"), MOUNTS)["fstype"] == "ext4"
    for text in (
        MOUNTS.splitlines()[0],
        MOUNTS.replace("ext4", "tmpfs"),
        MOUNTS + "3 2 0:2 / /workspace/evidence rw - tmpfs tmpfs rw\n",
    ):
        with pytest.raises(storage.StoragePressure):
            storage.mounted_volume(Path("/workspace/evidence/run"), text)
    with pytest.raises(storage.StoragePressure):
        storage.mounted_volume(Path("/tmp/evidence"), MOUNTS)


def test_budget_deduplicates_only_layout_and_reserves_archive_and_unpacked_copy():
    plan = bench.plan(bench.parse_args(["--output", "/unused"]))
    result = storage.budget(plan)
    assert len(result["oracle_cases"]) == 14
    assert storage.oracle_bytes(dict(shape=[3, 8, 4], dtype="fp32")) == 7296
    assert storage.oracle_bytes(dict(shape=[3, 8, 4], dtype="bf16")) == 5856
    assert result["required_empty_volume_free_bytes"] > result["evidence_limit_bytes"] * 3
    assert result["recommended_volume_gb"] > 40


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_compact_payload_preserves_values_and_rejects_excess_storage(tmp_path, dtype):
    case = dict(shape=[3, 8, 4], dtype=dtype, route="random", layout="contiguous")
    cpu, fixed, baseline, record = bench.cached_references(case, 386, tmp_path)
    payload = torch.load(Path(record["path"]) / "oracle.pt", weights_only=True)
    tensors = [t for group in payload.values() for t in group]
    assert sum(t.numel() * t.element_size() for t in tensors) == storage.oracle_bytes(case)
    assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in tensors)
    for got, expected in zip(fixed, bench.reference_outputs(cpu), strict=True):
        assert all(torch.equal(a, b) for a, b in zip(got, expected, strict=True))
    assert record["file_bytes"] <= storage.oracle_bytes(case) + storage.MIB


def test_low_space_rejects_before_creating_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(storage.StoragePressure, match="BLOCKED_STORAGE"):
        bench.cached_references(
            dict(shape=[3, 8, 4], dtype="fp32", route="random", layout="contiguous"), 386, tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_runtime_disk_pressure_stops_actual_cpu_worker(tmp_path, monkeypatch):
    marker = tmp_path / "pid"
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=0 if marker.exists() else 1000),
    )
    command = [
        sys.executable,
        "-c",
        "import os,time; from pathlib import Path; Path("
        + repr(str(marker))
        + ").write_text(str(os.getpid())); time.sleep(30)",
    ]
    with (tmp_path / "worker.log").open("w") as log:
        with pytest.raises(storage.StoragePressure):
            storage.supervised_run(
                command,
                cwd=tmp_path,
                env=dict(
                    os.environ,
                    RLK_ADALN_STORAGE_ROOT=str(tmp_path),
                    RLK_ADALN_DISK_FLOOR_BYTES="500",
                ),
                stdout=log,
                interval=0.2,
            )
    assert marker.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_disk_pressure_stops_descendant_after_leader_exits(tmp_path, monkeypatch):
    import signal
    import time

    marker = tmp_path / "child-pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess,time; subprocess.Popen("
        + repr([sys.executable, "-c", child]) + "); time.sleep(30)"
    )
    monkeypatch.setattr(
        storage.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=0 if marker.exists() else 1000),
    )
    with (tmp_path / "group.log").open("w") as log:
        try:
            with pytest.raises(storage.StoragePressure):
                storage.supervised_run(
                    [sys.executable, "-c", parent], cwd=tmp_path,
                    env=dict(os.environ, RLK_ADALN_STORAGE_ROOT=str(tmp_path),
                             RLK_ADALN_DISK_FLOOR_BYTES="500"),
                    stdout=log, interval=0.2,
                )
            pid = int(marker.read_text())
            status = Path(f"/proc/{pid}/stat")
            deadline = time.monotonic() + 2
            while True:
                try:
                    if status.read_text().split()[2] == "Z":
                        break
                except FileNotFoundError:
                    break
                assert time.monotonic() < deadline, "descendant survived worker cleanup"
                time.sleep(0.02)
        finally:
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_archival_checks_capacity_before_any_sendable_file(tmp_path, monkeypatch):
    from test_acceptance_reuse import minimal_batch
    import evidence_io

    out = minimal_batch(tmp_path)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(storage.StoragePressure):
        evidence_io.pack(out, partial=True)
    assert not Path(str(out) + ".tar.gz").exists()
    assert not (out / "recovery-index.json").exists()


def test_runtime_start_rejects_nonvolume_before_output_creation(tmp_path):
    import subprocess

    output = tmp_path / "unsafe-output"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_ticket0607.py")),
        "--start",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1 and "dedicated /workspace volume" in result.stderr
    assert not output.exists()


def test_failed_worker_preserves_exit_status_when_space_is_low_after_exit(tmp_path, monkeypatch):
    calls = []

    def free_space(path):
        calls.append(path)
        return SimpleNamespace(free=1000 if len(calls) == 1 else 0)

    monkeypatch.setattr(storage.shutil, "disk_usage", free_space)
    with (tmp_path / "error.log").open("w") as log:
        with pytest.raises(__import__("subprocess").CalledProcessError) as error:
            storage.supervised_run(
                [sys.executable, "-c", "raise SystemExit(7)"],
                cwd=tmp_path,
                env=dict(
                    os.environ,
                    RLK_ADALN_STORAGE_ROOT=str(tmp_path),
                    RLK_ADALN_DISK_FLOOR_BYTES="500",
                ),
                stdout=log,
                interval=5,
            )
    assert error.value.returncode == 7


def test_archive_footprint_counts_file_links_and_rejects_directory_aliases(tmp_path):
    target = tmp_path / "payload"
    target.write_bytes(b"1234")
    (tmp_path / "linked-payload").symlink_to(target)
    assert storage.tree_stats(tmp_path) == (8, 2)
    directory = tmp_path / "directory"
    directory.mkdir()
    (tmp_path / "linked-directory").symlink_to(directory, target_is_directory=True)
    with pytest.raises(storage.StoragePressure, match="directory symlinks"):
        storage.tree_stats(tmp_path)


def test_archive_relative_path_verifies_on_the_archive_filesystem(tmp_path, monkeypatch):
    from test_acceptance_reuse import minimal_batch
    import evidence_io

    out = minimal_batch(tmp_path)
    archive, status = evidence_io.pack(out, partial=True)
    monkeypatch.chdir(tmp_path)
    assert evidence_io.verify_archive(Path(archive.name), status["candidate_sha256"], partial=True)[
        "partial"
    ]


def test_inode_exhaustion_is_rejected_even_with_free_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=storage.GIB)
    )
    monkeypatch.setattr(
        storage.os, "statvfs", lambda path: SimpleNamespace(f_files=10000, f_favail=1)
    )
    with pytest.raises(storage.StoragePressure, match="free inodes"):
        storage.require_free(tmp_path, 1)


def test_resume_refuses_changed_storage_receipt(tmp_path):
    import run_ticket0607 as batch

    probe = tmp_path / "storage-check.json"
    probe.write_text("{}")
    state = {"storage_checks": [{"path": probe.name, "sha256": batch.sha(probe)}]}
    batch.verify_saved_evidence(tmp_path, state)
    probe.write_text("changed")
    with pytest.raises(RuntimeError, match="prior storage check changed"):
        batch.verify_saved_evidence(tmp_path, state)


def test_batch_space_reserves_archive_and_verification_on_resume(tmp_path, monkeypatch):
    plan = storage.budget(bench.plan(bench.parse_args(["--output", "/unused"])))
    (tmp_path / "existing-evidence").write_bytes(b"12345")
    monkeypatch.setattr(storage, "mounted_volume", lambda path: dict(mount="/workspace"))
    free = plan["required_empty_volume_free_bytes"] - 5
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    check = storage.check_batch(tmp_path, plan)
    assert check["evidence_bytes"] == 5
    assert check["disk_floor_bytes"] == (
        plan["archive_limit_bytes"] + plan["unpack_limit_bytes"] + storage.GIB
    )
    free -= 1
    with pytest.raises(storage.StoragePressure):
        storage.check_batch(tmp_path, plan)
