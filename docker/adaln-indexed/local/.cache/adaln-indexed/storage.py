"""CPU-only storage budget and guards for the indexed acceptance batch."""

import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

GIB = 1024**3
MIB = 1024**2
CONTROL_RESERVE = 256 * MIB
ARTIFACT_ALLOWANCE = 8 * GIB
VOLUME = Path("/workspace")


class StoragePressure(RuntimeError):
    pass


def oracle_bytes(case):
    batch, seq, hidden = case["shape"]
    elements = batch * seq * hidden
    modulation = 2 * batch * 3 * hidden
    width = {"fp32": 4, "bf16": 2}[case["dtype"]]
    # Inputs + fixed outputs use dtype; accuracy gold is FP32, baseline FP64.
    return (3 * elements + modulation) * (2 * width + 4 + 8) + batch * seq * 4


def budget(plan):
    unique = {}
    for case in plan["cases"]:
        key = (tuple(case["shape"]), case["dtype"], case["route"])
        unique[key] = dict(case)
    cases = [
        dict(case=c, tensor_bytes=oracle_bytes(c), file_limit_bytes=oracle_bytes(c) + MIB)
        for c in unique.values()
    ]
    oracle_limit = sum(c["file_limit_bytes"] for c in cases)
    evidence_limit = oracle_limit + ARTIFACT_ALLOWANCE
    archive_limit = math.ceil(evidence_limit * 1.02) + 64 * MIB
    unpack_limit = math.ceil(evidence_limit * 1.05) + 256 * MIB
    required = evidence_limit + archive_limit + unpack_limit + GIB
    return dict(
        schema="adaln-indexed-storage-v1",
        state="COMMAND_PREPARED",
        oracle_cases=cases,
        oracle_limit_bytes=oracle_limit,
        other_artifacts_limit_bytes=ARTIFACT_ALLOWANCE,
        evidence_limit_bytes=evidence_limit,
        archive_limit_bytes=archive_limit,
        unpack_limit_bytes=unpack_limit,
        control_reserve_bytes=CONTROL_RESERVE,
        required_empty_volume_free_bytes=required,
        recommended_volume_gb=math.ceil(required * 1.05 / 1_000_000_000 / 25) * 25,
        policy="one compact payload per logical case; 8 GiB bounded allowance for other evidence; "
        "archive and isolated verification coexist; filesystem overhead and control reserve",
        scope="CPU capacity model and enforced limits; not a measured GPU cache requirement",
    )


def tree_stats(directory):
    entries = list(directory.rglob("*"))
    if any(p.is_symlink() and p.is_dir() for p in entries):
        raise StoragePressure("directory symlinks cannot be safely bounded for archival")
    paths = [p for p in entries if p.is_file()]
    # Follow file symlinks: tar dereferences them, so archive size includes every copy.
    return sum(p.stat().st_size for p in paths), len(paths)


def mounted_volume(output, mountinfo=None):
    output = output.resolve()
    if not output.is_relative_to(VOLUME):
        raise StoragePressure("output must be on the dedicated /workspace volume")
    content = Path("/proc/self/mountinfo").read_text() if mountinfo is None else mountinfo
    found = None
    for line in content.splitlines():
        fields = line.split()
        separator = fields.index("-")
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if mount == str(VOLUME):
            found = dict(mount=mount, fstype=fields[separator + 1], source=fields[separator + 2])
    if found is None or found["fstype"] in {"overlay", "tmpfs", "ramfs", "squashfs"}:
        raise StoragePressure(
            "/workspace is not a dedicated data volume; check Pod Volume Disk configuration"
        )
    # A nested ephemeral mount must not shadow the output, even with a valid /workspace.
    for line in content.splitlines():
        fields = line.split()
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if mount != VOLUME and mount.is_relative_to(VOLUME) and output.is_relative_to(mount):
            raise StoragePressure(
                "nested output mount must be checked separately; use /workspace directly"
            )
    return found


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def require_free(path, required, *, files=4096):
    existing = existing_parent(path)
    available = shutil.disk_usage(existing).free
    filesystem = os.statvfs(existing)
    if filesystem.f_files and filesystem.f_favail < files:
        raise StoragePressure(
            f"BLOCKED_STORAGE: need {files} free inodes, have {filesystem.f_favail}"
        )
    if available < required:
        raise StoragePressure(
            f"BLOCKED_STORAGE: need {required / GIB:.2f} GiB free, have {available / GIB:.2f} GiB"
        )
    return available


def check_batch(output, plan):
    mount = mounted_volume(output)
    used = tree_stats(output)[0] if output.exists() else 0
    if used > plan["evidence_limit_bytes"]:
        raise StoragePressure(
            "batch evidence exceeds its locked storage budget; retain it and review scope"
        )
    required = (
        plan["evidence_limit_bytes"]
        - used
        + plan["archive_limit_bytes"]
        + plan["unpack_limit_bytes"]
        + GIB
    )
    available = require_free(output, required)
    return dict(
        mount,
        free_bytes=available,
        evidence_bytes=used,
        required_remaining_bytes=required,
        # Monitor filesystem growth without walking files during timed samples.
        # statvfs may expose a shared pool: unrelated writers must not move our threshold.
        disk_floor_bytes=plan["archive_limit_bytes"] + plan["unpack_limit_bytes"] + GIB,
    )


def supervised_run(command, *, cwd, env, stdout, interval=1.0):
    root = Path(env["RLK_ADALN_STORAGE_ROOT"])
    floor = int(env["RLK_ADALN_DISK_FLOOR_BYTES"])
    require_free(root, floor)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        while True:
            try:
                code = process.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                require_free(root, floor)
        if code:
            raise subprocess.CalledProcessError(code, command)
        require_free(root, floor)
    except BaseException:
        # Stop the worker and its compiler/sanitizer children before reserving evidence.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The leader can exit first while a descendant ignores SIGINT.
            # Always stop the remaining process group before returning control.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    return subprocess.CompletedProcess(command, code)
