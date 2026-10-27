"""Recorded storage-only recovery for the unmodified 88977ec acceptance image.

The candidate and all worker sources remain locked. Only the parent's shared-pool
threshold is replaced. --volume-gb is the user's Pod setting, not a quota query.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

GIB = 1024**3
LEGACY = {
    "storage.py": "9a838739184d7217a601eb46ffb3ba153a7b13fba2c6fab36923bf2b419fa562",
    "run_ticket0607.py": "c6c85bdee90481fcda0dce7009d51c38cd51429f5e26f3e42cc1109cbe5c3fec",
}
PLAN_SHA256 = "1e27461f5423ea20c77ff728d2eadfa45b0042664c3199541147f08128809e09"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy(root):
    helpers = root / ".cache/adaln-indexed"
    for name, expected in LEGACY.items():
        if digest(helpers / name) != expected:
            raise RuntimeError("recovery requires the unmodified 88977ec image: " + name)
    if (
        "storage" in sys.modules
        and Path(sys.modules["storage"].__file__).resolve() != helpers / "storage.py"
    ):
        raise RuntimeError("storage module already loaded from a different checkout")
    sys.path.insert(0, str(helpers))
    spec = importlib.util.spec_from_file_location(
        "legacy_recovery_batch", helpers / "run_ticket0607.py"
    )
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    return batch


def corrected_check(original, storage, output, plan, volume_gb):
    check = original(output, plan)
    capacity = volume_gb * 1_000_000_000
    # statvfs can report a host-wide pool. Declare the Pod volume and account its
    # contents separately; 5% margin is conservative planning, not a quota probe.
    volume_used = storage.tree_stats(storage.VOLUME)[0]
    margin = max(GIB, (capacity + 19) // 20)
    estimated_free = capacity - volume_used - margin
    if estimated_free < check["required_remaining_bytes"]:
        raise storage.StoragePressure(
            "BLOCKED_DECLARED_VOLUME: need "
            f"{check['required_remaining_bytes'] / GIB:.2f} GiB remaining; "
            f"configured {volume_gb} GB minus accounted files/margin leaves "
            f"{estimated_free / GIB:.2f} GiB"
        )
    check.update(
        filesystem_pool_free_bytes=check["free_bytes"],
        free_bytes=min(check["free_bytes"], estimated_free),
        declared_volume_bytes=capacity,
        accounted_volume_bytes=volume_used,
        assumed_volume_margin_bytes=margin,
        disk_floor_bytes=plan["archive_limit_bytes"] + plan["unpack_limit_bytes"] + GIB,
        storage_policy="shared-pool-recovery-v1; declared quota estimate, not quota telemetry",
    )
    return check


def verify_storage_failure(state):
    if (
        state.get("state") != "FAIL"
        or not str(state.get("error", "")).startswith("BLOCKED_STORAGE:")
    ):
        raise RuntimeError("recovery only accepts a recorded BLOCKED_STORAGE batch failure")


def retain_controller(source, audit):
    expected = digest(source)
    controller = audit / (expected + ".py")
    if controller.exists():
        if digest(controller) != expected:
            raise RuntimeError("retained recovery controller differs from executed source")
    else:
        temporary = controller.with_suffix(".py.tmp")
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise RuntimeError("recovery controller changed during retention")
        temporary.replace(controller)
    return controller


def finish_recovery(batch, out, plan):
    state = json.loads((out / "state.json").read_text())
    try:
        batch.record_storage(out, state, "recovery-complete", plan)
    except (OSError, RuntimeError) as exc:
        state.update(state="FAIL", error=str(exc))
        batch.record_failure(out, "storage-recovery", state)
        batch.write(out / "state.json", state)
        print("RECOVERY STORAGE CHECK FAILED:", exc, file=sys.stderr)
        return 1
    batch.write(out / "state.json", state)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume-gb", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("/opt/RL-Kernel-indexed"))
    parser.add_argument(
        "--check-only", action="store_true", help="CPU identity/storage checks only"
    )
    args = parser.parse_args(argv)
    if args.volume_gb <= 0:
        parser.error("--volume-gb must match the positive Pod Volume Disk setting")
    out = args.output.resolve()
    batch = load_legacy(args.root.resolve())
    from evidence_io import exclusive_output

    with exclusive_output(out):
        batch.verify_candidate(out)
        state = json.loads((out / "state.json").read_text())
        batch.verify_saved_evidence(out, state)
        verify_storage_failure(state)
        required = {batch.task_name(t) for t in sum(batch.tasks(), [])}
        if set(state["completed"]) != required or set(state.get("artifacts", {})) != required:
            raise RuntimeError("recovery requires all 17 suites and artifact receipts")
        if state.get("sessions"):
            raise RuntimeError("an accepted session already exists; do not rerun it")
        if digest(out / "storage-plan.json") != PLAN_SHA256:
            raise RuntimeError("recovery storage plan is not the locked v1 model")
        plan = json.loads((out / "storage-plan.json").read_text())
        original = batch.storage.check_batch

        def check(output, locked_plan):
            return corrected_check(original, batch.storage, output, locked_plan, args.volume_gb)

        probe = check(out, plan)
        print(
            f"STORAGE RECOVERY: declared={args.volume_gb} GB; "
            f"accounted={probe['accounted_volume_bytes'] / GIB:.2f} GiB; "
            f"fixed pool floor={probe['disk_floor_bytes'] / GIB:.2f} GiB; "
            "17 suites/artifacts retained; failed performance restarts as a new session attempt",
            flush=True,
        )
        if args.check_only:
            return 0
        audit = out / "storage-recovery"
        audit.mkdir(exist_ok=True)
        source = Path(__file__).resolve()
        controller = retain_controller(source, audit)
        batch.write(
            batch.next_attempt(audit, "policy"),
            dict(policy=probe, controller=controller.relative_to(out).as_posix(),
                 controller_sha256=digest(controller), legacy_helpers=LEGACY,
                 original_candidate_sha256=digest(out / "candidate.json"),
                 scope="parent storage supervision only; candidate and worker files unchanged"),
        )
        batch.storage.check_batch = check
        try:
            result = batch.run(out, "all", 1)
            return finish_recovery(batch, out, plan) if result == 0 else result
        finally:
            batch.storage.check_batch = original


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print("RECOVERY REFUSED:", exc, file=sys.stderr)
        raise SystemExit(1) from None
