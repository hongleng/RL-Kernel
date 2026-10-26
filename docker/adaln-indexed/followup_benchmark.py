"""Run a corrected benchmark beside immutable target evidence; never relabel old PASS."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    evidence, patched, out = (p.resolve() for p in (args.evidence, args.benchmark, args.output))
    if out == evidence or evidence in out.parents:
        raise RuntimeError("follow-up output must be outside the original batch")
    sys.path[:0] = [str(root), str(root / ".cache/adaln-indexed")]
    import run_ticket0607 as batch

    state = json.loads((evidence / "state.json").read_text())
    required = [batch.task_name(task) for task in sum(batch.tasks(), [])]
    if any(name not in state.get(k, {}) for k in ("completed", "artifacts") for name in required):
        raise RuntimeError("all 17 original suites and artifact steps must be complete")
    if state.get("sessions"):
        raise RuntimeError("this follow-up expects no accepted prior performance session")
    print("VERIFY original candidate, suites and artifact receipts", flush=True)
    batch.verify_candidate(evidence)
    batch.verify_saved_evidence(evidence, state)
    for task in sum(batch.tasks(), []):
        batch.check_result(
            evidence, task, evidence / state["completed"][batch.task_name(task)]["path"]
        )
    print("VERIFY target/compiler/runtime identity", flush=True)
    target = batch.target_identity()
    stable = {
        k: v for k, v in target.items() if k not in ("free_gpu_bytes", "available_host_bytes")
    }
    if stable != json.loads((evidence / "target.json").read_text()):
        raise RuntimeError("target/compiler/runtime changed")
    if target["free_gpu_bytes"] < 8 * 1024**3 or target["available_host_bytes"] < 16 * 1024**3:
        raise RuntimeError("BLOCKED_RESOURCE: >=8 GiB GPU / >=16 GiB host required")
    built = json.loads(
        (evidence / state["completed"]["cuda-correctness"]["path"] / "build.json").read_text()
    )
    binary = Path(built["binary"])
    if batch.sha(binary) != built["binary_sha256"]:
        raise RuntimeError("original fresh binary changed")
    baseline = json.loads((evidence / "source-sha256.json").read_text())
    plan = json.loads((evidence / "performance-plan.json").read_text())
    benchmark_path = ".cache/adaln-indexed/benchmark_adaln_indexed.py"
    patch_sha = batch.sha(patched)
    old_state_sha = batch.sha(evidence / "state.json")
    out.mkdir(parents=True, exist_ok=False)
    provenance: dict = dict(
        state="RUNNING",
        original_evidence=str(evidence),
        original_state_sha256=old_state_sha,
        original_candidate_sha256=batch.sha(evidence / "candidate.json"),
        original_benchmark_sha256=baseline[benchmark_path],
        benchmark_sha256=patch_sha,
        followup_driver_sha256=batch.sha(Path(__file__)),
        binary_sha256=built["binary_sha256"],
        target=stable,
        resources=target,
        sessions=[],
        scope="performance follow-up only; original batch remains unchanged; review pending",
    )
    shutil.copyfile(__file__, out / "followup-driver.py")
    batch.write(out / "provenance.json", provenance)
    reports = []
    try:
        deps = [p for p in sys.path if "site-packages" in p]
        for session in range(3):
            package = out / "sources" / str(session)
            shutil.copytree(evidence / "candidate-source", package)
            shutil.copyfile(patched, package / benchmark_path)
            changed = {k for k, v in batch.inventory(package).items() if baseline.get(k) != v}
            if changed != {benchmark_path} or set(batch.inventory(package)) != set(baseline):
                raise RuntimeError("only the benchmark may change in the follow-up source")
            (package / "rl_engine/_C.so").symlink_to(binary)
            dest = out / "sessions" / str(session)
            bootstrap = (
                "import sys,runpy; sys.path.extend("
                + repr(deps)
                + "); runpy.run_path("
                + repr(str(package / benchmark_path))
                + ",run_name='__main__')"
            )
            print("START performance", session, flush=True)
            batch.execute(
                [
                    sys.executable,
                    "-S",
                    "-c",
                    bootstrap,
                    "--output",
                    str(dest),
                    "--session",
                    str(session),
                ],
                out / f"performance-{session}.log",
                cwd=package,
                env=dict(os.environ, PYTHONPATH=str(package), OMP_NUM_THREADS="1"),
            )
            report = json.loads((dest / "report.json").read_text())
            if report["state"] != "EXECUTION_COMPLETE_REVIEW_PENDING":
                raise RuntimeError("performance session failed")
            for key in ("cases", "seed", "warmup", "repeat", "backends", "modes", "surfaces"):
                if report[key] != plan[key]:
                    raise RuntimeError("measurement inputs/plan changed: " + key)
            identity = report["environment"]
            if identity["binary_sha256"] != built["binary_sha256"]:
                raise RuntimeError("performance binary mismatch")
            for name, digest in identity["source_sha256"].items():
                expected = patch_sha if name == benchmark_path else baseline.get(name)
                if digest != expected:
                    raise RuntimeError("performance source mismatch: " + name)
            reports.append(report)
            provenance["sessions"].append(dict(path=str(dest), receipt=batch.receipt(dest)))
            batch.write(out / "provenance.json", provenance)
        spec = importlib.util.spec_from_file_location("corrected_benchmark", patched)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the corrected benchmark summary")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        batch.write(out / "summary.json", module.summarize_sessions(reports))
        batch.verify_candidate(evidence)
        batch.verify_saved_evidence(evidence, state)
        if batch.sha(evidence / "state.json") != old_state_sha:
            raise RuntimeError("original batch state changed")
        provenance["state"] = "EXECUTION_COMPLETE_REVIEW_PENDING"
    except (Exception, KeyboardInterrupt) as exc:
        provenance.update(state="FAIL", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        batch.write(out / "provenance.json", provenance)


if __name__ == "__main__":
    main()
