# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""User-run sequential ticket06/07 batch. --prepare only collects CPU nodes."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPERS = (
    ".cache/adaln-indexed/run_ticket05.py",
    ".cache/adaln-indexed/audit_ticket05_artifacts.py",
    ".cache/adaln-indexed/run_ticket0607.py",
    ".cache/adaln-indexed/test_ticket0607.py",
    ".cache/adaln-indexed/benchmark_adaln_indexed.py",
    ".cache/adaln-indexed/test_benchmark_adaln_indexed.py",
    ".cache/adaln-indexed/runpod-0607.md",
    ".cache/adaln-indexed/spec.md",
    ".cache/adaln-indexed/issues/06-target-qualification-and-strict.md",
    ".cache/adaln-indexed/issues/07-indexed-benchmark.md",
)
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inventory(root=ROOT):
    paths = [root / name for name in (*HELPERS, "pyproject.toml", "setup.py")]
    for name in ("rl_engine", "tests", "csrc", "benchmarks", "build_tools", "tools"):
        paths += [
            p
            for p in (root / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".so", ".pyc"))
        ]
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(paths))}


def tasks():
    correctness = [
        (b, s, None)
        for s, backends in (
            ("correctness", ("cuda", "triton", "integration")),
            ("shared", ("cuda", "triton")),
            ("large-offsets", ("cuda", "triton")),
            ("target", ("cuda", "triton")),
        )
        for b in backends
    ]
    return correctness, [(b, "sanitizer", t) for b in ("cuda", "triton") for t in TOOLS]


def task_name(task):
    backend, suite, tool = task
    return backend + "-" + (tool or suite)


def execute(command, log, *, cwd=ROOT, env=None):
    """No retry and no polling; preserve failing process output."""
    with log.open("x") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prepare(out, sessions=3):
    out.mkdir(parents=True, exist_ok=False)
    baseline = inventory()
    package = out / "candidate-source"
    for name in baseline:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(out / "source-sha256.json", baseline)
    with tarfile.open(out / "candidate-source.tar.gz", "w:gz") as archive:
        archive.add(package, arcname="candidate-source")
    write(
        out / "candidate.json",
        {
            "state": "COMMAND_PREPARED",
            "source_sha256": sha(out / "source-sha256.json"),
            "archive_sha256": sha(out / "candidate-source.tar.gz"),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "status": subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=all"],
                cwd=ROOT,
                text=True,
            ),
            "qualification": "CLOSED; candidate list is not a frozen support matrix",
        },
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for task in sum(tasks(), []):
        backend, suite, tool = task
        dest = out / "collection" / task_name(task)
        command = [
            sys.executable,
            str(ROOT / HELPERS[0]),
            backend,
            suite,
            "--output",
            str(dest),
            "--collect-only",
        ]
        if tool:
            command += ["--sanitizer", tool]
        execute(command, out / (task_name(task) + "-collection.log"), env=env)
    from benchmark_adaln_indexed import parse_args, plan

    write(out / "performance-plan.json", plan(parse_args(["--output", str(out / "performance")])))
    write(
        out / "node-lock.json",
        {
            task_name(t): json.loads((out / "collection" / task_name(t) / "nodes.json").read_text())
            for t in sum(tasks(), [])
        },
    )
    candidate = json.loads((out / "candidate.json").read_text())
    candidate["node_lock_sha256"] = sha(out / "node-lock.json")
    candidate["performance_plan_sha256"] = sha(out / "performance-plan.json")
    candidate["sessions"] = sessions
    write(out / "candidate.json", candidate)
    write(
        out / "state.json",
        {"state": "COMMAND_PREPARED", "completed": {}, "sessions": {}},
    )
    verify_candidate(out)
    print("COMMAND_PREPARED", out)


def verify_candidate(out):
    candidate = json.loads((out / "candidate.json").read_text())
    if sha(out / "source-sha256.json") != candidate["source_sha256"]:
        raise RuntimeError("source manifest identity changed")
    if sha(out / "candidate-source.tar.gz") != candidate["archive_sha256"]:
        raise RuntimeError("source archive identity changed")
    if sha(out / "node-lock.json") != candidate["node_lock_sha256"]:
        raise RuntimeError("node list identity changed")
    if sha(out / "performance-plan.json") != candidate["performance_plan_sha256"]:
        raise RuntimeError("performance plan identity changed")
    baseline = json.loads((out / "source-sha256.json").read_text())
    if inventory() != baseline or inventory(out / "candidate-source") != baseline:
        raise RuntimeError("source/harness identity changed; prepare a new candidate")


def target_identity():
    """Child query only; the coordinator never creates a CUDA context."""
    query = (
        "import torch,json,platform; "
        "assert torch.cuda.is_available() and torch.version.hip is None; "
        "import triton; "
        "print(json.dumps(dict(gpu=torch.cuda.get_device_name(), "
        "capability=torch.cuda.get_device_capability(), torch=torch.__version__, "
        "cuda=torch.version.cuda, triton=triton.__version__, python=platform.python_version(), "
        "free_gpu_bytes=torch.cuda.mem_get_info()[0])))"
    )
    result = json.loads(subprocess.check_output([sys.executable, "-c", query], text=True))
    for name, command in (
        (
            "driver",
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
        ),
        ("nvcc", ["nvcc", "--version"]),
        ("cxx", ["c++", "--version"]),
        ("sanitizer", ["compute-sanitizer", "--version"]),
    ):
        path = Path(shutil.which(command[0]) or command[0]).resolve()
        result[name] = dict(
            path=str(path),
            sha256=sha(path),
            version=subprocess.check_output(command, text=True),
        )
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    result["available_host_bytes"] = int(mem["MemAvailable"].split()[0]) * 1024
    result["executable"] = sys.executable
    result["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    import importlib.util

    spec = importlib.util.find_spec("triton")
    if spec is None or spec.origin is None:
        raise RuntimeError("Triton package identity unavailable")
    triton_root = Path(spec.origin).parent
    compiler_paths = [
        triton_root / name
        for name in (
            "backends/nvidia/compiler.py",
            "_C/libtriton.so",
            "backends/nvidia/bin/ptxas",
            "backends/nvidia/lib/libdevice.10.bc",
        )
    ]
    for name in ("ptxas", "cuobjdump"):
        tool = shutil.which(name)
        if tool is None:
            raise RuntimeError("missing " + name)
        compiler_paths.append(Path(tool).resolve())
    result["compiler_sha256"] = {str(p): sha(p) for p in compiler_paths}
    return result


def receipt(directory):
    """Bind all retained evidence, including builds and generated artifacts."""
    return {
        str(p.relative_to(directory)): sha(p)
        for p in directory.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    }


def verify_saved_evidence(out, state):
    for namespace in ("completed", "artifacts", "sessions"):
        for name, record in state.get(namespace, {}).items():
            if receipt(out / record["path"]) != record["receipt"]:
                raise RuntimeError(f"prior evidence changed: {namespace}/{name}")


def next_attempt(parent, name):
    parent.mkdir(parents=True, exist_ok=True)
    number = 0
    while True:
        target = parent / f"{name}-attempt-{number}"
        reserved = parent / f".{target.name}.reserved"
        if not target.exists() and not reserved.exists():
            # Reserve before the child can fail, even before output mkdir/import.
            reserved.touch(exist_ok=False)
            return target
        number += 1


def check_result(out, task, directory=None):
    dest = directory or out / "runs" / task_name(task)
    result = json.loads((dest / "result.json").read_text())
    if result["state"] != "PASS":
        raise RuntimeError(f"{dest}: not PASS")
    nodes = json.loads((dest / "nodes.json").read_text())
    expected = json.loads((out / "node-lock.json").read_text())[task_name(task)]
    if nodes != expected or not nodes:
        raise RuntimeError(f"{dest}: actual nodes differ from locked collection")
    baseline = json.loads((out / "source-sha256.json").read_text())
    recorded = json.loads((dest / "source-sha256.json").read_text())
    if any(baseline.get(name) != digest for name, digest in recorded.items()):
        raise RuntimeError("executed source differs from locked candidate")
    if (out / "target.json").exists():
        target = json.loads((out / "target.json").read_text())
        identity = json.loads((dest / "identity.json").read_text())
        for worker_key, target_key in (
            ("gpu", "gpu"),
            ("capability", "capability"),
            ("torch", "torch"),
            ("cuda_runtime", "cuda"),
            ("triton", "triton"),
        ):
            if identity[worker_key] != target[target_key]:
                raise RuntimeError("worker target identity mismatch: " + worker_key)
    reports = json.loads((dest / "reports.json").read_text())
    calls = {r["node"] for r in reports if r["phase"] == "call" and r["outcome"] == "passed"}
    if calls != set(nodes) or any(r["outcome"] != "passed" or r["xfail"] for r in reports):
        raise RuntimeError(f"{dest}: requires zero skip/xfail/xpass and all phases PASS")


def run(out, phase, sessions):
    state = json.loads((out / "state.json").read_text())
    try:
        verify_candidate(out)
        if sessions != json.loads((out / "candidate.json").read_text())["sessions"]:
            raise RuntimeError("session count differs from locked plan")
        identity = target_identity()
        prior = out / "target.json"
        stable = {
            k: v for k, v in identity.items() if k not in ("free_gpu_bytes", "available_host_bytes")
        }
        if prior.exists() and json.loads(prior.read_text()) != stable:
            raise RuntimeError("target/compiler/runtime identity changed; start a new batch")
        if not prior.exists():
            write(prior, stable)
        write(
            out
            / (
                "resources-"
                + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
                + ".json"
            ),
            identity,
        )
        if (
            identity["free_gpu_bytes"] < 8 * 1024**3
            or identity["available_host_bytes"] < 16 * 1024**3
        ):
            raise RuntimeError(
                "BLOCKED_RESOURCE: >=8 GiB free GPU / >=16 GiB available host required"
            )
        binary = ROOT / "rl_engine/_C.so"
        initial_binary = sha(binary) if binary.exists() else None
        # Recheck every previously accepted receipt before resuming; never splice identities.
        verify_saved_evidence(out, state)
        env = dict(os.environ, OMP_NUM_THREADS="1")
        correctness, sanitizers = tasks()
        selected = (
            correctness if phase == "correctness" else sanitizers if phase == "sanitizer" else []
        )
        if phase == "all":
            selected = correctness + sanitizers
        if phase in ("sanitizer", "performance"):
            required = correctness + (sanitizers if phase == "performance" else [])
            if any(task_name(t) not in state["completed"] for t in required):
                raise RuntimeError("prior required stages incomplete")
        for task in selected:
            name = task_name(task)
            if name in state["completed"]:
                check_result(out, task, out / state["completed"][name]["path"])
                continue
            verify_candidate(out)
            dest = next_attempt(out / "runs", name)
            backend, suite, tool = task
            command = [
                sys.executable,
                str(ROOT / HELPERS[0]),
                backend,
                suite,
                "--output",
                str(dest),
            ]
            if tool:
                command += ["--sanitizer", tool]
            print("START", name, flush=True)
            execute(command, out / (dest.name + ".log"), env=env)
            check_result(out, task, dest)
            verify_candidate(out)
            state["completed"][name] = dict(path=str(dest.relative_to(out)), receipt=receipt(dest))
            write(out / "state.json", state)
        if phase in ("all", "sanitizer", "performance"):
            extract_artifacts(out, state, env)
        if phase in ("all", "performance"):
            performance(out, state, sessions, env)
        verify_candidate(out)
        if (sha(binary) if binary.exists() else None) != initial_binary:
            raise RuntimeError("checkout _C.so changed")
        state["state"] = "EXECUTION_COMPLETE_REVIEW_PENDING"
    except (Exception, KeyboardInterrupt) as exc:
        state.update(state="FAIL", error=str(exc), traceback=traceback.format_exc())
        write(next_attempt(out / "failures", "batch"), state)
    write(out / "state.json", state)
    print(state["state"], out, flush=True)
    return int(state["state"] == "FAIL")


def extract_artifacts(out, state, env):
    """Only after every correctness and sanitizer suite. Keep raw runs immutable."""
    required: list[tuple[str, str, str | None]] = sum(tasks(), [])
    if any(task_name(t) not in state["completed"] for t in required):
        raise RuntimeError("generated-code stage requires all correctness and sanitizer suites")
    state.setdefault("artifacts", {})
    for task in required:
        name = task_name(task)
        if name in state["artifacts"]:
            continue
        verify_candidate(out)
        source = out / state["completed"][name]["path"]
        dest = next_attempt(out / "artifacts", name)
        dest.mkdir()
        backend, suite, _ = task
        if backend == "triton" and suite != "shared":
            for filename in ("identity.json", "nodes.json"):
                shutil.copyfile(source / filename, dest / filename)
            for directory in ("cache", "dump"):
                if (source / directory).exists():
                    shutil.copytree(source / directory, dest / directory)
            execute(
                [sys.executable, str(ROOT / HELPERS[1]), str(dest)], dest / "audit.log", env=env
            )
        elif backend != "triton":
            built = json.loads((source / "build.json").read_text())
            tool_path = shutil.which("cuobjdump")
            if tool_path is None:
                raise RuntimeError("missing cuobjdump")
            tool = Path(tool_path).resolve()
            write(
                dest / "extraction.json",
                dict(
                    binary=built["binary"],
                    binary_sha256=sha(Path(built["binary"])),
                    tool=str(tool),
                    tool_sha256=sha(tool),
                    scope="extraction only; arithmetic/coverage review pending",
                ),
            )
            for label, option in (
                ("version", "--version"),
                ("ptx", "--dump-ptx"),
                ("sass", "--dump-sass"),
                ("resources", "--dump-resource-usage"),
            ):
                execute(
                    [str(tool), option] + ([] if label == "version" else [built["binary"]]),
                    dest / (label + ".log"),
                    env=env,
                )
        else:
            write(
                dest / "shared-cache.json",
                dict(
                    scope="shared Triton cache in suite receipt; indexed audit not applicable",
                    suite_receipt=state["completed"][name],
                ),
            )
        state["artifacts"][name] = dict(path=str(dest.relative_to(out)), receipt=receipt(dest))
        write(out / "state.json", state)


def performance(out, state, sessions, env):
    # Fresh process without profiler/sanitizer; only reuse the pinned fresh native binary.
    from benchmark_adaln_indexed import summarize_sessions

    built = json.loads(
        (out / state["completed"]["cuda-correctness"]["path"] / "build.json").read_text()
    )
    deps = [p for p in sys.path if "site-packages" in p]
    for session in range(sessions):
        dest = next_attempt(out / "performance", str(session))
        if str(session) in state["sessions"]:
            record = state["sessions"][str(session)]
            if receipt(out / record["path"]) != record["receipt"]:
                raise RuntimeError("performance evidence changed")
            continue
        verify_candidate(out)
        package = out / "performance-source" / dest.name
        shutil.copytree(out / "candidate-source", package)
        (package / "rl_engine/_C.so").symlink_to(built["binary"])
        script = package / ".cache/adaln-indexed/benchmark_adaln_indexed.py"
        bootstrap = (
            "import sys,runpy; sys.path.extend("
            + repr(deps)
            + "); runpy.run_path("
            + repr(str(script))
            + ",run_name='__main__')"
        )
        command = [
            sys.executable,
            "-S",
            "-c",
            bootstrap,
            "--output",
            str(dest),
            "--session",
            str(session),
        ]
        execute(
            command,
            out / ("performance-" + dest.name + ".log"),
            cwd=package,
            env=dict(env, PYTHONPATH=str(package)),
        )
        report = json.loads((dest / "report.json").read_text())
        if report["state"] != "EXECUTION_COMPLETE_REVIEW_PENDING":
            raise RuntimeError("performance session failed")
        if report["environment"]["binary_sha256"] != built["binary_sha256"]:
            raise RuntimeError("performance loaded binary differs from fresh build")
        state["sessions"][str(session)] = dict(
            path=str(dest.relative_to(out)), receipt=receipt(dest)
        )
        write(out / "state.json", state)
    write(
        next_attempt(out / "summaries", "session-summary"),
        summarize_sessions(
            [
                json.loads((out / state["sessions"][str(i)]["path"] / "report.json").read_text())
                for i in range(sessions)
            ]
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="CPU collection only; no build/GPU")
    parser.add_argument(
        "--phase",
        choices=("all", "correctness", "sanitizer", "performance"),
        default="all",
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--start", action="store_true", help="prepare then run the complete batch")
    args = parser.parse_args(argv)
    if args.sessions < 2:
        parser.error("at least two independent performance sessions required")
    out = args.output.resolve()
    if args.prepare and args.start:
        parser.error("--prepare and --start are mutually exclusive")
    if args.prepare or args.start:
        was_new = not out.exists()
        try:
            prepare(out, args.sessions)
        except (Exception, KeyboardInterrupt) as exc:
            if was_new and out.exists():
                write(
                    next_attempt(out / "failures", "prepare"),
                    dict(state="FAIL", error=str(exc), traceback=traceback.format_exc()),
                )
            print("FAIL preparing:", exc, file=sys.stderr)
            return 1
        if args.prepare:
            return 0
    return run(out, args.phase, args.sessions)


if __name__ == "__main__":
    raise SystemExit(main())
