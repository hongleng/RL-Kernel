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
import time
import traceback
from zoneinfo import ZoneInfo

import storage

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
    ".cache/adaln-indexed/evidence_io.py",
    ".cache/adaln-indexed/test_acceptance_reuse.py",
    ".cache/adaln-indexed/test_publish_preflight.py",
    ".cache/adaln-indexed/test_disk_pressure.py",
    ".cache/adaln-indexed/storage.py",
    ".cache/adaln-indexed/test_storage.py",
    ".cache/adaln-indexed/acceptance-pr-v2.md",
    ".cache/adaln-indexed/spec.md",
    ".cache/adaln-indexed/issues/06-target-qualification-and-strict.md",
    ".cache/adaln-indexed/issues/07-indexed-benchmark.md",
)
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def record_failure(out, name, value):
    try:
        write(next_attempt(out / "failures", name), value)
    except OSError as exc:
        print(f"Cannot persist failure record: {exc}", file=sys.stderr, flush=True)


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
        started = time.monotonic()
        status = "FAIL"
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        try:
            if env and "RLK_ADALN_DISK_FLOOR_BYTES" in env:
                storage.supervised_run(command, cwd=cwd, env=env, stdout=stream)
            else:
                subprocess.run(
                    command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
            status = "COMPLETE"
        finally:
            elapsed = time.monotonic() - started
            try:
                write(
                    log.with_name(log.name + ".timing.json"),
                    dict(state=status, started=timestamp, elapsed_seconds=elapsed, command=command),
                )
            except OSError as exc:
                print(f"Cannot persist timing for {log.name}: {exc}", file=sys.stderr, flush=True)
                if status == "COMPLETE":
                    raise
            print(status, log.name, f"{elapsed:.1f}s", flush=True)


def isolated_command(package, code):
    deps = [p for p in sys.path if "site-packages" in p]
    paths = [str(package), str(package / ".cache/adaln-indexed")]
    bootstrap = (
        "import sys; sys.path[:0] = "
        + repr(paths)
        + "; sys.path.extend("
        + repr(deps)
        + "); "
        + code
    )
    return [sys.executable, "-S", "-c", bootstrap]


def cpu_preflight(package, dest):
    probe = (
        "import json; from pathlib import Path; import rl_engine; "
        "from rl_engine.backends.extension import _C, _EXT_AVAILABLE; "
        "print(json.dumps(dict(package=rl_engine.__file__, extension_available=_EXT_AVAILABLE, "
        "loaded_binary=getattr(_C,'__file__',None))),flush=True); "
        "assert Path(rl_engine.__file__).resolve().parent == Path("
        + repr(str(package / "rl_engine"))
        + "), 'CPU preflight imported another checkout'; "
        "assert not _EXT_AVAILABLE, 'CPU preflight requires an extension-free snapshot'; "
        "import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    )
    names = (
        ".cache/adaln-indexed/test_acceptance_reuse.py",
        ".cache/adaln-indexed/test_ticket0607.py",
        ".cache/adaln-indexed/test_benchmark_adaln_indexed.py",
        ".cache/adaln-indexed/test_publish_preflight.py",
        ".cache/adaln-indexed/test_disk_pressure.py",
        ".cache/adaln-indexed/test_storage.py",
        "tests/reference/test_adaln_indexed.py",
        "tests/runtime/test_adaln_strict_policy.py",
    )
    native_selection = (
        "not explicit_indexed_activation_resolves_without_enabling_cpu_select01 "
        "and not activated_strict_rechecks_actual_domain_before_execution "
        "and not ordinary_gpu_resolution_rechecks_shared_cpu_device_with_real_fallback"
    )
    execute(
        isolated_command(package, probe)
        + [
            *[str(package / n) for n in names],
            "-q",
            "-k",
            native_selection,
            "--junitxml=" + str(dest / "cpu.xml"),
        ],
        dest / "cpu.log",
        cwd=package,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
    )


def prepare(out, sessions=1, profile="pr-v2"):
    out.mkdir(parents=True, exist_ok=False)
    write(out / "candidate.json", dict(state="PREPARING", output_path=str(out)))
    write(out / "state.json", dict(state="PREPARING", completed={}, sessions={}))
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
            "output_path": str(out),
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
    collected = {}
    for task in sum(tasks(), []):
        backend, suite, tool = task
        dest = out / "collection" / task_name(task)
        selector = (backend, suite)
        if selector in collected:
            original = collected[selector]
            dest.mkdir(parents=True)
            for name in ("nodes.json", "node-details.json"):
                shutil.copyfile(original / name, dest / name)
            write(
                dest / "collection-reuse.json",
                dict(
                    source=str(original.relative_to(out)),
                    selector=list(selector),
                    nodes_sha256=sha(dest / "nodes.json"),
                    scope="collection only; every suite still executes independently",
                ),
            )
            continue
        collected[selector] = dest
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

    write(
        out / "performance-plan.json",
        plan(parse_args(["--output", str(out / "performance"), "--profile", profile])),
    )
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
    write(
        out / "storage-plan.json",
        storage.budget(json.loads((out / "performance-plan.json").read_text())),
    )
    candidate["storage_plan_sha256"] = sha(out / "storage-plan.json")
    candidate["sessions"] = sessions
    candidate["profile"] = profile
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
    if sha(out / "storage-plan.json") != candidate["storage_plan_sha256"]:
        raise RuntimeError("storage plan identity changed")
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
    for record in state.get("storage_checks", []):
        if sha(out / record["path"]) != record["sha256"]:
            raise RuntimeError("prior storage check changed")
    for namespace in ("completed", "artifacts", "sessions", "preflight"):
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
    from evidence_io import verify_reports

    verify_reports(nodes, reports)


def record_storage(out, state, label, plan):
    check = storage.check_batch(out, plan)
    path = next_attempt(out / "storage-checks", label)
    write(path, check)
    state.setdefault("storage_checks", []).append(
        dict(path=str(path.relative_to(out)), sha256=sha(path))
    )
    return check


def run(out, phase, sessions):
    state = json.loads((out / "state.json").read_text())
    try:
        verify_candidate(out)
        if sessions != json.loads((out / "candidate.json").read_text())["sessions"]:
            raise RuntimeError("session count differs from locked plan")
        storage_plan = json.loads((out / "storage-plan.json").read_text())
        verify_saved_evidence(out, state)
        storage_state = record_storage(out, state, "startup", storage_plan)
        if not state.get("preflight"):
            dest = next_attempt(out / "preflight", "cpu")
            dest.mkdir()
            write(out / "progress.json", dict(stage="CPU preflight", attempt=dest.name))
            print("START CPU preflight", flush=True)
            cpu_preflight(out / "candidate-source", dest)
            state["preflight"] = {
                "cpu": dict(path=str(dest.relative_to(out)), receipt=receipt(dest))
            }
            write(out / "state.json", state)
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
        state.update(state="RUNNING", error=None, traceback=None)
        write(out / "state.json", state)
        scratch = out / "runtime-tmp"
        scratch.mkdir(exist_ok=True)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            TMPDIR=str(scratch),
            CUDA_CACHE_PATH=str(out / "driver-cache"),
            RLK_ADALN_STORAGE_ROOT=str(out),
            RLK_ADALN_DISK_FLOOR_BYTES=str(storage_state["disk_floor_bytes"]),
        )
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
            storage_state = record_storage(out, state, name, storage_plan)
            env["RLK_ADALN_DISK_FLOOR_BYTES"] = str(storage_state["disk_floor_bytes"])
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
            if backend != "triton" and "cuda-correctness" in state["completed"]:
                shared = out / state["completed"]["cuda-correctness"]["path"] / "build.json"
                command += ["--reuse-build", str(shared)]
            if tool:
                command += ["--sanitizer", tool]
            write(out / "progress.json", dict(stage="suite", task=name, attempt=dest.name))
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
        record_storage(out, state, "completion", storage_plan)
        if (sha(binary) if binary.exists() else None) != initial_binary:
            raise RuntimeError("checkout _C.so changed")
        required = {task_name(t) for t in sum(tasks(), [])}
        complete = (
            set(state["completed"]) == required
            and set(state.get("artifacts", {})) == required
            and set(state["sessions"]) == {str(i) for i in range(sessions)}
        )
        state["state"] = "EXECUTION_COMPLETE_REVIEW_PENDING" if complete else "PHASE_COMPLETE"
        write(out / "progress.json", dict(stage=state["state"]))
    except (Exception, KeyboardInterrupt) as exc:
        state.update(state="FAIL", error=str(exc), traceback=traceback.format_exc())
        summary = (
            f"child process exited {exc.returncode}; see the stage log"
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        print("BATCH FAILED:", summary, file=sys.stderr, flush=True)
        record_failure(out, "batch", state)
    try:
        write(out / "state.json", state)
    except OSError as exc:
        state["state"] = "FAIL"
        print(
            f"Cannot persist state; last saved checkpoint retained: {exc}",
            file=sys.stderr,
            flush=True,
        )
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
        storage_state = record_storage(
            out, state, "stage", json.loads((out / "storage-plan.json").read_text())
        )
        env["RLK_ADALN_DISK_FLOOR_BYTES"] = str(storage_state["disk_floor_bytes"])
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
            extract_native(out, state, built, dest, tool, env)
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


NATIVE_EXTRACTION_OPTIONS = ("--version", "--dump-ptx", "--dump-sass", "--dump-resource-usage")


def extract_native(out, state, built, dest, tool, env):
    from evidence_io import verify_native_extraction

    if sha(Path(built["binary"])) != built["binary_sha256"]:
        raise RuntimeError("native binary hash mismatch before extraction")
    identity = dict(
        binary_sha256=built["binary_sha256"],
        tool_sha256=sha(tool),
        options=list(NATIVE_EXTRACTION_OPTIONS),
    )
    for record in state["artifacts"].values():
        metadata = out / record["path"] / "extraction.json"
        if not metadata.exists():
            continue
        prior = json.loads(metadata.read_text())
        if prior["identity"] == identity:
            verify_native_extraction(out, prior)
            write(dest / "extraction.json", dict(prior, reused_from=record["path"]))
            return
    for label, option in zip(
        ("version", "ptx", "sass", "resources"), NATIVE_EXTRACTION_OPTIONS, strict=True
    ):
        execute(
            [str(tool), option] + ([] if label == "version" else [built["binary"]]),
            dest / (label + ".log"),
            env=env,
        )
    metadata = dict(
        identity=identity,
        binary=built["binary"],
        binary_sha256=built["binary_sha256"],
        tool=str(tool),
        tool_sha256=sha(tool),
        output_path=str(dest.relative_to(out)),
        files_sha256={
            name: sha(dest / name)
            for name in ("version.log", "ptx.log", "sass.log", "resources.log")
        },
        scope="extraction only; independent per-suite execution and instruction review required",
    )
    verify_native_extraction(out, metadata)
    write(dest / "extraction.json", metadata)


def performance(out, state, sessions, env):
    # Fresh process without profiler/sanitizer; only reuse the pinned fresh native binary.
    from benchmark_adaln_indexed import summarize_sessions

    built = json.loads(
        (out / state["completed"]["cuda-correctness"]["path"] / "build.json").read_text()
    )
    for session in range(sessions):
        if str(session) in state["sessions"]:
            record = state["sessions"][str(session)]
            if receipt(out / record["path"]) != record["receipt"]:
                raise RuntimeError("performance evidence changed")
            continue
        storage_state = record_storage(
            out, state, "stage", json.loads((out / "storage-plan.json").read_text())
        )
        env["RLK_ADALN_DISK_FLOOR_BYTES"] = str(storage_state["disk_floor_bytes"])
        dest = next_attempt(out / "performance", str(session))
        verify_candidate(out)
        package = out / "performance-source" / dest.name
        shutil.copytree(out / "candidate-source", package)
        (package / "rl_engine/_C.so").symlink_to(built["binary"])
        script = package / ".cache/adaln-indexed/benchmark_adaln_indexed.py"
        command = isolated_command(
            package, "import runpy; runpy.run_path(" + repr(str(script)) + ",run_name='__main__')"
        ) + ["--output", str(dest), "--session", str(session)]
        locked = json.loads((out / "performance-plan.json").read_text())
        profile = json.loads((out / "candidate.json").read_text())["profile"]
        command += [
            "--profile",
            profile,
            "--warmup",
            str(locked["warmup"]),
            "--repeat",
            str(locked["repeat"]),
            "--seed",
            str(locked["seed"]),
            "--oracle-cache",
            str(out / "oracles"),
        ]
        write(out / "progress.json", dict(stage="performance", session=session, attempt=dest.name))
        print("START performance", session, dest.name, flush=True)
        execute(
            command,
            out / ("performance-" + dest.name + ".log"),
            cwd=package,
            env=dict(env, PYTHONPATH=str(package)),
        )
        from evidence_io import verify_performance

        report = verify_performance(dest, locked, session, built["binary_sha256"])
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
        "--cpu-preflight",
        action="store_true",
        help="isolated CPU checks of an extension-free snapshot; no collection/build/GPU",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "correctness", "sanitizer", "performance"),
        default="all",
    )
    parser.add_argument("--profile", choices=("pr-v2",), default="pr-v2")
    parser.add_argument("--sessions", type=int, choices=(1,), default=1)
    parser.add_argument("--start", action="store_true", help="prepare then run the complete batch")
    parser.add_argument(
        "--storage-plan",
        action="store_true",
        help="print CPU-only capacity budget; no output creation/GPU",
    )
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if sum((args.prepare, args.start, args.cpu_preflight, args.storage_plan)) > 1:
        parser.error(
            "--prepare, --start, --cpu-preflight and --storage-plan are mutually exclusive"
        )
    if args.storage_plan or args.start:
        from benchmark_adaln_indexed import parse_args, plan

        capacity = storage.budget(
            plan(parse_args(["--output", str(out), "--profile", args.profile]))
        )
        if args.storage_plan:
            summary = dict(
                oracle_payloads=len(capacity["oracle_cases"]),
                oracle_limit_gib=round(capacity["oracle_limit_bytes"] / storage.GIB, 3),
                other_artifacts_limit_gib=capacity["other_artifacts_limit_bytes"] / storage.GIB,
                archive_reserve_gib=round(capacity["archive_limit_bytes"] / storage.GIB, 3),
                unpack_reserve_gib=round(capacity["unpack_limit_bytes"] / storage.GIB, 3),
                required_empty_volume_free_gib=round(
                    capacity["required_empty_volume_free_bytes"] / storage.GIB, 3
                ),
                recommended_volume_gb=capacity["recommended_volume_gb"],
            )
            print(json.dumps(summary, indent=2))
            return 0
        # Fail before collection, preflight or any GPU query if storage is unsafe.
        try:
            storage.check_batch(out, capacity)
        except (OSError, storage.StoragePressure) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.cpu_preflight:
        out.mkdir(parents=True, exist_ok=False)
        cpu_preflight(ROOT, out)
        return 0
    if args.prepare or args.start:
        was_new = not out.exists()
        try:
            prepare(out, args.sessions, args.profile)
        except (Exception, KeyboardInterrupt) as exc:
            if was_new and out.exists():
                record_failure(
                    out,
                    "prepare",
                    dict(state="FAIL", error=str(exc), traceback=traceback.format_exc()),
                )
                state_path = out / "state.json"
                try:
                    saved = json.loads(state_path.read_text()) if state_path.exists() else {}
                    saved.update(state="FAIL", error=str(exc))
                    write(state_path, saved)
                except OSError as secondary:
                    print(f"Cannot persist preparation state: {secondary}", file=sys.stderr)
            print("FAIL preparing:", exc, file=sys.stderr)
            return 1
        if args.prepare:
            return 0
    from evidence_io import exclusive_output

    with exclusive_output(out):
        return run(out, args.phase, args.sessions)


if __name__ == "__main__":
    raise SystemExit(main())
