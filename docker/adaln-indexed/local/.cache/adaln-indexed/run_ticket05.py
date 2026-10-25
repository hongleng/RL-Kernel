"""User-run AdaLN acceptance. Collection never builds or runs GPU kernels.

Every invocation snapshots real source into a NEW output. CUDA/integration runs
build there; Triton L3 really has no extension. Active checkout _C.so is untouched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def files(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def selection(backend, suite):
    backend_filter = "not triton" if backend == "cuda" else "triton"
    indexed = [
        "tests/ops/norm/test_adaln_indexed.py",
        "tests/ops/norm/test_adaln_indexed_integration.py",
        "tests/runtime/test_adaln_indexed_activation.py",
    ]
    if backend == "integration":
        return [indexed[1]], "cuda_triton_indexed_small_bytes"
    if suite == "shared":
        return [
            "tests/ops/norm/test_adaln_modulation.py",
            "tests/backends/test_adaln_shared_bounds.py",
        ], (f"{backend_filter} and not real_image and not indexed")
    indexed += (
        [
            "tests/ops/norm/test_adaln_indexed_triton.py",
            "tests/runtime/test_adaln_indexed_triton.py",
        ]
        if backend == "triton"
        else ["tests/runtime/test_adaln_indexed.py"]
    )
    if suite == "large-offsets":
        return indexed, f"{backend_filter} and large_upstream"
    if suite == "target":
        return indexed, f"{backend_filter} and target_h3072"
    expr = (
        f"{backend_filter} and not large_upstream and not target_h3072 "
        "and not missing_extension_real_resolution and not cuda_triton_indexed_small_bytes"
    )
    if suite == "sanitizer":
        expr += (
            " and not launch_batch and not threads_geometry and not maximum_batch"
            " and not execution_fallback and not rechecks_support"
        )
    return indexed, expr


def build(out):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    assert torch.cuda.is_available() and torch.version.hip is None
    target = out / "build"
    target.mkdir()
    binding = Path("csrc/bindings/ops.cpp")
    blocks = re.findall(
        r"#if defined\(RLK_ADALN_CUDA_ENABLED\)\n(.*?)#endif", binding.read_text(), re.S
    )
    assert len(blocks) == 2, "review production binding structure"
    source = target / "adaln_bindings.cpp"
    source.write_text(
        "#include <torch/extension.h>\n"
        + blocks[0]
        + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        + blocks[1]
        + "\n}\n"
    )
    cuda = Path("csrc/cuda/norm/adaln_modulation.cu").resolve()
    os.environ["MAX_JOBS"] = "1"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, torch.cuda.get_device_capability()))
    cflags = ["-O2", "-std=c++17"]
    nvflags = ["-O3", "--fmad=false", "--ftz=false", "-lineinfo"]
    compiler_identity = {}
    for name, executable in {
        "cxx": os.environ.get("CXX", "c++"),
        "nvcc": str(Path(CUDA_HOME) / "bin/nvcc"),
        "ptxas": str(Path(CUDA_HOME) / "bin/ptxas"),
    }.items():
        path = Path(shutil.which(executable) or executable).resolve()
        compiler_identity[name] = {
            "path": str(path),
            "sha256": sha(path),
            "version": subprocess.check_output([str(path), "--version"], text=True),
        }
    extension = load(
        name="_C",
        sources=[str(source), str(cuda)],
        build_directory=str(target),
        extra_cflags=cflags,
        extra_cuda_cflags=nvflags,
        verbose=True,
    )
    binary = Path(extension.__file__).resolve()
    Path("rl_engine/_C.so").symlink_to(binary)
    write(
        out / "build.json",
        {
            "scope": "fresh AdaLN-only production CUDA + exact production AdaLN bindings",
            "full_extension": False,
            "compiler_identity": compiler_identity,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "nvcc_path": str(Path(CUDA_HOME) / "bin/nvcc"),
            "nvcc_sha256": sha(Path(CUDA_HOME) / "bin/nvcc"),
            "binary": str(binary),
            "binary_sha256": sha(binary),
            "sources": {str(p): sha(p) for p in (binding, source, cuda)},
            "cflags": cflags,
            "nvcc_flags": nvflags,
            "arch": os.environ["TORCH_CUDA_ARCH_LIST"],
            "build_files": files(target),
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    return 0


def worker(args, out):
    import pytest
    import torch
    import rl_engine
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    assert Path(rl_engine.__file__).resolve().parent == Path.cwd() / "rl_engine"
    assert _EXT_AVAILABLE == (args.backend != "triton" and not args.collect_only)
    if args.backend != "triton" and not args.collect_only:
        built = json.loads((out / "build.json").read_text())
        assert Path(_C.__file__).resolve() == Path(built["binary"])
        assert sha(Path(_C.__file__)) == built["binary_sha256"]
    identity = {
        "state": "COMMAND_PREPARED" if args.collect_only else "EXECUTION_STARTED",
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
        "package": rl_engine.__file__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip": torch.version.hip,
        "extension_available": _EXT_AVAILABLE,
        "loaded_binary": str(Path(_C.__file__).resolve()) if _EXT_AVAILABLE else None,
        "loaded_binary_sha256": sha(Path(_C.__file__)) if _EXT_AVAILABLE else None,
        "backend": args.backend,
        "suite": args.suite,
        "activation": "explicit IndexedAdaLNQualification via public resolver; default stays None",
        "cache": os.environ["TRITON_CACHE_DIR"],
        "dump": os.environ["TRITON_DUMP_DIR"],
        "options": {"enable_fp_fusion": False, "enable_reflect_ftz": False},
        "specialization": "B/S/H/eps/route broadcast/upstream presence/launch constexpr",
        "generated_code_review": "PENDING; no flags-only qualification",
    }
    try:
        import triton
    except ImportError:
        identity["triton"] = None
    else:
        root = Path(triton.__file__).parent
        identity["triton"] = triton.__version__
        identity["triton_path"] = str(root)
        identity["compiler_sha256"] = {
            str(p.relative_to(root)): sha(p)
            for p in (
                root / "backends/nvidia/compiler.py",
                root / "_C/libtriton.so",
                root / "language/extra/cuda/libdevice.py",
                root / "backends/nvidia/lib/libdevice.10.bc",
                root / "backends/nvidia/bin/ptxas",
            )
            if p.is_file()
        }
        ptxas = root / "backends/nvidia/bin/ptxas"
        if ptxas.is_file():
            identity["ptxas_version"] = subprocess.check_output(
                [str(ptxas), "--version"], text=True
            )
    if not args.collect_only:
        assert torch.cuda.is_available() and torch.version.hip is None
        identity.update(
            device=torch.cuda.current_device(),
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            free_gpu_bytes=torch.cuda.mem_get_info()[0],
        )
    write(out / "identity.json", identity)
    nodes, expr = selection(args.backend, args.suite)
    command = nodes + ["-q", "-rs", "-k", expr, "--junitxml=" + str(out / "results.xml")]
    if args.collect_only:
        command.append("--collect-only")
    write(out / "pytest-command.json", command)

    class Evidence:
        def __init__(self):
            self.nodes = []
            self.reports = []

        def pytest_collection_finish(self, session):
            self.nodes = [item.nodeid for item in session.items]
            write(out / "nodes.json", self.nodes)
            write(
                out / "node-details.json",
                [
                    {
                        "node": item.nodeid,
                        "parameters": {
                            key: repr(value)
                            for key, value in getattr(
                                getattr(item, "callspec", None), "params", {}
                            ).items()
                        },
                        "markers": [mark.name for mark in item.iter_markers()],
                        "source": str(item.path.relative_to(Path.cwd())),
                    }
                    for item in session.items
                ],
            )

        def pytest_runtest_logreport(self, report):
            self.reports.append(
                {
                    "node": report.nodeid,
                    "phase": report.when,
                    "outcome": report.outcome,
                    "xfail": hasattr(report, "wasxfail"),
                }
            )

        def pytest_sessionfinish(self, session, exitstatus):
            calls = {
                r["node"]
                for r in self.reports
                if r["phase"] == "call" and r["outcome"] == "passed" and not r["xfail"]
            }
            invalid = any(r["outcome"] != "passed" or r["xfail"] for r in self.reports)
            if not self.nodes or (not args.collect_only and (invalid or calls != set(self.nodes))):
                session.exitstatus = 1
            write(out / "reports.json", self.reports)

    result = int(pytest.main(command, plugins=[Evidence()]))
    identity["pytest_exit"] = result
    identity["generated_artifacts"] = {
        name: files(out / name) if (out / name).exists() else {} for name in ("cache", "dump")
    }
    # Test PASS remains separate from compiler/artifact/sanitizer acceptance.
    identity["state"] = (
        "COMMAND_PREPARED" if args.collect_only else ("PASS" if result == 0 else "FAIL")
    )
    write(out / "identity.json", identity)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("backend", choices=["cuda", "triton", "integration"])
    parser.add_argument(
        "suite", choices=["correctness", "sanitizer", "large-offsets", "target", "shared"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--sanitizer", choices=["memcheck", "racecheck", "initcheck", "synccheck"])
    parser.add_argument("--stage", choices=["build", "test"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.backend == "integration" and args.suite != "correctness":
        parser.error("integration uses correctness suite")
    if args.sanitizer and args.suite != "sanitizer":
        parser.error("--sanitizer requires sanitizer suite")
    out = args.output.resolve()
    if args.stage:
        return build(out) if args.stage == "build" else worker(args, out)
    checkout = Path.cwd().resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    for name in ("rl_engine", "tests", "csrc"):
        shutil.copytree(
            checkout / name,
            source / name,
            ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"),
        )
    shutil.copy2(checkout / "pyproject.toml", source / "pyproject.toml")
    runner = out / "run_ticket05.py"
    shutil.copy2(__file__, runner)
    write(
        out / "start.json",
        {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "status": subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=all"], text=True
            ),
            "checkout": str(checkout),
            "runner_sha256": sha(runner),
            "backend": args.backend,
            "suite": args.suite,
        },
    )
    (out / "tracked.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    write(out / "source-sha256.json", files(source))
    env = dict(
        os.environ,
        PYTHONPATH=str(source),
        RLK_ADALN_TRITON_ISOLATED="1",
        TRITON_CACHE_DIR=str(out / "cache"),
        TRITON_DUMP_DIR=str(out / "dump"),
        TRITON_KERNEL_DUMP="1",
        TRITON_STORE_BINARY_ONLY="0",
        TRITON_KERNEL_OVERRIDE="0",
        OMP_NUM_THREADS="1",
        RLK_ADALN_REAL_SHAPES="1" if args.suite == "target" else "0",
        RLK_ADALN_LARGE_OFFSETS="1" if args.suite == "large-offsets" else "0",
    )
    env.pop("RL_KERNEL_REQUIRE_EXT", None)
    if args.collect_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    deps = [p for p in sys.path if "site-packages" in p]
    bootstrap = (
        "import sys,runpy; sys.path.extend("
        + repr(deps)
        + "); runpy.run_path("
        + repr(str(runner))
        + ",run_name='__main__')"
    )
    base = [sys.executable, "-S", "-c", bootstrap, args.backend, args.suite, "--output", str(out)]
    if args.collect_only:
        base += ["--collect-only"]
    if not args.collect_only and args.backend != "triton":
        with (out / "build.log").open("w") as log:
            result = subprocess.run(
                base + ["--stage", "build"],
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            write(out / "result.json", {"state": "FAIL", "build_exit": result.returncode})
            return result.returncode
    versions = {}
    if not args.collect_only:
        for name, command in {
            "nvcc": ["nvcc", "--version"],
            "cxx": ["c++", "--version"],
            "driver": [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,name",
                "--format=csv,noheader",
            ],
            "sanitizer": ["compute-sanitizer", "--version"],
        }.items():
            try:
                r = subprocess.run(command, capture_output=True, text=True)
                versions[name] = {"exit": r.returncode, "output": r.stdout + r.stderr}
            except FileNotFoundError:
                versions[name] = {"state": "UNAVAILABLE"}
    write(out / "tools.json", versions)
    command = base + ["--stage", "test"]
    if args.sanitizer and not args.collect_only:
        command = [
            "compute-sanitizer",
            "--tool",
            args.sanitizer,
            "--target-processes",
            "all",
            "--error-exitcode",
            "99",
            "--log-file",
            str(out / "sanitizer.log"),
        ] + command
    write(out / "process-command.json", command)
    with (out / "console.log").open("w") as log:
        try:
            result = subprocess.run(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            write(out / "result.json", {"state": "FAIL", "launch_error": str(exc)})
            return 1
    state = "COMMAND_PREPARED" if args.collect_only and result.returncode == 0 else "FAIL"
    sanitizer_errors = None
    if not args.collect_only and result.returncode == 0:
        identity = json.loads((out / "identity.json").read_text())
        if identity["state"] == "PASS":
            state = "PASS"
        if args.sanitizer:
            log_path = out / "sanitizer.log"
            content = log_path.read_text() if log_path.is_file() else ""
            summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", content)
            hazards = re.findall(r"RACECHECK SUMMARY: (\d+) hazards", content)
            sanitizer_errors = {"errors": summaries, "hazards": hazards}
            warnings = re.findall(r"(\d+) warnings", content)
            sanitizer_errors["warnings"] = warnings
            if not (summaries or hazards) or any(int(n) for n in summaries + hazards + warnings):
                state = "FAIL"
    write(
        out / "result.json",
        {
            "state": state,
            "process_exit": result.returncode,
            "sanitizer": args.sanitizer,
            "sanitizer_errors": sanitizer_errors,
            "actual_scope": "nodes.json and reports.json",
            "qualification": "NOT OPENED; generated code and evidence review still required",
        },
    )
    print(f"{state}: {out}")
    return 0 if state in {"PASS", "COMMAND_PREPARED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
