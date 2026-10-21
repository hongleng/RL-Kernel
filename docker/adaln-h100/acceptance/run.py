"""H100: pinned preflight -> fresh sm_90 build -> full tests -> sanitizers -> benchmark."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--out", required=True, help="New evidence directory outside the source snapshot"
    )
    parser.add_argument("--cuda-home", default="/usr/local/cuda-12.6")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    assert args.jobs > 0
    pack = Path(__file__).resolve().parent
    root = pack.parent
    out = Path(args.out).resolve()
    assert not out.is_relative_to(root), "Keep evidence outside the source snapshot"
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    profile = json.loads((pack / "source-profile.json").read_text())
    environment = json.loads((pack / "environment.json").read_text())
    state = dict(
        state="preflight",
        commit=profile["commit"],
        runner_pid=os.getpid(),
        stages=[],
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    def save(name, value):
        temp = out / (name + ".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temp.replace(out / name)

    def digest(path):
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def capture(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)

    def verify():
        assert all(digest(root / p) == h for p, h in profile["files"].items()), "Source drift"
        if "binary" in state:
            assert digest(state["binary"]) == state["binary_sha256"], "Binary drift"

    def execute(stage, command):
        verify()
        record = dict(
            name=stage,
            command=command,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        state.update(state="running", active_stage=stage)
        state["stages"].append(record)
        save("status.json", state)
        print(stage, "->", out / (stage + ".log"), flush=True)
        with (out / (stage + ".log")).open("w") as log:
            process = subprocess.Popen(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            record["pid"] = process.pid
            save("status.json", state)
            record["exit_code"] = process.wait()
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save("status.json", state)
        assert record["exit_code"] == 0, f"{stage} failed; later stages not started"
        verify()
        return record

    save("status.json", state)
    try:
        verify()
        env = {
            key: os.environ[key]
            for key in (
                "HOME",
                "USER",
                "LANG",
                "LD_LIBRARY_PATH",
                "LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
            )
            if key in os.environ
        }
        assert "stubs" not in env.get("LD_LIBRARY_PATH", "")
        env.update(
            PATH=f"{Path(sys.executable).parent}:{args.cuda_home}/bin:/usr/local/bin:/usr/bin:/bin",
            CUDA_HOME=args.cuda_home,
            TORCH_CUDA_ARCH_LIST="9.0",
            MAX_JOBS=str(args.jobs),
            FORCE_CUDA="1",
            RL_KERNEL_REQUIRE_EXT="1",
            KERNEL_ALIGN_USE_FAST_MATH="0",
            KERNEL_ALIGN_FORCE_SM90="0",
            KERNEL_ALIGN_DET_GEMM_SM90="0",
            KERNEL_ALIGN_FORCE_ASCEND="0",
            KERNEL_ALIGN_NCU_LINEINFO="0",
            PYTHONNOUSERSITE="1",
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            RLK_ADALN_REAL_SHAPES="1",
            RLK_REPO_ROOT=str(root),
            TRITON_CACHE_DIR=str(out / "triton-cache"),
        )
        os.environ.clear()
        os.environ.update(env)
        sys.path.insert(0, str(root))
        import torch

        assert sys.version_info[:3] == tuple(map(int, environment["python"].split("."))), (
            sys.version
        )
        assert torch.__version__ == environment["torch"], torch.__version__
        assert torch.version.cuda == environment["cuda"], torch.version.cuda
        assert importlib.metadata.version("triton") == environment["triton"]
        assert torch.cuda.device_count() == 1, "Expose one H100 using CUDA_VISIBLE_DEVICES"
        name = torch.cuda.get_device_name(0)
        assert (
            "H100" in name and "MIG" not in name and torch.cuda.get_device_capability(0) == (9, 0)
        ), name
        nvcc = capture([f"{args.cuda_home}/bin/nvcc", "--version"])
        assert "release 12.6" in nvcc, nvcc
        sanitizer = f"{args.cuda_home}/bin/compute-sanitizer"
        assert shutil.which("ninja")
        save(
            "manifest.json",
            dict(
                environment=env,
                gpu=name,
                capability=[9, 0],
                python=sys.version,
                packages=dict(
                    sorted(
                        (d.metadata["Name"], d.version)
                        for d in importlib.metadata.distributions()
                        if d.metadata["Name"]
                    )
                ),
                nvcc=nvcc,
                compiler=capture(["c++", "--version"]),
                driver=capture(["nvidia-smi"]),
                sanitizer=capture([sanitizer, "--version"]),
                profile=profile,
            ),
        )
        assert not list((root / "rl_engine").glob("_C*.so")), (
            "Use the new source snapshot; old .so found"
        )
        execute(
            "build",
            [
                sys.executable,
                "setup.py",
                "build_ext",
                "--inplace",
                "--force",
                "--build-temp",
                str(out / "objects"),
                "--build-lib",
                str(out / "lib"),
            ],
        )
        binaries = list((root / "rl_engine").glob("_C*.so"))
        assert len(binaries) == 1, binaries
        binary = binaries[0]
        ninja = (out / "objects/build.ninja").read_text()
        assert "adaln_modulation.cu" in ninja and "--use_fast_math" not in ninja
        assert "sm_90" in ninja and "sm_90a" not in ninja
        shutil.copy2(out / "objects/build.ninja", out / "build.ninja")
        if (out / "objects/.ninja_log").exists():
            shutil.copy2(out / "objects/.ninja_log", out / "ninja-timing.log")
        assert digest(binary) == digest(out / "lib/rl_engine" / binary.name)
        identity_code = (
            "import torch,rl_engine._C as c,json; "
            'assert hasattr(c,"adaln_modulation_forward") '
            'and hasattr(c,"adaln_modulation_backward"); '
            'print(json.dumps({"path":c.__file__}))'
        )
        identity = capture([sys.executable, "-c", identity_code])
        assert (
            Path(json.loads(identity.strip().splitlines()[-1])["path"]).resolve()
            == binary.resolve()
        )
        targets = capture([f"{args.cuda_home}/bin/cuobjdump", "--list-elf", str(binary)])
        assert "sm_90" in targets, targets
        (out / "binary-targets.txt").write_text(targets)
        shutil.copy2(binary, out / binary.name)
        state.update(binary=str(binary), binary_sha256=digest(binary))
        env["ADALN_EXPECTED_BINARY_SHA256"] = state["binary_sha256"]
        save(
            "identity.json",
            dict(
                binary=state["binary"], binary_sha256=state["binary_sha256"], import_output=identity
            ),
        )
        expected = profile["expected_tests"]
        for stage in ("correctness", "racecheck", "memcheck", "benchmark"):
            dest = out / stage
            command = [sys.executable, str(pack / "worker.py"), stage, str(dest)]
            if stage in ("racecheck", "memcheck"):
                prefix = [sanitizer, "--tool", stage, "--error-exitcode", "7"]
                if stage == "racecheck":
                    prefix += [
                        "--racecheck-report",
                        "all",
                        "--racecheck-detect-level",
                        "info",
                        "--racecheck-num-workers",
                        "2",
                    ]
                command = prefix + command
            record = execute(stage, command)
            if stage in expected:
                result = json.loads((dest / "results.json").read_text())
                assert (
                    result["selected"] == result["passed"] == expected[stage]
                    and not result["invalid_reports"]
                ), result["invalid_reports"]
                record["passed"] = result["passed"]
            if stage in ("racecheck", "memcheck"):
                label = "RACECHECK SUMMARY:" if stage == "racecheck" else "ERROR SUMMARY:"
                summaries = [
                    line
                    for line in (out / (stage + ".log")).read_text().splitlines()
                    if label in line
                ]
                assert summaries and all(
                    all(int(n) == 0 for n in re.findall(r"\d+", line.split(label)[1]))
                    for line in summaries
                ), summaries
                record["zero_hazards_errors"] = summaries
            if stage == "benchmark":
                assert len(json.loads((dest / "benchmark.json").read_text())) == 54
                record["benchmark_points"] = 54
            save("status.json", state)
        state["state"] = "succeeded"
    except BaseException:
        state.update(state="failed", error=traceback.format_exc())
        (out / "failure.log").write_text(state["error"])
    finally:
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save("status.json", state)
        with tarfile.open(str(out) + ".tar.gz", "w:gz") as archive:

            def evidence_only(info):
                parts = Path(info.name).parts
                return (
                    None
                    if len(parts) > 1 and parts[1] in ("objects", "lib", "triton-cache")
                    else info
                )

            archive.add(out, arcname=out.name, filter=evidence_only)
        Path(str(out) + ".tar.gz.sha256").write_text(
            f"{digest(str(out) + '.tar.gz')}  {out.name}.tar.gz\n"
        )
    print(json.dumps(state, indent=2), flush=True)
    return state["state"] != "succeeded"


if __name__ == "__main__":
    raise SystemExit(main())
