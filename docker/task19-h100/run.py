#!/usr/bin/env python3
"""PR #707: verify a prebuilt image binary, run H100 checks, retain evidence."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

HEAD = "4c9ee6dcce483cf99a8385ef0a4da63fd3ed6b11"
BASE = "e9d52e95226b62d826f0a224bafad2cab3ed7538"
BASE_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "9859f8978cdfad549d72baa41d0b0bb7a5b46210a1446e09bf32600a968badb8"
)
BUILD = Path("/opt/task19-image-build")
GPU_PLAN = Path(__file__).with_name("gpu-plan.json")
SELECTION = [
    "tests/ops/test_op_accuracy.py",
    "tests/ops/logprob/test_deterministic_logp.py",
    "tests/ops/logprob/test_deterministic_logp_autograd.py",
    "tests/backends/cuda/test_fused_logp_chunked_backward.py",
    "tests/e2e/test_rl_kernel_loss_step.py::test_minimal_rl_loss_step_fused_logp_candidate_cuda",
    "-k",
    "not native and not sm90 and not source_locks "
    "and not long_sequence_backward_workspace and not guards_tensor_device",
]
VARIANTS = {
    "apply_fp32",
    "out",
    "indexed_out",
    "indexed_fp32",
    "online_out",
    "online_fp32",
    "online_indexed_out",
    "online_indexed_fp32",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def check_source(repo):
    require(git(repo, "rev-parse", "HEAD") == HEAD, "checkout does not match PR #707 pin")
    dirty = git(repo, "status", "--porcelain")
    require(not dirty, "checkout must be clean:\n" + dirty)
    git(repo, "cat-file", "-e", BASE + "^{commit}")
    require(
        not git(repo, "diff", BASE, HEAD, "--", "csrc", "build_tools", "setup.py"),
        "baseline and candidate native build sources differ",
    )


def environment(repo):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("KERNEL_ALIGN_", "FUSED_LOGP_", "DETERMINISTIC_LOGP_", "RL_KERNEL_")):
            del env[name]
    env.update(
        PYTHONPATH=str(repo),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCH_CUDA_ARCH_LIST="9.0",
        FORCE_CUDA="1",
        KERNEL_ALIGN_FORCE_SM90="0",
        KERNEL_ALIGN_USE_FAST_MATH="0",
        RL_KERNEL_REQUIRE_EXT="1",
        MAX_JOBS="4",
    )
    return env


def check_prebuilt(repo, build, torch):
    manifest = json.loads((build / "build.json").read_text())
    require(
        manifest["head"] == HEAD and manifest["baseline"] == BASE,
        "build source mismatch",
    )
    require(manifest["base_image"] == BASE_IMAGE, "build base image mismatch")
    require(manifest["gpu_plan_sha256"] == digest(GPU_PLAN), "build test selection mismatch")
    require(
        manifest["torch"] == str(torch.__version__)
        and manifest["cuda"] == torch.version.cuda
        and manifest["python_abi"] == __import__("sysconfig").get_config_var("SOABI")
        and manifest["torch_cxx11_abi"] == torch._C._GLIBCXX_USE_CXX11_ABI,
        "prebuilt binary runtime/ABI mismatch",
    )
    for name, checksum in manifest["files"].items():
        require(digest(repo / name) == checksum, "build source changed: " + name)
    require(
        manifest["runner_sha256"] == digest(Path(__file__)),
        "runner differs from image build",
    )
    require(
        manifest["builder_sha256"] == digest(Path(__file__).with_name("build_image.py"))
        and manifest["dockerfile_sha256"] == digest(Path(__file__).with_name("Dockerfile")),
        "build adapter or Dockerfile differs from image build",
    )
    binaries = list((repo / "rl_engine").glob("_C*.so"))
    require(len(binaries) == 1, "need exactly one prebuilt binary; no runtime compilation")
    require(
        digest(binaries[0]) == manifest["extension_sha256"],
        "prebuilt binary hash mismatch",
    )
    require(digest(build / "build.log") == manifest["build_log_sha256"], "build log changed")
    return manifest, binaries[0]


def command(repo, env, args, log=None):
    print("[task19]", " ".join(map(str, args)), flush=True)
    if log is None:
        subprocess.run(args, cwd=repo, env=env, check=True)
    else:
        print("[task19] log:", log, flush=True)
        with log.open("w") as output:
            subprocess.run(
                args,
                cwd=repo,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )


def load_plan():
    plan = json.loads(GPU_PLAN.read_text())
    require(
        plan["head"] == HEAD and plan["baseline"] == BASE and plan["selection"] == SELECTION,
        "test plan source/selection mismatch",
    )
    require(
        plan["nodeids"] and len(plan["nodeids"]) == len(set(plan["nodeids"])),
        "test plan must contain unique nodes",
    )
    return plan


def collect(repo, env):
    args = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        *SELECTION,
    ]
    result = subprocess.check_output(args, cwd=repo, env=env, text=True)
    nodes = [line for line in result.splitlines() if line.startswith("tests/") and "::" in line]
    require(sorted(nodes) == load_plan()["nodeids"], "collection differs from locked test nodes")
    print(result.splitlines()[-1], flush=True)
    return nodes


def check_reports(out, binary_hash):
    status = json.loads((out / "status.json").read_text())
    require(status["state"] == "PASS" and status["head"] == HEAD, "batch is not PASS at pin")
    cases = ET.parse(out / "gpu.xml").getroot().findall(".//testcase")
    plan = load_plan()
    require(json.loads((out / "gpu-plan.json").read_text()) == plan, "evidence test plan mismatch")
    observed_nodes = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(sorted(observed_nodes) == plan["nodeids"], "GPU suite does not match locked test nodes")
    require(
        all(
            not list(case) or all(c.tag not in {"skipped", "failure", "error"} for c in case)
            for case in cases
        ),
        "GPU suite contains skipped/failed/error cases",
    )
    report = json.loads((out / "comparison.json").read_text())
    require(
        report["backward"] == "native fused_logp_backward; FP32 upstream, input-dtype gradient",
        "comparison did not establish merged #495 native backward",
    )
    require(
        report["head"] == HEAD and report["baseline"] == BASE and not report["dirty"],
        "comparison source identity mismatch",
    )
    expected = {
        (dtype, shape, strided)
        for dtype in ("torch.float32", "torch.float16", "torch.bfloat16")
        for shape in ((6, 257), (2, 3, 4097), (1, 2, 151936))
        for strided in (False, True)
    }
    observed = {(c["dtype"], tuple(c["shape"]), c["strided"]) for c in report["cases"]}
    require(len(report["cases"]) == 18 and observed == expected, "18-case coverage mismatch")
    require(
        all(set(c["variant_sha256"]) == VARIANTS for c in report["cases"]),
        "missing native variant comparison",
    )
    hashes = [
        value
        for case in report["cases"]
        for value in (
            case["output_sha256"],
            case["gradient_sha256"],
            *case["variant_sha256"].values(),
        )
    ]
    require(
        all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes),
        "missing/invalid comparison tensor hashes",
    )
    source = json.loads((out / "source.json").read_text())
    require(
        set(report["python_source_sha256"])
        == {
            "rl_engine/backends/cuda/logprob/logp.py",
            "rl_engine/backends/musa/logprob/logp.py",
            "rl_engine/ops/logprob/fused_logp.py",
        },
        "missing shared/backend source fingerprints",
    )
    require(
        all(
            source["files"].get(name) == value
            for name, value in report["python_source_sha256"].items()
        ),
        "comparison source hashes differ from source manifest",
    )
    require(
        report["runner_sha256"]
        == source["files"].get("tools/validation/operators/compare_fused_logp_ownership.py"),
        "comparison runner hash differs from source manifest",
    )
    actual_env = json.loads((out / "environment.json").read_text())
    build = json.loads((out / "build.json").read_text())
    require(
        build["head"] == HEAD
        and build["baseline"] == BASE
        and build["base_image"] == BASE_IMAGE
        and build["extension_sha256"] == binary_hash
        and build["build_log_sha256"] == digest(out / "build.log")
        and build["runner_sha256"] == digest(out / "runner.py")
        and build["builder_sha256"] == digest(out / "builder.py")
        and build["dockerfile_sha256"] == digest(out / "Dockerfile")
        and build["gpu_plan_sha256"] == digest(out / "gpu-plan.json")
        and build["files"] == source["files"],
        "image build provenance mismatch",
    )
    require(
        status["image"] == actual_env["image"]
        and re.fullmatch(r"ghcr\.io/[^@\s]+@sha256:[0-9a-f]{64}", status["image"]),
        "missing published task19 image digest",
    )
    require(
        actual_env["cuda"] == report["cuda"] == "12.4"
        and actual_env["torch"] == report["torch"]
        and report["torch"].split("+")[0] == "2.4.1"
        and actual_env["gpu"] == report["gpu"]
        and "H100" in report["gpu"],
        "GPU/runtime report identity mismatch",
    )
    require(binary_hash == report["extension_sha256"], "native binary mismatch")
    fallback = json.loads((out / "sm90-fallback.json").read_text())
    require(
        fallback
        == {
            "state": "PASS",
            "experimental_tma_executed": False,
            "generic_native_forward": True,
            "native_fused_backward": True,
        },
        "SM90 wrapper fallback evidence missing",
    )
    return status


def checksums(out, binary_hash=None):
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(
            f"{binary_hash if p.name == 'native.so' and binary_hash else digest(p)}  {p.name}\n"
            for p in files
        )
    )


def verify(out):
    entries = (out / "SHA256SUMS").read_text().splitlines()
    names = set()
    actual_hashes = {}
    for entry in entries:
        checksum, name = entry.split("  ", 1)
        require(
            Path(name).name == name and name not in names,
            "unsafe/duplicate evidence name",
        )
        path = out / name
        require(not path.is_symlink() and path.is_file(), "missing/linked evidence: " + name)
        actual_hashes[name] = digest(path)
        require(actual_hashes[name] == checksum, "checksum mismatch: " + name)
        names.add(name)
    required = {
        "status.json",
        "environment.json",
        "source.json",
        "runner.py",
        "native.so",
        "build.log",
        "build.json",
        "builder.py",
        "Dockerfile",
        "gpu.log",
        "gpu.xml",
        "gpu-plan.json",
        "comparison.log",
        "comparison.json",
        "sm90-fallback.json",
        "nvidia-smi.txt",
    }
    require(required <= names, "missing required evidence members")
    source = json.loads((out / "source.json").read_text())
    require(
        source["head"] == HEAD and source["baseline"] == BASE,
        "source manifest identity mismatch",
    )
    check_reports(out, actual_hashes["native.so"])
    print(
        f"PASS: evidence checksums, source identity, {len(load_plan()['nodeids'])} tests, "
        "18 native-backward comparisons and SM90 fallback"
    )


def sm90_fallback(repo, out):
    from unittest.mock import patch

    import torch

    from rl_engine.backends.cuda.logprob.logp import FusedLogpSM90Op
    from rl_engine.backends.extension import _C

    # Bypass only the unused TMA-symbol constructor check. The real fallback
    # constructs FusedLogpGenericOp and executes this checkout's native extension.
    op = object.__new__(FusedLogpSM90Op)
    op._fallback = None
    torch.manual_seed(19)
    x = torch.randn(6, 257, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.randint(257, (6,), device="cuda")
    upstream = torch.linspace(-1, 1, 6, device="cuda", dtype=x.dtype)
    with patch.object(_C, "fused_logp_backward", wraps=_C.fused_logp_backward) as backward:
        value = op(x, y)
        grad = torch.autograd.grad(value, x, upstream)[0]
        require(backward.call_count == 1, "SM90 fallback did not execute native fused backward")
    ref = x.detach().float().requires_grad_()
    expected = ref.log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    expected_grad = torch.autograd.grad(expected, ref, upstream.float())[0]
    torch.testing.assert_close(value.float(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(grad, expected_grad.to(x.dtype), rtol=0.02, atol=0.002)
    require(
        type(op._fallback).__name__ == "FusedLogpGenericOp",
        "SM90 fallback is not generic",
    )
    save(
        out / "sm90-fallback.json",
        dict(
            state="PASS",
            experimental_tma_executed=False,
            generic_native_forward=True,
            native_fused_backward=True,
        ),
    )


def run(repo, out, env, image_ref):
    require(
        re.fullmatch(r"ghcr\.io/[^@\s]+@sha256:[0-9a-f]{64}", image_ref),
        "--image must be the published task19 GHCR image digest",
    )
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "runner.py")
    started = time.time()
    binary_hash = None
    state = dict(state="FAIL", head=HEAD, baseline=BASE, image=image_ref, started=started)
    try:
        import importlib.metadata
        import platform
        import sysconfig

        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        require(
            str(torch.__version__).split("+")[0] == "2.4.1" and torch.version.cuda == "12.4",
            "expected official torch 2.4.1 / CUDA 12.4 image",
        )
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "need one visible GPU",
        )
        gpu = torch.cuda.get_device_properties(0)
        require(
            "H100" in gpu.name
            and "MIG" not in gpu.name
            and torch.cuda.get_device_capability() == (9, 0)
            and gpu.total_memory >= 70 * 1024**3,
            "need a full H100 80GB, not MIG",
        )
        require(CUDA_HOME is not None, "CUDA development toolkit is missing")
        env["CUDA_HOME"] = CUDA_HOME
        manifest, binary = check_prebuilt(repo, BUILD, torch)
        shutil.copyfile(BUILD / "build.json", out / "build.json")
        shutil.copyfile(BUILD / "build.log", out / "build.log")
        shutil.copyfile(Path(__file__).with_name("build_image.py"), out / "builder.py")
        shutil.copyfile(Path(__file__).with_name("Dockerfile"), out / "Dockerfile")
        shutil.copyfile(GPU_PLAN, out / "gpu-plan.json")
        save(
            out / "source.json",
            dict(head=HEAD, baseline=BASE, files=manifest["files"]),
        )
        save(
            out / "environment.json",
            dict(
                image=image_ref,
                python=platform.python_version(),
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                gpu=gpu.name,
                memory=gpu.total_memory,
                python_abi=sysconfig.get_config_var("SOABI"),
                torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                packages={
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pytest", "ninja", "tabulate", "setuptools")
                },
                nvcc=subprocess.check_output(
                    [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
                ),
                cxx=subprocess.check_output([env.get("CXX", "c++"), "--version"], text=True),
                flags={
                    k: env[k]
                    for k in (
                        "TORCH_CUDA_ARCH_LIST",
                        "KERNEL_ALIGN_FORCE_SM90",
                        "KERNEL_ALIGN_USE_FAST_MATH",
                        "MAX_JOBS",
                        "CUDA_HOME",
                    )
                },
            ),
        )
        command(repo, env, ["nvidia-smi"], out / "nvidia-smi.txt")
        sys.path.insert(0, str(repo))
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE
        from rl_engine.runtime.registry import KernelRegistry

        require(
            _EXT_AVAILABLE and Path(_C.__file__).resolve() == binary.resolve(),
            "extension is missing or imported from an old checkout",
        )
        for symbol in (
            "fused_logp",
            "fused_logp_backward",
            "deterministic_logp_forward_fp32",
            "deterministic_logp_forward_with_lse",
            "deterministic_logp_backward",
            "deterministic_logp_forward_fp64",
            "deterministic_logp_backward_fp64",
            "deterministic_logp_probability",
            "deterministic_logp_row_sum",
            "deterministic_logp_selection",
        ):
            require(hasattr(_C, symbol), "native symbol missing: " + symbol)
        registry = KernelRegistry()
        require(
            type(registry.get_op("logp", device="cuda")).__name__ == "FusedLogpGenericOp",
            "generic registry silently fell back",
        )
        require(
            type(registry.get_op("logp_deterministic", device="cuda")).__name__
            == "DeterministicLogpCUDAOp",
            "deterministic registry silently fell back",
        )
        shutil.copyfile(_C.__file__, out / "native.so")
        binary_hash = digest(out / "native.so")
        command(
            repo,
            env,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml=" + str(out / "gpu.xml"),
                *SELECTION,
            ],
            out / "gpu.log",
        )
        command(
            repo,
            env,
            [
                sys.executable,
                "-m",
                "tools.validation.operators.compare_fused_logp_ownership",
                "--baseline",
                BASE,
                "--output",
                str(out / "comparison.json"),
            ],
            out / "comparison.log",
        )
        sm90_fallback(repo, out)
        state["state"] = "PASS"
        save(out / "status.json", state)
        check_reports(out, binary_hash)
    except BaseException:
        state["state"] = "FAIL"
        state["error"] = traceback.format_exc()
        raise
    finally:
        state["elapsed_seconds"] = time.time() - started
        try:
            save(out / "status.json", state)
            checksums(out, binary_hash)
        except OSError as error:
            print("[task19] evidence finalization failed:", error, file=sys.stderr)
            if state["state"] == "PASS":
                raise
        print("[task19]", state["state"], "evidence:", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "verify"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--image", help="Published task19 GHCR image@sha256 reference")
    args = parser.parse_args()
    if args.mode == "verify":
        require(args.out is not None, "--out is required")
        verify(args.out.resolve())
        return
    require(args.repo is not None, "--repo is required")
    repo = args.repo.resolve()
    check_source(repo)
    missing = [
        m
        for m in ("torch", "numpy", "pytest", "tabulate", "ninja")
        if importlib.util.find_spec(m) is None
    ]
    require(not missing, "missing packages: " + ", ".join(missing))
    env = environment(repo)
    os.environ.clear()
    os.environ.update(env)
    if args.mode == "preflight":
        collect(repo, env)
        print("CPU/collection preflight only; no GPU PASS or native build claimed")
    else:
        require(args.out is not None, "--out is required")
        require(args.image is not None, "--image is required")
        run(repo, args.out.resolve(), env, args.image)


if __name__ == "__main__":
    main()
