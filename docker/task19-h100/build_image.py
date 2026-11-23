"""GPU-less sm_90 build using the pinned project's unchanged build configuration."""

import os
import runpy
import subprocess
import sys
import sysconfig
from collections import deque
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (
    BASE,
    BASE_IMAGE,
    BUILD,
    GPU_PLAN,
    HEAD,
    check_prebuilt,
    check_source,
    digest,
    environment,
    git,
    require,
    save,
)


def main():
    repo = Path("/opt/task19-source")
    check_source(repo)
    if sys.argv[1:] == ["--check"]:
        import torch

        check_prebuilt(repo, BUILD, torch)
        print("PASS: final image source/binary/runtime identity; no GPU execution")
        return
    require(not sys.argv[1:], "unexpected build arguments")
    require(not list((repo / "rl_engine").glob("_C*.so")), "refuse a second native build")
    BUILD.mkdir(exist_ok=False)
    env = environment(repo)
    os.environ.clear()
    os.environ.update(env)
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    require(
        str(torch.__version__).split("+")[0] == "2.4.1" and torch.version.cuda == "12.4",
        "wrong image runtime",
    )
    require(CUDA_HOME is not None, "missing CUDA devel toolkit")
    # setup.py probes availability/capability even with TORCH_CUDA_ARCH_LIST set.
    # These two build-only answers select the normal H100 configuration, no GPU execution.
    stub = Path(CUDA_HOME) / "lib64/stubs"
    require((stub / "libcuda.so").is_file(), "missing CUDA driver link stub")
    os.environ["LIBRARY_PATH"] = str(stub)
    os.environ["CUDA_HOME"] = CUDA_HOME
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    sys.argv = [str(repo / "setup.py"), "build_ext", "--inplace"]
    # Redirect at fd level: also capture Ninja/nvcc output in the build record.
    with (BUILD / "build.log").open("w") as log:
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            ):
                runpy.run_path(str(repo / "setup.py"), run_name="__main__")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)
            if sys.exc_info()[0] is not None:
                with (BUILD / "build.log").open() as failed_log:
                    print("".join(deque(failed_log, maxlen=80)), file=sys.stderr)
    binaries = list((repo / "rl_engine").glob("_C*.so"))
    require(len(binaries) == 1, "build did not produce one binary")
    arch = subprocess.check_output(
        [str(Path(CUDA_HOME) / "bin/cuobjdump"), "--list-elf", str(binaries[0])],
        text=True,
    )
    require("sm_90" in arch, "binary has no sm_90 code")
    paths = git(repo, "ls-files").splitlines()
    save(
        BUILD / "build.json",
        dict(
            head=HEAD,
            baseline=BASE,
            base_image=BASE_IMAGE,
            files={p: digest(repo / p) for p in paths},
            runner_sha256=digest(Path(__file__).with_name("run.py")),
            builder_sha256=digest(Path(__file__)),
            dockerfile_sha256=digest(Path(__file__).with_name("Dockerfile")),
            gpu_plan_sha256=digest(GPU_PLAN),
            extension_sha256=digest(binaries[0]),
            build_log_sha256=digest(BUILD / "build.log"),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            python_abi=sysconfig.get_config_var("SOABI"),
            torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
            target="sm_90",
            experimental_sm90=False,
            fast_math=False,
            setup_argv=sys.argv,
            build_only_device_queries={
                "is_available": True,
                "get_device_capability": [9, 0],
            },
            flags={
                k: os.environ[k]
                for k in (
                    "TORCH_CUDA_ARCH_LIST",
                    "MAX_JOBS",
                    "KERNEL_ALIGN_FORCE_SM90",
                    "KERNEL_ALIGN_USE_FAST_MATH",
                    "CUDA_HOME",
                    "LIBRARY_PATH",
                )
            },
            nvcc=subprocess.check_output(
                [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
            ),
            cxx=subprocess.check_output(["c++", "--version"], text=True),
            architectures=arch,
        ),
    )
    print("Built once: sm_90 binary and provenance; no GPU PASS claimed", flush=True)


if __name__ == "__main__":
    main()
