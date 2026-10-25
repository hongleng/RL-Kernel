"""CPU-only checks of image source, pinned environment and exact collection."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def check_source(root):
    profile = json.loads((root / "image-acceptance/source-profile.json").read_text())
    for name, digest in profile["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    assert (root / "Dockerfile").read_text().splitlines()[0] == "FROM " + profile[
        "base_image"
    ] + " AS runtime"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert head == profile["source_sha"], (head, profile["source_sha"])
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    config = (root / ".git/config").read_text()
    assert "extraheader" not in config.lower() and "[remote " not in config.lower()
    for script in (root / ".cache/adaln-indexed").rglob("*.py"):
        ast.parse(script.read_text(), filename=str(script), feature_version=(3, 11))
    print("Source, Git identity, credential-free config and Python3.11 syntax PASS")


def check_extension_free_fallback(root):
    import torch
    import rl_engine
    from rl_engine.backends.extension import _EXT_AVAILABLE
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
    from rl_engine.runtime.registry import KernelRegistry
    from tests.helpers.adaln_indexed import assert_bytes, small_case

    assert Path(rl_engine.__file__).resolve().parent == root / "rl_engine"
    assert not _EXT_AVAILABLE, "CPU image checks must not import an external native binary"
    x, m, _, dy, dg = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2)
    assert trace["selected_backend"] == "TRITON_ADALN_MODULATION"
    y, gate = op(x, m[:2], eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg[:, :1]))
    rx, rm = x.detach().requires_grad_(), m.detach().requires_grad_()
    ry, rg = NativeAdaLNModulationOp()(rx, rm[:2], eps=7.0)
    rdx, rdm = torch.autograd.grad((ry, rg), (rx, rm), (dy, dg[:, :1]))
    for actual, expected in zip((y, gate, dx, dm), (ry, rg, rdx, rdm), strict=True):
        assert_bytes(actual, expected)
    assert trace["actual_backend"] == "pytorch" and trace["fallback"]
    assert trace["fallback_reason"] == "shared_triton_domain_unsupported"
    print("Extension-free Triton resolution and CPU fallback bytes/trace PASS")


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    check_source(root)
    expected = json.loads((root / "image-acceptance/environment.json").read_text())
    import torch
    import triton
    from triton.backends.nvidia.compiler import CUDAOptions

    assert sys.version_info[:3] == tuple(map(int, expected["python"].split(".")))
    assert torch.__version__ == expected["torch"] and torch.version.cuda == expected["cuda"]
    assert triton.__version__ == expected["triton"]
    assert "enable_reflect_ftz" in CUDAOptions.__dataclass_fields__
    check_extension_free_fallback(root)
    if len(sys.argv) == 2:
        candidate = Path(sys.argv[1])
        for actual, locked in (
            ("node-lock.json", "expected-node-lock.json"),
            ("performance-plan.json", "expected-performance-plan.json"),
        ):
            assert json.loads((candidate / actual).read_text()) == json.loads(
                (root / "image-acceptance" / locked).read_text()
            ), actual
        assert json.loads((candidate / "candidate.json").read_text())["state"] == "COMMAND_PREPARED"
        print("All 17 suite node lists and benchmark plan match the locked candidate")
    print("Pinned environment PASS; GPU qualification remains closed")


if __name__ == "__main__":
    main()
