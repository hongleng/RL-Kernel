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


def main():
    root = Path(__file__).resolve().parents[1]
    check_source(root)
    expected = json.loads((root / "image-acceptance/environment.json").read_text())
    import torch
    import triton
    from triton.backends.nvidia.compiler import CUDAOptions

    assert sys.version_info[:3] == tuple(map(int, expected["python"].split(".")))
    assert torch.__version__ == expected["torch"] and torch.version.cuda == expected["cuda"]
    assert triton.__version__ == expected["triton"]
    assert "enable_reflect_ftz" in CUDAOptions.__dataclass_fields__
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
