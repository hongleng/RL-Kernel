"""Lightweight source/runner checks only; does not import torch or launch a GPU task."""

import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys

pack = Path(__file__).resolve().parent
root = pack.parent
profile = json.loads((pack / "source-profile.json").read_text())
for path, expected in profile["files"].items():
    with (root / path).open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, path
for path in pack.glob("*.py"):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
sys.path.insert(0, str(pack))
from benchmark import summary  # noqa: E402 -- import from the checked package directory

assert summary([1.0, 2.0, 3.0, 4.0])["median_ms"] == 2.5
assert summary([1.0, 2.0, 3.0, 4.0])["first_half_ms"] == 1.5
orders = list(itertools.permutations(("cuda", "triton", "pytorch")))
for position in range(3):
    for backend in ("cuda", "triton", "pytorch"):
        assert (
            sum(orders[(r + s) % 6][position] == backend for r in range(6) for s in range(6)) == 12
        )
print("Source hashes, syntax, summaries and balanced ordering verified; no H100 execution.")

manifest = json.loads((pack / "test-manifest.json").read_text())
assert manifest["source_commit"] == profile["commit"]
assert set(manifest["stages"]) == {"correctness", "racecheck", "memcheck"}
for stage, nodes in manifest["stages"].items():
    assert nodes and len(nodes) == len(set(nodes)) == profile["expected_tests"][stage]
    assert all((root / node.split("::")[0]).is_file() for node in nodes)
    assert set(nodes) <= set(manifest["stages"]["correctness"])
correctness = manifest["stages"]["correctness"]
assert sum("real_image_shapes_forward_backward" in node for node in correctness) == 6
assert sum("relocation_with_all_launch_configurations" in node for node in correctness) == 18
assert any("gpu_fallback_forward_backward" in node for node in correctness)
environment = json.loads((pack / "environment.json").read_text())
assert environment["python"].startswith("3.11.") and environment["cuda"] == "12.6"
print("Reviewed test manifests and Python 3.11 syntax verified.")
