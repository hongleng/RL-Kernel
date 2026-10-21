# Establish the snapshot's import path before importing torch or rl_engine.
# ruff: noqa: E402
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["RLK_REPO_ROOT"])
sys.path.insert(0, str(root))
os.chdir(root)
mode, destination = sys.argv[1:3]
out = Path(destination)
out.mkdir(exist_ok=True)
import torch
import pytest
import rl_engine._C as ext
from rl_engine.runtime.registry import OpBackend, kernel_registry

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available()
assert (
    hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()
    == os.environ["ADALN_EXPECTED_BINARY_SHA256"]
)
for hidden in (7, 32, 64, 3072, 4095, 4096):
    _, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=hidden)
    assert (
        trace["selected_backend"] == OpBackend.CUDA_ADALN_MODULATION.name and not trace["fallback"]
    ), trace
print("PROFILE", torch.__version__, torch.cuda.get_device_name(0), "mode", mode, flush=True)

if mode != "benchmark":
    from test_plan import pytest_args

    manifest = json.loads((Path(__file__).parent / "test-manifest.json").read_text())
    args = [*pytest_args(mode), "--junitxml=" + str(out / "junit.xml")]

    class Results:
        def __init__(self):
            self.reports = []
            self.selected = []

        def pytest_collection_finish(self, session):
            self.selected = [item.nodeid for item in session.items]
            (out / "selected-tests.json").write_text(json.dumps(self.selected, indent=2) + "\n")

        def pytest_runtest_logreport(self, report):
            self.reports.append(
                {
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "xfail": hasattr(report, "wasxfail"),
                }
            )

    results = Results()
    code = int(pytest.main(args, plugins=[results]))
    torch.cuda.synchronize()
    invalid = [r for r in results.reports if r["outcome"] != "passed" or r["xfail"]]
    (out / "results.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "pytest_args": args,
                "pytest_exit_code": code,
                "selected": len(results.selected),
                "passed": sum(
                    r["when"] == "call" and r["outcome"] == "passed" for r in results.reports
                ),
                "invalid_reports": invalid,
                "reports": results.reports,
            },
            indent=2,
        )
        + "\n"
    )
    assert results.selected == manifest["stages"][mode], (
        "Test selection differs from reviewed manifest"
    )
    if mode == "correctness":
        assert sum("real_image_shapes" in node for node in results.selected) == 6
        assert (
            sum("relocation_with_all_launch_configurations" in node for node in results.selected)
            == 18
        )
    raise SystemExit(code or bool(invalid))

import benchmark

benchmark.main(out)
