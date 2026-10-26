"""CPU checks for isolated imports, event lifetime and shared extraction."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import benchmark_adaln_indexed as bench
import evidence_io as evidence
import run_ticket0607 as batch

spec = importlib.util.spec_from_file_location(
    "native_runner", Path(__file__).with_name("run_ticket05.py")
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_isolated_command_ignores_sitecustomize_and_prefers_candidate_package(tmp_path):
    package = tmp_path / "candidate"
    (package / "rl_engine").mkdir(parents=True)
    (package / "rl_engine/__init__.py").write_text('origin = "candidate"\n')
    poison = tmp_path / "editable"
    poison.mkdir()
    marker = tmp_path / "hook-ran"
    (poison / "sitecustomize.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n"
    )
    env = dict(os.environ, PYTHONPATH=str(poison), CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True, timeout=20)
    assert marker.exists(), "negative control must exercise the contaminating site hook"
    marker.unlink()
    result = subprocess.run(
        batch.isolated_command(package, "import rl_engine; print(rl_engine.origin)"),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert result.stdout.strip() == "candidate" and not marker.exists()


def test_sampling_reuses_one_initialized_event_pair_and_synchronizes_each_call(monkeypatch):
    log = []
    created = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}
            created.append(self)

        def record(self):
            log.append("record")

        def elapsed_time(self, end):
            assert log[-1] == "sync" and end is created[1]
            return 1.0

    monkeypatch.setattr(bench.torch.cuda, "Event", Event)
    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: log.append("sync"))
    monkeypatch.setattr(bench.torch.cuda, "reset_peak_memory_stats", lambda: None)
    for name in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(bench.torch.cuda, name, lambda: 0)
    events = bench.prepare_events()
    assert log == ["record", "record", "sync"]

    def op(*args, **kwargs):
        log.append("body")
        return ()

    inputs = (None,) * 5
    for _ in range(2):
        assert bench.sample_call(op, inputs, "forward", events)["event_ms"] == 1.0
    assert len(created) == 2 and log.count("body") == 2 and log.count("sync") == 5


def test_native_build_configuration_embeds_ptx_for_offline_review():
    options = native.build_options((9, 0))
    assert options["arch"] == "9.0+PTX"
    assert options["nvcc_flags"] == ["-O3", "--fmad=false", "--ftz=false", "-lineinfo"]


def test_native_extraction_reuses_only_matching_binary_tool_and_options(tmp_path):
    binary = tmp_path / "_C.so"
    binary.write_bytes(b"CPU binary fixture")
    built = dict(binary=str(binary), binary_sha256=evidence.sha(binary))
    tool = tmp_path / "cuobjdump"
    calls = tmp_path / "calls"
    tool.write_text(
        "#!"
        + sys.executable
        + "\nfrom pathlib import Path\n"
        + "with Path("
        + repr(str(calls))
        + ").open('a') as f: f.write('call\\n')\n"
        + "print('CPU extraction fixture')\n"
    )
    tool.chmod(0o755)
    state = {"artifacts": {}}
    first = tmp_path / "artifacts/first"
    first.mkdir(parents=True)
    batch.extract_native(tmp_path, state, built, first, tool, dict(os.environ))
    state["artifacts"]["cuda-correctness"] = dict(
        path="artifacts/first", receipt=batch.receipt(first)
    )
    second = tmp_path / "artifacts/second"
    second.mkdir()
    batch.extract_native(tmp_path, state, built, second, tool, dict(os.environ))
    shared = json.loads((second / "extraction.json").read_text())
    assert calls.read_text().splitlines() == ["call"] * 4
    assert shared["output_path"] == "artifacts/first"
    evidence.verify_native_extraction(tmp_path, shared)
    # Tool changes invalidate extraction reuse even with the identical binary.
    tool.write_text(tool.read_text() + "# changed tool identity\n")
    third = tmp_path / "artifacts/third"
    third.mkdir()
    batch.extract_native(tmp_path, state, built, third, tool, dict(os.environ))
    assert len(calls.read_text().splitlines()) == 8
    (first / "sass.log").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        evidence.verify_native_extraction(tmp_path, shared)
