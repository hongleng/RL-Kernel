# SPDX-License-Identifier: Apache-2.0
"""CPU behavior checks; these do not measure GPU performance."""

from pathlib import Path

import torch

import benchmark_adaln_indexed as bench


def test_seeded_inputs_keep_all_backends_on_identical_routes_and_upstreams():
    a = bench.make_inputs((2, 8, 2), torch.float32, "alternating", seed=386)
    b = bench.make_inputs((2, 8, 2), torch.float32, "alternating", seed=386)
    assert torch.equal(a[2], torch.tensor([[0, 1] * 4] * 2, dtype=torch.int32))
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].requires_grad and a[1].requires_grad
    assert not torch.equal(a[3], a[4])


def test_backward_only_prepares_forward_outside_timed_call_and_preserves_gradients():
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp

    inputs = bench.make_inputs((2, 8, 2), torch.float32, "alternating")
    observed = []
    native = NativeAdaLNModulationOp()

    def op(*args, **kwargs):
        observed.append("forward")
        return native(*args, **kwargs)

    call = bench.prepare_call(op, inputs, "backward-only")
    assert observed == ["forward"]
    gradients = call()
    assert observed == ["forward"]
    assert [tuple(t.shape) for t in gradients] == [(2, 8, 2), (4, 6)]
    assert torch.count_nonzero(gradients[0]) > 0
    assert inputs[0].grad is None
    for mode in ("forward", "forward+backward"):
        observed.clear()
        call = bench.prepare_call(op, inputs, mode)
        assert observed == []
        assert len(call()) == 2
        assert observed == ["forward"]


def test_summary_keeps_raw_samples_and_labels_without_performance_claims():
    import pytest

    result = bench.summarize([1.0, 2.0, 3.0, 10.0])
    assert result["samples_ms"] == [1.0, 2.0, 3.0, 10.0]
    assert result["median_ms"] == 2.5
    assert result["min_ms"] == 1.0 and result["max_ms"] == 10.0
    assert result["p95_ms"] == pytest.approx(8.95)
    for bad in ([], [float("nan")], [-1.0]):
        with pytest.raises(ValueError):
            bench.summarize(bad)
    args = bench.parse_args(["--output", "/tmp/unused", "--plan"])
    plan = bench.plan(args)
    assert {tuple(c["shape"]) for c in plan["cases"]} == {
        (2, 8, 2),
        (3, 4096, 3072),
        (3, 6889, 3072),
        (3, 6032, 3072),
    }
    assert plan["backends"] == ["cuda", "triton", "framework-select01"]
    assert plan["modes"] == list(bench.MODES)
    assert plan["state"] == "COMMAND_PREPARED"
    assert plan["warm_kernel"] == "NOT PROVIDED: no public preallocated kernel API"
    for argv in (["--repeat", "0"], ["--warmup", "-1"]):
        with pytest.raises(SystemExit):
            bench.parse_args(["--output", "/tmp/unused", *argv])


def test_balanced_order_and_candidate_failure_are_observable_at_public_seams():
    import pytest
    from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference

    orders = [bench.backend_order(i) for i in range(6)]
    assert len({tuple(row) for row in orders}) == 6
    assert all(sorted(row) == sorted(bench.BACKENDS) for row in orders)
    inputs = bench.make_inputs((2, 8, 2), torch.float32, "alternating")
    references = bench.reference_outputs(inputs)
    bench.validate_candidate(IndexedAdaLNFixedReference(), inputs, references)

    def wrong(x, m, **kwargs):
        y, gate = IndexedAdaLNFixedReference()(x, m, **kwargs)
        return y + 1.0, gate

    with pytest.raises(AssertionError):
        bench.validate_candidate(wrong, inputs, references)

    def broken(*args, **kwargs):
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        bench.prepare_call(broken, inputs, "forward+backward")()


def test_all_routes_dtypes_and_strided_upstreams_have_the_declared_inputs():
    for dtype in (torch.float32, torch.bfloat16):
        for route in bench.ROUTES:
            data = bench.make_inputs((2, 8, 2), dtype, route, layout="strided")
            assert all(t.dtype == (torch.int32 if i == 2 else dtype) for i, t in enumerate(data))
            assert all(not t.is_contiguous() for t in data)
            if route == "all0":
                assert data[2].sum() == 0
            if route == "all1":
                assert data[2].sum() == 16
            if route == "segmented":
                assert data[2][0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_cli_plan_and_missing_gpu_failure_preserve_report_and_exit_status(tmp_path):
    import os
    import subprocess
    import sys
    import json

    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_adaln_indexed.py")),
        "--output",
        str(tmp_path / "run"),
    ]
    result = subprocess.run(
        command + ["--plan"],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        timeout=20,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["state"] == "COMMAND_PREPARED"
    assert not (tmp_path / "run").exists()
    failed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        timeout=20,
    )
    assert failed.returncode == 1
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["state"] == "FAIL" and report["results"] == []
    assert "no performance result" in report["error"]


def test_session_statistics_reject_failed_or_mismatched_samples():
    import copy
    import pytest

    # Literal parser fixtures, never GPU performance evidence.
    report = dict(
        state="EXECUTION_COMPLETE_REVIEW_PENDING",
        session=0,
        seed=386,
        warmup=5,
        repeat=30,
        cases=[{"shape": [2, 8, 2]}],
        modes=["forward"],
        surfaces=["dispatch-production"],
        backends=["cuda"],
        environment=dict(gpu="fixture", binary_sha256="x", source_sha256={"p": "y"}),
        results=[
            dict(
                case={"shape": [2, 8, 2]},
                backend="cuda",
                mode="forward",
                surface="dispatch-production",
                median_ms=2.0,
            )
        ],
    )
    second = copy.deepcopy(report)
    second["session"] = 1
    second["results"][0]["median_ms"] = 4.0
    result = bench.summarize_sessions([report, second])
    assert result["groups"][0]["session_medians_ms"] == [2.0, 4.0]
    assert result["groups"][0]["median_ms"] == 3.0
    second["state"] = "FAIL"
    with pytest.raises(ValueError):
        bench.summarize_sessions([report, second])
    second["state"] = report["state"]
    second["environment"]["binary_sha256"] = "different"
    with pytest.raises(ValueError):
        bench.summarize_sessions([report, second])

    second = copy.deepcopy(report)
    second["session"] = 1
    for key in ("seed", "warmup", "repeat"):
        second[key] += 1
        with pytest.raises(ValueError, match="input/sampling plan mismatch"):
            bench.summarize_sessions([report, second])
        second[key] = report[key]


def test_framework_accuracy_uses_high_precision_sum_without_changing_indexed_bytes():
    import pytest
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
    from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference

    x = torch.zeros(1, 8, 2, requires_grad=True)
    m = torch.zeros(2, 6, requires_grad=True)
    index = torch.zeros(1, 8, dtype=torch.int32)
    dy = torch.zeros_like(x)
    dg = torch.zeros_like(x)
    dg[0, :, 0] = torch.tensor([2**24, 1, 1, 1, -(2**24), 1, 1, 1])
    inputs = (x, m, index, dy, dg)
    references = bench.reference_outputs(inputs)
    assert references[0][3][0, 4].item() == references[1][3][0, 4].item() == 3
    precise = bench.baseline_reference_outputs(inputs)
    assert precise[3][0, 4].item() == 6
    baseline_refs = (references[0], precise)
    bench.validate_candidate(NativeAdaLNModulationOp(), inputs, baseline_refs, fixed_bytes=False)
    bench.validate_candidate(IndexedAdaLNFixedReference(), inputs, references)

    def wrong(*args, **kwargs):
        y, gate = NativeAdaLNModulationOp()(*args, **kwargs)
        return y + 1, gate

    with pytest.raises(AssertionError):
        bench.validate_candidate(wrong, inputs, baseline_refs, fixed_bytes=False)
