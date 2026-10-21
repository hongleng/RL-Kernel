"""Balanced paired samples after correctness and sanitizer, without a speed gate."""

import csv
import gc
import itertools
import json
import statistics
import subprocess
import time


def summary(values):
    mid = len(values) // 2
    ordered = sorted(values)
    return dict(
        median_ms=statistics.median(values),
        p10_ms=ordered[int(0.1 * (len(values) - 1))],
        p90_ms=ordered[int(0.9 * (len(values) - 1))],
        cv=statistics.pstdev(values) / statistics.mean(values),
        first_half_ms=statistics.median(values[:mid]),
        second_half_ms=statistics.median(values[mid:]),
    )


def sample(op, x, m, dy, dg, mode):
    import torch

    x.grad = m.grad = None
    output = op(x, m) if mode == "backward_only" else None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    if mode == "forward":
        with torch.no_grad():
            output = op(x, m)
    else:
        if mode == "forward_backward":
            output = op(x, m)
        torch.autograd.backward(output, (dy, dg))
    end.record()
    end.synchronize()
    result = dict(
        cuda_ms=start.elapsed_time(end),
        wall_ms=1000 * (time.perf_counter() - wall),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_extra_bytes=torch.cuda.max_memory_allocated() - baseline,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    del output
    return result


def main(out):
    import torch
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
    from rl_engine.validation.operators.operator_specs import GtestAdaLNReference
    from rl_engine.contracts.numerical import load_contract, resolve_tolerance

    ops = dict(
        cuda=CudaAdaLNModulationOp(),
        triton=TritonAdaLNModulationOp(),
        pytorch=NativeAdaLNModulationOp(),
    )
    orders = list(itertools.permutations(ops))
    rows = []
    with (out / "samples.jsonl").open("x", buffering=1) as log:

        def emit(value):
            log.write(json.dumps(value) + "\n")

        def telemetry(label):
            query = (
                "--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,"
                "power.draw,temperature.gpu,utilization.gpu"
            )
            process = subprocess.run(
                ["nvidia-smi", query, "--format=csv"], text=True, capture_output=True, timeout=30
            )
            emit(
                dict(
                    kind="telemetry",
                    label=label,
                    exit_code=process.returncode,
                    value=process.stdout,
                    error=process.stderr,
                )
            )

        for dtype_name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            for seq in (4096, 6889, 6032):
                torch.manual_seed(386 + seq)
                x = torch.randn(1, seq, 3072, device="cuda", dtype=dtype, requires_grad=True)
                m = torch.randn(1, 9216, device="cuda", dtype=dtype, requires_grad=True)
                dy, dg = torch.randn_like(x), torch.randn(1, 1, 3072, device="cuda", dtype=dtype)
                key = dict(dtype=dtype_name, S=seq, H=3072, B=1)
                # GPU Native is an accuracy baseline; only CUDA/Triton claim byte replay.
                ref_x = x.detach().cpu().float().requires_grad_()
                ref_m = m.detach().cpu().float().requires_grad_()
                ref_y, ref_gate = GtestAdaLNReference().forward_fp32(ref_x, ref_m)
                ref_dx, ref_dm = torch.autograd.grad(
                    (ref_y, ref_gate), (ref_x, ref_m), (dy.cpu().float(), dg.cpu().float())
                )
                gold = (ref_y, ref_gate, ref_dx, ref_dm)
                expected = None
                for name, op in ops.items():
                    x.grad = m.grad = None
                    y, gate = op(x, m)
                    torch.autograd.backward((y, gate), (dy, dg))
                    values = [
                        v.detach().cpu().contiguous().view(torch.uint8)
                        for v in (y, gate, x.grad, m.grad)
                    ]
                    if name == "cuda":
                        expected = values
                    elif name == "triton":
                        assert all(
                            torch.equal(a, b) for a, b in zip(values, expected, strict=True)
                        ), name
                    for index, (actual, reference) in enumerate(
                        zip((y, gate, x.grad, m.grad), gold, strict=True)
                    ):
                        tolerance = resolve_tolerance(
                            load_contract(),
                            op_class="reduction",
                            dtype=dtype,
                            judgment="forward_accuracy" if index < 2 else "gradient_accuracy",
                        )
                        torch.testing.assert_close(
                            actual.detach().cpu().float(),
                            reference.detach(),
                            atol=tolerance.atol,
                            rtol=tolerance.rtol,
                        )
                    emit(
                        dict(
                            kind="qualification",
                            **key,
                            backend=name,
                            accuracy=True,
                            byte_replay=name != "pytorch",
                        )
                    )
                    del y, gate, values
                del expected, gold, ref_x, ref_m, ref_y, ref_gate, ref_dx, ref_dm
                for mode in ("forward", "backward_only", "forward_backward"):
                    x.grad = m.grad = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Condition the whole case for >=2 s; each backend gets >=10 calls.
                    began, calls = time.monotonic(), 0
                    while calls < 10 or time.monotonic() - began < 2:
                        for name in orders[calls % 6]:
                            sample(ops[name], x, m, dy, dg, mode)
                        calls += 1
                    emit(
                        dict(
                            kind="warmup",
                            **key,
                            mode=mode,
                            calls_per_backend=calls,
                            seconds=time.monotonic() - began,
                        )
                    )
                    measured = {name: [] for name in ops}
                    for round_index in range(6):
                        telemetry(f"{key}-{mode}-round{round_index}-before")
                        for index in range(6):
                            order = orders[(round_index + index) % 6]
                            for name in order:
                                value = sample(ops[name], x, m, dy, dg, mode)
                                measured[name].append(value)
                                emit(
                                    dict(
                                        kind="sample",
                                        **key,
                                        mode=mode,
                                        backend=name,
                                        round=round_index,
                                        sample=index,
                                        order=order,
                                        **value,
                                    )
                                )
                        telemetry(f"{key}-{mode}-round{round_index}-after")
                    for name, values in measured.items():
                        row = dict(
                            **key,
                            mode=mode,
                            backend=name,
                            samples=36,
                            **summary([v["cuda_ms"] for v in values]),
                        )
                        for metric in (
                            "peak_allocated_bytes",
                            "peak_extra_bytes",
                            "peak_reserved_bytes",
                        ):
                            row[metric] = max(v[metric] for v in values)
                        rows.append(row)
                        emit(dict(kind="summary", **row))
                        print(json.dumps(row), flush=True)
                    (out / "benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")
                del x, m, dy, dg
                gc.collect()
                torch.cuda.empty_cache()
    assert len(rows) == 54
    with (out / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
