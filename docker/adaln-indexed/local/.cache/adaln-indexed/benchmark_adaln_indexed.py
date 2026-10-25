# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""AdaLN-local indexed measurements. GPU execution is explicitly user-started."""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.operators.norm.benchmark_adaln_modulation import (  # noqa: E402
    _command_output,
    _sha256,
)
from tests.helpers.adaln_indexed import small_case  # noqa: E402

ROUTES = ("all0", "all1", "alternating", "random", "segmented")
MODES = ("forward", "backward-only", "forward+backward")
BACKENDS = ("cuda", "triton", "framework-select01")


def make_inputs(shape, dtype, route, *, seed=386, device="cpu", layout="contiguous"):
    """Generate once on CPU; preserve identical values across backend execution."""
    if route not in ROUTES or layout not in ("contiguous", "strided"):
        raise ValueError("unknown route or layout")
    b, s, h = shape
    if min(shape) < 1:
        raise ValueError("positive B/S/H required")
    x, m, _, dy, dg = small_case("cpu")
    if shape != (2, 8, 2) or seed != 386:
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(shape, generator=gen) * 2 + 0.3
        m = torch.randn(2 * b, 3 * h, generator=gen) * 0.5
        dy = torch.randn(shape, generator=gen)
        dg = torch.randn(shape, generator=gen)
    token = torch.arange(s)
    if route == "random":
        index = torch.randint(2, (b, s), generator=torch.Generator().manual_seed(seed + 1))
    else:
        row = {
            "all0": token * 0,
            "all1": token * 0 + 1,
            "alternating": token % 2,
            "segmented": (token >= s // 2).int(),
        }[route]
        index = row.expand(b, s).clone()
    values = (x, m, index, dy, dg)
    result = []
    for i, value in enumerate(values):
        value = value.detach().to(device=device, dtype=torch.int32 if i == 2 else dtype)
        if layout == "strided":
            value = torch.stack((value, torch.zeros_like(value)), -1)[..., 0]
        result.append(value.requires_grad_(i in (0, 1)))
    return tuple(result)


def prepare_call(op, inputs, mode):
    """Fresh graph per sample. Backward-only excludes its forward preparation."""
    if mode not in MODES:
        raise ValueError("unknown timing mode")
    x, m, index, dy, dg = inputs

    def forward():
        return op(x, m, modulate_index=index, eps=7.0)

    if mode == "forward":
        return forward
    outputs = forward() if mode == "backward-only" else None

    def backward():
        return torch.autograd.grad(outputs if outputs is not None else forward(), (x, m), (dy, dg))

    return backward


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("finite nonnegative samples required")
    ordered = sorted(samples)
    pos = 0.95 * (len(ordered) - 1)
    low, high = math.floor(pos), math.ceil(pos)
    return {
        "samples_ms": list(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p95_ms": ordered[low] + (ordered[high] - ordered[low]) * (pos - low),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--seed", type=int, default=386)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1 or args.session < 0:
        parser.error("warmup/session nonnegative; repeat positive")
    return args


def plan(args):
    sizes = [(2, 8, 2)]
    if not args.diagnostic_only:
        sizes += [(3, s, 3072) for s in (4096, 6889, 6032)]
    return {
        "schema": "adaln-indexed-benchmark-v1",
        "state": "COMMAND_PREPARED",
        "backends": list(BACKENDS),
        "modes": list(MODES),
        "cases": [
            dict(shape=list(shape), dtype=dtype, route=route, layout=layout)
            for shape in sizes
            for dtype in ("fp32", "bf16")
            for route in ROUTES
            for layout in ("contiguous", "strided")
        ],
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "session": args.session,
        "surfaces": ["public-backend-production", "dispatch-production"],
        "warm_kernel": "NOT PROVIDED: no public preallocated kernel API",
        "timing": "synchronized host wall primary; CUDA event auxiliary, validation included",
        "forward": "grad enabled; graph creation included, graph destruction after interval",
        "backward-only": "fresh graph prepared and synchronized outside interval per sample",
        "allocation": "inputs/index/random dy/dgate preallocated; outputs, gradients, copies, "
        "temporaries allocated inside public production calls; autograd.grad, no .grad",
        "order": "six backend permutations; session/case offsets; raw order retained",
        "cold": "first candidate validation forward/backward wall; JIT/cache lookup included; "
        "cache initially empty per process; reused specializations explicitly possible",
        "memory": "allocated/reserved peaks, baseline and allocated peak-minus-baseline; "
        "not a direct scratch-only measurement",
        "qualification": "NOT OPENED",
        "performance_conclusion": "NONE UNTIL TARGET EXECUTION/REVIEW",
    }


def backend_order(round_index):
    order = list(BACKENDS)
    if round_index % 6 >= 3:
        order.reverse()
    offset = round_index % 3
    return order[offset:] + order[:offset]


def evaluate(op, inputs, first_use=None):
    x, m, index, dy, dg = inputs
    if first_use is not None:
        torch.cuda.synchronize()
        start = time.perf_counter()
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    if first_use is not None:
        torch.cuda.synchronize()
        first_use["forward_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    if first_use is not None:
        torch.cuda.synchronize()
        first_use["backward_ms"] = (time.perf_counter() - start) * 1000
    return y, gate, dx, dm


def reference_outputs(inputs):
    from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference
    from rl_engine.validation.operators.operator_specs import GtestAdaLNReference

    cpu = tuple(t.detach().cpu().requires_grad_(i in (0, 1)) for i, t in enumerate(inputs))
    gold = tuple(
        t.float().detach().requires_grad_(i in (0, 1)) if i != 2 else t for i, t in enumerate(cpu)
    )
    return (
        evaluate(IndexedAdaLNFixedReference(), cpu),
        evaluate(GtestAdaLNReference().forward_fp32, gold),
    )


def validate_candidate(op, inputs, references, *, fixed_bytes=True, first_use=None):
    from rl_engine.contracts.numerical import load_contract, resolve_tolerance
    from tests.helpers.adaln_indexed import assert_bytes

    actual = evaluate(op, inputs, first_use)
    for i, (a, fixed, gold) in enumerate(zip(actual, *references, strict=True)):
        assert a.dtype == inputs[0].dtype
        if fixed_bytes or i == 1:
            assert_bytes(a, fixed)
        tolerance = resolve_tolerance(
            load_contract(),
            judgment="forward_accuracy" if i < 2 else "gradient_accuracy",
            op_class="reduction",
            dtype=a.dtype,
        )
        torch.testing.assert_close(
            a.detach().cpu().float(), gold, atol=tolerance.atol, rtol=tolerance.rtol
        )
    return actual


def explicit_backends(hidden):
    """Pin constructors; never allow resolver priority to replace the baseline."""
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.backends.shared.triton.norm.adaln_modulation import (
        TritonAdaLNModulationOp,
    )
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
    from rl_engine.runtime.adaln import AdaLNDispatch, INDEXED_PROFILE, SELECT01_PROFILE

    direct = dict(
        zip(
            BACKENDS,
            (
                CudaAdaLNModulationOp(),
                TritonAdaLNModulationOp(),
                NativeAdaLNModulationOp(),
            ),
            strict=True,
        )
    )
    dispatch = {}
    for backend, op in direct.items():
        trace = {
            "arithmetic_profile": (
                SELECT01_PROFILE if backend == "framework-select01" else INDEXED_PROFILE
            ),
            "requested_backend": backend,
            "fallback": False,
            "fallback_reason": None,
        }
        dispatch[backend] = AdaLNDispatch(op, trace, hidden=hidden)
    return direct, dispatch


def assert_backend(op, backend):
    expected = "pytorch" if backend == "framework-select01" else backend
    assert op.trace["actual_backend"] == expected, op.trace
    assert op.trace["arithmetic_profile"] == (
        "adaln-select01-framework-v1" if expected == "pytorch" else "adaln-indexed-fp32-tree-v1"
    )


def sample_call(op, inputs, mode):
    call = prepare_call(op, inputs, mode)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # CUDA events initialize lazily; allocate both before the measured interval.
    start.record()
    end.record()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start.record()
    wall_start = time.perf_counter()
    outputs = call()
    end.record()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - wall_start) * 1000
    result = dict(
        wall_ms=elapsed,
        event_ms=start.elapsed_time(end),
        allocated_before_bytes=before,
        reserved_before_bytes=reserved_before,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated() - before,
    )
    del outputs, call
    return result


def environment():
    import triton
    import rl_engine
    from rl_engine.backends.extension import _C

    binary = getattr(_C, "__file__", None)
    paths = [Path(__file__), ROOT / "pyproject.toml"]
    for name in ("rl_engine", "csrc", "tests/helpers"):
        paths += [
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h", ".json")
        ]
    triton_root = Path(triton.__file__).parent
    compiler = [
        triton_root / p
        for p in (
            "backends/nvidia/compiler.py",
            "_C/libtriton.so",
            "backends/nvidia/bin/ptxas",
            "backends/nvidia/lib/libdevice.10.bc",
        )
    ]
    return dict(
        python=sys.version,
        executable=sys.executable,
        pid=os.getpid(),
        package=rl_engine.__file__,
        loaded_binary=binary,
        binary_sha256=_sha256(binary),
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=triton.__version__,
        driver=_command_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ]
        ),
        compiler_sha256={str(p): _sha256(p) for p in compiler},
        nvcc=_command_output(["nvcc", "--version"]),
        source_sha256={str(p.relative_to(ROOT)): _sha256(p) for p in paths},
        cache=os.environ["TRITON_CACHE_DIR"],
        dump=os.environ["TRITON_DUMP_DIR"],
        launch=dict(cuda_threads=256, triton_num_warps=4, triton_reduction_tile=128),
        specialization="B/S/H/eps/broadcast/upstream-presence/launch; cache metadata retained",
    )


def main(argv=None):
    args = parse_args(argv)
    report = plan(args)
    if args.plan:
        print(json.dumps(report, indent=2))
        return 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # A new cache and process are mandatory for a session. No inherited warm cache.
    os.environ.update(
        TRITON_CACHE_DIR=str(out / "cache"),
        TRITON_DUMP_DIR=str(out / "dump"),
        TRITON_KERNEL_DUMP="1",
        TRITON_STORE_BINARY_ONLY="0",
        TRITON_KERNEL_OVERRIDE="0",
    )
    report.update(state="RUNNING", results=[], first_use=[], traces=[])
    try:
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("NVIDIA CUDA required; no performance result produced")
        report["environment"] = environment()
        with (
            contextlib.redirect_stdout(sys.stderr),
            (out / "samples.jsonl").open("x") as raw,
        ):
            for case_index, case in enumerate(report["cases"]):
                dtype = torch.float32 if case["dtype"] == "fp32" else torch.bfloat16
                cpu = make_inputs(
                    tuple(case["shape"]),
                    dtype,
                    case["route"],
                    seed=args.seed,
                    layout=case["layout"],
                )
                references = reference_outputs(cpu)
                data = make_inputs(
                    tuple(case["shape"]),
                    dtype,
                    case["route"],
                    seed=args.seed,
                    device="cuda",
                    layout=case["layout"],
                )
                direct, dispatch = explicit_backends(case["shape"][-1])
                for backend in backend_order(args.session + case_index):
                    first = dict(
                        case=case,
                        backend=backend,
                        label="first-use validation call; JIT/cache + production overhead",
                    )
                    first["cache_metadata_before"] = sorted(
                        str(p.relative_to(out)) for p in (out / "cache").rglob("*.json")
                    )
                    validate_candidate(
                        direct[backend],
                        data,
                        references,
                        fixed_bytes=backend != "framework-select01",
                        first_use=first,
                    )
                    first["cache_metadata_after"] = sorted(
                        str(p.relative_to(out)) for p in (out / "cache").rglob("*.json")
                    )
                    first["new_cache_metadata"] = sorted(
                        set(first["cache_metadata_after"]) - set(first["cache_metadata_before"])
                    )
                    report["first_use"].append(first)
                    validate_candidate(
                        dispatch[backend],
                        data,
                        references,
                        fixed_bytes=backend != "framework-select01",
                    )
                    assert_backend(dispatch[backend], backend)
                    report["traces"].append(
                        dict(
                            case=case,
                            backend=backend,
                            trace=dispatch[backend].trace.copy(),
                        )
                    )
                del references, cpu
                for surface, ops in (
                    ("public-backend-production", direct),
                    ("dispatch-production", dispatch),
                ):
                    for mode in MODES:
                        records: dict[str, list[dict]] = {b: [] for b in BACKENDS}
                        for iteration in range(args.warmup + args.repeat):
                            order = backend_order(args.session + case_index + iteration)
                            for backend in order:
                                measured = sample_call(ops[backend], data, mode)
                                if surface == "dispatch-production":
                                    assert_backend(ops[backend], backend)
                                if iteration >= args.warmup:
                                    row = dict(
                                        case=case,
                                        backend=backend,
                                        surface=surface,
                                        mode=mode,
                                        session=args.session,
                                        iteration=iteration - args.warmup,
                                        order=order,
                                        actual_backend=(
                                            "pytorch"
                                            if backend == "framework-select01"
                                            else backend
                                        ),
                                        **measured,
                                    )
                                    raw.write(json.dumps(row) + "\n")
                                    raw.flush()
                                    records[backend].append(measured)
                        for backend, samples in records.items():
                            report["results"].append(
                                dict(
                                    case=case,
                                    backend=backend,
                                    surface=surface,
                                    mode=mode,
                                    **summarize([r["wall_ms"] for r in samples]),
                                    event_summary=summarize([r["event_ms"] for r in samples]),
                                    memory_samples=samples,
                                )
                            )
                del data, direct, dispatch
        report["state"] = "EXECUTION_COMPLETE_REVIEW_PENDING"
    except (Exception, KeyboardInterrupt) as exc:
        import traceback

        report.update(state="FAIL", error=str(exc), traceback=traceback.format_exc())
    report["generated_sha256"] = {
        str(p.relative_to(out)): _sha256(p)
        for name in ("cache", "dump")
        for p in (out / name).rglob("*")
        if p.is_file()
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["state"], out)
    return int(report["state"] == "FAIL")


def summarize_sessions(reports):
    """Compare session medians only within the identical source/binary/target."""
    if len(reports) < 2 or len({r["session"] for r in reports}) != len(reports):
        raise ValueError("at least two distinct sessions required")
    identity_keys = (
        "gpu",
        "capability",
        "torch",
        "cuda",
        "triton",
        "driver",
        "binary_sha256",
        "source_sha256",
        "compiler_sha256",
        "launch",
    )
    identities = [{k: r["environment"].get(k) for k in identity_keys} for r in reports]
    if any(r["state"] != "EXECUTION_COMPLETE_REVIEW_PENDING" for r in reports):
        raise ValueError("failed or incomplete session")
    if any(identity != identities[0] for identity in identities):
        raise ValueError("session source/binary/environment mismatch")
    plan_keys = ("seed", "warmup", "repeat", "cases", "modes", "surfaces", "backends")
    plans = [{k: r[k] for k in plan_keys} for r in reports]
    if any(p != plans[0] for p in plans):
        raise ValueError("session input/sampling plan mismatch")
    grouped = []
    expected = None
    for report in reports:
        cells = {}
        for row in report["results"]:
            key = json.dumps(
                {k: row[k] for k in ("case", "backend", "mode", "surface")},
                sort_keys=True,
            )
            if key in cells:
                raise ValueError("duplicate result cell")
            cells[key] = row["median_ms"]
        if expected is None:
            expected = set(cells)
        if not cells or set(cells) != expected:
            raise ValueError("session coverage mismatch")
        grouped.append(cells)
    assert expected is not None
    return dict(
        state="REVIEW_PENDING",
        groups=[
            dict(
                json.loads(key),
                session_medians_ms=[g[key] for g in grouped],
                **summarize([g[key] for g in grouped]),
            )
            for key in sorted(expected)
        ],
        conclusion="No preset speedup threshold; negative/unstable results are valid",
    )


if __name__ == "__main__":
    raise SystemExit(main())
