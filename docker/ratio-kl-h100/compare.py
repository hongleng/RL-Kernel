from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(directory: Path):
    rows = {}
    metadata = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        metadata.append(payload["metadata"])
        for row in payload["results"]:
            key = (row["dtype"], tuple(row["shape"]), row["mask_density"])
            rows[key] = row
    return metadata, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_meta, base = load(args.base)
    head_meta, head = load(args.head)
    assert base and base.keys() == head.keys(), "base/head benchmark cases differ"
    assert all("H100" in item["gpu"] for item in base_meta + head_meta)
    assert all(item["warmup"] == 20 and item["iterations"] == 100 for item in base_meta + head_meta)

    lines = [
        "# H100 ratio_kl backward conclusion",
        "",
        "| dtype | shape | density | isolated speedup | fwd+bwd change | peak saved | gates |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    passed = True
    isolated_speedups = []
    for key in sorted(base, key=str):
        old, new = base[key], head[key]
        isolated_speedup = old["isolated_backward_median_ms"] / new["isolated_backward_median_ms"]
        isolated_speedups.append(isolated_speedup)
        forward_change = (
            new["forward_backward_median_ms"] / old["forward_backward_median_ms"] - 1
        )
        saved = old["incremental_peak_bytes"] - new["incremental_peak_bytes"]
        expected = new["expected_staging_saving_bytes"]
        memory_ok = abs(saved - expected) <= expected * 0.05
        nonregression_ok = isolated_speedup >= 1 / 1.05
        forward_ok = forward_change <= 0.03
        case_ok = memory_ok and nonregression_ok and forward_ok
        passed &= case_ok
        dtype, shape, density = key
        lines.append(
            f"| {dtype} | {list(shape)} | {density} | {isolated_speedup:.3f}x | "
            f"{forward_change:+.1%} | {saved / 2**20:.1f} MiB | "
            f"{'PASS' if case_ok else 'FAIL'} |"
        )
    median_speedup = statistics.median(isolated_speedups)
    passed &= median_speedup >= 1.10
    lines.extend(
        [
            "",
            f"Median isolated speedup: **{median_speedup:.3f}x**",
            f"Overall: **{'PASS' if passed else 'FAIL'}**",
            "",
        ]
    )
    args.output.write_text("\n".join(lines))
    print(args.output)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
