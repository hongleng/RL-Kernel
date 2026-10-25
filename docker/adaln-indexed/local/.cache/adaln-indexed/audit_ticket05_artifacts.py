"""Inspect retained indexed Triton artifacts; no GPU execution or qualification."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    tool_path = shutil.which("cuobjdump")
    assert tool_path, "put the target CUDA toolkit's cuobjdump on PATH"
    tool = Path(tool_path).resolve()
    tool_identity = {
        "path": str(tool),
        "sha256": sha(tool),
        "version": subprocess.check_output([str(tool), "--version"], text=True),
    }
    identity = json.loads((run / "identity.json").read_text())
    for directory, records in identity["generated_artifacts"].items():
        for name, digest in records.items():
            assert sha(run / directory / name) == digest, name
    assert not (run / "audit_ticket05_artifacts.py").exists()
    shutil.copyfile(__file__, run / "audit_ticket05_artifacts.py")
    entries = []
    assembly = run / "cuobjdump"
    assembly.mkdir(exist_ok=False)
    for metadata in sorted((run / "cache").rglob("*.json")):
        if metadata.name not in {"_rows.json", "_fold.json"}:
            continue
        data = json.loads(metadata.read_text())
        assert data["target"]["arch"] == 10 * identity["capability"][0] + identity["capability"][1]
        assert data["enable_fp_fusion"] is False and data["enable_reflect_ftz"] is False
        stem = metadata.with_suffix("")
        paths = {
            suffix: stem.with_suffix("." + suffix)
            for suffix in ["ttir", "ttgir", "llir", "ptx", "cubin", "source"]
        }
        assert all(p.is_file() for p in paths.values())
        ptx, ttir = paths["ptx"].read_text(), paths["ttir"].read_text()
        result = subprocess.run(
            [str(tool), "--dump-sass", str(paths["cubin"])],
            text=True,
            capture_output=True,
            check=True,
        )
        sass = assembly / (metadata.parent.name + "-" + metadata.stem + ".sass")
        sass.write_text(result.stdout)
        fmas = [line.strip() for line in ptx.splitlines() if re.search(r"\bfma\.", line)]
        entries.append(
            {
                "cache_key": metadata.parent.name,
                "kernel": metadata.stem,
                "num_warps": data["num_warps"],
                "target": data["target"],
                "range_ends": re.findall(r"tt.make_range \{end = (\d+)", ttir),
                "signature": next(
                    line.strip() for line in ttir.splitlines() if "tt.func public" in line
                ),
                "fp32_loop_lines": [
                    line.strip() for line in ttir.splitlines() if "scf.for" in line
                ],
                "ptx_fma": fmas,
                "ptx_ftz_lines": [line.strip() for line in ptx.splitlines() if ".ftz" in line],
                "ptx_atomic_lines": [
                    line.strip() for line in ptx.splitlines() if re.search(r"\b(atom|red)\.", line)
                ],
                "ptx_rn_ops": {
                    op: len(re.findall(r"\b" + op + r"\.rn\.f32", ptx))
                    for op in ["add", "sub", "mul", "div", "sqrt"]
                },
                "sass_ftz_count": result.stdout.count(".FTZ"),
                "sass_fma_count": result.stdout.count("FFMA") + result.stdout.count("HFMA"),
                "cuobjdump_sass": str(sass.relative_to(run)),
                "sass_sha256": sha(sass),
                "artifacts_sha256": {suffix: sha(path) for suffix, path in paths.items()},
            }
        )
    assert entries
    report = {
        "scope": "retained indexed specializations; static inspection, not formal proof",
        "run": str(run),
        "audit_script_sha256": sha(Path(__file__)),
        "cuobjdump": tool_identity,
        "nodes": json.loads((run / "nodes.json").read_text()),
        "node_artifact_attribution": "run-level set; no exhaustive node-to-specialization mapping",
        "entries": entries,
        "caveats": [
            "RN sqrt/div lowering can contain internal FTZ/FMA with rescaling/fixups.",
            "S1 BF16 gate add may lower to bf16 FMA(v,1,+0), equivalent final result.",
            "Numerical subnormal and rounding tests remain separate evidence.",
        ],
    }
    (run / "code-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "specializations": len(entries),
                "ptx_fma_entries": sum(bool(x["ptx_fma"]) for x in entries),
                "ptx_ftz_entries": sum(bool(x["ptx_ftz_lines"]) for x in entries),
                "ptx_atomic_entries": sum(bool(x["ptx_atomic_lines"]) for x in entries),
            }
        )
    )


if __name__ == "__main__":
    main()
