"""CPU-only evidence completeness, archive and local recovery checks."""

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import tarfile
import tempfile

import storage


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


@contextlib.contextmanager
def exclusive_output(out):
    import fcntl

    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("batch output is in use; stop the runner before recovery") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def verify_reports(nodes, reports):
    expected = {(node, phase) for node in nodes for phase in ("setup", "call", "teardown")}
    actual = [(r["node"], r["phase"]) for r in reports]
    if (
        not nodes
        or len(actual) != len(expected)
        or set(actual) != expected
        or any(r["outcome"] != "passed" or r["xfail"] for r in reports)
    ):
        raise RuntimeError(
            "suite reports require all setup/call/teardown PASS, zero skip/xfail/xpass"
        )


def cell(case, backend, surface, mode):
    return json.dumps([case, backend, surface, mode], sort_keys=True)


def verify_performance(directory, plan, session, binary_sha256):
    report = read(directory / "report.json")
    if report["state"] != "EXECUTION_COMPLETE_REVIEW_PENDING":
        raise RuntimeError("incomplete performance report")
    for key, value in plan.items():
        if key not in ("state", "session") and report.get(key) != value:
            raise RuntimeError("performance plan mismatch: " + key)
    if report["session"] != session or report["environment"]["binary_sha256"] != binary_sha256:
        raise RuntimeError("performance process/binary mismatch")
    strict = plan.get("schema") == "adaln-indexed-benchmark-pr-v2"
    if strict:
        pairs = {
            json.dumps([case, backend], sort_keys=True)
            for case in plan["cases"]
            for backend in plan["backends"]
        }
        for field in ("first_use", "traces"):
            observed = [
                json.dumps([r["case"], r["backend"]], sort_keys=True) for r in report[field]
            ]
            if len(observed) != len(pairs) or set(observed) != pairs:
                raise RuntimeError("missing performance validation evidence: " + field)
        for first in report["first_use"]:
            if not all(
                math.isfinite(first[k]) and first[k] >= 0 for k in ("forward_ms", "backward_ms")
            ):
                raise RuntimeError("invalid first-use evidence")
        for trace in report["traces"]:
            expected_backend = (
                "pytorch" if trace["backend"] == "framework-select01" else trace["backend"]
            )
            expected_profile = (
                "adaln-select01-framework-v1"
                if expected_backend == "pytorch"
                else "adaln-indexed-fp32-tree-v1"
            )
            if (
                trace["trace"]["actual_backend"] != expected_backend
                or trace["trace"]["arithmetic_profile"] != expected_profile
            ):
                raise RuntimeError("performance validation backend/profile mismatch")
        if [r["case"] for r in report["oracles"]] != plan["cases"]:
            raise RuntimeError("missing per-case oracle provenance")
    expected = {
        cell(case, backend, surface, mode)
        for case in plan["cases"]
        for backend in plan["backends"]
        for surface in plan["surfaces"]
        for mode in plan.get("surface_modes", {}).get(surface, plan["modes"])
    }
    rows = {}
    for row in report["results"]:
        key = cell(*(row[k] for k in ("case", "backend", "surface", "mode")))
        if key in rows or len(row["samples_ms"]) != plan["repeat"]:
            raise RuntimeError("duplicate/incomplete performance cell")
        if strict and (
            len(row["memory_samples"]) != plan["repeat"]
            or len(row["event_summary"]["samples_ms"]) != plan["repeat"]
        ):
            raise RuntimeError("missing memory/event samples")
        rows[key] = row["samples_ms"]
    if set(rows) != expected:
        raise RuntimeError("performance selection coverage mismatch")
    result_rows = {
        cell(*(r[k] for k in ("case", "backend", "surface", "mode"))): r for r in report["results"]
    }
    samples = {key: {} for key in expected}
    with (directory / "samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = cell(*(row[k] for k in ("case", "backend", "surface", "mode")))
            iteration = row["iteration"]
            if (
                key not in samples
                or row["session"] != session
                or iteration in samples[key]
                or iteration not in range(plan["repeat"])
                or row["actual_backend"]
                != ("pytorch" if row["backend"] == "framework-select01" else row["backend"])
                or not all(math.isfinite(row[k]) and row[k] >= 0 for k in ("wall_ms", "event_ms"))
            ):
                raise RuntimeError("invalid/duplicate raw performance sample")
            if strict:
                measured = result_rows[key]["memory_samples"][iteration]
                required = (
                    "wall_ms",
                    "event_ms",
                    "allocated_before_bytes",
                    "reserved_before_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "incremental_peak_allocated_bytes",
                )
                if (
                    any(
                        k not in measured
                        or measured[k] != row[k]
                        or not math.isfinite(measured[k])
                        or measured[k] < 0
                        for k in required
                    )
                    or result_rows[key]["event_summary"]["samples_ms"][iteration] != row["event_ms"]
                ):
                    raise RuntimeError("raw memory/event evidence mismatch")
            samples[key][iteration] = row["wall_ms"]
    for key, values in samples.items():
        if (
            sorted(values) != list(range(plan["repeat"]))
            or [values[i] for i in sorted(values)] != rows[key]
        ):
            raise RuntimeError("missing raw samples or report/sample mismatch")
    return report


def verify_native_extraction(out, extraction):
    directory = out / extraction["output_path"]
    if not directory.resolve().is_relative_to(out.resolve()):
        raise RuntimeError("native extraction path escapes batch")
    required = {"version.log", "ptx.log", "sass.log", "resources.log"}
    if set(extraction["files_sha256"]) != required:
        raise RuntimeError("incomplete native extraction file list")
    for name, digest in extraction["files_sha256"].items():
        path = directory / name
        if not path.stat().st_size:
            raise RuntimeError("empty native extraction: " + name)
        if sha(path) != digest:
            raise RuntimeError("changed native extraction: " + name)


def validate_tree(out, *, partial=False):
    candidate = read(out / "candidate.json")
    state = (
        read(out / "state.json")
        if (out / "state.json").exists()
        else dict(state="PREPARATION_INCOMPLETE", completed={}, artifacts={}, sessions={})
    )
    preparation_partial = partial and not any(
        state.get(k) for k in ("completed", "artifacts", "sessions")
    )
    missing = []
    for file, field in (
        ("source-sha256.json", "source_sha256"),
        ("candidate-source.tar.gz", "archive_sha256"),
        ("node-lock.json", "node_lock_sha256"),
        ("performance-plan.json", "performance_plan_sha256"),
    ):
        if preparation_partial and (field not in candidate or not (out / file).is_file()):
            missing.append(file)
            continue
        if sha(out / file) != candidate[field]:
            raise RuntimeError("candidate identity mismatch: " + file)
    if "storage_plan_sha256" in candidate:
        if sha(out / "storage-plan.json") != candidate["storage_plan_sha256"]:
            raise RuntimeError("storage plan identity mismatch")
    baseline = read(out / "source-sha256.json") if (out / "source-sha256.json").exists() else {}
    for name, digest in baseline.items():
        if sha(out / "candidate-source" / name) != digest:
            raise RuntimeError("candidate source mismatch: " + name)

    def retained(reference):
        original = Path(candidate.get("output_path", str(out)))
        path = out / Path(reference).relative_to(original)
        if not path.resolve().is_relative_to(out.resolve()):
            raise RuntimeError("referenced artifact escapes batch")
        return path

    if (
        "storage_plan_sha256" in candidate
        and state.get("completed")
        and not state.get("storage_checks")
    ):
        raise RuntimeError("missing runtime storage checks")
    for record in state.get("storage_checks", []):
        probe = out / record["path"]
        if not probe.resolve().is_relative_to(out.resolve()) or sha(probe) != record["sha256"]:
            raise RuntimeError("storage check identity mismatch")
        check = read(probe)
        if (
            check["mount"] != "/workspace"
            or check["free_bytes"] < check["required_remaining_bytes"]
        ):
            raise RuntimeError("invalid runtime storage check")
    lock = read(out / "node-lock.json") if (out / "node-lock.json").exists() else {}
    if not partial and (
        state["state"] != "EXECUTION_COMPLETE_REVIEW_PENDING"
        or set(state["completed"]) != set(lock)
        or set(state.get("artifacts", {})) != set(lock)
        or set(state["sessions"]) != {str(i) for i in range(candidate["sessions"])}
        or not state.get("preflight")
    ):
        raise RuntimeError(
            "required batch stages incomplete; use explicit --partial for recovery only"
        )
    for namespace in ("completed", "artifacts", "sessions", "preflight"):
        for name, record in state.get(namespace, {}).items():
            directory = out / record["path"]
            if not directory.resolve().is_relative_to(out.resolve()):
                raise RuntimeError("receipt path escapes batch")
            actual = {
                str(p.relative_to(directory)): sha(p)
                for p in directory.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
            }
            if not actual or actual != record["receipt"]:
                raise RuntimeError("evidence receipt mismatch: " + namespace + "/" + name)
            if namespace == "completed":
                if (
                    read(directory / "result.json")["state"] != "PASS"
                    or read(directory / "nodes.json") != lock[name]
                ):
                    raise RuntimeError("suite result/node mismatch")
                reports = read(directory / "reports.json")
                verify_reports(lock[name], reports)
                for required in (
                    "identity.json",
                    "console.log",
                    "results.xml",
                    "process-command.json",
                ):
                    path = directory / required
                    if not path.is_file() or (
                        required != "console.log" and not path.stat().st_size
                    ):
                        raise RuntimeError("missing suite evidence: " + required)
                import xml.etree.ElementTree as ET

                ET.parse(directory / "results.xml")
                if not (out / (directory.name + ".log")).is_file():
                    raise RuntimeError("missing suite outer log")
                identity = read(directory / "identity.json")
                for source, digest in read(directory / "source-sha256.json").items():
                    if read(out / "source-sha256.json").get(source) != digest:
                        raise RuntimeError("executed source mismatch")
                if name.startswith("triton-"):
                    if identity["extension_available"]:
                        raise RuntimeError("Triton isolation lost")
                else:
                    built = read(directory / "build.json")
                    if (
                        sha(retained(built["binary"])) != built["binary_sha256"]
                        or identity["loaded_binary_sha256"] != built["binary_sha256"]
                    ):
                        raise RuntimeError("loaded/retained binary mismatch")
                result = read(directory / "result.json")
                if result.get("sanitizer"):
                    errors = result["sanitizer_errors"]
                    if (
                        not errors
                        or not any(errors.values())
                        or any(int(v) for values in errors.values() for v in values)
                    ):
                        raise RuntimeError("sanitizer result incomplete")
                    if not (directory / "sanitizer.log").stat().st_size:
                        raise RuntimeError("missing sanitizer log")
            if namespace == "artifacts":
                if name.startswith("triton-") and name != "triton-shared":
                    if not read(directory / "code-audit.json")["entries"]:
                        raise RuntimeError("missing Triton specializations")
                elif name == "triton-shared":
                    read(directory / "shared-cache.json")
                else:
                    extraction = read(directory / "extraction.json")
                    if sha(retained(extraction["binary"])) != extraction["binary_sha256"]:
                        raise RuntimeError("extraction binary mismatch")
                    verify_native_extraction(out, extraction)
            if namespace == "preflight":
                import xml.etree.ElementTree as ET

                suite = ET.parse(directory / "cpu.xml").getroot()
                testsuites = list(suite) if suite.tag == "testsuites" else [suite]
                if (
                    not testsuites
                    or sum(int(s.get("tests", 0)) for s in testsuites) < 1
                    or any(
                        int(s.get(k, 0))
                        for s in testsuites
                        for k in ("failures", "errors", "skipped")
                    )
                ):
                    raise RuntimeError("CPU preflight incomplete")
            if namespace == "sessions":
                native = out / state["completed"]["cuda-correctness"]["path"] / "build.json"
                if not (out / ("performance-" + directory.name + ".log")).is_file():
                    raise RuntimeError("missing performance outer log")
                report = verify_performance(
                    directory,
                    read(out / "performance-plan.json"),
                    int(name),
                    read(native)["binary_sha256"],
                )
                for oracle in report.get("oracles", []):
                    payload = retained(oracle["path"]) / "oracle.pt"
                    if sha(payload) != oracle["sha256"]:
                        raise RuntimeError("missing/changed oracle payload")
    if state.get("completed"):
        read(out / "target.json")
    if not partial:
        summaries = list((out / "summaries").glob("session-summary-attempt-*"))
        if not summaries or not any(read(p).get("groups") for p in summaries if p.is_file()):
            raise RuntimeError("missing session summary")
    return dict(
        candidate_sha256=sha(out / "candidate.json"),
        state=state["state"],
        recovery="complete",
        acceptance="independent review pending",
        partial=partial,
        missing_preparation_files=missing,
    )


def pack(out, *, partial=False):
    out = out.resolve()
    with exclusive_output(out):
        return pack_locked(out, partial=partial)


def pack_locked(out, *, partial=False):
    status = validate_tree(out, partial=partial)
    size, files = storage.tree_stats(out)
    # Reserve an incompressible tar plus its simultaneously expanded verification copy.
    tar_bound = size + files * 8192 + size // 100 + storage.MIB
    storage.require_free(out.parent, tar_bound + size + files * 4096 + storage.CONTROL_RESERVE)
    status["recovery"] = "LOCAL_VERIFICATION_PENDING"
    # A link outside this batch could silently disappear during transport.
    for path in out.rglob("*"):
        if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(out)):
            raise RuntimeError("external or broken evidence symlink: " + str(path))
    archive = Path(str(out) + ".tar.gz")
    temporary = Path(str(archive) + ".tmp")
    if archive.exists() or temporary.exists():
        raise FileExistsError("archive exists; preserve it and choose another transfer filename")
    index = {
        str(p.relative_to(out)): sha(p)
        for p in out.rglob("*")
        if p.is_file() and p.name != "recovery-index.json"
    }
    (out / "recovery-index.json").write_text(json.dumps(index, indent=2) + "\n")
    # A failed tar never publishes a sendable archive/checksum.
    with tarfile.open(temporary, "w:gz", dereference=True) as stream:
        stream.add(out, arcname="evidence")
    temporary.replace(archive)
    Path(str(archive) + ".sha256").write_text(sha(archive) + "  " + archive.name + "\n")
    return archive, status


def verify_archive(archive, candidate_sha256, *, partial=False):
    archive = archive.resolve()
    digest = Path(str(archive) + ".sha256").read_text().split()[0]
    if sha(archive) != digest:
        raise RuntimeError("archive checksum mismatch")
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        expanded = sum(m.size for m in members if m.isfile())
        storage.require_free(
            archive.parent,
            expanded + len(members) * 4096 + storage.CONTROL_RESERVE,
            files=len(members) + 4096,
        )
    with tempfile.TemporaryDirectory(prefix="adaln-recovery-", dir=archive.parent) as temporary:
        root = Path(temporary)
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            if not members:
                raise RuntimeError("empty evidence archive")
            for item in members:
                target = root / item.name
                if (
                    not target.resolve().is_relative_to(root)
                    or not (item.isdir() or item.isfile())
                    or item.name.split("/")[0] != "evidence"
                ):
                    raise RuntimeError("unsafe archive member")
            for item in members:
                target = root / item.name
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(item) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1024 * 1024)
        out = root / "evidence"
        if sha(out / "candidate.json") != candidate_sha256:
            raise RuntimeError("wrong batch candidate identity")
        index = read(out / "recovery-index.json")
        actual = {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "recovery-index.json"
        }
        if not index or actual != index:
            raise RuntimeError("recovered file inventory mismatch")
        return validate_tree(out, partial=partial)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pack", type=Path, metavar="OUTPUT")
    mode.add_argument("--verify", type=Path, metavar="ARCHIVE")
    parser.add_argument("--candidate-sha256")
    parser.add_argument(
        "--partial", action="store_true", help="recover stopped/failed evidence; never acceptance"
    )
    parser.add_argument(
        "--send", action="store_true", help="runpodctl send after successful pack and content check"
    )
    args = parser.parse_args(argv)
    if args.verify:
        if not args.candidate_sha256 or args.send:
            parser.error("local --verify requires --candidate-sha256 and does not send")
        result = verify_archive(args.verify, args.candidate_sha256, partial=args.partial)
    else:
        archive, result = pack(args.pack, partial=args.partial)
        verify_archive(archive, result["candidate_sha256"], partial=args.partial)
        print("ARCHIVE", archive, "CANDIDATE", result["candidate_sha256"], flush=True)
        if args.send:
            subprocess.run(["runpodctl", "send", str(archive)], check=True)
            subprocess.run(["runpodctl", "send", str(archive) + ".sha256"], check=True)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
