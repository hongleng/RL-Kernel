"""Small indexed AdaLN PR-review bundle. CPU-only; not a complete batch backup.

Oracle tensors and unaccepted caches are deliberately omitted. This format has
its own verifier and never claims to satisfy the legacy full-backup verifier.
"""

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET

MIB = 1024**2
SUITES = {
    b + "-" + s
    for b, s in (
        ("cuda", "correctness"),
        ("triton", "correctness"),
        ("integration", "correctness"),
        *[(b, s) for s in ("shared", "large-offsets", "target") for b in ("cuda", "triton")],
        *[
            (b, s)
            for b in ("cuda", "triton")
            for s in ("memcheck", "racecheck", "initcheck", "synccheck")
        ],
    )
}
CODE = {".json", ".ttir", ".ttgir", ".llir", ".ptx", ".cubin", ".source"}
OMITTED = {
    "oracle.pt": (
        "runtime CPU reference cache; identity/seed/source hashes retained; tensors not verified"
    ),
    "duplicate source trees/build scratch": "one hash-checked candidate source archive retained",
    "raw cache/dump": "only accepted-session compiled code and audited specializations retained",
    "unaccepted performance data": "not accepted-session evidence; outer failure logs retained",
    "full backup receipts": (
        "original state retained; selected members rehashed; full backup not claimed"
    ),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda stream=stream: stream.read(MIB), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(name):
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or any(p in (".", "..") for p in path.parts)
    ):
        raise RuntimeError("unsafe relative evidence path")
    return path.as_posix()


def stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class Selection:
    def __init__(self, root):
        self.root = root.resolve()
        self.files = {}
        self.candidate = read(self.root / "candidate.json")
        self.original = Path(self.candidate.get("output_path", str(self.root)))

    def path(self, name):
        path = self.root / safe(name)
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise RuntimeError("missing or external review evidence: " + name)
        return path

    def take(self, name, reason, expected=None):
        name = safe(name)
        self.path(name)
        prior = self.files.get(name)
        if prior and prior["expected"] and expected and prior["expected"] != expected:
            raise RuntimeError("conflicting original evidence hashes: " + name)
        self.files[name] = dict(reason=reason, expected=expected or (prior or {}).get("expected"))
        return name

    def ref(self, reference):
        return safe(Path(reference).relative_to(self.original).as_posix())

    def json(self, name):
        return read(self.path(name))

    def require(self, record, names):
        for name in names:
            if name not in record["receipt"]:
                raise RuntimeError("missing mandatory original receipt entry: " + name)
            self.take(
                safe(record["path"]) + "/" + safe(name),
                "mandatory original evidence",
                record["receipt"][name],
            )

    def record(self, record, reason):
        directory = safe(record["path"])
        for name, digest in record["receipt"].items():
            safe(name)
            if (
                "/" not in name
                and Path(name).suffix in {".json", ".jsonl", ".xml", ".log", ".py", ".txt"}
                and name != "checkpoint.json"
            ):
                self.take(directory + "/" + name, reason, digest)
        return directory


def select(root):
    v = Selection(root)
    c = v.candidate
    state = v.json("state.json")
    lock = v.json("node-lock.json")
    if (
        state.get("state") != "EXECUTION_COMPLETE_REVIEW_PENDING"
        or set(lock) != SUITES
        or set(state.get("completed", {})) != SUITES
        or set(state.get("artifacts", {})) != SUITES
        or set(state.get("sessions", {})) != {"0"}
        or c.get("sessions") != 1
        or not state.get("preflight")
    ):
        raise RuntimeError("expected completed 17-suite pr-v2 batch and one accepted session")
    for name, field in (
        ("source-sha256.json", "source_sha256"),
        ("candidate-source.tar.gz", "archive_sha256"),
        ("node-lock.json", "node_lock_sha256"),
        ("performance-plan.json", "performance_plan_sha256"),
        ("storage-plan.json", "storage_plan_sha256"),
    ):
        v.take(name, "candidate identity and reproducible source", c[field])
    for name in ("candidate.json", "state.json", "target.json"):
        v.take(name, "original batch provenance")
    baseline = v.json("source-sha256.json")
    for record in state.get("storage_checks", []):
        v.take(record["path"], "storage supervision history", record["sha256"])
    native_hashes = set()
    for name, record in state["completed"].items():
        v.require(
            record,
            (
                "result.json",
                "nodes.json",
                "reports.json",
                "identity.json",
                "console.log",
                "results.xml",
                "process-command.json",
                "source-sha256.json",
            ),
        )
        d = v.record(record, "independent suite reports and logs")
        result = v.json(d + "/result.json")
        nodes = v.json(d + "/nodes.json")
        if result.get("state") != "PASS" or nodes != lock[name]:
            raise RuntimeError("suite result or node selection mismatch: " + name)
        verify_reports(nodes, v.json(d + "/reports.json"))
        for required in (
            "identity.json",
            "console.log",
            "results.xml",
            "process-command.json",
            "source-sha256.json",
        ):
            v.take(d + "/" + required, "suite audit")
        ET.parse(v.path(d + "/results.xml"))
        for source, digest in v.json(d + "/source-sha256.json").items():
            if baseline.get(source) != digest:
                raise RuntimeError("suite executed different source")
        identity = v.json(d + "/identity.json")
        v.take(Path(d).name + ".log", "suite outer log")
        if name.startswith("triton-"):
            if identity["extension_available"]:
                raise RuntimeError("Triton isolation lost")
        else:
            v.require(record, ("build.json",))
            built = v.json(d + "/build.json")
            if identity["loaded_binary_sha256"] != built["binary_sha256"]:
                raise RuntimeError("loaded native binary mismatch")
            native_hashes.add(built["binary_sha256"])
            v.take(
                v.ref(built["binary"]),
                "one content-deduplicated loaded native binary",
                built["binary_sha256"],
            )
        if result.get("sanitizer"):
            v.require(record, ("sanitizer.log",))
            errors = result["sanitizer_errors"]
            if (
                not errors
                or not any(errors.values())
                or any(int(x) for values in errors.values() for x in values)
            ):
                raise RuntimeError("incomplete or failed sanitizer evidence")
            if not v.path(d + "/sanitizer.log").stat().st_size:
                raise RuntimeError("missing sanitizer log")
            v.take(d + "/sanitizer.log", "raw sanitizer evidence")
    for name, record in state["artifacts"].items():
        d = v.record(record, "instruction audit metadata")
        if name == "triton-shared":
            v.require(record, ("shared-cache.json",))
            v.json(d + "/shared-cache.json")
        elif name.startswith("triton-"):
            v.require(record, ("code-audit.json",))
            audit = v.json(d + "/code-audit.json")
            if not audit["entries"]:
                raise RuntimeError("missing audited Triton specializations")
            for entry in audit["entries"]:
                candidates = [
                    p
                    for p in record["receipt"]
                    if p.startswith("cache/")
                    and Path(p).name == entry["kernel"] + ".json"
                    and Path(p).parent.name == entry["cache_key"]
                ]
                if len(candidates) != 1:
                    raise RuntimeError("ambiguous or missing audited kernel metadata")
                meta = candidates[0]
                v.take(d + "/" + meta, "audited compiler metadata", record["receipt"][meta])
                for suffix, digest in entry["artifacts_sha256"].items():
                    v.take(
                        d + "/" + str(Path(meta).with_suffix("." + suffix)),
                        "audited IR/PTX/cubin/source",
                        digest,
                    )
                v.take(d + "/" + entry["cuobjdump_sass"], "audited SASS", entry["sass_sha256"])
        else:
            v.require(record, ("extraction.json",))
            extracted = v.json(d + "/extraction.json")
            if set(extracted["files_sha256"]) != {
                "version.log",
                "ptx.log",
                "sass.log",
                "resources.log",
            }:
                raise RuntimeError("incomplete native instruction extraction")
            v.take(v.ref(extracted["binary"]), "loaded native binary", extracted["binary_sha256"])
            for file, digest in extracted["files_sha256"].items():
                source = safe(extracted["output_path"]) + "/" + file
                if not v.path(source).stat().st_size:
                    raise RuntimeError("empty native instruction evidence")
                v.take(source, "native instruction evidence", digest)
    for record in state["preflight"].values():
        v.require(record, ("cpu.xml", "cpu.log"))
        d = v.record(record, "CPU preflight (not GPU PASS)")
        xml = ET.parse(v.path(d + "/cpu.xml")).getroot()
        suites = list(xml) if xml.tag == "testsuites" else [xml]
        if (
            not suites
            or sum(int(x.get("tests", 0)) for x in suites) < 1
            or any(int(x.get(k, 0)) for x in suites for k in ("errors", "failures", "skipped"))
        ):
            raise RuntimeError("CPU preflight incomplete")
    plan = v.json("performance-plan.json")
    if plan.get("schema") != "adaln-indexed-benchmark-pr-v2" or len(plan["cases"]) != 18:
        raise RuntimeError("expected the locked 18-case pr-v2 performance plan")
    for session, record in state["sessions"].items():
        v.require(record, ("report.json", "samples.jsonl"))
        d = v.record(record, "accepted session report and original samples")
        report = v.json(d + "/report.json")
        binary = report["environment"]["binary_sha256"]
        executed_sources = report["environment"].get("source_sha256", {})
        if not executed_sources or any(
            baseline.get(p) != digest for p, digest in executed_sources.items()
        ):
            raise RuntimeError("accepted performance source differs from candidate")
        if binary not in native_hashes:
            raise RuntimeError("performance binary differs from suite binary")
        verify_performance(v.root / d, plan, int(session), binary)
        v.take("performance-" + Path(d).name + ".log", "accepted performance outer log")
        generated = report.get("generated_sha256", {})
        extensions = {Path(p).suffix for p in generated if p.startswith("cache/")}
        if not {".json", ".ptx", ".cubin"}.issubset(extensions):
            raise RuntimeError("missing accepted-session compiler metadata/PTX/cubin evidence")
        for relative, digest in generated.items():
            if relative.startswith("cache/") and Path(relative).suffix in CODE:
                v.take(d + "/" + safe(relative), "accepted-session compiler evidence", digest)
        for oracle in report["oracles"]:
            meta = v.ref(oracle["path"]) + "/identity.json"
            v.take(meta, "oracle reproducibility metadata; tensors omitted")
            identity = v.json(meta)
            value = identity["identity"]
            key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            if (
                identity["sha256"] != oracle["sha256"]
                or key != oracle["key"]
                or value["case"] != {k: x for k, x in oracle["case"].items() if k != "layout"}
                or value["seed"] != plan["seed"]
            ):
                raise RuntimeError("oracle provenance differs from accepted report")
            if any(baseline.get(p) != digest for p, digest in value["sources"].items()):
                raise RuntimeError("oracle generator/reference sources differ")
    summaries = [
        p
        for p in (v.root / "summaries").glob("session-summary-attempt-*")
        if p.is_file() and not p.name.endswith((".tmp", ".reserved"))
    ]
    if not summaries or not any(
        v.json(p.relative_to(v.root).as_posix()).get("groups") for p in summaries
    ):
        raise RuntimeError("missing accepted-session summary")
    for directory in ("summaries", "storage-recovery", "failures"):
        for p in sorted((v.root / directory).glob("*")):
            if (
                p.is_file()
                and not p.name.endswith((".tmp", ".reserved"))
                and not p.name.startswith(".")
            ):
                relative = p.relative_to(v.root).as_posix()
                v.take(relative, "summary or original failure/recovery metadata")
                if directory == "failures":
                    v.json(relative)  # Original JSON failure record, not failed tensors/caches.
                if directory == "storage-recovery" and p.name.startswith("policy-attempt-"):
                    policy = v.json(relative)
                    v.take(
                        policy["controller"],
                        "executed storage recovery controller",
                        policy["controller_sha256"],
                    )
                    if policy["original_candidate_sha256"] != sha(v.path("candidate.json")):
                        raise RuntimeError("storage recovery policy belongs to another candidate")
    for p in v.root.glob("performance-*.log"):
        v.take(p.name, "accepted/failed attempt outer history; failed samples/cache omitted")
    # Every selected receipt member must agree with its original, immutable receipt.
    for namespace in ("completed", "artifacts", "sessions", "preflight"):
        for record in state[namespace].values():
            for relative, digest in record["receipt"].items():
                name = safe(record["path"]) + "/" + safe(relative)
                if name in v.files:
                    v.take(name, v.files[name]["reason"], digest)
    return v


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


class DigestIO:
    def __init__(self, stream, label):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.label = label
        self.count = 0
        self.last = time.monotonic()

    def observe(self, data):
        self.digest.update(data)
        self.count += len(data)
        if time.monotonic() - self.last > 3:
            self.last = time.monotonic()
            print(f"{self.label} {self.count / MIB:.1f} MiB", flush=True)

    def read(self, size=-1):
        data = self.stream.read(size)
        self.observe(data)
        return data

    def write(self, data):
        n = self.stream.write(data)
        if n != len(data):
            raise OSError("short archive write")
        self.observe(data)
        return n

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()


def add_bytes(tar, name, data):
    item = tarfile.TarInfo(name)
    item.size = len(data)
    item.mode = 0o644
    tar.addfile(item, io.BytesIO(data))


def publish(pairs):
    owned = []
    try:
        for source, target in pairs:
            owned.append((source, target))
            os.link(source, target)
    except BaseException:
        for source, target in reversed(owned):
            try:
                if stamp(source)[:2] == stamp(target)[:2]:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    for source, _ in pairs:
        try:
            source.unlink()
        except OSError:
            print("NOTE published successfully; retained temporary", source, flush=True)


@contextlib.contextmanager
def locked(root):
    with (root / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("original runner/packer still holds the output lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def plan_size(v):
    known = set()
    total = 0
    for name, item in v.files.items():
        digest = item["expected"]
        if digest and digest in known:
            continue
        total += v.path(name).stat().st_size
        if digest:
            known.add(digest)
    return total


def extract(root, archive=None, max_mib=1024, plan_only=False):
    tool = Path(__file__).resolve()
    tool_stamp = stamp(tool)
    tool_data = tool.read_bytes()
    if stamp(tool) != tool_stamp:
        raise RuntimeError("extraction tool changed during startup")
    root = root.resolve()
    with locked(root):
        print("SELECT PR evidence; oracle tensor payloads will not be read", flush=True)
        v = select(root)
        estimated = plan_size(v)
        print(
            f"PLAN {len(v.files)} references; at most {estimated / MIB:.2f} MiB before gzip1; "
            "oracle.pt excluded; duplicate objects shared",
            flush=True,
        )
        if estimated > max_mib * MIB:
            largest = sorted(v.files, key=lambda p: v.path(p).stat().st_size, reverse=True)[:5]
            raise RuntimeError(
                "selection exceeds --max-mib; inspect these before increasing it: "
                + ", ".join(largest)
            )
        if plan_only:
            return dict(selected=len(v.files), maximum_bytes=estimated, omitted=OMITTED)
        archive = (
            (root.parent / (root.name + ".review.tar.gz"))
            if archive is None
            else archive.absolute()
        )
        if archive.resolve().is_relative_to(root):
            raise RuntimeError("review archive must be outside original output")
        tmp = Path(str(archive) + ".tmp")
        checksum = Path(str(archive) + ".sha256")
        checksum_tmp = Path(str(checksum) + ".tmp")
        for path in (archive, tmp, checksum, checksum_tmp):
            if path.exists() or path.is_symlink():
                raise FileExistsError("preserve existing transfer file: " + str(path))
        if shutil.disk_usage(archive.parent).free < estimated + estimated // 100 + 16 * MIB:
            raise RuntimeError("insufficient filesystem space for selected review archive")
        initial = {name: stamp(v.path(name)) for name in v.files}
        manifest = dict(
            schema="adaln-indexed-pr-review-v1",
            candidate_sha256=None,
            scope="PR review, not full batch backup or tensor-level recomputation",
            omitted=OMITTED,
            files={},
            objects={},
            tool_sha256=hashlib.sha256(tool_data).hexdigest(),
        )
        seen = {}
        inodes = {}
        started = time.monotonic()
        with tmp.open("xb") as raw:
            writer = DigestIO(raw, "ARCHIVE")
            with gzip.GzipFile(
                filename="", fileobj=writer, mode="wb", compresslevel=1, mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", copybufsize=MIB) as tar:
                    for number, (name, spec) in enumerate(v.files.items(), 1):
                        path = v.path(name)
                        if stamp(path) != initial[name]:
                            raise RuntimeError("selected evidence changed: " + name)
                        expected = spec["expected"]
                        if initial[name] in inodes:
                            digest, object_name = inodes[initial[name]]
                            if expected and digest != expected:
                                raise RuntimeError("original evidence hash mismatch: " + name)
                        elif expected and expected in seen:
                            # Deduplicate transport, not source verification: distinct files
                            # with equal recorded hashes must each still match their receipt.
                            with path.open("rb") as source:
                                data = DigestIO(source, "CHECK " + name)
                                for _data in iter(lambda data=data: data.read(MIB), b""):
                                    pass
                                digest = data.digest.hexdigest()
                            if digest != expected:
                                raise RuntimeError("original evidence hash mismatch: " + name)
                            object_name = seen[expected]
                            inodes[initial[name]] = (digest, object_name)
                        else:
                            object_name = f"objects/{len(manifest['objects']):06d}"
                            info = tarfile.TarInfo("review/" + object_name)
                            info.size = initial[name][2]
                            with path.open("rb") as source:
                                opened = os.fstat(source.fileno())
                                if (opened.st_dev, opened.st_ino) != initial[name][:2]:
                                    raise RuntimeError("selected file replaced: " + name)
                                data = DigestIO(source, "READ " + name)
                                tar.addfile(info, data)
                                digest = data.digest.hexdigest()
                            if expected and digest != expected:
                                raise RuntimeError("original evidence hash mismatch: " + name)
                            manifest["objects"][object_name] = dict(sha256=digest, bytes=info.size)
                            seen[digest] = object_name
                            inodes[initial[name]] = (digest, object_name)
                        if stamp(path) != initial[name]:
                            raise RuntimeError("selected evidence changed while reading: " + name)
                        manifest["files"][name] = dict(object=object_name, **spec)
                        if number % 50 == 0:
                            print(f"COPY {number}/{len(v.files)} selected references", flush=True)
                    for name, before in initial.items():
                        if stamp(v.path(name)) != before:
                            raise RuntimeError(
                                "selected evidence changed before publication: " + name
                            )
                    if stamp(tool) != tool_stamp:
                        raise RuntimeError("extraction tool changed during execution")
                    candidate_object = manifest["files"]["candidate.json"]["object"]
                    manifest["candidate_sha256"] = manifest["objects"][candidate_object]["sha256"]
                    add_bytes(tar, "review/tool.py", tool_data)
                    add_bytes(
                        tar,
                        "review/manifest.json",
                        (json.dumps(manifest, indent=2) + "\n").encode(),
                    )
            raw.flush()
            os.fsync(raw.fileno())
            archive_sha = writer.digest.hexdigest()
        with checksum_tmp.open("x") as stream:
            stream.write(archive_sha + "  " + archive.name + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish(((tmp, archive), (checksum_tmp, checksum)))
        print("ARCHIVE", archive, "CANDIDATE", manifest["candidate_sha256"], flush=True)
        print(
            f"EXTRACTED {archive.stat().st_size / MIB:.2f} MiB "
            f"in {time.monotonic() - started:.1f}s; "
            "LOCAL_REVIEW_BUNDLE_VERIFICATION_PENDING; full backup not claimed",
            flush=True,
        )
        return manifest


def verify_sources(view, max_bytes, materialize=False):
    expanded = 0
    expected = view.json("source-sha256.json")
    observed = {}
    with tarfile.open(view.path("candidate-source.tar.gz"), "r|*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                not name.parts
                or name.parts[0] != "candidate-source"
                or any(p == ".." for p in name.parts)
            ):
                raise RuntimeError("unsafe candidate source archive")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError("unexpected link in candidate source archive")
            expanded += member.size
            if member.size < 0 or expanded > max_bytes:
                raise RuntimeError("candidate source archive exceeds expanded size limit")
            relative = safe(PurePosixPath(*name.parts[1:]).as_posix())
            if relative in observed or relative not in expected:
                raise RuntimeError("unexpected candidate source member")
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            destination = view.root / "candidate-source" / relative
            if materialize:
                destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") if materialize else contextlib.nullcontext() as output:
                for data in iter(lambda stream=stream: stream.read(MIB), b""):
                    digest.update(data)
                    if output is not None:
                        output.write(data)
            observed[relative] = digest.hexdigest()
    if observed != expected:
        raise RuntimeError("candidate source archive differs from source manifest")


def verify(archive, candidate_sha, max_mib=1024, dest=None):
    archive = archive.resolve()
    if dest is not None:
        dest = Path(dest).absolute()
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("preserve existing local review directory: " + str(dest))
    expected_archive = Path(str(archive) + ".sha256").read_text().split()[0]
    print(
        "VERIFY review bundle checksum, objects, required coverage and original reports", flush=True
    )
    with tempfile.TemporaryDirectory(prefix="adaln-pr-review-", dir=archive.parent) as directory:
        root = Path(directory)
        objects = {}
        metadata = {}
        seen = set()
        total = 0
        with archive.open("rb") as raw:
            reader = DigestIO(raw, "VERIFY")
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                for member in tar:
                    name = safe(member.name)
                    if name in seen or not member.isfile():
                        raise RuntimeError("duplicate or non-file review member")
                    seen.add(name)
                    if member.size < 0:
                        raise RuntimeError("negative review object size")
                    total += member.size
                    if total > max_mib * MIB + 16 * MIB:
                        raise RuntimeError("review bundle exceeds extraction limit")
                    stream = tar.extractfile(member)
                    if name in {"review/manifest.json", "review/tool.py"}:
                        if member.size > 8 * MIB:
                            raise RuntimeError("oversized review metadata")
                        metadata[name] = stream.read()
                    elif name.startswith("review/objects/") and name[15:].isdigit():
                        key = name[len("review/") :]
                        path = root / key
                        path.parent.mkdir(exist_ok=True)
                        digest = hashlib.sha256()
                        with path.open("xb") as destination:
                            for data in iter(lambda stream=stream: stream.read(MIB), b""):
                                digest.update(data)
                                destination.write(data)
                        objects[key] = dict(sha256=digest.hexdigest(), bytes=member.size)
                    else:
                        raise RuntimeError("unexpected review archive member")
            for _data in iter(lambda: reader.read(MIB), b""):
                pass
            if reader.digest.hexdigest() != expected_archive:
                raise RuntimeError("review archive checksum mismatch")
        manifest = json.loads(metadata["review/manifest.json"])
        if (
            manifest["schema"] != "adaln-indexed-pr-review-v1"
            or manifest["objects"] != objects
            or manifest["omitted"] != OMITTED
        ):
            raise RuntimeError("review object inventory mismatch")
        if hashlib.sha256(metadata["review/tool.py"]).hexdigest() != manifest["tool_sha256"]:
            raise RuntimeError("extraction tool provenance mismatch")
        if manifest["candidate_sha256"] != candidate_sha:
            raise RuntimeError("wrong review batch candidate")
        view_root = root / "view"
        for name, spec in manifest["files"].items():
            path = view_root / safe(name)
            obj = spec["object"]
            if obj not in objects or (
                spec["expected"] and spec["expected"] != objects[obj]["sha256"]
            ):
                raise RuntimeError("selected file differs from original receipt")
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(root / obj, path)
        if sha(view_root / "candidate.json") != candidate_sha:
            raise RuntimeError("candidate bytes mismatch")
        if {x["object"] for x in manifest["files"].values()} != set(objects):
            raise RuntimeError("unreferenced review objects")
        v = select(view_root)
        if set(v.files) != set(manifest["files"]):
            raise RuntimeError("review selection missing or unexpected files")
        for name, spec in v.files.items():
            obj = manifest["files"][name]["object"]
            if spec["expected"] and spec["expected"] != objects[obj]["sha256"]:
                raise RuntimeError("original required evidence hash mismatch: " + name)
        verify_sources(v, max_mib * MIB, materialize=dest is not None)
        if dest is not None:
            shutil.copytree(view_root, dest)
            with (dest / "REVIEW_BUNDLE_MANIFEST.json").open("x") as stream:
                json.dump(manifest, stream, indent=2)
            print("REVIEW FILES", dest, flush=True)
        print(
            "REVIEW_BUNDLE_VERIFIED; oracle tensors and full backup deliberately excluded; "
            "GPU/instruction/performance acceptance still requires independent review",
            flush=True,
        )
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    export = sub.add_parser("extract")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--archive", type=Path)
    export.add_argument("--plan", action="store_true")
    export.add_argument("--max-mib", type=int, default=1024)
    check = sub.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--candidate-sha256", required=True)
    check.add_argument("--max-mib", type=int, default=1024)
    check.add_argument(
        "--dest",
        type=Path,
        help="keep verified, readable reports and source in a new local directory",
    )
    args = parser.parse_args(argv)
    if args.max_mib < 1:
        parser.error("--max-mib must be positive")
    if args.mode == "extract":
        extract(args.output, args.archive, args.max_mib, args.plan)
    else:
        verify(args.archive, args.candidate_sha256, args.max_mib, args.dest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, tarfile.TarError) as exc:
        print("REVIEW BUNDLE FAILED:", str(exc)[:1200], flush=True)
        raise SystemExit(1) from None
