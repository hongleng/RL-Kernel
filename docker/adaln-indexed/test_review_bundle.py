"""Literal CPU parser fixtures; none of these records are GPU execution evidence."""

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

import review_bundle as bundle

ASSETS = Path(__file__).parent


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def receipt(path):
    return {p.relative_to(path).as_posix(): bundle.sha(p) for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def completed(tmp_path):
    out = tmp_path / "batch"
    out.mkdir()
    source = out / "source-fixture.py"
    source.write_text("# literal CPU reproducibility fixture only\n")
    baseline = {"source-fixture.py": bundle.sha(source)}
    with tarfile.open(out / "candidate-source.tar.gz", "w:gz") as tar:
        tar.add(source, arcname="candidate-source/source-fixture.py")
    source.unlink()
    put(out / "source-sha256.json", baseline)
    lock = json.loads((ASSETS / "expected-node-lock.json").read_text())
    plan = json.loads((ASSETS / "expected-performance-plan.json").read_text())
    put(out / "node-lock.json", lock)
    put(out / "performance-plan.json", plan)
    put(out / "storage-plan.json", {"fixture": "CPU only"})
    put(out / "target.json", {"gpu": "CPU FIXTURE, NOT GPU RESULTS"})
    binary = out / "native.so"
    binary.write_bytes(b"literal native binary parser fixture, not executable")
    binary_sha = bundle.sha(binary)
    run_dirs = {}
    artifact_dirs = {}
    native_owner = out / "artifacts/cuda-correctness"
    native_owner.mkdir(parents=True)
    for name in ("ptx.log", "sass.log", "resources.log", "version.log"):
        (native_owner / name).write_text("literal CPU instruction parser fixture\n")
    native_files = {p.name: bundle.sha(p) for p in native_owner.iterdir()}
    for name in lock:
        d = out / "runs" / name
        d.mkdir(parents=True)
        run_dirs[name] = d
        sanitizer = name.split("-", 1)[1] in {"memcheck", "racecheck", "initcheck", "synccheck"}
        put(
            d / "result.json",
            dict(
                state="PASS",
                sanitizer="fixture" if sanitizer else None,
                sanitizer_errors={"fixture": [0]},
            ),
        )
        put(d / "nodes.json", lock[name])
        put(
            d / "reports.json",
            [
                dict(node=node, phase=phase, outcome="passed", xfail=False)
                for node in lock[name]
                for phase in ("setup", "call", "teardown")
            ],
        )
        put(
            d / "identity.json",
            dict(
                extension_available=not name.startswith("triton-"), loaded_binary_sha256=binary_sha
            ),
        )
        put(d / "source-sha256.json", baseline)
        put(d / "process-command.json", ["LITERAL CPU FIXTURE"])
        (d / "console.log").write_text("literal CPU suite console\n")
        (d / "results.xml").write_text('<testsuites><testsuite tests="1"/></testsuites>')
        if sanitizer:
            (d / "sanitizer.log").write_text("literal CPU tool log\n")
        if not name.startswith("triton-"):
            put(d / "build.json", dict(binary=str(binary), binary_sha256=binary_sha))
        (out / (d.name + ".log")).write_text("literal CPU outer suite log\n")
        a = out / "artifacts" / name
        a.mkdir(parents=True, exist_ok=True)
        artifact_dirs[name] = a
        if name == "triton-shared":
            put(a / "shared-cache.json", {"fixture": "not an indexed instruction audit"})
        elif name.startswith("triton-"):
            cache = a / "cache/key"
            cache.mkdir(parents=True)
            put(cache / "_rows.json", {"fixture": "compiler metadata"})
            hashes = {}
            for suffix in ("ttir", "ttgir", "llir", "ptx", "cubin", "source"):
                p = cache / ("_rows." + suffix)
                p.write_text("literal CPU kernel parser fixture " + suffix)
                hashes[suffix] = bundle.sha(p)
            (a / "audit.sass").write_text("literal CPU SASS parser fixture")
            put(
                a / "code-audit.json",
                {
                    "entries": [
                        dict(
                            cache_key="key",
                            kernel="_rows",
                            artifacts_sha256=hashes,
                            cuobjdump_sass="audit.sass",
                            sass_sha256=bundle.sha(a / "audit.sass"),
                        )
                    ]
                },
            )
        else:
            put(
                a / "extraction.json",
                dict(
                    binary=str(binary),
                    binary_sha256=binary_sha,
                    output_path=native_owner.relative_to(out).as_posix(),
                    files_sha256=native_files,
                ),
            )
    cpu = out / "preflight/cpu"
    cpu.mkdir(parents=True)
    (cpu / "cpu.xml").write_text(
        '<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0"/></testsuites>'
    )
    (cpu / "cpu.log").write_text("CPU preflight fixture")
    perf = out / "performance/0-attempt-1"
    perf.mkdir(parents=True)
    report = dict(
        plan,
        state="EXECUTION_COMPLETE_REVIEW_PENDING",
        session=0,
        environment={"binary_sha256": binary_sha, "source_sha256": baseline},
        results=[],
        first_use=[],
        traces=[],
        oracles=[],
        generated_sha256={},
    )
    raw = []
    for case in plan["cases"]:
        identity = dict(
            case={k: v for k, v in case.items() if k != "layout"},
            seed=plan["seed"],
            sources=baseline,
        )
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        oracle = out / "oracles" / key
        oracle.mkdir(parents=True, exist_ok=True)
        put(oracle / "identity.json", dict(identity=identity, sha256="0" * 64))
        # Huge sparse cache with deliberately unmatched content: it must never be read or verified.
        if not (oracle / "oracle.pt").exists():
            with (oracle / "oracle.pt").open("wb") as stream:
                stream.truncate(1024**3)
        report["oracles"].append(dict(case=case, key=key, path=str(oracle), sha256="0" * 64))
        for backend in plan["backends"]:
            report["first_use"].append(
                dict(case=case, backend=backend, forward_ms=1.0, backward_ms=1.0)
            )
            actual = "pytorch" if backend == "framework-select01" else backend
            profile = (
                "adaln-select01-framework-v1"
                if actual == "pytorch"
                else "adaln-indexed-fp32-tree-v1"
            )
            report["traces"].append(
                dict(
                    case=case,
                    backend=backend,
                    trace=dict(actual_backend=actual, arithmetic_profile=profile),
                )
            )
            for surface in plan["surfaces"]:
                for mode in plan["surface_modes"].get(surface, plan["modes"]):
                    row = dict(case=case, backend=backend, surface=surface, mode=mode)
                    memory = []
                    for iteration in range(plan["repeat"]):
                        measures = {
                            k: 1.0
                            for k in (
                                "wall_ms",
                                "event_ms",
                                "allocated_before_bytes",
                                "reserved_before_bytes",
                                "peak_allocated_bytes",
                                "peak_reserved_bytes",
                                "incremental_peak_allocated_bytes",
                            )
                        }
                        memory.append(measures)
                        raw.append(
                            dict(
                                row,
                                session=0,
                                iteration=iteration,
                                actual_backend=actual,
                                **measures,
                            )
                        )
                    report["results"].append(
                        dict(
                            row,
                            samples_ms=[1.0] * plan["repeat"],
                            memory_samples=memory,
                            event_summary={"samples_ms": [1.0] * plan["repeat"]},
                        )
                    )
    for suffix in ("json", "ptx", "cubin", "source", "ttir", "ttgir", "llir"):
        file = perf / ("cache/accepted/_rows." + suffix)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("literal accepted CPU compiler fixture " + suffix)
        report["generated_sha256"][file.relative_to(perf).as_posix()] = bundle.sha(file)
    put(perf / "report.json", report)
    (perf / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    (out / ("performance-" + perf.name + ".log")).write_text("literal accepted-session CPU fixture")
    (out / "performance-0-attempt-0.log").write_text("retained storage failure history")
    (out / "performance/0-attempt-0").mkdir()
    (out / "performance/0-attempt-0/NOT_REQUIRED_DATA").write_bytes(b"not accepted")
    put(out / "summaries/session-summary-attempt-0", {"groups": ["CPU fixture"]})
    state = dict(
        state="EXECUTION_COMPLETE_REVIEW_PENDING",
        completed={},
        artifacts={},
        sessions={"0": dict(path=perf.relative_to(out).as_posix(), receipt=receipt(perf))},
        preflight={"cpu": dict(path=cpu.relative_to(out).as_posix(), receipt=receipt(cpu))},
        storage_checks=[],
    )
    for name in lock:
        state["completed"][name] = dict(
            path=run_dirs[name].relative_to(out).as_posix(), receipt=receipt(run_dirs[name])
        )
        state["artifacts"][name] = dict(
            path=artifact_dirs[name].relative_to(out).as_posix(),
            receipt=receipt(artifact_dirs[name]),
        )
    put(out / "state.json", state)
    fields = {
        "source-sha256.json": "source_sha256",
        "candidate-source.tar.gz": "archive_sha256",
        "node-lock.json": "node_lock_sha256",
        "performance-plan.json": "performance_plan_sha256",
        "storage-plan.json": "storage_plan_sha256",
    }
    put(
        out / "candidate.json",
        dict(
            output_path=str(out),
            sessions=1,
            profile="pr-v2",
            **{field: bundle.sha(out / file) for file, field in fields.items()},
        ),
    )
    return out


def forbid_oracle_reads(monkeypatch):
    original = Path.open

    def opened(path, *args, **kwargs):
        if path.name == "oracle.pt":
            pytest.fail("oracle payload was read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)


def test_review_bundle_omits_bulk_data_and_verifies_all_pr_coverage(completed, monkeypatch):
    forbid_oracle_reads(monkeypatch)
    plan = bundle.extract(completed, plan_only=True)
    assert plan["maximum_bytes"] < 16 * bundle.MIB
    manifest = bundle.extract(completed)
    assert not any(p.endswith("oracle.pt") or "NOT_REQUIRED_DATA" in p for p in manifest["files"])
    assert len(manifest["objects"]) < len(manifest["files"])
    archive = Path(str(completed) + ".review.tar.gz")
    destination = completed.parent / "readable-review"
    checked = bundle.verify(archive, manifest["candidate_sha256"], dest=destination)
    assert (destination / "performance/0-attempt-1/samples.jsonl").is_file()
    assert (destination / "candidate-source/source-fixture.py").is_file()
    assert not list(destination.rglob("oracle.pt"))
    with pytest.raises(FileExistsError):
        bundle.verify(archive, manifest["candidate_sha256"], dest=destination)
    assert checked["omitted"] == bundle.OMITTED
    assert (completed / "oracles").exists()


def test_missing_raw_sample_is_rejected_before_writing(completed):
    path = completed / "performance/0-attempt-1/samples.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="missing raw samples"):
        bundle.extract(completed)
    assert not Path(str(completed) + ".review.tar.gz.tmp").exists()


def test_selected_receipt_corruption_is_not_hidden_by_dedup(completed):
    (completed / "runs/triton-shared/console.log").write_text("changed since original receipt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        bundle.extract(completed)
    assert not Path(str(completed) + ".review.tar.gz").exists()


def test_wrong_batch_and_corrupt_archive_are_rejected(completed):
    manifest = bundle.extract(completed)
    archive = Path(str(completed) + ".review.tar.gz")
    with pytest.raises(RuntimeError, match="wrong review batch"):
        bundle.verify(archive, "wrong candidate")
    checksum = Path(str(archive) + ".sha256")
    checksum.write_text("0" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bundle.verify(archive, manifest["candidate_sha256"])


def test_size_guard_is_preflight_and_does_not_silently_omit(completed):
    with pytest.raises(RuntimeError, match="exceeds --max-mib"):
        bundle.extract(completed, max_mib=1)
    assert not Path(str(completed) + ".review.tar.gz.tmp").exists()


def test_missing_mandatory_receipt_entry_cannot_be_replaced_by_current_file(completed):
    state = json.loads((completed / "state.json").read_text())
    del state["completed"]["triton-correctness"]["receipt"]["identity.json"]
    put(completed / "state.json", state)
    with pytest.raises(RuntimeError, match="missing mandatory original receipt"):
        bundle.extract(completed)
    assert not Path(str(completed) + ".review.tar.gz.tmp").exists()


def test_unfinished_summary_is_not_parsed_or_exported(completed):
    (completed / "summaries/session-summary-attempt-99.tmp").write_text("incomplete JSON")
    manifest = bundle.extract(completed)
    assert not any(p.endswith(".tmp") for p in manifest["files"])


def test_packer_does_not_publish_when_output_lock_is_held(completed):
    with bundle.locked(completed):
        with pytest.raises(RuntimeError, match="holds the output lock"):
            bundle.extract(completed)
    assert not Path(str(completed) + ".review.tar.gz").exists()


def test_publication_refuses_destination_created_during_extraction(completed, monkeypatch):
    archive = Path(str(completed) + ".review.tar.gz")
    original = bundle.publish

    def concurrent(pairs):
        archive.write_bytes(b"foreign archive must survive")
        return original(pairs)

    monkeypatch.setattr(bundle, "publish", concurrent)
    with pytest.raises(FileExistsError):
        bundle.extract(completed)
    assert archive.read_bytes() == b"foreign archive must survive"


def test_write_failure_preserves_originals_and_no_final_archive(completed, monkeypatch):
    original = bundle.DigestIO.write
    calls = []

    def fail(writer, data):
        calls.append(True)
        if len(calls) > 1:
            raise OSError("injected archive write failure")
        return original(writer, data)

    monkeypatch.setattr(bundle.DigestIO, "write", fail)
    with pytest.raises(OSError):
        bundle.extract(completed)
    assert (completed / "state.json").exists()
    assert not Path(str(completed) + ".review.tar.gz").exists()
    assert not Path(str(completed) + ".review.tar.gz.sha256").exists()


def test_accepted_session_code_is_retained_and_missing_ptx_refused(completed):
    selected = bundle.select(completed)
    assert "performance/0-attempt-1/cache/accepted/_rows.ptx" in selected.files
    (completed / "performance/0-attempt-1/cache/accepted/_rows.ptx").unlink()
    with pytest.raises(RuntimeError, match="missing or external review evidence"):
        bundle.extract(completed)


def test_successful_triton_session_without_code_inventory_is_rejected(completed):
    path = completed / "performance/0-attempt-1/report.json"
    value = json.loads(path.read_text())
    value["generated_sha256"] = {}
    put(path, value)
    with pytest.raises(RuntimeError, match="missing accepted-session compiler"):
        bundle.extract(completed)


def test_tool_update_during_selection_is_not_recorded_as_executed_code(
    completed, monkeypatch, tmp_path
):
    tool = tmp_path / "review-tool.py"
    tool.write_bytes(Path(bundle.__file__).read_bytes())
    monkeypatch.setattr(bundle, "__file__", str(tool))
    original = bundle.select

    def changed(root):
        value = original(root)
        tool.write_text("# replaced tool after selection started")
        return value

    monkeypatch.setattr(bundle, "select", changed)
    with pytest.raises(RuntimeError, match="tool changed"):
        bundle.extract(completed)
    assert not Path(str(completed) + ".review.tar.gz").exists()


def test_original_failure_reason_is_preserved_without_failed_session_data(completed):
    put(
        completed / "failures/batch-attempt-0", dict(state="FAIL", error="BLOCKED_STORAGE: fixture")
    )
    manifest = bundle.extract(completed)
    assert "failures/batch-attempt-0" in manifest["files"]
    assert not any("NOT_REQUIRED_DATA" in name for name in manifest["files"])


def rewrite_archive(archive, mutate):
    data = {}
    with tarfile.open(archive) as source:
        for member in source:
            data[member.name] = source.extractfile(member).read()
    mutate(data)
    temporary = Path(str(archive) + ".rewrite")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as output:
        for name, value in data.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            import io

            output.addfile(info, io.BytesIO(value))
    temporary.replace(archive)
    Path(str(archive) + ".sha256").write_text(bundle.sha(archive))


def test_missing_required_data_cannot_pass_after_rechecksumming_bundle(completed):
    manifest = bundle.extract(completed)
    archive = Path(str(completed) + ".review.tar.gz")

    def remove(data):
        metadata = json.loads(data["review/manifest.json"])
        spec = metadata["files"].pop("performance/0-attempt-1/samples.jsonl")
        del metadata["objects"][spec["object"]]
        del data["review/" + spec["object"]]
        data["review/manifest.json"] = json.dumps(metadata).encode()

    rewrite_archive(archive, remove)
    with pytest.raises(RuntimeError, match="missing or external review evidence"):
        bundle.verify(archive, manifest["candidate_sha256"])


def test_tool_provenance_tamper_is_rejected_after_new_archive_checksum(completed):
    manifest = bundle.extract(completed)
    archive = Path(str(completed) + ".review.tar.gz")
    rewrite_archive(
        archive, lambda data: data.update({"review/tool.py": b"untrusted changed tool"})
    )
    with pytest.raises(RuntimeError, match="tool provenance mismatch"):
        bundle.verify(archive, manifest["candidate_sha256"])


def test_unsafe_archive_member_is_never_written(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("../escape")
        output.addfile(member)
    Path(str(archive) + ".sha256").write_text(bundle.sha(archive))
    with pytest.raises(RuntimeError, match="unsafe relative"):
        bundle.verify(archive, "unused")
    assert not (tmp_path / "escape").exists()


def test_old_full_backup_cli_is_no_longer_default_pr_command(tmp_path):
    import subprocess
    import sys

    command = [sys.executable, "-S", str(ASSETS / "local/.cache/adaln-indexed/evidence_io.py")]
    result = subprocess.run(
        command + ["--pack", str(tmp_path / "missing")], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "use review_bundle.py extract" in result.stderr
    result = subprocess.run(
        command
        + ["--verify", str(tmp_path / "batch.review.tar.gz"), "--candidate-sha256", "unused"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "require review_bundle.py verify" in result.stderr


def test_successful_session_source_must_match_candidate(completed):
    path = completed / "performance/0-attempt-1/report.json"
    report = json.loads(path.read_text())
    report["environment"]["source_sha256"] = {"source-fixture.py": "wrong source"}
    put(path, report)
    with pytest.raises(RuntimeError, match="performance source differs"):
        bundle.extract(completed)
