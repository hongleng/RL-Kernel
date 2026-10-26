"""CPU regression checks at acceptance harness interfaces; no target results."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch

import benchmark_adaln_indexed as bench
import evidence_io as evidence
import run_ticket0607 as batch

spec = importlib.util.spec_from_file_location(
    "ticket05", Path(__file__).with_name("run_ticket05.py")
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_default_pr_selection_cannot_silently_run_the_legacy_budget():
    selected = bench.plan(bench.parse_args(["--output", "/unused", "--profile", "pr-v2"]))
    assert len(selected["cases"]) == 18
    assert selected["warmup"] == 3 and selected["repeat"] == 12
    assert selected["surface_modes"]["dispatch-production"] == ["forward+backward"]
    cells = len(selected["cases"]) * 3 * sum(map(len, selected["surface_modes"].values()))
    assert cells == 216 and cells * selected["repeat"] == 2592
    assert all(c["route"] in ("alternating", "segmented", "random") for c in selected["cases"])
    for dtype in ("fp32", "bf16"):
        assert (
            dict(shape=[3, 6889, 3072], dtype=dtype, route="random", layout="strided")
            in selected["cases"]
        )
    default = bench.plan(bench.parse_args(["--output", "/unused"]))
    assert default == selected
    with pytest.raises(SystemExit):
        bench.parse_args(["--output", "/unused", "--profile", "legacy-v1"])


def test_binary_reuse_checks_source_flags_arch_tools_abi_and_actual_bytes(tmp_path):
    binary = tmp_path / "_C.so"
    binary.write_bytes(b"literal CPU identity fixture, not an extension")
    identity = dict(
        sources={"kernel.cu": "source"},
        flags=["--fmad=false"],
        arch="9.0",
        tools={"nvcc": "tool hash"},
        abi=False,
        runtime="torch fixture",
    )
    record = dict(binary=str(binary), binary_sha256=native.sha(binary), reuse_identity=identity)
    assert native.validate_reuse(record, copy.deepcopy(identity)) == binary
    for key in identity:
        changed = copy.deepcopy(identity)
        changed[key] = "changed"
        with pytest.raises(RuntimeError, match="identity mismatch"):
            native.validate_reuse(record, changed)
    binary.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="binary hash mismatch"):
        native.validate_reuse(record, identity)


def test_cached_oracle_shares_math_preserves_real_layout_and_rejects_corruption(tmp_path):
    case = dict(shape=[2, 8, 2], dtype="fp32", route="alternating", layout="contiguous")
    cpu, refs, base, record = bench.cached_references(case, 386, tmp_path)
    assert not record["reused"]
    strided = dict(case, layout="strided")
    views, shared, _, second = bench.cached_references(strided, 386, tmp_path)
    assert second["reused"] and second["key"] == record["key"]
    assert all(not t.is_contiguous() and t.is_leaf for t in views)
    assert views[0].requires_grad and views[1].requires_grad
    for group, expected in zip(shared, bench.reference_outputs(views), strict=True):
        for actual, gold in zip(group, expected, strict=True):
            assert torch.equal(actual, gold)
    from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference

    bench.validate_candidate(IndexedAdaLNFixedReference(), views, shared)
    for change in (dict(case, route="all1"), dict(case, dtype="bf16")):
        assert bench.cached_references(change, 386, tmp_path)[3]["key"] != record["key"]
    assert bench.cached_references(case, 387, tmp_path)[3]["key"] != record["key"]
    (Path(record["path"]) / "oracle.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="identity/hash mismatch"):
        bench.cached_references(case, 386, tmp_path)


def test_interrupted_oracle_is_retained_and_rebuilt(tmp_path):
    case = dict(shape=[2, 8, 2], dtype="fp32", route="alternating", layout="contiguous")
    record = bench.cached_references(case, 386, tmp_path)[3]
    directory = Path(record["path"])
    (directory / "identity.json").unlink()
    (directory / "oracle.pt").write_bytes(b"partial write")
    assert not bench.cached_references(case, 386, tmp_path)[3]["reused"]
    assert (
        directory.with_name(directory.name + ".incomplete-0").joinpath("oracle.pt").read_bytes()
        == b"partial write"
    )


def minimal_batch(tmp_path):
    out = tmp_path / "batch"
    out.mkdir()
    source = out / "candidate-source"
    source.mkdir()
    (source / "fixture.py").write_text("# literal recovery fixture\n")
    records = {
        "source-sha256.json": {"fixture.py": evidence.sha(source / "fixture.py")},
        "node-lock.json": {"cuda-correctness": ["node"]},
        "performance-plan.json": {"cases": []},
        "state.json": {"state": "FAIL", "completed": {}, "artifacts": {}, "sessions": {}},
    }
    for name, record in records.items():
        batch.write(out / name, record)
    with tarfile.open(out / "candidate-source.tar.gz", "w:gz") as archive:
        archive.add(source, arcname="candidate-source")
    batch.write(
        out / "candidate.json",
        dict(
            source_sha256=evidence.sha(out / "source-sha256.json"),
            archive_sha256=evidence.sha(out / "candidate-source.tar.gz"),
            node_lock_sha256=evidence.sha(out / "node-lock.json"),
            performance_plan_sha256=evidence.sha(out / "performance-plan.json"),
            sessions=1,
        ),
    )
    (out / "failure.log").write_text("original failure retained\n")
    return out


def test_partial_recovery_is_explicit_and_verifies_content_and_batch_identity(tmp_path):
    out = minimal_batch(tmp_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        evidence.pack(out)
    archive, status = evidence.pack(out, partial=True)
    assert status["recovery"] == "LOCAL_VERIFICATION_PENDING"
    result = evidence.verify_archive(archive, status["candidate_sha256"], partial=True)
    assert result["partial"] and result["recovery"] == "complete" and result["state"] == "FAIL"
    with pytest.raises(RuntimeError, match="wrong batch"):
        evidence.verify_archive(archive, "wrong candidate", partial=True)


def test_resume_receipt_follows_shared_binary_and_refuses_changed_target(tmp_path):
    target = tmp_path / "build"
    target.mkdir()
    binary = target / "_C.so"
    binary.write_bytes(b"fixture binary")
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "_C.so").symlink_to(binary)
    state = dict(completed={"suite": dict(path="worker", receipt=batch.receipt(worker))})
    batch.verify_saved_evidence(tmp_path, state)
    binary.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="prior evidence changed"):
        batch.verify_saved_evidence(tmp_path, state)


def test_empty_archive_is_not_recovered_even_with_matching_checksum(tmp_path):
    archive = tmp_path / "empty.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    Path(str(archive) + ".sha256").write_text(evidence.sha(archive))
    with pytest.raises(RuntimeError, match="empty"):
        evidence.verify_archive(archive, "candidate")


def test_pack_failure_never_sends_or_publishes_checksum(tmp_path):
    out = minimal_batch(tmp_path)
    (out / "outside").symlink_to(tmp_path / "missing")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sender = bindir / "runpodctl"
    marker = tmp_path / "sent"
    sender.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\n')
    sender.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(Path(evidence.__file__)), "--pack", str(out), "--partial", "--send"],
        env=dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"]),
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0 and not marker.exists()
    assert not Path(str(out) + ".tar.gz.sha256").exists()


def test_missing_required_report_cannot_be_archived_as_complete(tmp_path):
    out = minimal_batch(tmp_path)
    state = json.loads((out / "state.json").read_text())
    dest = out / "runs/native"
    dest.mkdir(parents=True)
    (dest / "result.json").write_text('{"state":"PASS"}')
    state["completed"] = {"cuda-correctness": dict(path="runs/native", receipt=batch.receipt(dest))}
    batch.write(out / "state.json", state)
    with pytest.raises(FileNotFoundError):
        evidence.pack(out, partial=True)


def test_performance_selection_and_raw_sample_failures(tmp_path):
    # Literal CPU parser data. This does not assert a GPU PASS.
    case = dict(shape=[2, 8, 2], dtype="fp32", route="alternating", layout="strided")
    plan = dict(
        cases=[case],
        backends=["cuda"],
        surfaces=["dispatch-production"],
        modes=["forward", "forward+backward"],
        surface_modes={"dispatch-production": ["forward+backward"]},
        repeat=2,
    )
    row = dict(case=case, backend="cuda", surface="dispatch-production", mode="forward+backward")
    report = dict(
        plan,
        state="EXECUTION_COMPLETE_REVIEW_PENDING",
        session=0,
        environment={"binary_sha256": "binary"},
        results=[dict(row, samples_ms=[1.0, 2.0])],
    )
    batch.write(tmp_path / "report.json", report)
    samples = [
        dict(row, iteration=i, session=0, actual_backend="cuda", wall_ms=float(i + 1), event_ms=1.0)
        for i in range(2)
    ]
    raw = tmp_path / "samples.jsonl"
    raw.write_text("".join(json.dumps(s) + "\n" for s in samples))
    evidence.verify_performance(tmp_path, plan, 0, "binary")
    for bad in (samples[:1], samples + samples[:1], [dict(samples[0], mode="forward"), samples[1]]):
        raw.write_text("".join(json.dumps(s) + "\n" for s in bad))
        with pytest.raises(RuntimeError):
            evidence.verify_performance(tmp_path, plan, 0, "binary")


def test_tar_io_failure_never_publishes_sendable_archive(tmp_path, monkeypatch):
    out = minimal_batch(tmp_path)

    def disk_failure(*args, **kwargs):
        raise OSError("injected archive disk failure")

    monkeypatch.setattr(tarfile.TarFile, "add", disk_failure)
    with pytest.raises(OSError, match="disk failure"):
        evidence.pack(out, partial=True)
    assert not Path(str(out) + ".tar.gz").exists()
    assert not Path(str(out) + ".tar.gz.sha256").exists()
    assert (out / "failure.log").read_text() == "original failure retained\n"


def test_recovery_rejects_missing_log_even_after_rechecksumming_archive(tmp_path):
    out = minimal_batch(tmp_path)
    archive, status = evidence.pack(out, partial=True)
    broken = tmp_path / "missing-log.tar.gz"
    with tarfile.open(archive) as original, tarfile.open(broken, "w:gz") as changed:
        for member in original.getmembers():
            if member.name == "evidence/failure.log":
                continue
            changed.addfile(member, original.extractfile(member) if member.isfile() else None)
    Path(str(broken) + ".sha256").write_text(evidence.sha(broken))
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        evidence.verify_archive(broken, status["candidate_sha256"], partial=True)


def test_running_batch_refuses_pack_and_parallel_resume(tmp_path):
    out = minimal_batch(tmp_path)
    with evidence.exclusive_output(out):
        with pytest.raises(RuntimeError, match="in use"):
            evidence.pack(out, partial=True)
        with pytest.raises(RuntimeError, match="in use"):
            with evidence.exclusive_output(out):
                pass
    evidence.pack(out, partial=True)


def test_suite_cannot_resume_or_archive_with_missing_teardown_or_duplicate_call():
    reports = [
        dict(node="node", phase=p, outcome="passed", xfail=False)
        for p in ("setup", "call", "teardown")
    ]
    evidence.verify_reports(["node"], reports)
    for bad in (reports[:2], reports + [reports[1]], [dict(r, outcome="skipped") for r in reports]):
        with pytest.raises(RuntimeError, match="setup/call/teardown"):
            evidence.verify_reports(["node"], bad)


def test_complete_archive_is_readable_after_relocation_and_requires_generated_code(tmp_path):
    # Literal parser fixtures only; neither a native binary nor GPU evidence.
    out = minimal_batch(tmp_path)
    candidate = evidence.read(out / "candidate.json")
    candidate["output_path"] = str(out)
    run = out / "runs/cuda-attempt-0"
    run.mkdir(parents=True)
    binary = run / "build/_C.so"
    binary.parent.mkdir()
    binary.write_bytes(b"CPU fixture")
    digest = evidence.sha(binary)
    data = {
        "result.json": {"state": "PASS", "sanitizer": None},
        "nodes.json": ["node"],
        "reports.json": [
            dict(node="node", phase=p, outcome="passed", xfail=False)
            for p in ("setup", "call", "teardown")
        ],
        "identity.json": {"loaded_binary_sha256": digest},
        "build.json": {"binary": str(binary), "binary_sha256": digest},
        "source-sha256.json": evidence.read(out / "source-sha256.json"),
        "process-command.json": ["CPU parser fixture"],
    }
    for file, value in data.items():
        batch.write(run / file, value)
    (run / "console.log").touch()
    (run / "results.xml").write_text('<testsuites><testsuite tests="1"/></testsuites>')
    (out / (run.name + ".log")).write_text("CPU fixture log")
    artifact = out / "artifacts/native"
    artifact.mkdir(parents=True)
    for file in ("ptx.log", "sass.log", "resources.log", "version.log"):
        (artifact / file).write_text("CPU extraction parser fixture\n")
    batch.write(
        artifact / "extraction.json",
        dict(
            binary=str(binary),
            binary_sha256=digest,
            output_path="artifacts/native",
            files_sha256={
                file: evidence.sha(artifact / file)
                for file in ("ptx.log", "sass.log", "resources.log", "version.log")
            },
        ),
    )
    preflight = out / "preflight/cpu"
    preflight.mkdir(parents=True)
    (preflight / "cpu.xml").write_text(
        '<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0"/></testsuites>'
    )
    (preflight / "cpu.log").write_text("CPU fixture\n")
    performance = out / "performance/0-attempt-0"
    performance.mkdir(parents=True)
    plan = dict(
        cases=[{}], backends=["cuda"], surfaces=["dispatch-production"], modes=["forward"], repeat=1
    )
    row = dict(case={}, backend="cuda", surface="dispatch-production", mode="forward")
    batch.write(out / "performance-plan.json", plan)
    candidate["performance_plan_sha256"] = evidence.sha(out / "performance-plan.json")
    batch.write(out / "candidate.json", candidate)
    batch.write(
        performance / "report.json",
        dict(
            plan,
            state="EXECUTION_COMPLETE_REVIEW_PENDING",
            session=0,
            environment={"binary_sha256": digest},
            results=[dict(row, samples_ms=[1.0])],
        ),
    )
    (performance / "samples.jsonl").write_text(
        json.dumps(
            dict(row, iteration=0, session=0, actual_backend="cuda", wall_ms=1.0, event_ms=1.0)
        )
        + "\n"
    )
    (out / ("performance-" + performance.name + ".log")).write_text("CPU fixture")
    state = dict(
        state="EXECUTION_COMPLETE_REVIEW_PENDING",
        completed={},
        artifacts={},
        sessions={},
        preflight={},
    )
    for namespace, name, directory in (
        ("completed", "cuda-correctness", run),
        ("artifacts", "cuda-correctness", artifact),
        ("sessions", "0", performance),
        ("preflight", "cpu", preflight),
    ):
        state[namespace][name] = dict(
            path=str(directory.relative_to(out)), receipt=batch.receipt(directory)
        )
    batch.write(out / "state.json", state)
    batch.write(out / "target.json", {"gpu": "CPU parser fixture"})
    (out / "summaries").mkdir()
    batch.write(out / "summaries/session-summary-attempt-0", {"groups": [row]})
    archive, status = evidence.pack(out)
    assert not evidence.verify_archive(archive, status["candidate_sha256"])["partial"]
    (artifact / "sass.log").write_text("")
    state["artifacts"]["cuda-correctness"]["receipt"] = batch.receipt(artifact)
    batch.write(out / "state.json", state)
    with pytest.raises(RuntimeError, match="empty native extraction"):
        evidence.validate_tree(out)


def test_evidence_hash_streams_instead_of_loading_a_whole_archive(tmp_path, monkeypatch):
    import hashlib

    path = tmp_path / "archive-fixture"
    path.write_bytes(b"streamed evidence")

    def forbid_whole_file_read(*args, **kwargs):
        raise AssertionError("whole-file allocation")

    monkeypatch.setattr(Path, "read_bytes", forbid_whole_file_read)
    assert evidence.sha(path) == hashlib.sha256(b"streamed evidence").hexdigest()


def test_partial_recovery_handles_a_prepare_failure_before_locks_or_state_exist(tmp_path):
    out = tmp_path / "interrupted-prepare"
    out.mkdir()
    batch.write(out / "candidate.json", {"state": "PREPARING", "output_path": str(out)})
    (out / "collection-failure.log").write_text("original CPU collection failure\n")
    archive, status = evidence.pack(out, partial=True)
    assert status["recovery"] == "LOCAL_VERIFICATION_PENDING"
    result = evidence.verify_archive(archive, status["candidate_sha256"], partial=True)
    assert result["state"] == "PREPARATION_INCOMPLETE"
    assert set(result["missing_preparation_files"]) == {
        "source-sha256.json",
        "candidate-source.tar.gz",
        "node-lock.json",
        "performance-plan.json",
    }
    assert not (out / "state.json").exists()
    with pytest.raises((RuntimeError, KeyError, FileNotFoundError)):
        evidence.validate_tree(out)


def test_v2_performance_recovery_requires_first_use_trace_memory_and_event_evidence(tmp_path):
    plan = dict(
        schema="adaln-indexed-benchmark-pr-v2",
        cases=[{}],
        backends=["cuda"],
        surfaces=["dispatch-production"],
        modes=["forward+backward"],
        repeat=1,
    )
    measured = dict(
        wall_ms=1.0,
        event_ms=1.0,
        allocated_before_bytes=0,
        reserved_before_bytes=0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        incremental_peak_allocated_bytes=1,
    )
    row = dict(case={}, backend="cuda", surface="dispatch-production", mode="forward+backward")
    report = dict(
        plan,
        state="EXECUTION_COMPLETE_REVIEW_PENDING",
        session=0,
        environment={"binary_sha256": "fixture"},
        first_use=[dict(case={}, backend="cuda", forward_ms=1.0, backward_ms=1.0)],
        traces=[
            dict(
                case={},
                backend="cuda",
                trace=dict(actual_backend="cuda", arithmetic_profile="adaln-indexed-fp32-tree-v1"),
            )
        ],
        oracles=[dict(case={}, key="fixture")],
        results=[
            dict(
                row,
                samples_ms=[1.0],
                memory_samples=[measured],
                event_summary={"samples_ms": [1.0]},
            )
        ],
    )
    batch.write(tmp_path / "report.json", report)
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(dict(row, session=0, iteration=0, actual_backend="cuda", **measured)) + "\n"
    )
    evidence.verify_performance(tmp_path, plan, 0, "fixture")
    for field in ("first_use", "traces", "oracles"):
        bad = copy.deepcopy(report)
        bad[field] = []
        batch.write(tmp_path / "report.json", bad)
        with pytest.raises(RuntimeError):
            evidence.verify_performance(tmp_path, plan, 0, "fixture")
    for field in ("memory_samples", "event_summary"):
        bad = copy.deepcopy(report)
        bad["results"][0].pop(field)
        batch.write(tmp_path / "report.json", bad)
        with pytest.raises(KeyError):
            evidence.verify_performance(tmp_path, plan, 0, "fixture")
