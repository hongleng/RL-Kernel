# SPDX-License-Identifier: Apache-2.0
"""CPU acceptance of batch receipts and failure propagation; no GPU timings."""

import json
import subprocess
import sys

import pytest

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ticket0607", Path(__file__).with_name("run_ticket0607.py")
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_failed_stage_keeps_log_and_stops_caller(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(subprocess.CalledProcessError):
        batch.execute(
            [sys.executable, "-c", 'print("failure retained"); raise SystemExit(3)'],
            log,
        )
    assert "failure retained" in log.read_text()
    with pytest.raises(FileExistsError):
        batch.execute([sys.executable, "-c", "pass"], log)


@pytest.mark.parametrize(
    "invalid", ["skip", "xfail", "missing-call", "changed-node", "changed-source"]
)
def test_result_parser_rejects_incomplete_or_mismatched_evidence(tmp_path, invalid):
    dest = tmp_path / "runs/cuda-correctness"
    dest.mkdir(parents=True)
    records = {
        "result.json": {"state": "PASS"},
        "nodes.json": ["public-node"],
        "reports.json": [
            {
                "node": "public-node",
                "phase": "call",
                "outcome": "passed",
                "xfail": False,
            }
        ],
        "source-sha256.json": {"source.py": "same"},
    }
    (tmp_path / "source-sha256.json").write_text(json.dumps({"source.py": "same"}))
    (tmp_path / "node-lock.json").write_text(json.dumps({"cuda-correctness": ["public-node"]}))
    for name, value in records.items():
        (dest / name).write_text(json.dumps(value))
    batch.check_result(tmp_path, ("cuda", "correctness", None))
    if invalid == "skip":
        records["reports.json"][0]["outcome"] = "skipped"
    elif invalid == "xfail":
        records["reports.json"][0]["xfail"] = True
    elif invalid == "missing-call":
        records["reports.json"] = []
    elif invalid == "changed-node":
        records["nodes.json"] = ["different"]
    else:
        records["source-sha256.json"] = {"source.py": "different"}
    for name, value in records.items():
        (dest / name).write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        batch.check_result(tmp_path, ("cuda", "correctness", None))


def test_identity_verifier_rejects_manifest_archive_and_node_tampering(tmp_path):
    for name in ("source-sha256.json", "candidate-source.tar.gz", "node-lock.json"):
        (tmp_path / name).write_text("original")
    candidate = {
        "source_sha256": batch.sha(tmp_path / "source-sha256.json"),
        "archive_sha256": batch.sha(tmp_path / "candidate-source.tar.gz"),
        "node_lock_sha256": batch.sha(tmp_path / "node-lock.json"),
    }
    (tmp_path / "candidate.json").write_text(json.dumps(candidate))
    for name in ("source-sha256.json", "candidate-source.tar.gz", "node-lock.json"):
        (tmp_path / name).write_text("tampered")
        with pytest.raises(RuntimeError, match="identity changed"):
            batch.verify_candidate(tmp_path)
        (tmp_path / name).write_text("original")


def test_resuming_failed_work_uses_new_attempt_without_overwriting(tmp_path):
    first = batch.next_attempt(tmp_path, "cuda-correctness")
    first.mkdir()
    (first / "failed.log").write_text("original failure")
    second = batch.next_attempt(tmp_path, "cuda-correctness")
    assert second != first
    assert not second.exists()
    assert (first / "failed.log").read_text() == "original failure"


def test_resume_reserves_attempt_even_if_child_dies_before_output_creation(tmp_path):
    first = batch.next_attempt(tmp_path, "startup")
    (tmp_path / (first.name + ".log")).write_text("import failed before mkdir")
    second = batch.next_attempt(tmp_path, "startup")
    assert second != first
    assert not first.exists() and not second.exists()


def test_generated_code_stage_refuses_partial_correctness_before_any_tool_launch(tmp_path):
    state = {"completed": {"cuda-correctness": {}}}
    with pytest.raises(RuntimeError, match="requires all correctness and sanitizer"):
        batch.extract_artifacts(tmp_path, state, {})
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("changed", ["suite", "artifact", "session"])
def test_resume_checks_every_namespace_even_when_task_names_collide(tmp_path, changed):
    state = {"completed": {}, "artifacts": {}, "sessions": {}}
    for namespace, directory in (
        ("completed", "suite"),
        ("artifacts", "artifact"),
        ("sessions", "session"),
    ):
        dest = tmp_path / directory
        dest.mkdir()
        (dest / "evidence").write_text("original")
        state[namespace]["same-name"] = {"path": directory, "receipt": batch.receipt(dest)}
    batch.verify_saved_evidence(tmp_path, state)
    (tmp_path / changed / "evidence").write_text("changed")
    with pytest.raises(RuntimeError, match="prior evidence changed"):
        batch.verify_saved_evidence(tmp_path, state)
