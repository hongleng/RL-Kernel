"""Create a clean image context from the fixed acceptance source and bundled runners."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

SOURCE_SHA = "69bd47f72a24d1177580d9b11b6efdfeb6c6bab5"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent
    repo = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=assets, text=True
        ).strip()
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", SOURCE_SHA],
            cwd=repo,
            check=True,
        )
        with tarfile.open(archive) as source:
            source.extractall(out, filter="data")
    shutil.copytree(
        assets / "acceptance",
        out / "acceptance",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("Dockerfile", "requirements.txt"):
        shutil.copy2(assets / name, out / name)
    files = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    manifest = json.loads((out / "acceptance/test-manifest.json").read_text())
    assert manifest["source_commit"] == SOURCE_SHA
    profile = dict(
        commit=SOURCE_SHA,
        comparison_base="16b81175874ecf11d5ff05c00de3b69e9457591b",
        files=files,
        expected_tests={stage: len(nodes) for stage, nodes in manifest["stages"].items()},
    )
    (out / "acceptance/source-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    subprocess.run([sys.executable, str(out / "acceptance/selfcheck.py")], check=True)


if __name__ == "__main__":
    main()
