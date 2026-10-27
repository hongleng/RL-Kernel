"""Export the pinned source with standalone Git identity and local acceptance tools."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent
    config = json.loads((assets / "source.json").read_text())
    repo = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=assets, text=True
    ).strip()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", "--quiet", str(out)], check=True)
    # A local shallow fetch retains the real commit without copying checkout credentials.
    subprocess.run(
        ["git", "fetch", "--quiet", "--no-tags", "--depth=1", repo, config["source_sha"]],
        cwd=out,
        check=True,
    )
    subprocess.run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=out, check=True)
    # Overlay bytes are locked separately from the operator commit.
    actual = {
        str(p.relative_to(assets / "local")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (assets / "local").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual != config["acceptance_sha256"]:
        raise RuntimeError("acceptance overlay differs from source.json lock")
    shutil.copytree(assets / "local", out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    metadata = out / "image-acceptance"
    metadata.mkdir()
    for name in (
        "selfcheck.py",
        "source.json",
        "environment.json",
        "requirements.txt",
        "expected-node-lock.json",
        "expected-performance-plan.json",
        "expected-storage-plan.json",
        "README.md",
    ):
        shutil.copyfile(assets / name, metadata / name)
    shutil.copyfile(assets / "Dockerfile", out / "Dockerfile")
    shutil.copyfile(assets / "Dockerfile.dockerignore", out / "Dockerfile.dockerignore")
    files = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(out).parts
    }
    (metadata / "source-profile.json").write_text(
        json.dumps(dict(config, files=files), indent=2) + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
