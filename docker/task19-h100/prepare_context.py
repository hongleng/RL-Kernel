"""Export only pinned Git objects and image tools; no network or compilation."""

import argparse
import shutil
import subprocess
from pathlib import Path

from run import BASE, HEAD, check_source


def prepare(repo, out):
    check_source(repo)
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    for commit in (HEAD, BASE):
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "--quiet",
                "--no-tags",
                "--depth=1",
                str(repo.resolve()),
                commit,
            ],
            check=True,
        )
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", HEAD], check=True)
    # No worktree pointer, remote, credentials, local cache or old binary is copied.
    config = (source / ".git/config").read_text()
    if "[remote " in config or "extraheader" in config.lower():
        raise RuntimeError("unexpected Git credentials/remote in context")
    assets = Path(__file__).resolve().parent
    tools = out / "tools"
    tools.mkdir()
    for name in (
        "run.py",
        "test_run.py",
        "build_image.py",
        "Dockerfile",
        "README.md",
        "gpu-plan.json",
    ):
        shutil.copyfile(assets / name, tools / name)
    shutil.copyfile(assets / "Dockerfile", out / "Dockerfile")
    shutil.copyfile(assets / "Dockerfile.dockerignore", out / "Dockerfile.dockerignore")
    check_source(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo, args.out.resolve())
