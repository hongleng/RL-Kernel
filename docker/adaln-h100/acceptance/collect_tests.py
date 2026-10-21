"""Collect one acceptance stage without running tests; review before updating its manifest."""

import argparse
import json
import os
from pathlib import Path
import sys

from test_plan import FILTERS, pytest_args


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=FILTERS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["RLK_ADALN_REAL_SHAPES"] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    class Collection:
        def pytest_collection_finish(self, session):
            args.out.write_text(
                json.dumps([item.nodeid for item in session.items], indent=2) + "\n"
            )

    return pytest.main([*pytest_args(args.stage), "--collect-only", "-q"], plugins=[Collection()])


if __name__ == "__main__":
    raise SystemExit(main())
