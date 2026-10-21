"""The fixed AdaLN test scope used by collection and H100 execution."""

TEST_FILES = [
    "tests/ops/norm/test_adaln_modulation.py",
    "tests/validation/operators/test_adaln_gtest.py",
    "tests/backends/test_adaln_shared_bounds.py",
    "tests/runtime/test_adaln_dispatch.py",
]
FILTERS = {
    "correctness": "",
    "racecheck": "cuda and not real_image and not indexed and not maximum_batch",
    "memcheck": "(cuda or real_image or launch_geometry) and not indexed",
}


def pytest_args(stage):
    args = [*TEST_FILES, "-vv", "-ra", "-p", "no:cacheprovider"]
    if FILTERS[stage]:
        args += ["-k", FILTERS[stage]]
    return args
