"""CPU-only checks of the evidence CLI, not fake GPU validation."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("task19_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EvidenceCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        for name in (
            "environment.json",
            "runner.py",
            "builder.py",
            "Dockerfile",
            "build.log",
            "gpu.log",
            "comparison.log",
            "nvidia-smi.txt",
        ):
            (self.out / name).write_text("test fixture only\n")
        (self.out / "native.so").write_bytes(b"not a real native binary; CPU fixture")
        source_files = {
            "rl_engine/backends/cuda/logprob/logp.py": "0" * 64,
            "rl_engine/backends/musa/logprob/logp.py": "1" * 64,
            "rl_engine/ops/logprob/fused_logp.py": "2" * 64,
            "tools/validation/operators/compare_fused_logp_ownership.py": "3" * 64,
        }
        runner.save(
            self.out / "source.json",
            dict(head=runner.HEAD, baseline=runner.BASE, files=source_files),
        )
        image = "ghcr.io/hongleng/rl-kernel/task19-h100@sha256:" + "a" * 64
        runtime = dict(torch="2.4.1", cuda="12.4", gpu="NVIDIA H100 80GB", image=image)
        runner.save(self.out / "environment.json", runtime)
        runner.save(self.out / "status.json", dict(state="PASS", head=runner.HEAD, image=image))
        runner.save(
            self.out / "build.json",
            dict(
                head=runner.HEAD,
                baseline=runner.BASE,
                base_image=runner.BASE_IMAGE,
                files=source_files,
                extension_sha256=runner.digest(self.out / "native.so"),
                build_log_sha256=runner.digest(self.out / "build.log"),
                runner_sha256=runner.digest(self.out / "runner.py"),
                builder_sha256=runner.digest(self.out / "builder.py"),
                dockerfile_sha256=runner.digest(self.out / "Dockerfile"),
                gpu_plan_sha256=runner.digest(runner.GPU_PLAN),
            ),
        )
        shutil.copyfile(runner.GPU_PLAN, self.out / "gpu-plan.json")
        self.write_xml()
        cases = [
            dict(
                dtype=dtype,
                shape=shape,
                strided=strided,
                output_sha256="0" * 64,
                gradient_sha256="1" * 64,
                variant_sha256={v: "2" * 64 for v in runner.VARIANTS},
            )
            for dtype in ("torch.float32", "torch.float16", "torch.bfloat16")
            for shape in ((6, 257), (2, 3, 4097), (1, 2, 151936))
            for strided in (False, True)
        ]
        runner.save(
            self.out / "comparison.json",
            dict(
                head=runner.HEAD,
                baseline=runner.BASE,
                dirty="",
                backward="native fused_logp_backward; FP32 upstream, input-dtype gradient",
                cases=cases,
                extension_sha256=runner.digest(self.out / "native.so"),
                python_source_sha256={
                    k: v for k, v in source_files.items() if k.startswith("rl_engine/")
                },
                runner_sha256=source_files[
                    "tools/validation/operators/compare_fused_logp_ownership.py"
                ],
                **runtime,
            ),
        )
        runner.save(
            self.out / "sm90-fallback.json",
            dict(
                state="PASS",
                experimental_tma_executed=False,
                generic_native_forward=True,
                native_fused_backward=True,
            ),
        )

    def write_xml(self, count=None, child=""):
        nodes = runner.load_plan()["nodeids"]
        if count is None:
            count = len(nodes)
        cases = []
        for i, node in enumerate(nodes[:count]):
            path, name = node.split("::", 1)
            classname = path.removesuffix(".py").replace("/", ".")
            cases.append(
                f'<testcase classname="{classname}" name="{name}">'
                f'{child if i == 0 else ""}</testcase>'
            )
        (self.out / "gpu.xml").write_text(
            "<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>"
        )

    def verify(self):
        runner.checksums(self.out)
        return subprocess.run(
            [sys.executable, str(SCRIPT), "verify", "--out", str(self.out)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_complete_selected_evidence_is_accepted(self):
        self.assertEqual(self.verify().returncode, 0)

    def test_missing_binary_is_rejected_even_with_fresh_checksums(self):
        (self.out / "native.so").unlink()
        self.assertNotEqual(self.verify().returncode, 0)

    def test_skipped_or_missing_gpu_cases_are_rejected(self):
        total = len(runner.load_plan()["nodeids"])
        for count, child in ((total, "<skipped/>"), (total - 1, "")):
            with self.subTest(count=count, child=child):
                self.write_xml(count, child)
                self.assertNotEqual(self.verify().returncode, 0)

    def test_replaced_node_is_rejected_even_at_the_same_count(self):
        path = self.out / "gpu.xml"
        path.write_text(path.read_text().replace('name="test_', 'name="wrong_test_', 1))
        self.assertNotEqual(self.verify().returncode, 0)

    def test_old_pytorch_backward_claim_is_rejected(self):
        path = self.out / "comparison.json"
        report = json.loads(path.read_text())
        report["backward"] = "existing PyTorch VJP; CUDA native forward and variants"
        runner.save(path, report)
        self.assertNotEqual(self.verify().returncode, 0)

    def test_wrong_source_is_rejected(self):
        runner.save(self.out / "source.json", dict(head="wrong batch", baseline=runner.BASE))
        self.assertNotEqual(self.verify().returncode, 0)

    def test_wrong_matrix_or_variant_is_rejected(self):
        path = self.out / "comparison.json"
        report = json.loads(path.read_text())
        report["cases"][0]["variant_sha256"].pop("indexed_out")
        runner.save(path, report)
        self.assertNotEqual(self.verify().returncode, 0)

    def test_corrupt_selected_bytes_are_rejected(self):
        runner.checksums(self.out)
        (self.out / "native.so").write_bytes(b"corrupt")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "verify", "--out", str(self.out)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)

    def test_prebuilt_checks_reject_source_binary_and_abi_drift(self):
        repo = self.out / "repo"
        (repo / "rl_engine").mkdir(parents=True)
        binary = repo / "rl_engine/_C.test.so"
        binary.write_bytes(b"binary")
        source = repo / "source.py"
        source.write_text("source")
        torch = SimpleNamespace(
            __version__="2.4.1",
            version=SimpleNamespace(cuda="12.4"),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
        )
        manifest = dict(
            head=runner.HEAD,
            baseline=runner.BASE,
            base_image=runner.BASE_IMAGE,
            files={"source.py": runner.digest(source)},
            torch="2.4.1",
            cuda="12.4",
            python_abi=sysconfig.get_config_var("SOABI"),
            torch_cxx11_abi=False,
            runner_sha256=runner.digest(SCRIPT),
            builder_sha256=runner.digest(SCRIPT.with_name("build_image.py")),
            dockerfile_sha256=runner.digest(SCRIPT.with_name("Dockerfile")),
            gpu_plan_sha256=runner.digest(runner.GPU_PLAN),
            extension_sha256=runner.digest(binary),
            build_log_sha256=runner.digest(self.out / "build.log"),
        )
        runner.save(self.out / "build.json", manifest)
        runner.check_prebuilt(repo, self.out, torch)
        for filename in (source, binary):
            original = filename.read_bytes()
            filename.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                runner.check_prebuilt(repo, self.out, torch)
            filename.write_bytes(original)
        torch.version.cuda = "12.6"
        with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
            runner.check_prebuilt(repo, self.out, torch)
        torch.version.cuda = "12.4"
        binary.unlink()
        with self.assertRaisesRegex(RuntimeError, "no runtime compilation"):
            runner.check_prebuilt(repo, self.out, torch)

    def test_runtime_has_no_build_command(self):
        import ast

        tree = ast.parse(SCRIPT.read_text())
        runtime = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        text = ast.get_source_segment(SCRIPT.read_text(), runtime)
        self.assertNotIn("build_ext", text)
        self.assertNotIn("setup.py", text)
        with patch.object(runner, "command", side_effect=AssertionError("must not build")):
            fake = SimpleNamespace(
                __version__="wrong",
                version=SimpleNamespace(cuda="wrong"),
                _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
            )
            # A missing binary/provenance never initiates a recovery build.
            with self.assertRaises(FileNotFoundError):
                runner.check_prebuilt(self.out, self.out / "missing", fake)


if __name__ == "__main__":
    unittest.main()
