import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_image", ROOT / "checks/check_image.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ImageContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fields = self.root / "srt/arg_groups/fields"
        self.fields.mkdir(parents=True)
        self.field_file = self.fields / "model.py"
        self.field_file.write_text("\n".join(f"{name}: str" for name in sorted(check.REQUIRED_FIELDS)))
        (self.fields / "spec.py").write_text('ALGORITHMS = ["DSPARK"]\n')
        override = self.root / "srt/arg_groups/model_overrides"
        override.mkdir()
        (override / "deepseek_v4.py").write_text("def select(): return platform.is_sm90\n")
        self.versions = {"sglang": "0.5.22", "flashinfer-python": "0.7.1"}

    def test_supported_packaging_contract(self):
        self.assertEqual(check.validate(self.root, self.versions), self.versions)

    def test_local_version_suffix_is_accepted(self):
        self.versions["sglang"] = "0.5.22+cu130"
        check.validate(self.root, self.versions)

    def test_different_sglang_release_is_rejected(self):
        self.versions["sglang"] = "0.5.23"
        with self.assertRaisesRegex(RuntimeError, "Unexpected SGLang"):
            check.validate(self.root, self.versions)

    def test_old_or_invalid_flashinfer_is_rejected(self):
        for version in ["0.6.17", "unknown"]:
            with self.subTest(version=version):
                self.versions["flashinfer-python"] = version
                with self.assertRaisesRegex(RuntimeError, "FlashInfer"):
                    check.validate(self.root, self.versions)

    def test_missing_server_field_is_rejected(self):
        self.field_file.write_text(self.field_file.read_text().replace("model_path: str", ""))
        with self.assertRaisesRegex(RuntimeError, "model_path"):
            check.validate(self.root, self.versions)

    def test_missing_dspark_is_rejected(self):
        (self.fields / "spec.py").write_text('ALGORITHMS = ["EAGLE"]\n')
        with self.assertRaisesRegex(RuntimeError, "DSpark"):
            check.validate(self.root, self.versions)

    def test_missing_sm90_selection_is_rejected(self):
        (self.root / "srt/arg_groups/model_overrides/deepseek_v4.py").write_text("pass\n")
        with self.assertRaisesRegex(RuntimeError, "SM90"):
            check.validate(self.root, self.versions)

    def test_checks_do_not_import_inference_packages(self):
        check.validate(self.root, self.versions)
        for name in ["sglang", "torch", "flashinfer"]:
            self.assertNotIn(name, sys.modules)

    def test_pinned_base_and_arguments_only_entrypoint(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("lmsysorg/sglang:v0.5.22-cu130@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e", dockerfile)
        entrypoint = next(line for line in dockerfile.splitlines() if line.startswith("ENTRYPOINT "))
        self.assertEqual(json.loads(entrypoint.removeprefix("ENTRYPOINT ")), [
            "/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m", "sglang.launch_server",
        ])
        self.assertIn("CMD []", dockerfile)
        self.assertIn("RUN python3 /usr/local/lib/sglang-pcai/check_image.py", dockerfile)

    def test_prs_cannot_publish_or_write_registry_cache(self):
        workflow = (ROOT / ".github/workflows/build.yml").read_text()
        guard = "github.event_name != 'pull_request'"
        self.assertIn("push: ${{ " + guard + " }}", workflow)
        self.assertIn("cache-to: ${{ " + guard + " &&", workflow)
        self.assertIn("if: " + guard, workflow)
        self.assertNotIn("pull_request_target", workflow)
        self.assertIn("needs: license", workflow)
        self.assertIn("platforms: linux/amd64", workflow)
        for tag in ["type=sha,format=long", "type=ref,event=branch", "type=semver,pattern={{version}}"]:
            self.assertIn(tag, workflow)

    def test_build_context_is_allowlisted(self):
        lines = (ROOT / ".dockerignore").read_text().splitlines()
        self.assertEqual(lines, ["**", "!Dockerfile", "!checks/", "!checks/check_image.py"])

    def test_license_exists(self):
        self.assertIn("Apache License", (ROOT / "LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
