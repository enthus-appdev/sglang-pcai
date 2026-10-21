"""CPU-only packaging tripwires; deliberately never import the inference engine."""

import ast
import importlib.metadata
import importlib.util
import json
import re
from pathlib import Path

EXPECTED_SGLANG = "0.5.22"
REQUIRED_FIELDS = {
    "model_path", "revision", "host", "port", "served_model_name", "tp_size",
    "moe_runner_backend", "flashinfer_mxfp4_moe_precision", "context_length",
    "chunked_prefill_size", "max_running_requests", "mem_fraction_static",
    "speculative_algorithm", "speculative_draft_model_revision",
    "default_chat_template_kwargs", "enable_metrics",
}


def validate(root, versions):
    if versions["sglang"].split("+", 1)[0] != EXPECTED_SGLANG:
        raise RuntimeError(f"Unexpected SGLang version: {versions['sglang']}")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", versions["flashinfer-python"])
    if not match or tuple(map(int, match.groups())) < (0, 6, 18):
        raise RuntimeError("Hopper MXFP4/FP8 requires FlashInfer >=0.6.18")

    fields_dir = root / "srt/arg_groups/fields"
    fields = set()
    for path in fields_dir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                fields.add(node.target.id)
    missing = REQUIRED_FIELDS - fields
    if missing:
        raise RuntimeError(f"Required server fields missing: {sorted(missing)}")
    if "DSPARK" not in (fields_dir / "spec.py").read_text():
        raise RuntimeError("Bundled DSpark configuration missing")
    override = root / "srt/arg_groups/model_overrides/deepseek_v4.py"
    if "is_sm90" not in override.read_text():
        raise RuntimeError("DeepSeek V4 SM90 backend selection missing")
    return versions


def main():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang package not found")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "torch", "flashinfer-python", "transformers")
    }
    validate(Path(spec.origin).parent, versions)
    print(json.dumps({"packaging_checks": "passed", "versions": versions}, sort_keys=True))
    print("Static packaging checks only; GPU/model qualification is not performed.")


if __name__ == "__main__":
    main()
