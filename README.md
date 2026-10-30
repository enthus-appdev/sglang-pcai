# sglang-pcai

Pinned [SGLang](https://github.com/sgl-project/sglang) serving images for **HPE Private Cloud AI (PCAI)**. Build/publishing conventions follow [vllm-pcai](https://github.com/enthus-appdev/vllm-pcai).

## Why this image exists

The target PCAI interface accepts **arguments, not an executable override**. The inspected upstream image has NVIDIA's entrypoint but no default server command. This wrapper supplies:

```text
/opt/nvidia/nvidia_entrypoint.sh python3 -m sglang.launch_server <PCAI arguments>
```

There are **no SGLang patches, additional Python dependencies, baked model weights or weight conversions**. The build performs CPU-only installed-version/source tripwires; it does not import the engine, download a model or run GPU inference. Publishing an image does not deploy it to PCAI.

## Base image

- SGLang **v0.5.22**, CUDA13 image, **linux/amd64**.
- Upstream source: `baa9befa381511cd4d56f83980951e5a45fffc6d`.
- Base: `docker.io/lmsysorg/sglang:v0.5.22-cu130@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e`.
- Observed upstream image configuration declares FlashInfer 0.7.1. The build checks installed package metadata, required launch fields, DSPARK configuration and DeepSeek V4 SM90 selection without importing GPU packages.

These checks establish packaging/source expectations, **not** two-H100-NVL fit, numerical correctness, long-context stability or performance.

**Observed PCAI startup blocker (2026-10-21):** the TP2 trial launched this image but repeatedly failed during NCCL communicator initialization with `No space left on device` while extending `/dev/shm/nccl-*` to 34,210,180 bytes (32.625 MiB segment payload). It never established model-serving readiness. The serving container's total SHM capacity was not measurable with available permissions. Resolve shared-memory provisioning or qualify a targeted NCCL workaround before treating the example below as deployable; lowering context is not an established fix.

## Build and publish

GitHub Actions workflow: [`.github/workflows/build.yml`](.github/workflows/build.yml).

| Event | Result |
|---|---|
| Pull request to `main` | Run tests/license check and build the image, including installed-image tripwires. No image or registry-cache writes. |
| Push to `main` | Publish `:main`, `:latest` and `:sha-<full-commit>` to GHCR. |
| Push a `v*` semver tag | Publish version, major/minor, major and SHA tags; stable semver releases also receive the metadata action's `latest` tag. |
| Manual workflow dispatch | Build/publish the selected ref; `latest` follows default-branch/stable-release metadata rules. |

Registry: **`ghcr.io/enthus-appdev/sglang-pcai`**. Deployment should use the published digest or full SHA tag, not mutable `latest`.

The `License Check` must pass before the build job starts. Builds use Buildx, registry-backed `:buildcache`, OCI metadata/provenance and the workflow's built-in `GITHUB_TOKEN`; no personal token secret is required. CUDA extension compilation/sccache machinery from vllm-pcai is intentionally omitted because this image does not compile or patch CUDA operators.

**One-time GHCR visibility:** GitHub creates container packages private by default, even for public repositories. After first publication, an owner must verify/set the package's intended visibility in package settings and verify an anonymous pull if PCAI requires public access. Otherwise configure supported registry credentials. A public source repository alone is not evidence of a public image.

Manual build/push, only when explicitly desired:

```bash
docker build --platform linux/amd64 -t ghcr.io/enthus-appdev/sglang-pcai:local .
docker push ghcr.io/enthus-appdev/sglang-pcai:local
```

The pinned base is large; allow disk space for compressed/unpacked layers. Building does not require GPUs. Runtime model loading does.

## PCAI arguments

Select the published custom image and allocate the intended GPU count in PCAI. Provide **only server arguments**, not `python3 -m ...`, Docker `--gpus`, `--ipc`, capability or volume options.

Example **unqualified smoke configuration**, not a production rollout instruction:

```text
--model-path deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 --load-format safetensors --host 0.0.0.0 --port 8080 --served-model-name deepseek-v4-flash --tensor-parallel-size 2 --moe-runner-backend flashinfer_mxfp4 --flashinfer-mxfp4-moe-precision fp8 --kv-cache-dtype fp8_e4m3 --context-length 32768 --max-running-requests 1 --chunked-prefill-size 1024 --mem-fraction-static 0.95 --cuda-graph-max-bs-decode 1 --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 --default-chat-template-kwargs={"thinking":true,"reasoning_effort":"high"} --enable-metrics --trust-remote-code
```

The FP8 precision switch selects **FP8 activations with the original FP4 expert weights**, not a separately requantized checkpoint. The reference snapshot is not an assertion about the currently deployed model revision; verify intended weights/hashes before comparison.

After a separately approved target-only correctness check, bundled 0731 DSpark can be investigated with:

```text
--speculative-algorithm DSPARK --speculative-draft-model-revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

Let the checkpoint infer its draft geometry: gamma5, verify window6. Do not supply a separate preview EAGLE checkpoint. Verify PCAI's actual argv serialization, particularly the JSON argument.

Runtime requirements:

- Writable model/cache storage and sufficient quota; Hub model loading is a runtime operation, not part of the image build. No credentials belong in the image or public configuration.
- Explicit `--host 0.0.0.0 --port 8080`: `EXPOSE` alone does not change SGLang's defaults.
- Confirm actual serving-container RAM, `/dev/shm`, driver/GPU topology and cold-start probe window. Host IPC is not automatically mandatory, nor is a historical64MiB SHM observation proof of compatibility.
- Preserve platform authentication; separately verify backend API/admin routes, probes and metric scraping. This wrapper adds no authentication layer.
- `/health` and, when enabled, `/metrics` are upstream SGLang routes. They are not model-output correctness proofs. No vLLM `/collect_env` route or vLLM CPU-KV-offload configuration is carried over.
- A SGLang memory fraction of0.95 is not semantically identical to vLLM utilization0.95. The small smoke context/concurrency does not qualify262K or six concurrent requests.

No default model, resource allocation or deployment is embedded in the image.

## Local checks and upgrades

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

Dependabot checks Docker and GitHub Actions daily. A base update requires a reviewed tag/digest change, corresponding version/source-contract test updates and a fresh image build. Do not relax failing tripwires merely to make an upgrade green. GPU/quality/context tests remain separate, explicitly authorized work.

## Layout

```text
Dockerfile                    # pinned upstream + tripwire + server entrypoint
checks/check_image.py         # CPU-only installed-image contract check
tests/test_image_contract.py  # local contract and publishing-policy regressions
.github/workflows/build.yml   # PR validation and GHCR publication
.github/dependabot.yml        # base/action update proposals
```

## License

Repository files: [Apache-2.0](LICENSE). Upstream SGLang and other image components retain their respective licenses. This repository does not vendor model weights or chat templates.
