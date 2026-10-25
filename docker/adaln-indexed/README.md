# Indexed AdaLN Actions image and manual RunPod batch

This infrastructure branch builds an image for operator source
`5490223a850fe9cb1efd5c3cceeb0fb964ee5c1c`. It does not change that operator
branch or enable indexed qualification. The local ticket06/07 tools are explicit
copies under `local/.cache/adaln-indexed/`, not dependencies on a developer checkout.

The base image digest in `source.json` preserves the previous successful
CUDA12.6.3 / Python3.11.17 / Torch2.12.1+cu126 / Triton3.7.1 environment and its
split dependency layers. The new workdir `/opt/RL-Kernel-indexed` keeps old shared
source outside this batch. Pulling the image still transfers several GiB.

`prepare_context.py` makes a local shallow Git checkout of the pinned commit,
without remotes or credentials, then adds the acceptance assets. This preserves
the real HEAD required by the ticket05/0607 runners. Plain `git archive` would
omit Git metadata and break their identity recording. No local `.venv`, loaded
extension, build output, GPU cache or historical GPU evidence enters the context.

## Build

The existing registered workflow name is retained; execute this branch's version:

```bash
gh workflow run adaln-h100-image.yml --repo hongleng/RL-Kernel \
  --ref infra/adaln-indexed-image -f infrastructure_ref=infra/adaln-indexed-image
```

Actions checks source hashes, HEAD, Python3.11 syntax, pinned environment and pip
dependencies, runs 158 CPU harness/reference/policy checks plus an extension-free
Triton-to-CPU fallback self-check, then collects all
17 suites without GPU execution or extension compilation. Actual node IDs and
the performance plan must equal the reviewed candidate locks. Failures stop the
build. No numerical qualification is inferred from image build success.

Five CPU-input policy cases assume native CUDA selection during resolution:
`test_explicit_indexed_activation_resolves_without_enabling_cpu_select01` and the
three `test_activated_strict_rechecks_actual_domain_before_execution` parameters,
plus `test_ordinary_gpu_resolution_rechecks_shared_cpu_device_with_real_fallback`.
They are explicitly deselected in this extension-free image build, not counted as
passed. Native activation and domain checks remain target evidence requirements.
The self-check verifies actual source import, absence of a native binary, real
Triton selection, CPU fallback reason and forward/backward bytes. Local validation
must use isolated Python `-S` with explicit site-packages to prevent editable import
hooks from loading another checkout's native extension.

The artifact `adaln-indexed-image-evidence` contains the published image digest,
source/infra/workflow commits, CPU JUnit, exact node lists, performance plan,
source hashes and installed packages. Use `image-reference.txt`'s digest in
RunPod. Ensure the new GHCR package is publicly readable, or configure read-only
image pull credentials; a public repository alone does not make its package public.

## RunPod (user initiated)

Use a complete H100, non-MIG, with at least 64GiB host RAM recommended and ample
evidence disk for all retained builds, caches and generated instructions. Start
the container with `sleep infinity`; no GPU job runs automatically.

Save the image digest, Actions URL and package inventory in the evidence parent
directory. The batch output itself must not already exist:

```bash
cd /opt/RL-Kernel-indexed
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p /workspace/evidence
cp /opt/adaln-indexed-image-checks/image-packages.txt /workspace/evidence/
python .cache/adaln-indexed/run_ticket0607.py --start \
  --output /workspace/evidence/adaln-indexed-new --sessions 3
```

Run in tmux for a detachable terminal. The order is CPU prepare, target identity
and resource checks, fresh AdaLN-only builds and full correctness/shared/resource
cases, four sanitizers, generated-code extraction, then three performance sessions.
Failures stop later stages. Resume with the same output and `--phase` only when
source/environment/evidence identities still match; retain failed attempts.

Return the full evidence, including cache/dump and generated instructions bound
by receipts. Do not reuse the old archive rule that excluded JIT caches. Transfer
the archive and checksum to the local computer (runpodctl send/receive or configured
SSH), verify them and inspect logs before terminating the Pod. runpodctl is not
preinstalled by this Dockerfile; it is not required to run the batch.

Independent kernel-only timing remains unavailable. Target correctness, sanitizer
coverage, generated instruction semantics and performance acceptance require
independent review after execution; indexed qualification stays closed.
