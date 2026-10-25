# Indexed AdaLN CUDA/Triton follow-up

Status: approved local specification with user amendments; implementation may begin at ticket 01. Q1 arithmetic and Q2 strict delivery are approved. This document does not authorize publication, commits, long builds or GPU acceptance runs.

## Problem Statement

Indexed AdaLN currently executes a PyTorch select01 fallback on every device. It implements two-bank token routing, but its framework backward has no declared token reduction order and is rejected in strict mode. Callers need dedicated NVIDIA CUDA and Triton forward/backward implementations with an explicit arithmetic profile, independent accuracy validation and byte-level replay evidence.

This is an explicitly requested indexed follow-up. It is not an unfulfilled Phase A requirement of issue #386, which excluded edit-specific conditioning. Shared AdaLN behavior and generic validation contracts remain in their existing scope.

## Solution

Keep the current operator interface. Add dedicated indexed CUDA/Triton execution, preserving explicit fallback for valid unsupported inputs in ordinary mode. Deliver strict support for the independent `adaln-indexed-fp32-tree-v1` arithmetic profile after the claimed implementations pass their own acceptance gates.

The first profile uses the shared hidden-dimension arithmetic and FP32 token-ascending left folds independently for each sample, bank and hidden channel. This is a versioned arithmetic choice, not a permanent restriction on every future indexed implementation. Performance optimization first preserves arithmetic order; a different fixed reduction tree needs a separately declared and verified profile.

## User Stories

1. As a training caller, I want token index 0/1 to select my sample's first/second bank, so conditioning identity survives batching.
2. As a caller, I want a `[1,S]` index to broadcast across samples while each sample retains its own modulation rows.
3. As a caller, I want FP32 and BF16 y/gate outputs with the same dtype and device as x.
4. As a training caller, I want correct dx and both banks' shift/scale/gate gradients for arbitrary mixed routes.
5. As a gate-only caller, I want x.grad to remain None and unused shift/scale slices to receive zero when the modulation tensor participates.
6. As a caller of strided views, I want correct values and gradients without having to prepare contiguous inputs myself.
7. As a numerical reviewer, I want every reduction, multiplication, rounding point and output cast to have one declared meaning.
8. As a reviewer, I want independent CPU accuracy gold and a separate fixed-order byte reference, so a matching production bug cannot establish correctness.
9. As a strict caller, I want unsupported configurations and unqualified fallback to fail closed.
10. As an ordinary caller, I want previously valid indexed eps and geometry outside the new kernel domain to retain explicit framework fallback.
11. As a debugger, I want resolution and successful execution observations to identify the actual backend/profile and fallback reason accurately.
12. As a GPU operator user, I want batch position and declared launch changes to preserve forward/backward bytes within the supported arithmetic profile.
13. As a reviewer, I want empty banks to stay +0 even when selected tokens in the other bank contain NaN/Inf.
14. As a caller, I want gate selection to preserve original dtype bits, including BF16 payloads.
15. As a performance reviewer, I want comparable indexed measurements against the existing select01 fallback, with raw samples and overhead disclosed.
16. As a shared AdaLN user, I want this follow-up to preserve existing shared routing, arithmetic and support boundaries.

## Implementation Decisions

### Public semantics and execution domain

- x is nonempty `[B,S,H]`, modulation is `[2B,3H]` ordered shift/scale/gate; all floating inputs share FP32 or BF16 dtype and device. Index is `[B,S]` or `[1,S]`, same device, bool/uint8/int8/int16/int32/int64, values only 0 or 1. For sample b, index 0 selects row b and index 1 row B+b. Both y and gate are `[B,S,H]`; index has no gradient.
- Dedicated NVIDIA CUDA and NVIDIA Triton initially target H<=4096, B<=65535, positive S, int32-safe logical element counts and accessed offsets for all inputs, outputs and temporary buffers. Grid design must not accidentally turn B into an unsupported 2B grid dimension. Padded inactive lanes must not access memory; bounds proofs distinguish accessed offsets from masked integer expressions.
- Validate shape/dtype/device/eps and invalid index values before running an accelerated path. Check metadata bounds before contiguous copies, output allocation or launch. Strided inputs/upstreams, including expanded views, may be copied internally; autograd edges must return gradients correctly to their source views.
- Illegal inputs raise errors. Valid but unsupported inputs may use existing select01 fallback in ordinary mode. Resolution knows only part of the input, so execution must re-evaluate actual geometry, eps and indexed mode before choosing the final path. No blanket exception-to-fallback handler may hide programming or launch errors.
- Kernel eps is rounded to finite FP32, nonnegative, and a positive input must not round to zero. FP32 subnormals, including 2^-149, are supported. Existing finite nonnegative eps outside this domain retain ordinary framework fallback. Strict rejects that unqualified fallback. eps=0 with zero variance propagates NaN as declared, without an invented stabilizing branch.

### Arithmetic profile

- Determinism means repeated results for fixed inputs/environment; bitwise equality is a comparison across declared executions; the numerical contract defines correct arithmetic and its exceptions. Fixed S order alone proves none of the other arithmetic choices.
- Pad H to the next power of two and use the shared lower/upper pairwise FP32 tree for mean, variance and dX hidden reductions. Use the existing shared normalization, rstd and explicit dX arithmetic order. RN operations, verified preservation of required subnormals on the target GPU/compiler/instruction profile, and no compiler reassociation or FMA contraction that changes rounding points are required.
- Normalize x and compute affine intermediates in FP32. Cast y and dx once at the output boundary; preserve the shared FP32-to-BF16 rounding policy.
- For each `(b,bank,h)`, initialize three FP32 accumulators to +0. Walk s in ascending logical order. Only selected tokens contribute: dy for dshift, FP32-rounded dy*norm for dscale, and FP32 dgate-output upstream for dgate. Round each addition to FP32. The dscale product must round before addition; an FMA with a single rounding is a different arithmetic profile. Cast the final sums once to modulation dtype.
- No cross-sample sum, atomics, reassociated segment sums or mask-times-zero substitution. All three empty-bank gradients are +0, including their sign bits. A nonempty bank also starts from +0; a single -0 contribution does not receive an identity-payload guarantee.
- Forward gate directly copies selected original-dtype bits. Accumulated dgate has arithmetic semantics and no NaN-payload identity guarantee. Normalization/affine arithmetic NaN payloads are outside byte equality; their classification and isolation are tested separately. Signed zero, finite subnormals and relevant infinities follow the declared arithmetic rather than an allclose-only check.

### Autograd semantics

- A custom Function must distinguish absent output gradients from explicitly supplied zero tensors; `set_materialize_grads(False)` is the expected mechanism, subject to behavior tests.
- If grad_y is absent, return dx=None, skip normalization backward, and make shift/scale slices +0 if modulation participates through gate. If gate upstream is absent, its modulation slice is +0. Explicit zero dy still executes the declared arithmetic and may interact with nonfinite norm.
- Empty-bank slice zeros are distinct from a whole input that is unused or frozen. Frozen leaf grad remains None. No new higher-order-gradient guarantee is introduced.

### References and validation ownership

- Independent accuracy gold uses CPU FP32 leaves, independent LayerNorm/autograd normalization and an independently written indexed affine VJP with selected-token folds. It does not call production normalization/reduction helpers. Gold outputs/gradients remain FP32; candidate output and gradient dtypes are checked separately from tolerance comparisons.
- A separate CPU byte reference spells out the fixed H tree, explicit dX and bank folds. A double-precision sqrt rounded to FP32 matches the existing correctly rounded reference strategy. Its own literal tests must reject concrete wrong arithmetic. It must not silently replace the existing framework fallback or its API domain.
- Registry/dispatch, backend wrappers/kernels, AdaLN references, specialized validation, tests, benchmark and operator documentation own the change. Reuse existing within-backend statistics logic where appropriate while keeping the accuracy gold independent.
- Inputs determine the indexed profile. Do not add a redundant caller profile argument, generic schema migration, compatibility shim or performance-selection framework.

### Strict policy and observations

- Strict is a delivery requirement, not an incremental slice's default. During development, unsupported/unvalidated indexed strict execution remains rejected. Complete strict policy and an AdaLN-local default-closed qualification mechanism are implemented before final evidence collection; acceptance exercises the same activation/execution path intended for delivery. Activation after review must not rewrite arithmetic, dispatch selection or launch behavior. A core path change requires new affected evidence. Opening strict is gated on independent evidence for each claimed backend/target runtime, including complete forward/backward tests.
- Preserve the distinction between an arithmetic policy check and hardware evidence. No automatic hardware qualification, binary attestation, backward-observed flag or profiler claim is inferred from successful forward trace. Do not introduce a runtime certification database or repurpose generic provenance to implement this feature.
- Strict resolution/execution check the applicable profile, NVIDIA device, hidden/mode and support-domain constraints. Ordinary fallback records the actual implementation and reason; a failed call clears previous successful observations. A profile string or another backend's PASS cannot qualify this implementation.
- CUDA/Triton are compared byte-for-byte within their pinned, qualified environment and the declared nonfinite exceptions. Neither inherits the parent's shared H100 PASS, and cross-device generality is not assumed.

## Testing Decisions

Approved public seams (L3, L2, L1 respectively):

1. Existing AdaLN resolver followed by its returned callable and trace: routing, fallbacks, strict policy, values, gradients and failed-call observations.
2. Existing public backend constructors/callables: direct CUDA/Triton launch-parameter and supported-domain checks. Native callable remains the observable legacy fallback baseline.
3. Accuracy-gold callable and fixed-byte-reference callable: independent literal validation and candidate comparisons. No private kernel helper or production reduction graph is a correctness oracle.

Use existing AdaLN semantic, backend-boundary, dispatch, validation and benchmark test conventions. A test must assert behavior through these seams and reject a concrete wrong implementation; one test/minimal implementation per red-green slice. Do not build a horizontal bank of all tests before implementing the first behavior.

- Start with a small B2/S8/H2 FP32 case, separate sample/bank tags, mixed int32 routes and nonzero dy/dgate. The old framework path must fail the actual-backend expectation rather than accidentally pass as the accelerated implementation.
- Cover mixed/all0/all1, empty banks, broadcast routes, batch moves, every index dtype, strided/expanded inputs/upstreams, frozen inputs, gate-only/y-only/both, and explicit zero versus absent upstream.
- Literal cancellation cases distinguish ascending fold from a reordered tree for dshift/dscale/dgate. A separate literal distinguishes FMA from multiply-rounded-then-add. Use epsilon/input choices that actually realize the expected normalized value; do not assume norm=1 from a convenient input.
- With all tokens routed to bank0, identical x/M/eps/dtype/dy must yield byte-equal indexed bank0 dshift/dscale and shared gradients. Test bank1 +0 independently. Shared dgate is a passthrough while indexed dgate is a sum, so do not assert their direct equality. Independent gold remains mandatory.
- Cover BF16 ties, raw gate payloads, signed zero, subnormal eps, eps=0, invalid eps, legal kernel-domain-external eps, selected nonfinite propagation and empty-bank isolation. Accuracy, raw-byte and classification checks have distinct purposes.
- Test H around powers of two and 4096; B around 65535; int32 address boundaries with metadata rejection rather than giant allocation. Positive B boundaries use nonzero dx and sample identity tags. Test every declared CUDA threads and Triton num_warps/reduction_tile configuration across logical batch placement/padding.
- Compare identical logical token sets. Appending genuinely contributing tokens changes parameter gradients and is not an invariance test. Any zero-contribution padding fixture specifies route and upstream arithmetic.
- Run H3072 and S4096/6889/6032 with FP32/BF16 and realistic segmented routing, plus mixed/dispersed routing. Retain shared regression coverage.
- Fresh CUDA build and per-process source/binary identity precede target evidence. Record actual collected nodes, skips/xfails, tool/runtime/hardware versions and sanitizer summaries. Explicitly integrate CUDA/Triton raw-byte comparison after both are implemented. Distinguish actual PASS, dependency/device SKIP and COMMAND_PREPARED. Run relevant CUDA/Triton memcheck/racecheck cases early enough to catch shared-buffer hazards; add initcheck/synccheck where justified. Host meta PASS or uninstrumented kernels do not establish GPU acceptance.
- Benchmark old select01 versus indexed CUDA/Triton on the same inputs/routes/upstreams. Separate forward, backward-only and forward+backward; disclose validation/copies/allocation, peak memory, cold compilation/cache identities, warm timing, backend order, raw samples and variability. No speedup acceptance threshold is assumed.

## Out of Scope

- Shared arithmetic/support changes, projection, gated residual, final AdaLN, model integration or complete Diffusers/trajectory equivalence.
- ROCm/MUSA/NPU qualification, arbitrary new dtypes, unqualified compiler/runtime combinations or an implicit higher-order autograd promise.
- A new parallel reduction profile, atomics, general contract/provenance/CI rewrites, environment deployment or public evidence upload as side effects of implementation.

## Further Notes

- Parent #559 was rechecked OPEN at head/review base `beb5bff59d2a9611b3d12eeeec775ac6d757321b`; target test-qwenimage is `16b81175874ecf11d5ff05c00de3b69e9457591b`. Retain the parent branch for review. After merge, verify actual landed content before migrating only indexed commits.
- Semantic call-site reference is Diffusers revision `784ae091a80d568f089842d766cac34637eef86e`; Triton source investigation used v3.7.1 commit `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`. These references do not establish a model checkpoint fingerprint or a new GPU execution profile.
- Main contract/experience provenance and exact local development entries are indexed in the companion review file, keeping issue text free of checkout-specific implementation paths.
- Downloads over 1GiB, runs over 10 minutes or uncertain-duration builds/tests require commands prepared for the user to execute. No model or large repository download is needed. Publication and commits remain outside this preparation authorization.
