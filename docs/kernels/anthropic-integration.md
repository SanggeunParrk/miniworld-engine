# Anthropic kernel integration

This integration builds on [Anthropic's biomolecular inference work](https://github.com/anthropics/uplifting-biomolecular-modeling),
pinned at `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. See [project direction](../standards.md)
for the acknowledgment and the distinction between upstream and our contribution.

## Source and attribution

The optimization kits and shared runtime are imported under
`third_party/anthropic/upstream/`. Stock models and model weights are excluded.
The upstream licenses, NOTICE, and per-kit third-party notices are retained.
[UPSTREAM.json](../../third_party/anthropic/UPSTREAM.json) records the original Git
blob and SHA-256 of every imported file; [INVENTORY.md](../../third_party/anthropic/INVENTORY.md)
lists source files containing GPU-kernel syntax, including variants and test
sources. This source inventory does not mean every file has been executed.

`miniworld-engine dev import-anthropic CHECKOUT third_party/anthropic` verifies the pinned
revision and original bytes before copying. Do not rerun it over a runtime tree
containing local builds without preserving those build artifacts first.
Compiled binaries are local artifacts, excluded from Git. On another stack,
rebuild with the upstream build recipes and record the new ABI and hashes.

## Engine entry points

### Release routing audit (2026-10-05)

The upstream remote HEAD still matches the pinned revision above. All 1,296
carried `common/opt_core` files match the import manifest. The current Torch
environment imports 33 META packages; `pallas_glut` imports in the isolated
JAX/Tokamax environment from `.validation/anthropic-jax.env`. The combined
34-package connection audit does not mean every package imports in Torch alone,
or that every architecture-specific variant executes on A100.

Public APB-based modules now call the release's `fast` selector with its actual
registered geometry (`dit_h16d48`, `pf_h16d24`, or `msarow_h8d32`) instead of
unconditionally pinning `apb_attn` with `cell=None`. The selection, including any
explicit stock choice, remains visible in `anthropic_selection`. Generic engine
geometries without a release cell retain an explicitly named APB composition.
They are not labelled as the release's measured full-module winner. A named
primitive row still propagates its refusal; it is not silently replaced.

The provider distinguishes eager and graph timing columns. Warm up and capture
under the same policy:

```python
from miniworld_engine.integrations.anthropic import execution_policy

with torch.no_grad(), execution_policy(timing="graph"):
    # Run warm-up, correctness reference calls and CUDA Graph capture here.
    output = module(*inputs)
```

TriangleAttention's default `anthropic_row="auto"` resolves to `block:fast`
(core-only `fast` with Q/K normalization). Explicit rows remain available.
Shared conditioning now uses upstream DTK's `mod_period` / `gate_period` APIs;
broadcasts inside the row geometry still materialize the correct broadcast order.

The Anthropic SWA Atom DiT composition supports `torch.compile(fullgraph=True)`
through four primitive custom ops in `integrations/anthropic_swa_ops.py`.
They call the unchanged upstream RMS modulation, gate/residual, SwiGLU and
gather-attention entry points; loading and row validation happen outside Dynamo
tracing. Projections, RoPE and window-index construction remain in the compiled
graph. The block is not wrapped as one opaque op. Upstream's inference-only and
value-dtype limits remain: gather attention accepts BF16/FP16 values and refuses
FP32 values. Its K=129 row remains an explicitly opted-in candidate, locally
qualified on A100. See the [compiled SWA comparison](../records/a100-swa-anthropic-compile-20261005.md)
for actual execution evidence, correctness checks and all six protocol shapes.

A100 single-direction TriMul now enters `trimul_native.face.serve` directly,
including the release's size/tile lookup. The bidirectional operation still
requires the documented native K1/two-contraction/K3 composition because the
upstream face has no bidirectional contract. The local CUDA rebuild cannot pass
the release cubin-hash identity check. After manifest and loader checks, the
adapter therefore replays the **unchanged original byte vectors** with the
upstream development option `ignore_build=True`, once per device/process;
any applicable byte mismatch raises. Qualification passes 31 original cases,
with four cases skipped by the release's own unsupported-exact rules. Expected
outputs, sources, and release manifests are not rewritten. This validates the
rebuilt outputs, not identity to the original compiler's binary or upstream
certification of this software stack.

The Protenix v2 `8.0|3.7` recipe enables tier-selected token attention but does
not enable its H100 `dit_fused` stack. MiniWorld module compositions remain
explicit adapters to our shapes/residual/mask contracts; they are not claimed
to be an unmodified full Protenix model. In particular, the release has measured
token attention cells for sample counts 1 and 5: requesting 2 samples makes its
`fast` selector explicitly choose stock SDPA. Both the previous pinned-row
comparison and the new release-policy comparison must be interpreted with
their recorded selections.

```python
import torch
from miniworld_engine.modules import Transition, TriangleMultiplication, TriangleAttention

transition = Transition(128, implementation="anthropic").cuda().bfloat16().eval()
trimul = TriangleMultiplication(128, implementation="anthropic").cuda().bfloat16().eval()
attention = TriangleAttention(128, implementation="anthropic", anthropic_row="block:triattn_native").cuda().eval()

with torch.no_grad():
    y = transition(x)
    z = trimul(pair, residue_mask)
    a = attention(pair, residue_mask)
```

Use the normal model weight-loading and BF16 activation/projection setup.
These modules retain their existing residual convention. Single-direction
TriMul accepts outgoing or incoming; `BidirectionalTriangleMultiplication` composes
native K1/K3 with two contractions. TriangleAttention supports both core-only rows (e.g. `triattn_native`) and full
upstream surrounds (`block:triattn_native`). The latter packs weights and connects
upstream LN + projections, the attention core, and gate + output projection. It
uses an out-of-place engine residual add because the upstream residual mode
mutates the input. It preserves starting/ending frames and key-only residue masks.
The selected prologue/epilogue plan is recorded in `anthropic_selection`; a plan
may explicitly select statements for an unsupported fused surround. Q/K RMSNorm
requires a core-only row. Primitive and full-module timings are distinct.

`implementation="anthropic"` is explicit. The existing `miniworld` dispatch and
training paths are unchanged. A forced `engine_backend="triton"` rejects the
Anthropic backend, which may execute CUDA. Unsupported module families refuse
the backend instead of silently running another implementation.

The inference provider API requires `torch.no_grad()` or `torch.inference_mode()`. It
rejects grad-enabled calls before launching, and rejects active training
dropout. TriMul modules additionally expose an explicit `anthropic_row="native_rebuilt"`
training baseline: unchanged native K1/K3 forward, PyTorch recomputation/cuBLAS
backward, external row dropout/residual. Both single-direction and bidirectional
modules support it. Training scope and validation: `archive/docs-20260928:docs/anthropic/trimul-training.md`.
The newer `anthropic_row="training_saved"` keeps the previous training fusion
boundaries and saved tensors, replaces the front/F567 kernels with Anthropic CUDA
derivatives, and reuses the existing Triton/cuBLAS backward. Both TriMul module
classes support this explicit row. These rows are distinct; the native baseline
does not have an optimized backward. Inference packed weight caches are rebuilt
after parameter replacement, in-place updates, state-dict loading, or dtype/device
changes. Construct model Parameters outside inference_mode so they retain
version counters.

## Primitive providers and carried kernels

`miniworld_engine.integrations.anthropic` exposes inference wrappers for
triangle multiplication, triangle attention, Transition, LayerNorm, shared
pair-bias attention, and atom attention. For providers returning `(output,
selection)`, retain the selection in run records. `provider(name)` exposes the
upstream provider and its named rejection/selection rules. `carried_kernel(name)`
routes any upstream META-registered kernel to the imported copy, including
atom-window, gather attention, template embedding, and DiT primitives. These
lower-level surfaces preserve upstream signatures and dependencies;
`operation(name)` exposes `msa_fused.msa_triton`, `msa_opm`, `msa_pwa`, and
`msa_pwa2`. These imported operation surfaces require callers to honor their
shape, dtype, mask, and numerical contracts; they do not
claim model-level integration or autograd support.

Source checkouts find the vendored tree automatically. Installed packages can
set `MINIWORLD_ANTHROPIC_ROOT` to the imported `upstream/` directory. A different
already-loaded `opt_core` is rejected to avoid accidentally measuring another copy.

## Local H100 binary builds

The release includes binaries for other software stacks. Our node02 stack is
PyTorch 2.10.0+cu128, CPython 3.10, with a CUDA-12.9-capable driver. Some CUDA-13
cubins cannot be loaded by this driver. A loader failure is an unsupported build,
not a performance result.

For TriMul, `anthropic_row="native_rebuilt"` explicitly selects a locally rebuilt
payload. Set `TRIMUL_NATIVE_BUILD_DIR` to its `build/` directory; the same parent
must retain `python/`, `csrc/`, and regenerated `testvectors/`. The original CUDA
sources must match UPSTREAM.json. This row is measured against an independent
reference and is not advertised as upstream's bitwise-certified `native_exact`.

Record separately: original source hashes, compiler/header versions, local binary
hashes, loader checks, and independent output checks. Updating binary checksum
manifests does not transfer upstream's numerical or performance certification.

## Qualification and profiling

### A100 native TriangleAttention local build (2026-10-04)

The `triattn_native` v11 package includes an sm80 CUDA member. On the cssb stack
(PyTorch 2.13.0+cu129, CPython 3.12, nvcc 12.9), rebuild `triattn_sm80_ext` with
the upstream recipe in a separate scratch copy. A successful build alone is not
enough: the release loader requires a checksum entry for the new ABI's binary.
The following registration verifies all 1,296 carried upstream files against
the pinned Git revision, the source hashes in the compiler's build record, and
the binary hash. It writes `SHA256SUMS` and `LOCAL_BUILD.json` **inside the new
ABI directory**, leaving the original release manifest, CUDA sources, loader,
and expected test vectors unchanged.

```bash
# SOURCE is the pinned Git checkout; ROOT is a separate scratch copy containing common/opt_core.
export MINIWORLD_ANTHROPIC_ROOT="$ROOT"
V11="$ROOT/common/opt_core/opt_core/kernels/triattn/triattn_native/pkg/v11"
# Run inside the target PyTorch environment, with its nvcc on PATH and CUDA_HOME set.
python "$V11/tools/build_prebuilt.py" --exts triattn_sm80_ext --no-loadcheck
python -m miniworld_engine.tools.record_anthropic_triattn_build \
  --root "$ROOT" --source "$SOURCE" \
  --stack torch2.13.0+cu129-cpython-312-x86_64-linux-gnu
# On an allocated A100:
python -m pytest tests/integrations/test_anthropic_triattn_a100_gpu.py -q
python benchmarks/runners/bench.py target=triangle_attention level=module mode=inference \
  'implementations=[anthropic]' compile=false +triattn_anthropic_row=block:triattn_native \
  min_seq_len=384 max_seq_len=768 seq_len_step=384 mask_prob=0.1
```

Validated on A100 80GB PCIe (Slurm job 63482): all three original A100 byte-gate
vectors match bitwise and select `cuda_80`; 18 module cases cover L128/384/768,
starting/ending attention, absent/sparse/fully masked keys, FP32 reference checks,
and CUDA Graph replay after changing the inputs. Three CPU tests check that
selection and serving refusals propagate. This is a local rebuild of unchanged
upstream sources, not a claim of upstream certification for the new ABI.

The block adapter uses the upstream provider through a strict callable core.
The upstream `pair_fused` string `tier:<row>` silently substitutes `flash_triattn`
after a refusal; that behavior is inappropriate for an explicitly named benchmark.
The adapter now raises instead and records the successful core selection in
`anthropic_selection['core']`. It remains inference-only. The public module
dispatcher now admits the integrated Anthropic module families listed below.

### A100 module integration (2026-10-04)

Use `implementation="anthropic"` and `torch.no_grad()` / `inference_mode()`.
`integrations.anthropic_modules` connects upstream kernels with ordinary torch
GEMMs and tensor transforms. It enters before MiniWorld's CUDA/Triton fast paths;
MSA requests no longer accidentally execute this engine's own OPM/PWA kernels.
These are inference compositions, not new upstream fused kernels or backward implementations.

| Public module | Anthropic execution | A100 numerical/graph coverage |
| --- | --- | --- |
| TriangleMultiplication, BidirectionalTriangleMultiplication | v5 sm80 K1/K3 + cuBLAS | L128/384; C64/128/256/384, bidirectional C64/128 |
| TriangleAttention | strict native sm80 core + upstream surround | L128/384/768, starting/ending, sparse/absent/empty keys |
| Transition | `pf` at C128/256; explicit `v2@bm32bh32w4s2il1` at C384 | module L128/384 C128; additional primitive C256/384 |
| LayerNorm, RMSNorm | upstream LN / DTK RMSNorm | L128/384 C128 |
| AdaptiveLayerNorm, ConditionedTransition | DTK LN/modulation/SwiGLU/gate + cuBLAS | L128/384, shared conditioning |
| AttentionPairBias, AugmentedAttentionPairBias | `apb_attn` + upstream norms/gates + cuBLAS | L128/384; masks, multiple batches, changed weights |
| token/atom DiTBlock | augmented attention + conditioned transition | L128/384; token 768/384/128 and atom 128/128/16 |
| BiasOnlyDiTBlock | APB with explicit zero Q/K + DTK + cuBLAS | L128/384; token 768/384/128, H16×48 |
| Bias-only / bidirectional TriangleAttention | projected APB composition + DTK gate | L128/384; starting/ending, self-attention and bias-only |
| PairformerBlock | connected TriMul + TriAttn + Transition children | L128/384; self-attention/bias-only × separate/bidirectional TriMul |
| OuterProductMean | `msa_opm.forward_mask_norm` | L128/384, S64, C_m64/C_z128/H32; masks and changed weights |
| OuterProductMean, normalize_before_proj=False | `opm_core`, explicit a1 tile, then masked count division | L128/384, C_m64/C_z256/H32 |
| MSAPairWeightedAveraging | `msa_pwa.forward_masked` | L128/384, S64, C_m64/C_z128, H8×32; masks and changed weights |
| SwiGLUFFN | DTK SwiGLU + cuBLAS | L128/384 C128 |
| SWA3DRoPEAttention, SWADiTBlock | gather attention + DTK + torch RoPE + cuBLAS | L128/384, H4×32, half-window64, padded/empty sequences |

SWA uses **129** keys for a ±64 sliding window. This is an explicitly opted-in
upstream candidate cell qualified locally; it is not the release's different
32-query/128-key atom-window operation. Atom DiT remains dense attention; its
C_pair=16 projection uses separate LN and GEMM because upstream `ln_proj.pair_bias`
does not serve that width. All-masked APB batches retain the module's uniform
finite-logit convention. Graph replay reads changed activations;
fresh eager calls observe changed weights.

Primitive qualification also exercises atom-window, gather attention, template
embedding, both APB rows, DTK, MSA LN+linear, LN+linear, and scalar OPM core.
For scalar OPM C_z256 the release's default tile exceeds A100 shared memory;
`bench_anthropic.py --family opm_core --width 256 --row a1` explicitly uses the
release's smaller `CFG_VARIANTS['a1']`. A bare Transition `v2` has no selected A100
launch; specify the configuration above. TriMul `v4` does not serve C384; native
sm80 does. Refusals are retained in qualification records rather than relabelled
as passing runs.

All **34 META-registered packages** are accessible through `carried_kernel(name)`.
The loader imports their canonical `opt_core.kernels.<name>` package, preserving
relative imports; top-level imports had broken several providers. `catalog()`
returns the complete upstream metadata without loading optional frameworks.
This does not imply every variant runs on A100: sm90a/sm100a instructions still
require their GPU. Supported dimensions and differentiation remain the upstream
provider's contract. Torch module compositions above reject gradient recording
and active training dropout; the separate backward entry points below expose
only the gradients the original row implements.

### JAX and backward entry points (2026-10-04)

`pallas_call(op, ..., word=...)` connects all seven upstream serving faces:
attention, triangle attention block, triangle multiplication, transposed masked
GLU, transition, layer norm and outer product mean. It defaults to `strict=True`,
so a named row's refusal is propagated. Arrays and gradients remain JAX-native.
`triangle_attention_xla` and `triangle_multiplication_xla` also expose native FFI
providers directly. Native XLA TriAttn runs on A100; native XLA TriMul retains its
H100-only architecture refusal. The differentiable Pallas TriMul rows run on A100.

| Entry / row | Qualification |
| --- | --- |
| `cd_ln`, `cd_transition` | JAX forward and input/parameter VJPs |
| `cd_triatt`, `pallas_attn` | JAX attention forward and Q/K/V/bias VJPs |
| `cd_trimul`, `fpf_trimul_xlabwd` | JAX forward and input/parameter VJPs; the latter explicitly uses the upstream XLA backward |
| `mlp_transition`, `glut`, `cd_opm`, `opm_two_launch` | JAX forward |
| `triattn_xla` | A100 native FFI attention forward |
| `pallas_components().attention_core` | Actual aligned N128 stage, explicit cc=8.0; independent of the measured tier table |
| `layer_norm_backward_dx` / `ef2_ln_bwd_dx` | Torch native LN input gradient, with/without residual gradient; no affine gradients |
| `transition_autograd` | `esm_kd3`, `esm_kd3:lean`, `esm_t15_kd3`: A100 forward and frozen-weight input gradients, C256/H1024 |

The coarse measured `fpf_core` bucket refuses N128 because its N199 benchmark
was not tile-aligned. That refusal is preserved; the explicit component API
qualifies the actual aligned stage without rewriting the table. GLUT uses the
release's Tokamax **0.0.12**, with an explicit `Config` to avoid parsing the host
application's command-line flags.

JAX qualification uses a separate CUDA12 environment with `jax[cuda12]==0.10.2`,
`dm-haiku`, and `tokamax==0.0.12`. It unsets `LD_LIBRARY_PATH` so Pixi's CUDA
libraries cannot override the JAX plugin's libraries, and sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Local launcher:

```bash
# Run inside an allocated A100 job, from this repository.
bash /home/psk6950/miniworld-engine-scratch/codex_anthropic_all/jax_run.sh \
  -m pytest tests/integrations/test_anthropic_jax_gpu.py -q
```

The JAX ending triangle-attention block's original Pallas VJP showed 15.2%
input-gradient relative error when transposition created entirely masked rows.
That input is **not qualified**. Block gradient qualification requires at least
one valid key in each attention row in either orientation; this limitation does
not apply to the separately tested Torch module empty-mask contract. The existing
Torch bidirectional TriAttn reference itself returns NaN for entirely empty keys;
its two empty-mask comparisons are skipped explicitly, while absent and sparse
masks are compared in both modes.

The ESM input-gradient rows use a separate environment with Transformers fork
`ef32577f55da19a4989cd7b22e004dc43a4998cb`, exactly the upstream ESMFold2 pin.
`esm_kd3` and its lean variant use the original saved-activation/recomputation
backward; `esm_t15_kd3` uses the original A100 forward tile with that backward.
No generic Torch recomputation is substituted by this adapter. Local launcher:

```bash
bash /home/psk6950/miniworld-engine-scratch/codex_anthropic_all/esm_run.sh \
  -m pytest tests/integrations/test_anthropic_esm_backward_gpu.py -q
```

Local cssb environment (inside Pixi and an allocated A100 job):

```bash
source .validation/anthropic-a100.env
python -m pytest tests/integrations/test_anthropic_modules_a100_gpu.py \
  tests/integrations/test_anthropic_trimul_a100_gpu.py \
  tests/integrations/test_anthropic_triattn_a100_gpu.py -q
python benchmarks/runners/bench.py target=dit_atom level=module mode=inference \
  'implementations=[anthropic]' compile=false min_seq_len=16 max_seq_len=16
```

The local TriMul build lives outside the imported tree, with original `python/`
and `csrc/` siblings. Reproduce it with the release's builder:

```bash
PYTHONPATH="$TRIMUL_PACKAGE/python" python -m trimul_native.build --archs sm_80 \
  --units trimul_k1_sm80,trimul_k3_sm80 --src "$TRIMUL_PACKAGE/csrc" \
  --out "$TRIMUL_PACKAGE/build" --nvcc "$CONDA_PREFIX/bin/nvcc" --jobs 2
export TRIMUL_NATIVE_BUILD_DIR="$TRIMUL_PACKAGE/build"
export ANTHROPIC_TRIMUL_BUILD_DIR="$TRIMUL_NATIVE_BUILD_DIR" # pristine benchmark label
```

Environment setup and results are specific to the local ABI; source copies,
binary hashes, exact rows, successes and refusals are recorded in
[A100 qualification](../gpus/a100/anthropic-all-20261004.json).

### H100 qualification

The first H100 campaign uses L=384/768, nonzero projections, explicit masks where
supported, and independent FP32 formulas. It records output and residual-update
error separately so the residual cannot hide a wrong update. Timings exclude
compilation and one-time weight packing and use warmed CUDA Graph replay.
Candidate processes are separate; these results are not interleaved A/B trials.

NCU reports include the hierarchical Tensor Core roofline, HBM/L2 traffic,
occupancy, and scheduler statistics. NCU replay durations are separate from
CUDA Graph timings. A high SM percentage alone is not a roofline conclusion.
Assess each shape's actual bottleneck; reaching a hardware roof does not rule
out reducing algorithmic work or memory traffic.

Results and limitations: the H100 campaign report, `archive/docs-20260928:docs/anthropic/h100-audit.md`.
For reproducible qualification (set PYTHONPATH and runtime paths for your stack):

```bash
python benchmarks/runners/bench_anthropic.py --family trimul --length 384 --width 128 \
  --row native_rebuilt --output /tmp/adoption/results/trimul.json
python benchmarks/runners/bench_anthropic.py --family triattn --module --length 768 \
  --row block:triattn_native --output /tmp/adoption/results/attention.json
python benchmarks/runners/profile_anthropic.py --run-dir /tmp/adoption --lanes 1 --only trimul
python benchmarks/runners/profile_anthropic.py --run-dir /tmp/adoption --lanes 1 --only triattn --modules
```

Gather attention's `ensure_sorted=True` performs a host boolean check of GPU
indices. Sort indices once and pass `ensure_sorted=False` for graph capture;
include sorting in end-to-end timings if index sets change per call. Source
imports and low-level access do not imply all model adapters or genomics kits
have been executed. This campaign does not change the running MiniWorld training
job or claim an end-to-end model speedup.
