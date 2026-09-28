# Anthropic inference integration

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

```python
import torch
from miniworld_engine.modules import Transition, TriangleMultiplication, TriangleAttention

transition = Transition(128, implementation="anthropic", anthropic_row="v2").cuda().eval()
trimul = TriangleMultiplication(128, implementation="anthropic", anthropic_row="v4").cuda().eval()
attention = TriangleAttention(128, implementation="anthropic", anthropic_row="block:triattn_native").cuda().eval()

with torch.no_grad():
    y = transition(x)
    z = trimul(pair, residue_mask)
    a = attention(pair, residue_mask)
```

Use the normal model weight-loading and BF16 activation/projection setup.
These modules retain their existing residual convention. Single-direction
TriMul accepts outgoing or incoming; this adapter does not claim a fused
bidirectional kernel. TriangleAttention supports both core-only rows (e.g. `triattn_native`) and full
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
