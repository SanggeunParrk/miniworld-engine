# Modules and backends

`miniworld_engine.modules`: one file per op family (see the module docstring of `modules/__init__.py` for the map). Every op is a
`BackendModule` (`modules/primitives.py`).

## The BackendModule contract

```python
class Transition(BackendModule, MasterWeights):
    version = 1                                   # human-managed, see "Versions"
    backends = ("pytorch", "anthropic", "cuda", "triton")

    def __init__(self, ..., backend="pytorch"):
        super().__init__()
        ...
        self.set_backend(backend)                 # ValueError if not in `backends`

    def _forward_pytorch(self, x): ...            # the reference
    def _forward_cuda(self, x): ...
    def _supports_cuda(self, x) -> str | None: ...   # None = runs; a string = why not
    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict: ...
```

- `backends` lists the names; each needs a `_forward_<name>` (checked at class creation, `pytorch` required).
- `forward` calls `_forward_<self.backend>`. For a non-reference backend it first calls `unsupported(...)`, i.e.
  `_supports_<backend>(*args, **kwargs)`; a reason string raises `BackendUnsupported`. **A backend never falls back** to another one.
  A backend without `_supports_<name>` claims every call.
- `backend` is a plain attribute, not in the state dict: one checkpoint loads into every backend, and `set_backend` switches it.
  Classes keep the parameter / buffer names of the engine's earlier modules, so older state dicts load.
- `make_inputs` returns the forward kwargs of one measurement call (random activations in `dtype`, key masks with padding via
  `key_mask`, structural inputs). It is the only module-specific part of the measurers and never changes the forward math.
- `wide(*refs)` is the "at least fp32" compute dtype: fp32 for bf16 / fp16 / fp32 operands, fp64 when any operand is fp64. Statements
  pinned to fp32 use it, so the fp64 gold run of the accuracy measurer stays fp64 end to end.

## Backends

| backend | what | judged by the accuracy rule |
|---|---|---|
| `pytorch` | the reference: the single source of truth of the op's math | it is the reference |
| `cuda` | the engine's hand-written kernels for the card of the input tensor: H100 (sm_90a) or A100 (sm_80), through `kernels/<family>/cuda` and `kernels/integrations`; any other card is refused by `_supports_cuda` | yes |
| `triton` | the engine's Triton kernels (no TMA, any CUDA card), launch configs from `kernels/<family>/configs/<card>.json` | yes |
| `anthropic` | Anthropic's uplifting-biomolecular-modeling release in `third_party/anthropic/upstream`, loaded by `modules/_anthropic.py` (revision and file checksums checked against `third_party/anthropic/UPSTREAM.json`; the release itself is not in git; `MINIWORLD_ANTHROPIC_ROOT` overrides the root) | no: informational |
| `cueq` | NVIDIA cuEquivariance 0.12 (`cuequivariance_torch`), loaded by `modules/_cueq.py`; `_supports_cueq` refuses exactly the calls the library would run in its own PyTorch formula | no: informational |

External backends are connected as they are: they state in `_supports_<name>` what they run and are not changed to pass anything.

A `cuda` backend picks the card's path from the input tensor's device capability inside `_supports_cuda` / `_forward_cuda`
(see [architecture.md](architecture.md#how-a-call-reaches-the-cards-kernels)). A kernel that fails to build or load raises
`kernels._nvcc.KernelBuildError`; it is an error, never a reason to run another kernel or backend.

Which classes have which backends is the `backends` tuple of each class; the per-family H100 notes in [h100/](README.md#h100-kernel-notes)
describe the `cuda` / `triton` ranges.

## Selectors

Some ops have one class per model; a selector builds the model's own class (not a wrapper, so the state dict has that class's keys):

| selector | map | example |
|---|---|---|
| `TokenDiT(model, ...)` | `TOKEN_DIT_MODELS` in `token_dit.py` | `af3`, `protenix`, `openfold3`, `esmfold2`, ... -> `TokenDiTProtenix`; `boltz2`, `rosettafold3`, `chai1`, `miniworld`, `miniworld_bias_only` -> their classes |
| `AtomDiT(model, ...)` | `ATOM_DIT_MODELS` in `atom_dit.py` | `af3` / `openfold3` / `intellifold2` -> `AtomDiTOpenFold3`; `protenix` / `opendde` -> `AtomDiTProtenix`; `esmfold2`, `miniworld`, `boltz2`, `rosettafold3`, `chai1` |
| `TriangleAttention(..., model=)` | `triangle.py` | `chai1` -> `TriangleAttentionChai1`; `af3` -> `TriangleAttentionAF3`; `rf3`; `chai1_confidence` |
| `TriangleMultiplication(..., model=)` | `triangle.py` | `rf3` -> `TriangleMultiplicationRF3` |

The requirements' `model` column lists the selector keys of each row; the measurers build the class directly by name.

`protenix_v2` is an alias of `protenix` for `TriangleAttention` / `TriangleMultiplication` (Protenix-v2 runs Protenix's triangle updates
unchanged), so a family maps to one key across all selectors. A `PairformerBlock` takes separate keys for its two kinds of triangle
update: `PairformerConfig(model=..., triattn_model=None, trimul_model=None)` (None = `model`); RoseTTAFold3's fast mode is
`model="rf3", trimul_model="miniworld"` (RF3 triangle attention, the plain multiplication).

## Multi-fold contract of the DiT blocks (trunk_seed x diffusion_seed)

Every `TokenDiT*` and pair-taking `AtomDiT*` block (and `TokenDiTStack` / `AtomDiTStack`) takes several folds of one input in ONE call;
the fold axis is the blocks' `B` axis:

| input | layout | meaning |
|---|---|---|
| `single` | `[A, B, L, d]` | A diffusion samples (diffusion_seeds) of each of B folds (trunk_seeds) |
| `cond` | `[A, B, L, d_cond]` | fold `b`'s conditioning; one per fold expanded over A (stride 0) is the sampler step's form and is read once |
| `pair` | token `[B, L, L, d_pair]`, atom `[B, nwin, 32, 128, d_pair]` | fold `b`'s own pair (trunk) |
| `mask` | `[B, L]` | the key mask; the folds of one input share it: one `[1, L]` row expanded over B (stride 0) |
| `uid` | `[B, N]` | Chai-1's conformer ids, likewise one row expanded over B |

Sample `(a, b)` (row `single[a, b]`) attends with fold `b`'s pair bias and conditioning: the result equals B separate calls of
`[A, 1, ...]` (one per fold), which is how the tests check it (tests/cpu/test_requests_api.py, tests/gpu/test_requests_api_gpu.py). Inside the kernels the `A x B` rows are laid
out `[A][B]` and the grouped-bias attention reads sample `s`'s bias from pair `s mod B`. The requirement rows with stream `token_folds` /
`atom_folds` (the `depths` column = the fold count B) are this layout.

Served on: `pytorch` and `triton` everywhere; `cuda` for bfloat16 (or master) inference: on H100 in one call over the B folds (the grouped-bias
kernels; the token block within its inference gate, e.g. L a multiple of 128; the atom block: every key window), on A100 the token block
within its sm_80 gate and the atom block as one sm_80 call per fold (the sm_80 atom kernels take one pair; N a multiple of 64). A B > 1
call in fp32 or with a gradient is refused with a reason on both cards (the TF32 kernels and the backward keep one pair bias). FoldForge
measured its grouped 5 x 5 sampler at 1.5x end to end for assemblies up to 256 tokens.

## DiT stacks and the shared pair norm (`modules/dit_stack.py`)

`TokenDiTStack(model, n_blocks, ...)` / `AtomDiTStack(model, n_blocks, ...)` hold `n_blocks` blocks of one model's own class in a
`ModuleList` `blocks` (state-dict keys `blocks.{i}.<block names>`) and run them in order; they add no math. Every block gets the same
`pair` / `cond` / `mask` / `uid` objects, so each block's hoisted pair bias and conditioning tables are made once per fold;
`extra_pair_bias` / `compute_dtype` (token) and `uid` (Chai-1's atom blocks) are forwarded. `AtomDiTStack` refuses the pair-free
ESMFold2 / MiniWorld atom blocks (their stack is `SWAAtomTransformer`). `set_backend` sets every block's backend.

A released AF3-family stack keeps ONE pair LayerNorm and one projection of `n_blocks * n_head` rows (AF3's token DiT: one per super block
of four, whose row-concatenation is that projection). `from_shared_pair(norm_weight, projection, n_blocks=, n_head=, norm_bias=None)` gives
each block `attention.ln_pair.weight` (a copy) and its `n_head` rows of `attention.to_bias.weight`; `stack.load_shared_pair(...)` copies
them in. The LayerNorm offset is dropped: it adds `W_proj beta`, one constant per head for every (query, key), which the softmax cancels.

## Parallel residual (`parallel_residual`, `ParallelResidual`)

A parallel stage whose parts own their residuals (`y_i = x + delta_i`; Chai-1's MSA block) updates `x` by the sum of the deltas:
`parallel_residual(x, *ys, backend="pytorch" | "triton")` = `x + sum_i (y_i - x)`, the differences and the sum in fp32, one rounding.
`"triton"` (`kernels/parallel_residual/triton`) is one pass over the tensors (1 to 4 branches, CUDA, one dtype); the backward is torch
(`d/dy_i = g`, `d/dx = (1 - n) g`). There is no `"cuda"` backend: the Triton pass already streams at the memory bandwidth (H100, bf16, pair
768 x 768 x 256 with three branches: 1.2 GB read + written in 0.49 ms, about 3.1 TB/s against the card's 3.35), so a hand-written kernel
has nothing left to gain. Measured on the requirement rows (graph timing, L2 flushed; eval, ms):

| row | H100 pytorch | H100 triton | A100 pytorch | A100 triton |
|---|---|---|---|---|
| MSA [1, 1024, 768, 64], 2 branches | 1.356 | 0.138 | 2.381 | 0.250 |
| pair [1, 768, 768, 256], 3 branches | 5.640 | 0.490 | 9.965 | 0.887 |
| pair [1, 768, 768, 256], 2 branches | 4.012 | 0.393 | 7.078 | 0.721 |

`PairformerBlock(parallel=True)` keeps its own statements for its parallel schedule.

## Dead residual modules (the `dead` flag; the loader is not part of the engine)

`Transition`, `AttentionPairBias`, `TriangleAttention`, `TriangleMultiplication`, `OuterProductMean` and `MSAPairWeightedAveraging` keep a
plain class attribute `dead = False`: a flagged instance returns its residual input at inference (no grad, eval), on every backend. The
engine does NOT ship the function that sets it (2.4.x `architecture.checkpoints.dead_modules.mark_dead_modules` /
`clear_dead_modules`, removed with `architecture/` in 2.5.0) and will not export one: deciding that an instance is dead is a property of
a checkpoint (its parameters fall below a threshold that the model's own kit chose, e.g. Protenix-v2's ~5e-37 weights against a 1e-30
bound), so it belongs to the model side that loads checkpoints (FoldForge `modules/dense/dead.py`, MiniWorld's loader), next to the
other checkpoint conversions. The engine's contract is only the flag: set `module.dead = True` after loading and the module returns its
input at inference; `False` (the default) runs it. Decision 2026-11-11 (FoldForge request R15: declined, scope).

## Public CUDA-graph capture API (`miniworld_engine.capture`)

`miniworld_engine` is a namespace package (no `__init__.py`: the prebuilt kernel wheels install into it), so nothing can be exported
from the package root; the public capture API is the module `miniworld_engine.capture`:

| name | meaning |
|---|---|
| `static_weights()` | context manager: weights stay fixed between graph replays (inference graphs only); a capture inside it serves from, and fills, the eager weight-pack caches, so a replay runs no packing kernel |
| `static_inputs()` | context manager: the conditioning / pair inputs of hoisted per-item tables keep their contents for the life of the graph (and the weights too); a replay re-runs none of their producing kernels |
| `clear_weight_caches(*modules)` | drop every weight / input-derived cache of the engine (all stores registered with `kernels._capture.weight_cache`, the parameter-form cache, the TF32 operand cache) and, for the given modules, the per-instance caches; call it before replacing weights (a second model, a test that builds many) and drop graphs captured before the clear |
| `weight_caches()` | the registered stores by name (inspection) |

A new cache in kernel code registers itself: `_STORE = _capture.weight_cache("family.name")` (a dict), or for a per-instance attribute
`_capture.instance_cache_attr("_name")`. `tests/cpu/test_capture_api.py` scans the kernel sources so that a module-level dict that is
neither registered nor listed as not weight-derived fails the CPU tests.

## Versions

`BackendModule.version` (default 1, set explicitly on every requirement class) is human-managed: bump it when a change to the class's
math, its reference or a backend adapter can change an accuracy or speed result; not for a refactor or a comment. The judge owes a
new measurement of every case of a class whose version changed.

`BackendModule.backend_versions` (class attribute, backend -> int, missing = 1) versions one backend's adapter: bump an entry when a
change alters only that backend's results (e.g. an external library rewired). A key `"<backend>:<card>"` (e.g. `"cuda:a100_sm80"`)
versions that backend on one card only and wins over the plain `"<backend>"` key on that card. Example, `Transition`:
`{"anthropic": 2, "cuda:h100_sm90": 4, "cuda:a100_sm80": 4, "triton:a100_sm80": 2}`. See [judge.md](judge.md).

## Precision helpers

- `MasterWeights` (mixin): `set_master_weights()` pins fp32 master parameters under a bulk bf16 cast; with bf16 activations the
  module runs under bf16 autocast (`master_autocast`) or bf16 GEMMs with fp32 weight gradients (`master_linear`). This is the
  `master` dtype of the requirements.
- `LayerNorm` keeps its affine in fp32 under a bf16 cast (`_Fp32ParamsMixin` in `norm.py`).
