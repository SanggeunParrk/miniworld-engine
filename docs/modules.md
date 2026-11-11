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
