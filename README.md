# miniworld-engine

PyTorch modules of AF3-style structure-prediction models (pair stack, MSA ops, token / atom diffusion transformers, sampler glue,
ProteinMPNN, losses), each with a PyTorch reference and hand-written H100 kernels as alternative backends, plus the tooling that
states what each module must serve and judges whether a backend is accurate and fast enough.

Scope: **H100 (sm90) only.** The `cuda` backends are built for sm_90a; the launch-config tables, the prebuilder's default target and
the judge's recorded environment are H100.

## The four parts

| part | where | what |
|---|---|---|
| requirements | `src/miniworld_engine/requirements/` | `requirements_module.csv`: every (module class, dims, options) a downstream model builds, with lengths, dtypes, modes. `requirements_kernel.csv`: every kernel a module backend launches, its family, role and human-managed `version`. |
| modules + backends | `src/miniworld_engine/modules/` | One `BackendModule` per op. `pytorch` is the reference; `cuda` / `triton` are the engine's kernels; `anthropic` / `cueq` are external baselines. A backend states its range in `_supports_<backend>` and never falls back. |
| kernels | `src/miniworld_engine/kernels/` | Kernel families (CUDA and Triton), per-GPU launch-config tables, the prebuilder (CUDA artifacts ahead of time) and the pretuner (launch-config tables). |
| measure + judge | `src/miniworld_engine/measure/`, `src/miniworld_engine/judge/` | Measurers for accuracy (fp64 gold) and speed (CUDA graphs); the judge turns records into tokens and decides which cases are owed a measurement, by human-managed versions. |

## Quick usage

Build a module with a backend:

```python
import torch
from miniworld_engine.modules import TriangleMultiplication, TokenDiT

tri = TriangleMultiplication(128, backend="cuda").cuda()     # H100 kernels
ref = TriangleMultiplication(128).cuda()                     # backend="pytorch", the reference
ref.load_state_dict(tri.state_dict())                        # same parameter names for every backend
tri.set_backend("triton")                                    # switch later; ValueError if the class has no such backend

dit = TokenDiT(model="boltz2", ...)                          # selector: builds the model's own class (TokenDiTBoltz2)
```

A call outside a backend's range raises `BackendUnsupported` with the reason; it is never run by another backend.

Prebuild the CUDA kernels (CPU node, no GPU needed):

```sh
python -m miniworld_engine.kernels._prebuild --list            # plan from the code + requirements, what is missing
python -m miniworld_engine.kernels._prebuild --jobs 16 --cutlass $CUTLASS_PATH
```

Measure and judge (the measurers need a GPU; the judge runs on a CPU):

```sh
python -m miniworld_engine.judge owed --plan $DIR             # which cases are owed; writes configs + groups G1..G4
# one 1-GPU, 10-minute job per group: for each line "<kind> <module> <cfg>" of $DIR/G<i>
python -m miniworld_engine.measure.<kind> <cfg> --out $DIR/out/<kind>.<module>.jsonl
python -m miniworld_engine.judge issue $DIR/out               # records made by current versions -> tokens
python -m miniworld_engine.judge status
```

## Documentation

[docs/README.md](docs/README.md) is the index: requirements, modules, kernels, measurement, judge, and per-family H100 kernel notes.

## License

`MIT AND Apache-2.0 AND BSD-3-Clause`. MiniWorld Engine's own code is MIT (copyright 2026 Sanggeun Park, `LICENSE`); the code derived
from Anthropic's uplifting-biomolecular-modeling release, and any file whose header says `SPDX-License-Identifier: Apache-2.0`, is
Apache-2.0 (with Anthropic's `NOTICE`), and the FlashAttention-3 portions and the
CUTLASS-instantiated object code in the prebuilt kernels are BSD-3-Clause. Details, per-file attributions and the licence texts are in
[`licenses/THIRD_PARTY_NOTICES.md`](licenses/THIRD_PARTY_NOTICES.md). The external release used by the `anthropic` backends is cloned
locally under `third_party/anthropic/upstream/` (git-ignored, not redistributed).
