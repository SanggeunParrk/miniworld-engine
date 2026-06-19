# miniworld-kernels

Dedicated GPU kernel-development repo for MiniWorld / AF3-style ops. The idea is
to **cut one op out of the full model and optimize it in isolation**:

- A **kernel** (`kernels/<unit>/`) is a chunk you deliberately chose to fuse and
  hand-optimize. It owns its backend implementations (`triton/`, `cute/`,
  `cuda/`) plus a PyTorch `reference.py`, a public `interface.py`, and a
  `benchmark/` dir holding that kernel's results.
- A **module** (`modules/<op>/`) is a part cut from the model (e.g.
  `triangle_multiplication`). It only *connects* kernels — it has **no**
  `triton/cute/cuda` folders — and keeps its own `benchmark/` results.

Kernels and modules were consolidated here out of `team-gm`
(`src/team_gm/modules/`, across `psk/benchmark`, `perf/trimul`, `miniworld`,
`exp/miniworld`) and the FlashAttentionBias repo (cute trimul work).

## Layout

```
src/miniworld_kernels/
├── _typecheck.py                 # standalone team_gm.typecheck shim
├── kernels/                      # fusion units: backends + per-kernel results
│   ├── tm1/  tm2/                #   left/right- and output-gated GEMM kernels
│   │   ├── reference.py interface.py
│   │   ├── triton/ cute/ cuda/   #   backend implementations
│   │   └── benchmark/            #   this kernel's results
│   ├── transition/ layernorm/ adaln/
│   ├── triangle_attention/ augmented_attention/
│   ├── bias_only_attention/ gated_projection/
│   ├── fused_ln_mask/            #   LN+mask fusion (used by the trimul cute path)
│   └── __init__.py               #   flat re-export bridge: kernels.triton_tm1, ...
└── modules/                      # model ops: connect kernels (NO backend folders)
    ├── triangle_multiplication/  #   module.py (connects tm1/tm2/LN; pytorch/triton/
    │   ├── module.py             #     cute/cuequivariance via ImplementationType)
    │   ├── reference.py interface.py baseline_dtv1.py
    │   └── benchmark/            #   this op's results
    ├── triangle_attention/ transition/ adaptive_layernorm/ augmented_attention/
    ├── exceptions.py             #   ImplementationType (pytorch/triton/cuda/cute/cuequivariance)
    ├── primitives.py ops.py      #   shared connecting utilities (LayerNorm, Linear, gates)
    └── __init__.py
scripts/bench.py                  # THE single bench entry (hydra, team-gm style)
config/bench.yaml                 # bench config (kernel, implementations, compile, ...)
benchmark/logs/                   # raw SLURM bench logs
cute-env/                         # CuTeDSL pixi env (cu128 torch + cutlass-dsl + quack)
tests/run_bench.sbatch            # single SLURM launcher for scripts/bench.py
```

In each kernel's `triton/`: `main.py` is the `psk/benchmark` variant (canonical),
`perf.py`/`miniworld.py` are alternates. Each vendored file carries a
`# vendored from team-gm <branch>@<sha>` header. Vendored kernel bodies
(`**/triton/*.py`, `**/cuda/*.py`) are not linted.

## Benchmarking

One entry point — `scripts/bench.py`, driven by `config/bench.yaml` (hydra):

```bash
# (on a GPU; team-gm env has torch 2.10+cu128 + triton + cuequivariance)
PY=/home/psk6950/team-gm/.pixi/envs/default/bin/python
PYTHONPATH=src $PY scripts/bench.py kernel=triangle_multiplication \
    implementations=[pytorch,triton,cuequivariance] compile=false
# or submit: sbatch tests/run_bench.sbatch
```

`kernel=` selects the op (`triangle_multiplication`, `triangle_attention`,
`transition`, `adaptive_layernorm`, `augmented_attention_token/atom`).
`compile=true` benches the `torch.compile`'d variant — no separate script.
Results land in `benchmark/<gpu>/`; raw logs in `benchmark/logs/`.

The **cute** path is `implementations=[cute]` (an `ImplementationType.CUTE`
implementation of `triangle_multiplication` that connects the tm1/tm2/fused-LN
cute kernels). It needs the `cute-env` (cutlass-dsl + quack), so run
`scripts/bench.py` with the cute-env python for that.

## Status

Restructured into the kernels/modules split above. The triangle_multiplication
cute path wins end-to-end at L ≥ 768 on H100 (≈1.75 ms at L=1024); the
from-scratch single-megakernel tm2 (`kernels/tm2/cute/tm2_cute_kernel.py`) is WIP.

## Toolchain

```bash
ruff check src/
ruff format src/
ty check
```
