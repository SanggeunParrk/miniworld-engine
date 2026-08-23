# layernorm — standalone LayerNorm kernel workspace

This folder owns the standalone LayerNorm kernel work.

The existing `triton/` and `cuda/` subfolders are preserved as legacy baselines.
New implementation work should happen in this folder root and its backend
subfolders, not in a separate sibling kernel directory.

## Current baselines

- `pytorch`: `torch.nn.functional.layer_norm`
- `triton`: legacy vendored `triton_layernorm`
- `cuequivariance`: `cuequivariance_ops_torch.fused_layer_norm_torch.layer_norm_transpose`
  with `layout="nd->nd"`

## Files

- `reference.py`: PyTorch reference and small `nn.Module` wrapper
- `interface.py`: public entrypoint for the new kernel
- `compile_native.py`: `custom_op` entrypoints for `atomic` / `partial` / `dispatch`
  compile paths
- Kernel benchmarks live under `benchmarks/kernels/layernorm/`; generated
  results live under that target's `artifacts/`.

The default sweep is:

- `M = L^2` for `L ∈ {384, 512, 768, 1024}`
- `D ∈ {128, 256, 384, 512, 768}`

## Benchmark policy

This kernel is **not** an excuse to freestyle benchmarking.

- The preferred benchmark path in this repo is the unified team-gm-style
  harness: `benchmarks/runners/bench.py` +
  `benchmarks/modules/<module>/configs/bench.yaml`.
- Temporary LayerNorm probes should stay untracked until promoted into
  `benchmarks/kernels/layernorm/`.
- Keep benchmark output aligned with the team-gm flow so logs and reports are
  directly comparable to the rest of the repo.
- Do not treat quick `python - <<'PY'` experiments as benchmark artifacts.

## Comparing against an existing kernel (not PyTorch)

`benchmarks/runners/plot_csv.py` defaults to speedup-vs-PyTorch, but PyTorch is
not always the kernel we are trying to beat. Pass `--baseline <backend>` to
render speedup bar plots against an existing kernel instead. Historical reports:

- `bench_layernorm_vs_triton.*` — `--baseline triton` (legacy vendored
  kernel). Forward is the *same* fused kernel (≈1.00×); the win is the
  partial-reduction backward dispatch: **1.32–1.39× training at d=768**, ~1.05–1.14×
  at mid d, neutral at d=128 (dispatch falls back to atomic there).
- `bench_layernorm_vs_cuequiv.*` — `--baseline cuequivariance`. Ours is
  **1.10–2.03× inference** and **1.30–1.82× training** (margin grows with d; cuEquiv
  degrades badly past d=512).

New results must be re-rendered from CSV, not from captured stdout.

## CuTeDSL / H100 investigation

Historical CuTeDSL/H100 investigation notes found that a cute / H100-specific
rewrite did not justify becoming the default path. Two old bench modes backed it:

- `bench.py --suite inference_tune` — inference-only, reports achieved HBM bandwidth
  (% of 3.35 TB/s peak); compares pytorch / triton / `triton/lowreg.py` /
  quack CuTeDSL LN inference.
- `bench.py --suite cute_training` — training, our triton LN vs quack CuTeDSL RMSNorm
  (backward proxy; quack ships no LN backward).

Headline: **inference is at the bandwidth wall (82–90% peak) — no rewrite helps**;
**backward has real headroom — a CuTeDSL-style persistent `sm_count` grid is
1.2–2.3× faster than our triton backward, the gap growing with d** (1.79× training
at d=768). The inference `lowreg` variant is a confirmed negative result (kept as a
documented probe).

## Note

Current benchmark figures are grouped bar plots rendered from CSV. Latency uses
"lower is better"; speedup uses "higher is better".

The current `layernorm_kernel` path auto-dispatches the backward between three
implementations:

- `atomic` — legacy Triton backward with direct `atomic_add` (small D)
- `partial` — partial-buffer backward, `[cdiv(M, block_m), D]` then final reduction
- `persistent` — persistent grid-stride backward (`triton/persistent.py`): a fixed
  `NUM_SM * waves` grid, vectorized 2D tiles, only `[grid, D]` partial rows. Ported
  from quack's CuTeDSL backward algorithm; see the archived `CUTE_INVESTIGATION.md`.

The present H100 heuristic (`compile_native.py:_dispatch_bwd`) is:

- `D >= 384` -> **persistent** (beats partial 1.06–1.22×, matches quack cute at
  D=768 within ~2%; 1.76× over the atomic path at D=768/M=1M)
- `D == 256` -> partial when `M >= 262144`, else atomic
- `D < 256` -> atomic

Measured: persistent vs the old partial at M=1048576 — D=384 1.06×, D=512 1.06×,
D=768 1.18×; the win grows with D because the old partial's scalar per-row loop +
`next_pow2(N)` register pressure hurt most at large D.

### Portability (non-H100 GPUs) — per-GPU dispatch cache

All three backward impls are **correct on any CUDA arch** and self-tune there:
plain Triton (recompiled per arch, autotune re-picks BLOCK_M / warps / stages); the
persistent kernel reads the live SM count and uses **no Hopper-only features** (no
clusters / DSMEM / TMA). So `miniworld` LayerNorm is safe on Ampere / Ada / etc.

The *crossover* (which of the three is fastest at a given `d`, `M`) was only measured
on H100, so on other GPUs we **don't guess — we measure once and cache** it
(`dispatch_cache.py`):

- **H100 (sm_90):** uses the measured static heuristic directly (`d>=384`
  persistent / `d==256` partial-or-atomic / `d<256` atomic) — no calibration.
- **Other GPUs:** the first time a `(d, M-bucket)` is seen, the backward times the
  three paths on the real tensors, picks the winner, and writes it to a per-GPU
  JSON under `src/miniworld_engine/autotune/data/ln_bwd_dispatch/<gpu>.json`. Later runs
  (and re-imports as a submodule) read the cache and dispatch instantly. The cache
  only ever selects among *correct* kernels, so it can never affect numerics — a
  stale/corrupt cache at worst picks a slower valid path, and any error falls back
  to the static heuristic.

Controls (`miniworld_engine.settings`, via `settings.configure(...)`) — these were the
`MINIWORLD_LN_AUTOTUNE` / `MINIWORLD_LN_BWD` environment variables until settings.py replaced
every `MINIWORLD_*` switch with a field; nothing reads the variables now:

- `layernorm_dispatch="auto"` (default) `| "off" | "force"` — `off` always uses the
  static heuristic; `force` calibrates even on H100.
- `layernorm_bwd_path="persistent"|"partial"|"atomic"|"cuda"` — hard override, bypasses cache.

The calibrated path is cached in-repo at
`src/miniworld_engine/autotune/data/ln_bwd_dispatch/<gpu_key>.json` (committed to git);
there is no `~/.cache` / env-var cache location.

Calibration is skipped during CUDA-graph capture (falls back to static / cache).
