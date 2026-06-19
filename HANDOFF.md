# miniworld-kernels — handoff (2026-06-16)

Status doc for the next agent. The repo was bootstrapped by consolidating GPU
kernels out of `team-gm` and the `FlashAttentionBias` repo into a dedicated,
self-contained kernel-development repo. **Goal: clean per-op folder structure +
the best kernel per op.**

---

## 1. What this repo is

Per-operation kernel repo. Each op folder holds a PyTorch reference + Triton /
CuTeDSL / CUDA implementations + a benchmark, side by side:

```
src/miniworld_kernels/
├── _typecheck.py                 # standalone shim for team_gm.typecheck (no team-gm dep)
└── kernels/
    ├── triangle_multiplication/  # composite op family
    │   ├── tm1/   reference.py interface.py + {triton,cute,cuda,benchmark}/
    │   ├── tm2/   reference.py interface.py + {triton,cute,cuda,benchmark}/
    │   └── trimul/reference.py interface.py + {triton,cute,cuda,benchmark}/
    ├── transition/        {triton,cute,cuda,benchmark}/ + reference/interface
    ├── layernorm/         {triton,cuda,benchmark}/
    ├── adaln/             {triton,benchmark}/
    ├── gated_projection/  {triton,benchmark}/
    ├── bias_only_attention/ {triton,benchmark}/
    ├── triangle_attention/  {triton,cute,benchmark}/
    └── augmented_attention/ {triton,benchmark}/
cute-env/                 # single CuTeDSL pixi env (pixi.toml + pixi.lock only)
tests/                    # cross-cutting bench/verify utilities
```

In each op's `triton/`: `main.py` = team-gm **psk/benchmark** variant (bench tag
`tn`), `perf.py` = **perf/trimul** variant (bench tag `nv`/dtv1), `miniworld.py`
= **miniworld** branch variant (where it exists). Every vendored file has a
`# vendored from team-gm <branch>@<sha>` header.

---

## 2. Provenance (vendored from team-gm = git@github.com:CSSB-SNU/team-gm)

Branch SHAs at vendoring time:

| branch            | short SHA | role                                   |
|-------------------|-----------|----------------------------------------|
| `psk/benchmark`   | `e085d6d` | `triton/main.py` (tn) + CUDA + adaln + bias_only |
| `perf/trimul`     | `3fbb02b` | `triton/perf.py` (nv) + `trimul/triton/dtv1.py`  |
| `miniworld`       | `7c3c67e` | `triton/miniworld.py` + `gated_projection`       |
| `exp/miniworld`   | `32e3897` | `layernorm`, `augmented_attention/compute_efficient.py` |

Original kernel location in team-gm: `src/team_gm/modules/kernels/`.
CUDA sources came from `src/team_gm/modules/kernels/cuda/{layernorm,transition}/`
(build artifacts `.so/.o` intentionally NOT vendored).

The FlashAttentionBias port modules `triangle_multiplication/` and `transition/`
(both untracked there) were **copied** (not moved) into `kernels/`. The cute env
that lived at `FlashAttentionBias/triangle_multiplication/tm1/cute/` was promoted
to the repo-level `cute-env/` and its scripts split into per-op `cute/` dirs.

---

## 3. Key decisions / gotchas

- **team_gm dependency cut**: the only intra-team_gm import was
  `from team_gm import typecheck`. Replaced everywhere with
  `from miniworld_kernels._typecheck import typecheck`. The shim is a no-op
  unless `SHOULD_TYPECHECK=true` (faithful to team-gm). No team-gm path/import
  deps remain (verified via grep).
- **Benches are self-contained**: every bench previously loaded team-gm kernels
  via `TEAM_GM_SRC` env + importlib path hacks. All rewired to import the
  vendored baselines as `miniworld_kernels.kernels.*`. No more `/home/psk6950/
  team-gm` or `.team-gm-perf-trimul` dependency.
- **cute cross-imports**: the cute scripts (launch.py, tm2_cute.py,
  bench_trimul.py, fused_ln_mask.py, ...) use bare imports of each other and
  now live in three sibling dirs (`tm1/cute`, `tm2/cute`, `trimul/cute`). A
  small `sys.path` bootstrap prelude was injected at the top of each script
  that cross-imports, adding all three cute dirs (+ the repo `src/` root) to
  `sys.path`. Bare `from launch import ...` style is preserved. **Possible
  future cleanup**: convert to proper package-relative imports.
- **Vendored kernel bodies are NOT linted** (`**/triton/*.py`, `**/cuda/*.py`
  are `["ALL"]`-ignored in ruff per-file-ignores) — they are faithful copies.
- **team-gm `tests/modules/test_*.py` were NOT vendored**: they test the model
  layer (`team_gm.modules.primitives.Transition`,
  `team_gm.modules.attentions.TriangleAttention`, `.utils`), not the standalone
  kernels. Kernel-level correctness lives in the moved
  `*/cute/verify.py` / `_verify_*.py` scripts and the `reference.py` files.
- **canonical = psk/benchmark (provisional)**: `triton/main.py` is the
  psk/benchmark variant. tm1/tm2 `main` vs `miniworld` have identical public
  signatures (`triton_tm1(x, WL, WLg, WR, WRg)`, `triton_tm2(x, x_out, Wg, Wp)`)
  so they are drop-in swappable. The head-to-head bench below decides the winner.

---

## 4. OUTSTANDING WORK (start here)

### 4a. Confirm the best tm1/tm2 Triton variant (GPU)
A head-to-head bench was written and submitted to SLURM but was still
**pending (PD, no free GPU)** at handoff time:

```
tests/bench_select_triton.py   # main vs miniworld, L in {384,512,768,1024}, D=128, bf16
# submitted as: srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 ...
#   (job name mwk_sel; job 8760 was PD/Resources)
```

Run it (login node has NO GPU driver — must srun onto h100 partition):

```bash
PY=/home/psk6950/FlashAttentionBias/.pixi/envs/default/bin/python   # has triton+torch
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 --time=00:15:00 \
  bash -c "cd /home/psk6950/miniworld-kernels && PYTHONPATH=src $PY tests/bench_select_triton.py"
```

It prints per-L winner + a verdict line per op. If `miniworld` wins, swap
`triton/main.py` <- `triton/miniworld.py` for that op (signatures match, so the
benches keep working unchanged). User's expectation: psk/benchmark (main) wins.

### 4b. Confirm the end-to-end cute path still reproduces the headline numbers
`trimul/cute/bench_trimul.py` is the flagship 4-way bench (pt / nv-triton /
cuequivariance / cute). It needs the **cute-env** (cutlass-dsl + quack), not the
FA env. Expected on H100, bf16, B=1, D=128: cute ≈ **1.71 ms at L=1024**, wins
at L≥512. To run, install the cute-env first:

```bash
cd /home/psk6950/miniworld-kernels/cute-env && pixi install        # heavy: cu128 torch + cutlass-dsl + quack
# then (on a GPU), with src/ + the cute dirs on path — bench_trimul has the bootstrap built in:
srun ... bash -c "cd /home/psk6950/miniworld-kernels && pixi -C cute-env run python \
  src/miniworld_kernels/kernels/triangle_multiplication/trimul/cute/bench_trimul.py"
```

### 4c. Clean up the FlashAttentionBias repo (only after 4a/4b pass)
These are untracked in FA and were **copied** here, so deletion is safe once the
move is verified:
- `FlashAttentionBias/triangle_multiplication/`
- `FlashAttentionBias/transition/`
- `FlashAttentionBias/.team-gm-perf-trimul/`

### 4d. (optional) commit + push
The repo is a fresh git repo (`git@github.com:SanggeunParrk/miniworld-kernels`,
branch `main`, only "Initial commit"). Nothing here is committed yet. Suggested
first commit: the whole `src/`, `tests/`, `cute-env/`, `pyproject.toml`,
`.gitignore`, README, this HANDOFF.

---

## 5. Quick verification (no GPU)

```bash
PY=/home/psk6950/FlashAttentionBias/.pixi/envs/default/bin/python
cd /home/psk6950/miniworld-kernels
$PY -m py_compile $(find src tests -name '*.py')        # all compile
PYTHONPATH=src $PY -c "from miniworld_kernels._typecheck import typecheck; print('shim ok')"
/home/psk6950/FlashAttentionBias/.pixi/envs/default/bin/ruff check src/ tests/   # "All checks passed"
```

---

## 6. Background / perf context (why the cute path wins)

The TriMul cute win comes NOT from faster GEMMs (tm1/tm2 are ~par with cuequiv;
tm2 cute baseline literally IS cuequiv's SM90 kernel) but from a `[B,D,L,L]`
direct-write layout that turns the 4-D einsum contraction into a flat cuBLAS
`bmm` and lets LayerNorm fuse the layout flips. Per-stage (L=1024): fused_ln_mask
0.23 / tm1 0.43 / bmm 0.45 / LN_out+transpose 0.34 / tm2 0.29 ms. The contraction
bmm (~600 TFLOPS, ~60% of peak) is already near-optimal; the headroom is in
fusing the memory-bound LN/transpose glue (a megakernel, ~15-30% est.), not in
rewriting GEMMs. The from-scratch CuTeDSL tm2 (`tm2/cute/tm2_cute_kernel.py`) is
WIP — full GMMA+epilogue+TMA-store path is now written (past the early-return
checkpoint the old README describes); correctness/hang status needs GPU re-check.
