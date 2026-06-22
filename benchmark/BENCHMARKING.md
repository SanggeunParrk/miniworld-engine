# Benchmarking convention

**Every benchmark in this repo ships a table AND a graph together.** A table
alone hides trends; a graph alone hides exact numbers. Always produce both, and
save them next to each other so a result is never just a wall of text.

**All benchmark results live inside the kernel they belong to:**
`src/miniworld_kernels/kernels/<kernel>/benchmark/`. The raw log, the rendered
markdown report, and the PNGs all go in that one folder — NOT in this top-level
`benchmark/` directory (which is only for cross-kernel/unified runs and these
docs).

## Workflow (example: `layernorm_linear`)

Let `K=src/miniworld_kernels/kernels/layernorm_linear/benchmark`.

1. **Run** the bench on a GPU compute node, capturing stdout into `$K`. Uses the
   repo's unified pixi env (`.pixi/`); `--frozen` keeps the cu12 TE core fix in
   place (a bare `pixi run`/`install` re-pins cu13 — see pyproject `fix-te-cu12`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --gres=gpu:h100:1 \
     --mem=64G --cpus-per-task=8 --time=00:30:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m miniworld_kernels.kernels.<kernel>.bench"' \
     | tee "$K/<name>.out"
   ```
2. **Render** the table + graphs into the same `$K`. **Never run this on the
   login node** — route it through `srun` (CPU only, no `--gres`). matplotlib is
   in the unified env; `LD_LIBRARY_PATH` picks up its libstdc++ (`CXXABI_1.3.15`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --mem=16G --cpus-per-task=4 --time=00:10:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
       python scripts/plot_bench.py '"\"$K/<name>.out\" \"$K\""' --name <name>"'
   ```
   This writes `<name>.md` (markdown tables + embedded graphs) and one PNG per
   metric (`_fwd.png`, `_fwd_bwd.png`) into `$K`.

`scripts/plot_bench.py` parses the standard bench-output format
(`=== M=.. d_in=.. d_out=.. ===` blocks + `torch.compile`/`TE` timing lines), so
any bench that prints in that format gets table + graph for free.

## Conventions

- **Results go in `kernels/<kernel>/benchmark/`** — log + report + PNGs together.
- **Both forward and forward+backward** when the op is used in training.
- Bold the faster backend in each table cell; "lower is better" on graphs.
- Report numerical agreement (max abs error, relative Frobenius error, cosine)
  alongside latency — never just speed.

## Existing reports

- `kernels/layernorm_linear/benchmark/layernorm_linear_square_sweep.md` — TE
  `LayerNormLinear` vs `torch.compile`, d_in=d_out ∈ {128,256,384,512,768} ×
  M ∈ {16384,65536,262144} (H100, bf16).
