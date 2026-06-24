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
- `bench.py`: temporary standalone bench, but it must stay in **team-gm bench
  format** so `scripts/plot_bench.py` can parse it
- `benchmark/`: raw logs, markdown report, and graphs

The default sweep is:

- `M = L^2` for `L ∈ {384, 512, 768, 1024}`
- `D ∈ {128, 256, 384, 512, 768}`

## Benchmark policy

This kernel is **not** an excuse to freestyle benchmarking.

- The preferred benchmark path in this repo is still the unified team-gm-style
  harness: `scripts/bench.py` + `config/bench.yaml` + `tests/run_bench.sbatch`.
- `layernorm/bench.py` exists only because standalone LayerNorm is not yet wired
  into that harness.
- Keep its output format aligned with the team-gm flow so the resulting `.out`
  log, `.md` report, and `.png` graphs are directly comparable to the rest of
  the repo.
- Do not treat quick `python - <<'PY'` experiments as benchmark artifacts.

## Run

Bench on a compute node:

```bash
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
  --cpus-per-task=8 --mem=64G --time=00:30:00 \
  bash -lc 'cd /home/psk6950/miniworld-kernels && \
    export PYTHONPATH=src && \
    export LD_LIBRARY_PATH=.pixi/envs/default/lib:${LD_LIBRARY_PATH:-} && \
    ./.pixi/envs/default/bin/python -m miniworld_kernels.kernels.layernorm.bench \
      | tee src/miniworld_kernels/kernels/layernorm/benchmark/bench_layernorm.out'
```

Render the markdown report and graphs on a compute node as well:

```bash
srun --partition=h100 --account=cssb --qos=cssb_h100 --mem=16G --cpus-per-task=4 --time=00:10:00 \
  bash -lc 'cd /home/psk6950/miniworld-kernels && \
    export LD_LIBRARY_PATH=.pixi/envs/default/lib:${LD_LIBRARY_PATH:-} && \
    ./.pixi/envs/default/bin/python scripts/plot_bench.py \
      src/miniworld_kernels/kernels/layernorm/benchmark/bench_layernorm.out \
      src/miniworld_kernels/kernels/layernorm/benchmark \
      --title "LayerNorm baselines MD sweep (H100, bf16)" \
      --name bench_layernorm'
```

## Note

Each metric PNG now contains both views in one figure:

- x-axis is `M`
- each `D` gets its own subplot
- bars are grouped by backend
