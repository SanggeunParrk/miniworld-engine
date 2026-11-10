# Benchmarks

Speed figures and tables of the engine's requirement cases, one folder per card (`h100_sm90/`, `a100_sm80/`). Everything here is
**generated from results that already exist**; nothing in this folder measures:

    python -m miniworld_engine.measure.plot --gpu h100_sm90                 # from the card's speed tokens (judge/tokens/<gpu>/)
    python -m miniworld_engine.measure.plot --gpu h100_sm90 --module Transition
    python -m miniworld_engine.measure.plot --gpu h100_sm90 --source RUN_DIR --out /scratch/plots   # from speed records not issued yet

(Run it in a CPU Slurm job like every Python step; it needs matplotlib, which the dev environment has.)

Layout of `<gpu>/`:

- `summary.svg` + the table in `README.md`: **one figure over every class**, the geometric mean of the speedup vs PyTorch per class,
  backend and mode (eval / train), over the cases both ran;
- `summary_af3.png`: the same over only the requirement rows AlphaFold 3 uses (the rows whose `source` names `af3`;
  `--model-summaries` picks the models);
- `README.md`: the summary and the index (classes, case counts, figure counts, the source and the speed measurer version);
- `<Module>/L__<slice>.svg`: **L sweep**, median latency (log) and speedup vs PyTorch over the case length, one column per
  (dtype, mode): the engine (CUDA, gold) against the external baselines (Anthropic orange, cuEquivariance teal) and the PyTorch
  reference (grey). The engine's Triton rung is in the tables, not the figures (`--backends` draws others);
- `<Module>/D-<key>__L<length>__<slice>.svg`: **D sweep**, the same over one channel width of the constructor (`d_pair`,
  `d_hidden`, `normalized_shape`, ...) at one length, the other constructor arguments fixed;
- `<Module>/README.md`: every figure with its table (median ms per backend and speedup vs PyTorch).

A slice is one (model, stream, constructor kwargs, depth, samples, timing). Only `status: ok` CUDA-graph timings are drawn;
capture fallbacks (`eager`) are marked in the tables and left out of the figures; unsupported / skipped / failed cases are gaps.
How the numbers are taken: [docs/measure.md](../docs/measure.md); the order the engine's backends must keep:
[docs/ladder.md](../docs/ladder.md).

Regenerate the card's folder after every measurement cycle that changed speed results, and commit it with the tokens. Model-level
benchmarks (FoldForge against the Anthropic kits, the 5I28 benchmark) belong to FoldForge, not here.
