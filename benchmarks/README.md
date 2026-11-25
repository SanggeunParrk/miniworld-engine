# Benchmarks

Two kinds of folders live here.

| folder | kind | what |
|---|---|---|
| `h100_sm90/`, `a100_sm80/`, `5090_sm120/` | **generated** | speed figures and tables per card, regenerated from the judge's speed tokens by `foldjet_engine.measure.plot` (below). There is no `b200_sm100/` folder yet. |
| `paper/` | paper data | the paper's experiments E1-E6: drivers, raw records, CSVs, figures ([paper/README.md](paper/README.md), [PAPER_DATA_SUMMARY.md](paper/PAPER_DATA_SUMMARY.md), [PAPER_PREWORK_STATUS.md](paper/PAPER_PREWORK_STATUS.md)) |
| `opm_win/` | experiment record | OuterProductMean at large MSA depth on the H100 vs the Anthropic kits: protocol, findings, rejected variants ([README](opm_win/README.md); release 2.5.11; named by a comment in `kernels/integrations/opm_train.py`) |
| `opm_bigs/` | experiment record | OuterProductMean large-S profiling drivers and their job logs (release 2.5.11) |
| `msa_depth/` | experiment record | OuterProductMean / MSAPairWeightedAveraging speed at large MSA depth (`bench_msa_depth.py`; release 2.5.11) |
| `fp8_5090/` | experiment record | RTX 5090 FP8 rate probes and A-B drivers named in [docs/fp8.md](../docs/fp8.md) |
| `fp8_e2e/` | experiment record | raw table `results_rf3_triattn.txt` named in [docs/fp8.md](../docs/fp8.md) |
| `bias_logits_modes.py` | experiment driver | speed / memory of the bias-only token DiT logit modes ([docs/bias_logits.md](../docs/bias_logits.md)) |

The experiment records are kept as they were run (paths inside job scripts may name the scratch trees they ran in); nothing in the
package imports them. Model-level benchmarks (FoldForge against the Anthropic kits, the 5I28 benchmark) belong to FoldForge, not here.

## Generated per-card folders

Speed figures and tables of the engine's requirement cases, one folder per card. Everything in them is
**generated from results that already exist**; nothing in them measures:

    python -m foldjet_engine.measure.plot --gpu h100_sm90                 # from the card's speed tokens (judge/tokens/<gpu>/)
    python -m foldjet_engine.measure.plot --gpu a100_sm80
    python -m foldjet_engine.measure.plot --gpu 5090_sm120
    python -m foldjet_engine.measure.plot --gpu h100_sm90 --module Transition
    python -m foldjet_engine.measure.plot --gpu h100_sm90 --source RUN_DIR --out /scratch/plots   # from speed records not issued yet

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

What the speedups mean: the baseline is the engine's own `pytorch` backend (the reference implementation of each class, eager PyTorch
kernels captured in the same CUDA graph, TF32 allowed, not `torch.compile`d), timed on the same card, case and inputs. They are
per-module numbers, not end-to-end model numbers. `anthropic` / `cueq` are the external baselines as they ship; an empty cell means the
baseline does not serve that class or mode.

A slice is one (model, stream, constructor kwargs, depth, samples, timing). Only `status: ok` CUDA-graph timings are drawn;
capture fallbacks (`eager`) are marked in the tables and left out of the figures; unsupported / skipped / failed cases are gaps.
How the numbers are taken: [docs/measure.md](../docs/measure.md); the order the engine's backends must keep:
[docs/ladder.md](../docs/ladder.md).

Regenerate the card's folder after every measurement cycle that changed speed results, and commit it with the tokens.
