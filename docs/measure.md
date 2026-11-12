# Measurement

`miniworld_engine.measure`: two measurers and their config. They only measure: one record per case and backend to stdout and to a
JSONL file. They judge, cache and skip nothing; [the judge](judge.md) reads the records.

## Config (`measure/config.py`)

A YAML or TOML file. Every key but `modules` / `backends` is optional:

```yaml
modules: all                         # class names of miniworld_engine.modules, or "all"
backends: [pytorch, cuda, triton, anthropic, cueq]   # "pytorch" is always added (the reference)
external: [anthropic, cueq]          # baselines: informational light
rows: []                             # requirements_module.csv data-row numbers (1 = first row under the header); [] = all
dtypes: []                           # filter: bfloat16 | float32 | master
modes: []                            # filter: eval | train
lengths: all                         # min | all | [128, 256] (intersected with the row's)
depths: min                          # same, for the MSA depth
samples: min                         # min | all (the row's eval_samples / train_samples)
device: cuda
seed: 0
fixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}
# speed only (top level or under speed:)
timing: [graph]
warmup: 5
iterations: 50
flush_l2: true
compile_mode: default
```

Configs in `measure/configs/`: `accuracy.yaml` and `speed.yaml` are the canonical runs the judge compares against (every row, every
length, min depth, min samples, every backend); `all.yaml` is an example (min length, all samples, external backends only).

A **case** is one (row, constructor variant, length, depth, samples, dtype, mode). Constructor kwargs are the row's `dims` + `options`;
`a=x|y` expands into variants. The class is built by name (`module_class`); a class whose `__init__` takes `config` gets the kwargs that
are fields of that config type. `build` constructs, moves to the device, redraws every floating parameter from a seeded generator
(matrices `N(0, 1/fan_in)`, 1-D weights `1 + 0.1 N`, other vectors `0.2 N`, so no zero-initialised projection hides the math), casts to
the case dtype (`master`: fp32 parameters with every `MasterWeights` pinned; inputs bf16; forward under bf16 autocast) and sets the mode.
Inputs come from the class's `make_inputs` with a generator seeded per case: the same values for every backend.

**Shared conditioning (`*_shared_cond` streams).** FoldForge passes a conditioning shared by the samples (or by the samples of each fold) as
a stride-0 `expand` (`AtomDiTStack`: cond `[1, 1, N, d]` expanded to `[S, 1, N, d]`; `TokenDiTStack`: `[L, d]` expanded over S; the R18 grouped
sampler: one cond per fold `[1, F, ., d]` expanded over S), and the DiT backends detect that layout and take dedicated shared paths
(hoisted conditioning tables, `cond_bwd_shared`, one launch for all samples). A requirements row whose stream ends in `_shared_cond`
(`atom_single_shared_cond`, `token_single_shared_cond`, `atom_folds_shared_cond`, `token_folds_shared_cond`; source "FoldForge shared cond")
measures that layout: `make_inputs` gives `cond` as one base tensor expanded over the samples (`primitives.shared_cond_stream`), and the
measurers keep it expanded (`config.keeps_expanded(case)`): accuracy copies the base (in float64 for the gold) and expands it again with
the same strides for the gold, the reference and every backend run, and the base is the leaf whose gradient is measured (`grad:cond`
`[1, F, ., d]`: the sum over the samples, as the gradient of a conditioning the caller expands); speed holds the base and expands it inside
every captured call. Every other case is measured exactly as before: its inputs are copied densely (an expanded input of an ordinary
case, e.g. the `token_folds` cond, becomes a dense copy in accuracy and stays the given view in speed), so no measurer version changed.

`settings(cfg)` = seed, `fixed` and the speed keys: how a case is measured. Keys that only select cases are left out, so a subset run
measures each case exactly as the full run does.

## Accuracy (`measure/accuracy.py`)

```sh
python -m miniworld_engine.measure.accuracy CONFIG.yaml --out DIR_OR_FILE.jsonl [--module X ...] [--backend B ...] [--rows N ...] [--lengths min|all|128,256]
```

Per case the module is built once and deep-copied for each run:

- **gold**: the `pytorch` backend in float64 on the same dtype-rounded parameters and inputs cast up (`module.double()`; the fp32-pinned
  LayerNorm affine becomes fp64; every `wide(...)` statement runs in fp64); no autocast.
- **reference**: the `pytorch` backend at the case dtype, with TF32 allowed for fp32 matmuls (the engine's fp32 kernels compute in TF32
  too).
- **every backend** runs with TF32 allowed (user decision 2026-11-10: "TF32 on for all, always"; `measure/config.py` `TF32_BACKENDS`).
- **each other backend** of the config the class has, with TF32 on like the reference. The range check
  (`unsupported`) runs first in the call's grad / autocast context; a refusal is recorded `unsupported` with its reason.

Measured tensors: the forward outputs and, in `train`, the gradient of every floating input and every parameter for one seeded random
upstream gradient (the same for every run). Training noise (`nn.Dropout`, `p_drop`, `coordinate_noise`) is off: random masks are not
numerics.

Error of a tensor: `max|x - gold| / max|gold|` (`max|x - gold|` when the gold is all zero). Per tensor:

```
tol    = max(err_pytorch, fixed[dtype])
GREEN    err_backend <= tol
YELLOW   tol < err_backend <= 5 * tol
RED      err_backend > 5 * tol          (NaN / inf is RED)
```

The call's light is its worst tensor's (largest `err_backend / tol`), named in the record (`tensor`). The light of `anthropic` / `cueq`
is informational. If the gold or the reference runs out of memory the case is `skipped` for every backend (it cannot be judged), not
a failure; any other exception is `error` with the text.

Record fields: the case (`row`, `module`, `model`, `stream`, `kwargs`, `length`, `depth`, `samples`, `dtype`, `mode`, `backend`), `gpu` (the
card key of the device measured on, `h100_sm90` / `a100_sm80`), `versions` (for that card; see [judge.md](judge.md)), `meta` (kind, measurer / config / prebuild versions, settings, environment), and the result (`status`,
`light`, `err_backend`, `err_pytorch`, `tol`, `tensor`, `fixed`, or `reason`). `ACCURACY_TRACEBACK=1` prints tracebacks.

## Speed (`measure/speed.py`)

```sh
python -m miniworld_engine.measure.speed CONFIG.yaml --out FILE.jsonl [--timing graph compile+graph]
```

One record per (case, backend, timing mode): `median_ms`, `p10_ms`, `p90_ms`, `n`, `speedup_vs_pytorch` (pytorch median / backend
median, same case and timing mode), or `status` `unsupported` / `failed` / `skipped` with the reason.

| timing | what is timed |
|---|---|
| `graph` (default) | the step (eval: forward under `no_grad`; train: forward + backward) warmed up eagerly on a side stream, captured once into a `torch.cuda.CUDAGraph` with static inputs, replayed. No CPU in the number. |
| `compile+graph` | the same capture of `torch.compile(_forward_<backend>, dynamic=False)`, warmed up until nothing recompiles, one throwaway capture, then the kept capture with `error_on_recompile`. Records `compiled_graphs` and `graph_breaks`. `reduce-overhead` / `max-autotune` are refused (their own CUDA graphs would nest). |
| `eager` | the step called directly; `diagnosis_only: true`, never a speed result. |

Policies:

- **Capture failure**: a backend whose step cannot be captured (or compiled) is timed eagerly instead; the record keeps its timing mode
  and adds `timed_as: eager` and `capture_error`. An eager time includes CPU time the GPU waits for: an upper bound. A step that cannot
  run even eagerly is `failed`. Out of memory is `skipped`.
- The range check runs once outside the captured region, on the same static inputs, in the step's grad / autocast context; the region
  calls `_forward_<backend>` directly.
- Train: every floating leaf input and every parameter requires grad; a fixed seeded upstream gradient; `.grad = None` inside the
  captured region (the production `zero_grad(set_to_none=True)` step). No optimizer.
- `master`: fp32 parameters, bf16 inputs, forward under bf16 autocast inside the region.
- L2: a 256 MB write before every timed call, outside its events (`flush_l2: false` for warm L2).
- Queueing: the GPU is held by `torch.cuda._sleep` until the CPU has enqueued every timed call.
- Statistics: median, p10, p90 of the per-call times. Same weights (one seeded build per case, deep-copied) and inputs for every backend.
- TF32 as in accuracy: allowed for every backend; recorded as `tf32_matmul`.
- Clocks are not locked: compare numbers from one job. The record's `env` carries GPU, torch, CUDA, TF32 flags, L2 / warmup / iterations.

`SPEED_TRACEBACK=1` prints tracebacks.

## Versions of the measurers

`ACCURACY_VERSION`, `SPEED_VERSION` (the measurer files) and `CONFIG_VERSION` (`config.py`: what a case means, how modules and inputs
are built) are human-managed: bump one when a change can alter a record, not for a refactor. They are recorded in every record's `meta`.
2026-11-11: `ACCURACY_VERSION` 3, `SPEED_VERSION` 3, `CONFIG_VERSION` 1 (v2 and v3 of the measurers: TF32 allowed for the Triton rung,
then for every backend).
