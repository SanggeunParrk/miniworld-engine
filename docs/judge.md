# Judge

`miniworld_engine.judge` decides, from the measurers' records and human-managed versions, which requirement cases have a current
result (a **token**) and which are owed a measurement. It never measures; the measurers never decide. Every comparison is of version
numbers and plain values a person sets or can read.

## What a result depends on

Each record (and the token made from it) carries:

| part | values | who changes it |
|---|---|---|
| case | module, model, stream, constructor kwargs, length, depth, samples, dtype, mode, backend, kind (`accuracy` / `speed`), timing | the requirements / the canonical configs |
| versions | the module class's `version`; the `version` of every kernel `requirements_kernel.csv` lists for the class whose role serves the mode; the `version` of those families' `h100_sm90.json` tables | a person (class, kernel); the pretuner's merge (table) |
| measurer | `ACCURACY_VERSION` / `SPEED_VERSION`, `CONFIG_VERSION`, `PREBUILD_VERSION` | a person |
| settings | seed, `fixed` standard, speed settings of the canonical config | a person (config file) |
| env | GPU capability, torch, CUDA, Triton versions | the installation |

The current state ("now") is computed from the code and the canonical configs `measure/configs/accuracy.yaml` and `speed.yaml`. The
judge runs on a CPU; its target environment capability is fixed to H100 (9.0).

## Verdicts

For every case the canonical configs ask for (every row, every length, min depth, min samples, every backend the class has, each
timing of `speed.yaml`):

| verdict | when | effect |
|---|---|---|
| carried | the token's versions, settings, measurer versions and env equal now (or a measurer / env difference was answered `keep`) | nothing to do |
| owed | no token; a module, kernel or table version changed; the settings changed; or a `sign ... remeasure` answer covers the difference | measured by the next run |
| pending | a measurer / config / prebuild version or the env changed and nobody answered yet | neither carried nor re-run; blocks a release only |

A pending difference is one question per (part, old, new), answered once for every case it affects:

```sh
python -m miniworld_engine.judge sign measurer:speed 1 2 keep --reason "statistics refactor, same timing"
python -m miniworld_engine.judge sign env:torch 2.13.0+cu129 2.13.1+cu129 remeasure --reason "new cuBLAS"
```

Answers are appended to `judge/answers.jsonl` (the latest answer wins).

## Tokens

`judge/tokens/<Module>.jsonl`, one JSON per line, sorted by case key, e.g.

```
accuracy | LayerNorm | - | atom_single | bias=False;elementwise_affine=True;normalized_shape=128 | L1024 | DNone | S1 | float32 | train | cuda
```

A token holds the case, versions, measurer / config / prebuild versions, settings, env, the result fields (`status`, `light`, errors,
`tol`, `median_ms`, `speedup_vs_pytorch`, `timed_as`, ...), the evidence (host, GPU, record path) and the issue time. For an engine
backend's accuracy token, `result.pass` is `light == GREEN`. Tokens of cases no longer required are kept and reported as unused.

`issue` turns a record into a token only if its module / kernel / table versions, measurer and config versions, prebuild version and
settings equal now; env differences do not block issuing (they surface as pending questions in `owed`). A new token replaces the older
token of the same case.

## Commands

```sh
python -m miniworld_engine.judge owed [--plan DIR] [--redo failed ...] [--again pytorch cueq anthropic ...]
python -m miniworld_engine.judge issue RECORDS...       # files or directories of *.jsonl (*.plans.jsonl ignored)
python -m miniworld_engine.judge sign PART OLD NEW keep|remeasure --reason TEXT
python -m miniworld_engine.judge status                 # token counts per (kind, backend, status)
```

- `owed` prints carried / owed / pending counts per kind, the reasons for owed cases, and the open questions with the `sign` command
  to answer each.
- `--redo STATUS ...`: also owe current tokens whose result has that status (after a measurer fix that changes no other result, so its
  version stays).
- `--again BACKEND ...`: also owe current tokens of those backends (a person asked for it).
- `--plan DIR` writes the owed run: `DIR/cfg/<kind>.<Module>.yaml` per (kind, module), the canonical config narrowed to the owed rows,
  lengths, backends (plus `pytorch`), dtypes and modes (a superset of the owed cases when they do not form a product: the extra cases are
  measured again, harmlessly), and four groups `DIR/G1` .. `DIR/G4` balanced by owed-case count, one line `<kind> <Module> <cfg path>`
  each.

## The measurement cycle (Slurm, H100)

1. On a CPU allocation: `python -m miniworld_engine.judge owed --plan DIR`.
2. One job per group, each **1 GPU, `--time=00:10:00`**. The job reads `DIR/G<i>` and for every line runs
   `python -m miniworld_engine.measure.<kind> <cfg> --out DIR/out/<kind>.<Module>.jsonl`. Records are appended and flushed one by one, so
   a job stopped at its time limit keeps every case it finished.
3. On a CPU allocation: `python -m miniworld_engine.judge issue DIR/out`.
4. Repeat from step 1: cases a job did not reach are still owed and go into the next plan. Stop when `owed` reports 0 owed; answer
   any pending questions with `sign`.

There is no runner script for step 2 in the tree; the job script is the loop above.

## When to bump what

| change | bump |
|---|---|
| a module's math, its reference, or a backend adapter that can change a result | the class `version` |
| a kernel's algorithm that can change a result | its `version` in `requirements_kernel.csv` |
| a launch-config table's content | automatic (the pretuner's merge increments the table `version`) |
| how a case is built or what a record means | `CONFIG_VERSION` |
| the accuracy rule, the gold, the measured tensors | `ACCURACY_VERSION` |
| how a step is timed | `SPEED_VERSION` |
| how kernels are built (flags, toolchain handling) | `PREBUILD_VERSION` |
| a refactor, a comment, a rebuild | nothing |
