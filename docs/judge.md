# Judge

`miniworld_engine.judge` decides, from the measurers' records and human-managed versions, which requirement cases have a current
result (a **token**) and which are owed a measurement. It never measures; the measurers never decide. Every comparison is of version
numbers and plain values a person sets or can read.

## What a result depends on

Each record (and the token made from it) carries:

| part | values | who changes it |
|---|---|---|
| case | module, model, stream, constructor kwargs, length, depth, samples, dtype, mode, backend, kind (`accuracy` / `speed`), timing | the requirements / the canonical configs |
| versions | the module class's `version`; the class's `backend_versions` entry for the case's backend on the card (`"<backend>:<card>"`, e.g. `"cuda:a100_sm80"`, else `"<backend>"`, else 1: one adapter, e.g. an external library rewired); for float32 cases the backend's entry of `measure/config.py` `TF32_RULE` (the TF32 rule it is measured under); the `version` of every kernel `requirements_kernel.csv` lists for the class whose role serves the mode, where a kernel with a card suffix (`_sm80`, `_sm90`, `_sm90a`, ...) counts only for that card; the `version` of those families' `<card>.json` tables. Kernel and table versions count only for `cuda` / `triton` cases | a person (class, kernel); the pretuner's merge (table) |
| measurer | `ACCURACY_VERSION` / `SPEED_VERSION`, `CONFIG_VERSION`, `PREBUILD_VERSION` | a person |
| settings | seed, `fixed` standard, speed settings of the canonical config | a person (config file) |
| env | GPU capability, torch, CUDA, Triton versions | the installation |

The current state ("now") is computed from the code and the canonical configs `measure/configs/accuracy.yaml` and `speed.yaml`, for
one card: `--gpu h100_sm90` (default) or `--gpu a100_sm80` (`GPUS` in `judge/__init__.py`). The judge runs on a CPU; the environment
capability it compares against is the card's (9.0 or 8.0).

## Verdicts

For every case the canonical configs ask for (every row, every length, min depth, min samples, every backend the class has, each
timing of `speed.yaml`):

| verdict | when | effect |
|---|---|---|
| carried | the token's versions, settings, measurer versions and env equal now (or a measurer / env difference was answered `keep`) | nothing to do |
| owed | no token; a module, backend-adapter, TF32-rule (float32 cases), kernel or table version changed; the settings changed; or a `sign ... remeasure` answer covers the difference | measured by the next run |
| pending | a measurer / config / prebuild version or the env changed and nobody answered yet | neither carried nor re-run; blocks a release only |

Only a kernel (or table) listed both in the token and now is compared: a kernel row that enters or leaves `requirements_kernel.csv`
for a class (a new kernel, or a correction of which kernels a class launches) owes nothing by itself. A class that starts launching
another kernel in a way that changes results bumps its own `version`.

A pending difference is one question per (part, old, new), answered once for every case it affects:

```sh
python -m miniworld_engine.judge sign measurer:speed 1 2 keep --reason "statistics refactor, same timing"
python -m miniworld_engine.judge sign env:torch 2.13.0+cu129 2.13.1+cu129 remeasure --reason "new cuBLAS"
```

Answers are appended to `judge/answers.jsonl` (the latest answer wins).

## Tokens

`judge/tokens/<card>/<Module>.jsonl` (`h100_sm90/`, `a100_sm80/`), one JSON per line, sorted by case key, e.g.

```
accuracy | LayerNorm | - | atom_single | bias=False;elementwise_affine=True;normalized_shape=128 | L1024 | DNone | S1 | float32 | train | cuda
```

A token holds the case, versions, measurer / config / prebuild versions, settings, env, the result fields (`status`, `light`, errors,
`tol`, `median_ms`, `speedup_vs_pytorch`, `timed_as`, ...), the evidence (host, GPU, record path) and the issue time. For an engine
backend's accuracy token, `result.pass` is `light == GREEN`. Tokens of cases no longer required are kept and reported as unused.

`issue` skips records of another card (the record's `gpu`, else its environment). It turns a record into a token only if its module / kernel / table versions, measurer and config versions, prebuild version and
settings equal now; env differences do not block issuing (they surface as pending questions in `owed`). A new token replaces the older
token of the same case.

## Commands

```sh
python -m miniworld_engine.judge [--gpu h100_sm90|a100_sm80] owed [--plan DIR] [--groups N] [--redo failed ...]
                                   [--again pytorch cueq anthropic ...] [--again-module M ...] [--again-kind accuracy|speed]
python -m miniworld_engine.judge [--gpu CARD] issue RECORDS...    # files or directories of *.jsonl (*.plans.jsonl ignored)
python -m miniworld_engine.judge sign PART OLD NEW keep|remeasure --reason TEXT
python -m miniworld_engine.judge [--gpu CARD] status               # token counts per (kind, backend, status)
```

- `owed` prints carried / owed / pending counts per kind, the reasons for owed cases, and the open questions with the `sign` command
  to answer each.
- `--redo STATUS ...`: also owe current tokens whose result has that status (after a measurer fix that changes no other result, so its
  version stays).
- `--gpu CARD` (before the command): the card whose tokens and versions are used; default `h100_sm90`. `sign` answers are shared by
  the cards.
- `--again BACKEND ...`: also owe current tokens of those backends (a person asked for it). `--again-module M ...`: also owe current
  tokens of those classes (a measurer fix that changes only their results), optionally only of the kinds in `--again-kind`.
- `--plan DIR` writes the owed run: `DIR/cfg/<kind>.<Module>.yaml` per (kind, module), the canonical config narrowed to the owed rows,
  lengths, backends (plus `pytorch`), dtypes and modes (a superset of the owed cases when they do not form a product: the extra cases are
  measured again, harmlessly), and `--groups` (default 4) groups `DIR/G1`, `DIR/G2`, ... balanced by owed-case count, one line
  `<kind> <Module> <cfg path>` each.

## The measurement cycle (Slurm, per card)

Run the cycle on the card the results are for (records carry the card they were measured on, and `issue` drops records of another card).

1. On a CPU allocation: `python -m miniworld_engine.judge --gpu CARD owed --plan DIR`.
2. One job per group, each **1 GPU, `--time=00:10:00`**. The job reads `DIR/G<i>` and for every line runs
   `python -m miniworld_engine.measure.<kind> <cfg> --out DIR/out/<kind>.<Module>.jsonl`. Records are appended and flushed one by one, so
   a job stopped at its time limit keeps every case it finished.
3. On a CPU allocation: `python -m miniworld_engine.judge --gpu CARD issue DIR/out`.
4. Repeat from step 1: cases a job did not reach are still owed and go into the next plan. Stop when `owed` reports 0 owed; answer
   any pending questions with `sign`.

`tools/slurm/cycle.sh PLAN_DIR [owed options]` runs steps 1-3 once (plus a prebuild of anything the cache lacks); site settings
(`SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`, `GPU_GRES`, `ENGINE_ENV`, `CUTLASS_PATH`, `MATHDX_HOME`, `CARD` (default `h100_sm90`), `ARCH`
(prebuild target, default `sm90a`; `sm80` for A100), `PARALLEL` (measurement jobs at once, default 4), `PLAN_GROUPS`, `NODELIST`) come
from the environment. An A100 cycle: `CARD=a100_sm80 ARCH=sm80 tools/slurm/cycle.sh PLAN_DIR`. Run it again while `owed` reports
owed cases.

## When to bump what

| change | bump |
|---|---|
| a module's math or its reference that can change a result | the class `version` |
| one backend's adapter only (e.g. an external library wired to other kernels) | the class `backend_versions[backend]` |
| one backend on one card only (e.g. the A100 `cuda` path) | the class `backend_versions["<backend>:<card>"]` |
| a new kernel row in `requirements_kernel.csv` alone | nothing |
| how a backend uses TF32 (float32 matmuls) | its entry in `measure/config.py` `TF32_RULE` |
| a kernel's algorithm that can change a result | its `version` in `requirements_kernel.csv` |
| a launch-config table's content | automatic (the pretuner's merge increments the table `version`) |
| how a case is built or what a record means | `CONFIG_VERSION` |
| the accuracy rule, the gold, the measured tensors | `ACCURACY_VERSION` |
| how a step is timed | `SPEED_VERSION` |
| how kernels are built (flags, toolchain handling) | `PREBUILD_VERSION` |
| a refactor, a comment, a rebuild | nothing |
