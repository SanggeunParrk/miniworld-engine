# Architecture

How the pieces of the engine fit together, and where to find each one in the source tree. The code is right where this page and the
code disagree. The other pages each cover one part in detail: [requirements.md](requirements.md), [modules.md](modules.md),
[kernels.md](kernels.md), [measure.md](measure.md), [judge.md](judge.md), [ladder.md](ladder.md).

## The pipeline

```
requirements/requirements_module.csv      what downstream models build: (module class, dims, options, lengths, dtypes, modes)
requirements/requirements_kernel.csv      which kernels each class's backends launch (family, role, version)
        │
        ├──► modules/            one BackendModule per op: pytorch (reference) | cuda | triton | anthropic | cueq
        │        │ _forward_cuda / _supports_cuda pick the card's path from the device capability
        │        ▼
        │    kernels/<family>/   cuda/ (sm_90a, sm_80 [, sm_100a]) + triton/ (portable) + configs/<card>.json
        │    kernels/integrations/   the glue that composes several families into one module path
        │
        ├──► kernels/_prebuild.py   plan from code + requirements; build every CUDA artifact ahead of time (CPU node)
        ├──► kernels/_pretune.py    time launch-config candidates on the GPU; merge into configs/<card>.json
        │
        ├──► measure/accuracy.py    fp64 gold vs the reference vs every backend -> records (JSONL)
        ├──► measure/speed.py       CUDA-graph timing of every backend          -> records (JSONL)
        │
        └──► judge/                 records + human-managed versions -> tokens (judge/tokens/<card>/<Module>.jsonl)
                                    -> which cases are owed a measurement
                 │
                 ▼
             measure/plot.py        tokens -> benchmarks/<card>/ (figures + tables); never measures
```

Each arrow reads the requirements tables. Nothing below `requirements/` decides by itself what has to be served: the prebuilder builds
only the kernel variants some row needs, the measurers and the judge enumerate cases from the rows, and a backend that leaves a row out
records that row as `unsupported` (visible in the tokens).

## 1. Requirements

`src/miniworld_engine/requirements/`. `requirements_module.csv` has one row per distinct (module class, stream, dims, options) that a
downstream model constructor builds (FoldForge's AF3-family models and MiniWorld), plus rows the engine keeps itself (`engine-own`).
`requirements_kernel.csv` lists every kernel a module backend launches, with its family, its role (`forward`, `forward_infer`,
`forward_train`, `backward`) and a human-managed `version`. A kernel's range is the union of its module rows, under the derive rule.
See [requirements.md](requirements.md).

Kernel names carry the card they are built for: a suffix `_sm90` / `_sm90a` is H100, `_sm80` is A100, and no suffix means a Triton or
card-independent kernel. 2026-11-11: 690 kernel rows (374 sm_90, 191 sm_80, 125 without a suffix). No row names an sm_100 kernel.

## 2. Modules and backends

`src/miniworld_engine/modules/`, one file per op family (the map is the docstring of `modules/__init__.py`). Every op is a
`BackendModule` (`modules/primitives.py`):

- `backends` lists the backends the class implements, each one a `_forward_<name>` method. `pytorch` is always present and is the
  reference.
- `_supports_<name>(*args)` returns `None` (runs) or a reason string. `forward` raises `BackendUnsupported` with that reason. A
  backend never falls back to another backend.
- `version` (class) and `backend_versions` (per backend, optionally per card as `"<backend>:<card>"`) are human-managed. The judge
  reads both.
- `make_inputs(...)` builds the forward kwargs of one measurement call. It is the only module-specific code the measurers use.

Selectors (`TokenDiT(model=...)`, `AtomDiT(model=...)`, `TriangleAttention(..., model=...)`, `TriangleMultiplication(..., model=...)`)
return the model's own class. See [modules.md](modules.md).

### How a call reaches the card's kernels

The `cuda` backend of a class is one method pair, `_supports_cuda` / `_forward_cuda`. It branches on the device capability of its
input tensor, and nothing else: there is no global switch and no environment flag. The pattern, as in `Transition`
(`modules/transition.py`):

```python
def _supports_cuda(self, x):
    if a100(x):                                   # capability (8, 0)  -> the sm_80 range
        return self._supports_cuda_sm80(x)
    why = h100_unsupported(x) or ...              # anything but (9, 0) -> a reason string
    ...                                           # shapes / dtypes the sm_90a builds take

def _forward_cuda(self, x):
    if a100(x):
        return kernels.transition.cuda.transition_sm80(...)     # fused_sm80 / fused_wide_sm80
    return kernels.transition.cuda.transition_sm90a(...)        # fused_sm90a / fused_wide_sm90a / fused_generic_sm90a
```

The helpers are `a100` and `h100_unsupported` in `modules/norm.py`, and `on_h100` / `num_sms` in `kernels/_device.py`. Some classes
put the decision in one route function instead, for example `TriangleAttention._cuda_route` in `modules/triangle.py`, which sends
capability (8, 0) to `_cuda_route_sm80` and refuses anything other than (9, 0). Within a card, a dispatch function in the family
picks the kernel variant from shapes and dtypes, and, for a native kernel with several schedules, from the launch-config table
(`_launch_config.choose`).

The `triton` backend runs the same Triton kernels on any CUDA card (no TMA). Its launch configs come from
`kernels/<family>/configs/<card>.json`. A card without a table uses each kernel's decorator default, with a warning.

The external backends wrap upstream code as it ships: `anthropic` (`modules/_anthropic.py`, Anthropic's uplifting-biomolecular-modeling
release under `third_party/anthropic/`) and `cueq` (`modules/_cueq.py`, NVIDIA cuEquivariance 0.12). They are baselines. The engine
does not change them to make them pass anything.

### Kernel builds fail loudly

CUDA kernels are torch extensions or driver-API cubins, built through `kernels/_nvcc.py` (`load_extension`, `load_inline_extension`,
`cubin`). The runtime first looks for the prebuilt bundle (`miniworld_engine/_prebuilt/`, installed by the per-arch kernels wheel),
then for the local JIT cache (`~/.cache/miniworld_engine_jit`). If neither has the artifact, the runtime compiles it. If a build or
load fails, the runtime raises `_nvcc.KernelBuildError`, chained to the original exception. A build failure is never a reason to
take another path. Support decisions (shape, dtype, arch, a missing toolkit) are made before any build and come back as
`_supports_cuda` reasons.

## 3. Kernel families per card

`src/miniworld_engine/kernels/<family>/` holds `cuda/` (hand-written CUDA), `triton/` (Triton kernels) and `configs/` (launch-config
tables and the pretuner's `candidates.json`). The `_*.py` files next to the families are shared infrastructure.

| card | key | arch | status | where the code is |
|---|---|---|---|---|
| H100 | `h100_sm90` | sm_90a | served, measured, judged (tokens, benchmarks) | `kernels/<family>/cuda/` files with `sm90` / `h100` in the name or without a card suffix, `kernels/integrations/*h100*.py`, `token_dit*.py`, `local_dit_h100.py`, ... |
| A100 | `a100_sm80` | sm_80 | served, measured, judged (tokens, benchmarks) | `*_sm80*` files and `sm80/` folders in the families, `kernels/integrations/*_sm80*.py` |
| B200 | `b200_sm100` | sm_100a | **not supported.** About 100 sm_100a sources are in the tree (`sm100/` folders, `*_sm100*.cu`, `integrations/csrc/sm100/`), but almost every `_supports_cuda` refuses capability (10, 0), no kernel row names them, and there are no B200 tables, tokens or benchmarks | as named |
| any CUDA card | — | — | the `triton` backend; launch configs only for H100 and A100 | `kernels/<family>/triton/` |

Families and the module classes they serve are listed in [kernels.md](kernels.md#families). The H100 design notes are in
[h100/](README.md#h100-kernel-notes).

## 4. Prebuilder and pretuner

- `kernels/_prebuild.py` (CPU node, no GPU): `RECIPES` maps each family to `prebuild(target, rows)`, which calls the family's own
  loaders under `_nvcc.planning()` for every variant its requirement rows need. The result is a plan of artifacts, each with a
  content key computed by `_codegraph.py` (from the sources and headers reached, the flags and the toolchain versions). The
  prebuilder builds whatever the cache lacks, in parallel, and the run can be resumed. Targets: `--arch sm90a` (default) and
  `sm80`. `sm100` exists as a target, but nothing measures what it builds.
- `kernels/_prebuild_cuda.py`: builds every tuning candidate of the native `choose` sites. It is only needed for the pretuner.
- `kernels/_pretune.py` (GPU): runs the speed config's cases once, and at each configured launch times the candidates
  (`configs/candidates.json`) whose outputs agree. `--merge` writes the top 4 per key into `configs/<card>.json` and increments the
  table's `version` when its content changed.

See [kernels.md](kernels.md).

## 5. Measure, judge, tokens

- `measure/config.py` turns a YAML config and the requirement rows into cases (row, constructor variant, length, depth, samples,
  dtype, mode), builds the module with seeded random weights, and computes the versions a result depends on (`versions(module, mode,
  gpu)`).
- `measure/accuracy.py` compares every backend with a float64 gold run of the `pytorch` backend. The light is
  `tol = max(err_pytorch, fixed[dtype])`, and only `cuda` and `triton` are held to it.
- `measure/speed.py` captures each backend's step once in a CUDA graph and replays it, flushing L2 before every call. Every backend
  runs with TF32 allowed (`TF32_BACKENDS`).
- `judge/` (CPU) issues a token per case from records made by the current versions. It reports every required case as `carried`,
  `owed` or `pending`, and `--plan` writes the owed run as Slurm work groups. `tools/slurm/cycle.sh` runs one plan → prebuild →
  measure → issue cycle (`CARD=a100_sm80 ARCH=sm80` for an A100 cluster).
- Tokens are kept per card: `src/miniworld_engine/judge/tokens/<card>/<Module>.jsonl`. The answers to the judge's questions are in
  `judge/answers.jsonl`.

See [measure.md](measure.md) and [judge.md](judge.md). The speed order the engine's backends must keep (cuda < triton < pytorch) is in
[ladder.md](ladder.md).

## 6. Benchmarks

`python -m miniworld_engine.measure.plot --gpu <card>` reads the card's speed tokens and regenerates `benchmarks/<card>/`: summary
figures and tables, plus the L and D sweeps per class. It never measures. See [../benchmarks/README.md](../benchmarks/README.md).

## Directory map

```
README.md, LICENSE, licenses/            front door; MIT for the engine, third-party notices
pyproject.toml, pixi.lock                package metadata; the development environment
docs/                                    this documentation; docs/h100/ = per-family H100 design notes
benchmarks/<card>/                       generated speed figures and tables (h100_sm90, a100_sm80)
tools/slurm/cycle.sh                     one judge cycle on a Slurm cluster
third_party/anthropic/                   UPSTREAM.json / INVENTORY.md of the Anthropic release; the release itself goes in upstream/ (not in git)
src/miniworld_engine/
  requirements/                          requirements_module.csv, requirements_kernel.csv, README.md (the rules)
  modules/                               BackendModule classes (one file per op family; mpnn/ for ProteinMPNN)
    primitives.py                        BackendModule, BackendUnsupported, MasterWeights, Linear, wide(), ...
    _anthropic.py, _cueq.py              loaders of the external baselines
    dit_variant_cuda.py                  the cuda glue of the Boltz-2 / RoseTTAFold3 / Chai-1 DiT blocks
  kernels/
    <family>/{cuda,triton,configs}/      ~40 kernel families (triangle_attention, trimul_inproj, transition, layernorm, swa_dit, mpnn_*, ...)
    integrations/                        module-level compositions per card (*_h100*.py, *_sm80*.py, token_dit*.py, local_dit*.py, csrc/)
    _nvcc.py                             nvcc discovery, extension / cubin builds, the artifact cache, KernelBuildError
    _codegraph.py                        content keys of build artifacts (include closure, flags, toolchain)
    _launch_config.py, _shape_key.py     launch-config tables: gpu_key, @configured (Triton), choose (native), shape buckets
    _prebuild.py, _prebuild_cuda.py      ahead-of-time builds (plan from code + requirements)
    _pretune.py                          launch-config tuning and table merge
    _capture.py, _compile.py, _device.py, _cast*.py, _tf32.py, _dot_precision.py, _fastmath.py, _tiles.py, _h100_master.py
  measure/                               config.py, accuracy.py, speed.py, plot.py, configs/{accuracy,speed,all}.yaml
  judge/                                 __init__.py (the judge), tokens/<card>/, answers.jsonl
```

Size (2026-11-11): about 116k lines of Python and 187k lines of CUDA (`.cu` / `.cuh`) under `src/miniworld_engine`, plus the
generated tokens and benchmarks.

## Where to start reading

| question | read |
|---|---|
| what does module X compute? | `_forward_pytorch` of its class in `modules/` (the reference) |
| which kernels run for X on H100 / A100? | `_supports_cuda` / `_forward_cuda` of the class, then the family's `cuda/__init__.py` or `kernels/integrations/`; the kernel rows of X in `requirements_kernel.csv` |
| why was a call refused? | the `BackendUnsupported` text; in records and tokens it is `status: unsupported` with `reason` |
| is X accurate / fast on card C? | `judge/tokens/<C>/X.jsonl` (`kind: accuracy` → `light`, `kind: speed` → `median_ms`, `speedup_vs_pytorch`); `benchmarks/<C>/X/README.md` |
| what must be re-measured after a change? | `python -m miniworld_engine.judge --gpu <C> owed` and the "When to bump what" table in [judge.md](judge.md) |
