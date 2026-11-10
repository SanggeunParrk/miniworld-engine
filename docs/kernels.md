# Kernels

`miniworld_engine.kernels`: one directory per kernel family, plus shared infrastructure in `_*.py` files. Modules reach kernels only
through their `cuda` / `triton` backends (see [modules.md](modules.md)); a kernel has no range or tolerance of its own (both come from
its module rows, [requirements.md](requirements.md)).

## Families

A family directory holds `cuda/` (hand-written CUDA: torch extensions or driver-API cubins), `triton/` (Triton kernels, no TMA) or
both, and `configs/` (launch-config tables). `kernels/integrations/` holds the H100 glue that composes several families into one
module path (TriMul, triangle attention, token / bias-only / local DiT, OPM and PWA training, APB).

| family | serves (module classes) | note |
|---|---|---|
| `triangle_attention` | TriangleAttention* , MSAPairWeightedAveraging (dropout helpers) | [h100/triangle_attention.md](h100/triangle_attention.md) |
| `trimul_inproj` | TriangleMultiplication, BidirectionalTriangleMultiplication | [h100/trimul.md](h100/trimul.md) |
| `transition`, `conditioned_transition` | Transition; the ConditionedTransition inside the DiT blocks | [h100/transition.md](h100/transition.md) |
| `outer_product_mean` | OuterProductMean | [h100/outer_product_mean.md](h100/outer_product_mean.md) |
| `pair_weighted_averaging` | MSAPairWeightedAveraging | [h100/pair_weighted_averaging.md](h100/pair_weighted_averaging.md) |
| `layernorm`, `layernorm_linear`, `rmsnorm` | LayerNorm and the norms inside other modules | [h100/layernorm.md](h100/layernorm.md) |
| `augmented_attention`, `adaln`, `rope`, `gated_projection` | AttentionPairBias, token / atom DiT attention and conditioning | [h100/attention_pair_bias.md](h100/attention_pair_bias.md), [h100/token_dit.md](h100/token_dit.md), [h100/local_atom_dit.md](h100/local_atom_dit.md) |
| `bias_only_dit` | TokenDiTMiniWorldBiasOnly | [h100/bias_only_dit.md](h100/bias_only_dit.md) |
| `dit_variants` | Boltz-2 / RoseTTAFold3 / Chai-1 token and atom DiT | [h100/dit_variants.md](h100/dit_variants.md) |
| `swa_dit` | AtomDiTMiniWorld, AtomDiTESMFold2, SWAAtomTransformer | [h100/swa_atom_dit.md](h100/swa_atom_dit.md) |
| `sampler_ops` | AugmentNoise, AlignEuler, PoolAtoms | [h100/sampler_ops.md](h100/sampler_ops.md) |
| `smooth_lddt` | SmoothLddtLoss | [h100/smooth_lddt.md](h100/smooth_lddt.md) |
| `pair_embed_adaln`, `pair_embed_geom`, `token_pair_init` | PairEmbedAdaLN / Geom / CA, TokenPairInit | [h100/pair_embed.md](h100/pair_embed.md) |
| `mpnn_*` (7 families) | ProteinMPNN | [h100/mpnn.md](h100/mpnn.md) |
| `bias_only_attention`, `tm1`, `tm2` | no module reaches them (no requirements_kernel.csv rows) | — |

Shared helpers: `_cast_cuda` (one-launch fp32/bf16 casts for the H100 paths), `_cast` (the Triton counterpart), `_h100_master`
(bf16 casts of live fp32 masters), `_capture` (weight-pack caches that are correct under CUDA-graph capture), `_compile` (how an entry
point is exposed to `torch.compile`), `_device` (H100 checks, SM count), `_tf32` (round-to-nearest TF32 operands for Triton fp32 dots),
`_dot_precision` (TF32 or IEEE from torch's own switch), `_fastmath` (one-instruction sigmoid), `_tiles` (tile visit order).

## Launch configs (`_launch_config.py`)

Tables: `kernels/<family>/configs/<gpu_key>.json`, for H100 `h100_sm90.json`:

```json
{"gpu": "...", "source": "...", "version": 3,
 "ops": {"<op>": {"<dtypes>|<shape>[|<flags>]": [{"kwargs": {...}, "num_warps": 4, "num_stages": 3, "ms": 0.24}, ...]}}}
```

- Per (op, key) at most the top 4 configs, fastest first; `ms` is what they were ranked by (reference only).
- Key: the sorted `+`-joined floating dtypes of the tensor arguments; the shape (`M=640,N=128`: `M` a row bucket or `L` a length bucket,
  other names exact widths) or a native kernel's dims; the kernel's other int / bool key arguments.
- `gpu_key` = `<model>_sm<major><minor>` from the device name (`NVIDIA H100 80GB HBM3` -> `h100_sm90`); SKUs of one model and arch share a
  table. A new GPU is new files and nothing else.
- `version`: an integer the pretuner's merge increments when the table's content changes. The lookup does not read it; the judge
  does (it is one of the versions a result depends on, [judge.md](judge.md)).

Lookup:

- Triton: `@configured(op, key=..., default=..., prune=...)` wraps a `@triton.jit` kernel. A launch uses rank 1 of the exact
  (dtype, shape bucket, flags) entry that the kernel's `prune` keeps. No entry, an unknown GPU or no CUDA: the kernel's own default,
  warned once per (op, gpu). `scratch=("DQ",)` names tensor arguments whose contents after the launch are config-dependent
  intermediates (per-block partials or split slots whose layout follows the tile config, summed by a later kernel); only the pretuner
  reads it (it does not compare them between candidates). Declared on `augmented_attention_bwd_split_triton` (`DQ`),
  `layernorm_bwd_split_triton` (`PART_DW`, `PART_DB`), `layernorm_bwd_split_mmajor_triton` (`PDG`, `PDB`) and
  `mpnn_relative_position_bwd_reduce_triton` (`partial_table_ptr`, `partial_bias_ptr`). A partial buffer with a fixed layout (fixed
  slice count, e.g. the `PART` of the pair-embedding table gradients, `sampler_ops`' `PARTS`) is compared and is not scratch.
- CUDA: `choose(family, op, dims, dtype, candidates, near=("M",), run=None)` returns the table entry for `dims` (exact, else the nearest
  measured dims: same dtype and non-`near` fields, least summed |log ratio| over `near`) if it is among the kernel's `candidates`,
  else `candidates[0]` (warned once).
- Nothing is measured by default: deterministic and safe under CUDA-graph capture and `torch.compile`.
- `tune(True)` (opt-in, in process): the first call at each exact shape times the stored top-K with `triton.testing.do_bench` and keeps
  the winner for the process. Inside a capture or a compile trace nothing is timed (the winner already found, else rank 1). Warm up
  eagerly, then capture.

Shape buckets (`_shape_key.py`): token lengths `128..768` (`TOKEN_SHAPES`), atom lengths `1024..8192` (`ATOM_SHAPES`). `token_key` /
`atom_key` bucket a length (`shape[-2]` of the pre-flatten activation); `both_key` buckets the row count (every leading axis multiplied
out) for kernels launched from several levels, so a pair `L=1024` and an atom `A=1024` do not share a config. Mapping is floor with
clamping at both ends. Width axes are exact and packed by sorted name into one int64, with a digit derived from the axis names so a
launcher that names a different axis set misses rather than collides.

Families with `@configured` ops but no `h100_sm90.json` (OPM, PWA, some Triton ops of other families) run their decorator defaults
until a table entry exists.

## Prebuilder (`_prebuild.py`)

Builds every hand-written CUDA / C++ artifact ahead of time, in parallel, on a CPU node:

```sh
python -m miniworld_engine.kernels._prebuild [--arch sm90a] [--jobs N] [--only FAMILY ...] [--match REGEX]
                                             [--list] [--check] [--rows] [--compare INDEX.json]
                                             [--stop-after SECONDS] [--reverse] [--cutlass DIR] [--mathdx DIR]
```

- **Plan from code + requirements.** `RECIPES` maps each family to a `prebuild(target, rows)` function that yields calls of the family's
  own loaders (the functions the runtime calls) for every variant its requirement rows ask for. `rows` = `family_rows(family)`: the
  module rows of the classes that launch the family's kernels, by the derive rule. A kernel no row needs is not built. The calls run
  under `_nvcc.planning()`: each `load_extension` / `load_inline_extension` / `cubin` records what it would build and returns a stand-in.
  Nothing is recorded from a model run.
- **Explicit arch.** `--arch` (default `sm90a`) picks the variants and the SM count (132) some builds specialise for; no device is
  queried. Builds run with `TORCH_CUDA_ARCH_LIST` set to the target.
- **Build cache.** Artifacts go to `~/.cache/miniworld_engine_jit` (`$XDG_CACHE_HOME` respected): `torch_extensions/<name>-<key>/` and
  `<family>/<stem>-<key>.cubin`. The key (`_codegraph.py`) is computed from the content each artifact compiles: the entry sources and the
  `#include` closure actually reached (normalised: comments and whitespace do not count), the flags, torch / CUDA / Python versions,
  library versions for CUTLASS / cuBLASDx. It decides only whether an artifact must be (re)built; it is never used to judge a result.
  `miniworld_engine/_prebuilt/` (same layout) is looked up first, then the local cache.
- **Resumable.** An artifact already present in any root is skipped; a build writes to a temporary path and renames into place under a
  flock, so a killed job leaves no partial artifact. `--stop-after` starts no build after a time limit; resubmit to continue.
- **Parallel.** A process pool, one compiler per worker; workers = min(CPUs, memory / `PEAK_GB`) with `PEAK_GB = 5.0`.
- **Checks.** `--list` prints the plan and what is missing; `--check` cross-checks the plan against the code (every build call site
  reached, every `.cu` / `.cpp` compiled by some artifact) and `requirements_kernel.csv`; `--rows` prints each family's rows;
  `--compare` diffs the plan against an earlier `prebuilt-<arch>.json`.
- After a build, `prebuilt-<arch>.json` in the cache root lists the plan's artifacts by relative path (the content of a bundle).
- External headers: CUTLASS 4.2 (`--cutlass` / `CUTLASS_PATH`) and mathdx / cuBLASDx (`--mathdx` / `MATHDX_HOME`).
- Triton kernels are out of scope (they JIT in seconds).
- Native `choose` sites: only each site's default and the configs the tables hold are planned (what the runtime can launch). The other
  tuning candidates are built by `_prebuild_cuda` (below).

### Candidate prebuild (`_prebuild_cuda.py`, opt-in, for tuning)

```sh
python -m miniworld_engine.kernels._prebuild_cuda [--arch sm90a] [--list] [--check] [--jobs N] [--match REGEX]
                                                  [--stop-after SECONDS] [--cutlass DIR] [--mathdx DIR]
```

- A native kernel picks its schedule with `_launch_config.choose(..., candidates, run=...)`; each candidate is a compiled artifact (or
  several share one). The pretuner skips a candidate whose artifact is absent (`not prebuilt`). This tool builds them all: for every
  site in `SITES` and every requirements shape that reaches it (`family_rows` of the site's family, read as the family's recipe reads
  them), the full candidate list the site passes to `choose`, through the family's own loader.
- Sites (H100): `transition/cuda/fused_sm90a.py::_select_schedule` (D128 fused Transition, n = 4 / 2; candidates
  `transition_candidates(sms, backward, n)`, one build per `(ctas, dw_repl, save, n)`: eval the non-saving forward, train the saving
  forward and the backward) and `trimul_inproj/cuda/h100_inference.py::_select_config` (packaged K1 / K3 inference; every K1 / K3
  variant is an instantiation inside the one unit of its `(cz, ch)`, so the candidates add no artifact). 2026-11-10: 54 + 72 keys,
  384 + 1686 candidates, 37 + 7 artifacts, of which 15 (Transition backward) were not built by `_prebuild`.
- Same planning (`_prebuild.record`), content keys, cache, parallel builder and resume as `_prebuild`. It writes no
  `prebuilt-<arch>.json`: candidates are not part of a bundle.
- `--list` prints per site the keys (`op dtype|dims`) with their candidate counts and artifacts; `--check` fails when a `choose(` call
  under `kernels/` is not one of the sites (add the new site's enumeration to `SITES`).

`PREBUILD_VERSION` (human-managed) is bumped when a change to how the prebuilder builds can make the kernels compute or run differently.
Every measurement record carries it; the judge asks once whether results made with the previous builds still stand.

## Pretuner (`_pretune.py`) — in progress

Being written while this page was written; described from its current docstring and code. It fills the launch-config tables on the
GPU from the launches the requirements reach.

```sh
python -m miniworld_engine.kernels._pretune --candidates                      # CPU: write kernels/<family>/configs/candidates.json
python -m miniworld_engine.kernels._pretune [--run DIR] [--config YAML] [--modules M ...] [--rows N ...] \
       [--deadline EPOCH] [--hard-deadline EPOCH]                             # GPU: tune, one process per GPU
python -m miniworld_engine.kernels._pretune --status [--run DIR]
python -m miniworld_engine.kernels._pretune --merge [--run DIR ...] [--dry-run] # CPU: shards -> <gpu>.json tables
```

- **Candidates** (once, then a reviewed file `candidates.json`): per configured Triton op, its decorator default(s), the old engine's
  config space (read from git `HEAD`) or a generated neighbourhood of the default, plus the configs the current table holds; at most 64.
  Native `choose` sites pass their own candidates.
- **Work list** = every case of the speed config (`measure/configs/speed.yaml` by default) for the `cuda` / `triton` backends. Each
  case's step (the speed measurer's `make_step`) runs once eagerly with the pretuner installed as `_launch_config.PRETUNE`. At every
  configured launch whose (gpu, op, key) no shard holds yet, every candidate that survives `prune` is launched, its outputs compared with
  the first candidate's (every tensor argument except the kernel's declared `scratch`, all restored before each candidate; relative max
  error within 2e-2 for 16-bit and 1e-3 for 32-bit tensors, equal values including equal infinities and NaN against NaN do not differ;
  a different result is recorded failed as `mismatch`), then timed with `do_bench` (L2 flushed, median). The launch proceeds with the
  winner. A candidate whose CUDA extension is not prebuilt is skipped (`not prebuilt`); nothing is compiled; `_prebuild_cuda` builds
  them. Nothing is timed under capture or a compile trace.
- **Units and resume.** A unit is one case (`r<row>-<n>`), claimed by creating `<run>/claims/<unit>.json` exclusively and finished with
  `<run>/done/<unit>.json`; a claim of a dead process is taken over; a unit claimed three times is failed. Shards
  `<run>/shards/<job>_<host>_<pid>.jsonl` hold one record per tuned key; keys already in a shard are not re-timed. `--deadline` starts no
  unit after it; `--hard-deadline` abandons the current unit. Parallel = one 1-GPU job per GPU on one run directory.
- **Merge**: top 4 per key into `kernels/<family>/configs/<gpu>.json`; re-tuned keys replaced, others kept; the table `version` is
  incremented by one only when the parsed content changed. That increment is what makes the judge owe the cases of the family's modules.

`PRETUNE_VERSION` exists in the file but is not among the versions the judge compares; a table change is seen through the table
`version`.
