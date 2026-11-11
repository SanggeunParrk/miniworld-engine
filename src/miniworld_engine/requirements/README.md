# Requirements

What every engine module must serve, and how a backend of it is judged.

## Files

- `requirements_module.csv` — the single source of shapes, dtypes and modes. One row per distinct
  (class, stream, dims, options): `module,model,stream,lengths,depths,dims,options,dtypes,modes,eval_samples,train_samples,source`.
  Streams `token_folds` / `atom_folds` (TokenDiT* / AtomDiT* rows): the multi-fold layout `[S samples, F folds, L, d]`, one pair per fold, F = the
  row's `depths` (docs/modules.md, multi-fold contract).
  `module` is a class of `miniworld_engine.modules`; `model` the selector keys that build it; `dtypes` among
  `float32|master` (master = fp32 parameters, bf16 activations, forward under bf16 autocast: this is the only bf16 case, the engine has no
  bf16-parameter requirement, so no row says `bfloat16`; a class without parameters runs its bf16 activations as `master`); `modes` among
  `eval|train`.
  The trunk-side classes (pairformer / evoformer: TriangleMultiplication*, TriangleAttention*, OuterProductMean,
  MSAPairWeightedAveraging, the Bidirectional* classes, PairformerBlock, the trunk Transitions) are `master` only: FoldForge builds the
  trunk as master (FoldForge's `modules/engine/modules.py`); its `--precision fp32` runs the trunk in fp32 with TF32 off, which the `pytorch`
  reference serves. Exception: the diffusion head's conditioning Transitions (FoldForge's `engine/graph.py make_conditioning_transition`, n = 2:
  pair 128 / 256 / 512, single 384 / 768) run fp32 + TF32 under `af3_default` / `model_default`, so those five Transition rows keep
  `float32|master` (user decisions 2026-11-10).
  `dims` / `options` are constructor arguments: `options` holds only arguments that change the class's math (not ones the class
  ignores, e.g. a flag only its caller reads) and, for the model classes (`TokenDiT*` / `AtomDiT*`), only values that differ from the
  class default (each model class defaults to its release's values). Rows with equal (module, stream, dims, options) are one row: the
  union of lengths / dtypes / modes / selector keys, the per-mode sample counts kept, all sources listed.
  `source` says who builds the row (FoldForge / MiniWorld models, or `engine-own`: kept by the engine although no
  downstream constructor builds it).
- `requirements_kernel.csv` — `name,family,modules,role,version`: each kernel of the engine, the module classes whose backend
  launches it, and its role. A kernel has no range or tolerance of its own: both come from its module rows. Roles:
  `backward` computes gradients (a recompute launched inside a backward included); `forward_train` writes what only a backward reads
  and runs only on the grad path; `forward_infer` runs only on no-grad paths; `forward` runs on both. A kernel that no module launches
  (only kernel checks / drivers call it) is not listed. The diffusion sampler's glue (`kernels/sampler_ops`) and the smooth lDDT loss
  are module classes with their own rows: `AugmentNoise`, `AlignEuler` (eval only, fp32 coordinates), `PoolAtoms` (bf16 pooling, eval and train: FoldForge R17) and
  `SmoothLddtLoss`; each family is one shape-independent build per arch (the extensions `sampler_ops_sm90a` / `sampler_ops_sm80`, the smooth lDDT cubin; the
  shapes are runtime launch parameters), prebuilt when some row of its classes is. The La-Proteina pair embeddings and the token-pair
  initialisation (`kernels/pair_embed_adaln`, `kernels/pair_embed_geom`, `kernels/token_pair_init`) are module classes with their own
  rows too (`PairEmbedAdaLN`, `PairEmbedGeom`, `PairEmbedCA`, `TokenPairInit`; parameters inside the class, eval and train, bf16 pair /
  fp32 z out); each family is one shape-independent extension per card (`pair_embed_adaln_sm90` / `_sm80`, `pair_embed_geom_sm90` / `_sm80`,
  `token_pair_init_sm90` / `_sm80`).
  `version` is human-managed: bump a kernel's when its algorithm changes a result (not for a refactor, a comment or a rebuild); the
  judge (`miniworld_engine.judge`) then owes a new measurement of every case of its modules in the modes its role serves.

## Derive rule (kernel range = its module rows)

| role            | module rows of its classes | modes it must serve |
|-----------------|----------------------------|---------------------|
| `forward`       | every row                  | the rows' modes     |
| `forward_infer` | rows with `eval`           | eval                |
| `forward_train` | rows with `train`          | train               |
| `backward`      | rows with `train`          | train               |

The kernel's lengths, dims and dtypes are the union over those rows. Every kernel must resolve to at least one row.

## Accuracy (one rule, every engine backend)

For every requirements row and every (dtype, mode) an engine backend claims, the backend, the `pytorch` backend and a gold run
get the same parameters and inputs:

- **gold**: the `pytorch` backend in float64 on the same (dtype-rounded) parameters and inputs, cast up (`module.double()`; a
  parameter the module pins to fp32 under bf16, the LayerNorm affine, becomes fp64 too; every statement computed "in at least fp32",
  `primitives.wide`, runs in fp64), no autocast;
- **error** of a tensor: `max|x - gold| / max|gold|` (`max|x - gold|` where the gold is all zero). Measured tensors: the forward
  output(s) and, in `train`, the gradient of every floating input and every parameter for one random upstream gradient (the same for
  every run). `err_backend` and `err_pytorch` are both reported.

A backend is accurate when, for every tensor, either criterion holds: (1) it is at least as accurate as the `pytorch` backend
(`err_backend <= err_pytorch`), or (2) it is within a fixed standard (`err_backend <= fixed[dtype]`). As lights:

    tol = max(err_pytorch, fixed[dtype])
    GREEN   err_backend <= tol
    YELLOW  tol < err_backend <= 5 * tol
    RED     err_backend > 5 * tol            (inaccurate; a NaN / inf error is RED)

A call's light is its worst tensor's. Every backend, the `pytorch` reference included, runs with TF32 products allowed in float32
(the engine's fp32 kernels compute in TF32 as well; `measure/config.py` `TF32_BACKENDS`); the gold is float64. The fixed standard is a setting of the measurement config (`fixed:`), by default
bfloat16 `8e-3` (about one bf16 ulp at 1.0), master `8e-3`, float32 `1e-5`. `master` = fp32 parameters, bf16 activations, the forward under
`torch.autocast("cuda", bfloat16)`. Training noise (dropout, coordinate noise) is off in every run: its masks are random draws, not
numerics. The weights are random (seeded), so no zero-initialised projection hides the math.

The rule binds only the engine's own backends (our kernels). External backends (`anthropic`, `cueq`) are comparison baselines
connected as they are: they are measured and shown the same way, but their light is informational; their `_supports_<name>` states
what they run.

The measurer is `miniworld_engine.measure.accuracy` (config: `miniworld_engine.measure.config`, examples in `measure/configs/`). It
is module-agnostic: the only module-specific part is each class's `make_inputs` (the forward kwargs of one call: random activations,
key masks with padding, structural inputs). It only measures — one record per (case, backend) to stdout and a JSONL file — and
judges, caches and skips nothing.

## Support and speed

- Support: a backend's range is its `_supports_<name>` (None = runs; a string = why not). An engine backend must
  cover every row / dtype / mode it is the backend for.
- Speed: measured separately, warm, under CUDA graphs or compiled; eager only as the marked fallback of a step that cannot be
  captured (`timed_as: eager`, see Speed).

## Speed (`miniworld_engine.measure.speed`)

The measurer only measures: give it modules and backends (an accuracy-format config plus the speed keys) and it prints one line and
writes one JSONL record per (row, length, depth, samples, dtype, mode, backend, timing mode): `median_ms`, `p10_ms`, `p90_ms`, `n`, and
`speedup_vs_pytorch` (pytorch median / backend median, same case and timing mode), `timed_as: eager` + `capture_error` when the
step could not be captured / compiled and was timed eagerly (below), or `status` `unsupported` / `failed` with the reason; a case that runs out of memory is `skipped` (in the accuracy measurer: the
float64 gold or the reference does not fit, so the case cannot be judged), never a failure.
It keeps no state, skips nothing and judges nothing; a separate component reads the records.

    python -m miniworld_engine.measure.speed my_speed.yaml --out /scratch/speed.jsonl [--timing graph compile+graph]

    # my_speed.yaml: the accuracy config's keys (modules / all, backends, filters) plus
    timing: [graph, compile+graph]      # default [graph]; "eager" = diagnosis only
    warmup: 5                           # eager warm-up steps before capture, and warm replays after it
    iterations: 100                     # timed calls; the statistics are over these per-call times
    flush_l2: true                      # cold L2 before every timed call
    compile_mode: default               # or max-autotune-no-cudagraphs (never a mode with Inductor's own CUDA graphs)

Timing modes, and why:

| mode | what is timed | includes | excludes |
|------|---------------|----------|----------|
| `graph` (default) | the step (eval: forward under `no_grad`; train: forward + backward) captured once in a manual `torch.cuda.CUDAGraph` with static inputs, replayed | the backend's kernels and the PyTorch glue as eager kernels | all CPU: Python, autograd, dispatch, launches |
| `compile+graph` | the same capture of `torch.compile(_forward_<backend>, dynamic=False)`, after warm-up until nothing recompiles (`error_on_recompile` during capture) | the backend's kernels, the glue fused by Inductor (an opaque kernel is a graph break; both sides still compile) | all CPU, including Dynamo guards and graph-break re-entry |
| `eager` | the step called directly | the GPU work plus any CPU time the GPU waits for | — (never a speed result; `diagnosis_only: true`) |

- Never eager: an eager loop of a small module measures the CPU. Under a manual graph a compiled and an eager kernel cost the same; the
  "compiled slowdowns" seen in eager loops were guards and custom-op dispatch, which a graph removes. `compile+graph` gives every
  backend the fusion a compiled model would give the reference, so a kernel is credited only with what it adds over compiled PyTorch.
  Inductor's `reduce-overhead` (its own CUDA graph trees) is refused: it would nest in the manual capture and adds input copies.
- A backend whose `graph` / `compile+graph` step fails (it cannot be captured, e.g. a host copy or sync inside, or cannot be
  compiled) is timed eagerly instead (uncompiled, same module, inputs and settings): the record keeps its timing mode and says
  `status: ok`, `timed_as: eager` and `capture_error` (the exception text). An eager time includes the CPU time the GPU waits for, so it
  is an upper bound, not comparable one-to-one with a captured time. Only a step that fails eagerly too is `failed` (its reason holds
  both errors); out of memory is `skipped`.
- The backend's range check (`module.unsupported(**inputs)`) runs once outside the region on the same static inputs, in the step's
  grad / autocast context; the region calls `_forward_<backend>` (a check may sync with the host, which a capture cannot hold).
- Train: every floating leaf input and every parameter requires grad, the upstream gradient is fixed and seeded, `.grad = None` inside the
  captured region (the production `zero_grad(set_to_none=True)` step: fresh gradients written, no memset, no accumulation add). No
  optimizer. `master`: fp32 parameters, the forward under bf16 autocast inside the region.
- L2: a 256 MB write before every timed call, outside its events (cold, as Triton's `do_bench`); `flush_l2: false` for warm L2.
- Queueing: the GPU is held by `torch.cuda._sleep` until the CPU has enqueued every timed call, so no replay waits for its launch.
- Same weights for every backend (one seeded build per case, deep-copied), same inputs (`make_inputs` with one seeded generator), fixed
  seeds. GPU clocks are not locked (no root on the cluster): compare numbers from one job; the record carries GPU, torch, CUDA, TF32 flags.
