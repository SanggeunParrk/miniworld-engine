# GPU kernel development guide

Hand-off document for an agent that develops the fast path of the model's blocks on **one GPU architecture** of its own
(H100 / A100 / A6000 / a new card). It is written so that this page plus the benchmark harness is enough to develop every
module without asking: which blocks exist, their shapes and dtypes, what "correct" and "fast enough" mean, how to measure,
what to test, what to document. The worked example is the B200 (sm_100a) port; its pages under `docs/gpus/b200/` show
every deliverable in its final form.

Read first, in this order: this page, [contributing.md](contributing.md) (gates, naming, where things go),
[../standards.md](../standards.md) (the bar), [../gpus/template.md](../gpus/template.md) (the page you will write).

## 1. What "done" means

A module is done on a card when **all** of these hold, measured with the harness of section 5 on an otherwise idle card:

1. **Correct.** Output and every gradient (inputs, parameters) against an fp32 PyTorch reference are no worse than the
   engine's existing path on that card (the Triton path, or the previous CUDA path), at every registry shape of section 3,
   in every dtype the module supports (section 2), with ragged masks and with random non-zero weights (section 6).
2. **Fast.** Faster than the strongest available baseline at the same dtype and regime: `torch.compile`d PyTorch always;
   cuEquivariance and Anthropic's kernels where they have the op (the columns of the measurement tables). Report the ratio
   against the fastest other row, not against eager PyTorch.
3. **Graph-capturable and compile-clean.** One opaque `torch.library` op per fused region (with a `fake`), no host
   synchronisation, no data-dependent allocation; the module captures in a CUDA graph and traces under `torch.compile`
   without a graph break (`settings.compile_wrap="custom_op"`, the default).
4. **Training included** wherever the block is trained: forward with saves, backward, fp32 weight gradients, dropout
   semantics of the reference, deterministic where the reference is.
5. **Gated and registered.** Dispatched only on the architectures it was built and verified for (compute-capability gate),
   falls back to the existing path otherwise, has rows in `kernels/registry/registry.csv`, and passes `pixi run ci`.
6. **Documented.** One page per module under `docs/gpus/<gpu>/<module>/`, the completion tables of
   `docs/gpus/<gpu>/<gpu>.md`, a CHANGELOG entry, every number reproducible from a committed command.

## 2. Precision policy (applies to every measurement and every kernel)

| item | rule |
|---|---|
| default training / sampling precision | **fp32 master parameters + bf16 activations** (AMP `bf16-mixed`: `model.float()`, forward under `torch.autocast(bfloat16)`, loss in fp32). Kernels receive fp32 parameters and bf16 activations; they cast the parameters to the kernel dtype *inside* their autograd Function (outside autograd) and hand back fp32 weight gradients that were never rounded to bf16. A cast left in autograd rounds the gradient and costs a copy per call: do not. |
| "bf16 parameters" | an explicit exception (a deployment that stores bf16 weights). Measure it only when that is the point, and say so in the table (`+param_master=false`). Norm gammas stay fp32 in every case. |
| trunk modules (TriMul, TriAttn, Transition, OPM, PWA, APB, pair Linears) | bf16 activations only; fp32 falls back to the PyTorch path. Do not build fp32 kernels for them unless asked. |
| diffusion transformer blocks (token DiT, bias-only token DiT, AF3 atom DiT, SWA atom DiT and their adaLN / conditioned transition / attention parts) | **both** `bf16-mixed` **and** `fp32`. fp32 means fp32 storage and residual stream with TF32 tensor-core matmuls (`allow_tf32=true`, the harness default; the baselines run TF32 too) and fp32 softmax / norm statistics. Draw and tabulate the two precisions separately, each against the PyTorch of the same precision. |
| accumulation | every MMA accumulates in fp32; softmax statistics, LayerNorm statistics, RoPE tables, LSE fp32. |
| FP8 | record only (an option / a table row), never the default path or a representative table. |
| activation checkpointing | off in every measurement and baseline; do not present a checkpointed configuration as a baseline. |
| randomness | training rows carry the reference's dropout (`dropout=` / row-wise masks); the graph replay check for stochastic modules compares against an eager run with the same seed. |

## 3. The model shape registry

Authoritative source: `src/miniworld_engine/kernels/registry/registry_module.csv` (module, stream, lengths, dims, dtypes,
modes, eval / train augmentation). The streams and their axes:

| stream | axis | values | notes |
|---|---|---|---|
| `token_pair` / `token_single` / `msa_token` | L (tokens) | 128, 256, 384, 512, 640, 768 | the harness `seq_len`; the main-figure point is L = 384 |
| `atom_single` / `atom_pair` | S (atoms) | 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192 | the harness runs atom targets at `seq_len * 8` atoms (tokens x 8); **S must be a multiple of 128** (callers pad; `seqused` masks the padding); the main-figure point is 4096 atoms (`seq_len=512`) |
| augmentation A | samples sharing one weight set | inference 5 (and 1), training 48 | diffusion blocks only (`eval_augmentation` / `train_augmentation`); trunk modules are A = 1. Harness `n_augment: auto` picks 5 / 48 |

Module rows (dims are the registered widths; every row exists for `eval` and `train` unless noted; dtype `bf16` means
bf16 only, `bf16|fp32` both):

| module (harness target) | stream | registered dims | dtypes |
|---|---|---|---|
| TriMul outgoing / incoming (`triangle_multiplication`) | token_pair | d_pair = d_hidden in 64, 128, 256, 384 (the B200 page also serves 512) | bf16 (registry also lists fp32 for the PyTorch path) |
| TriMul bidirectional (`triangle_multiplication_bidirectional`) | token_pair | d_pair = d_hidden = 128 (64 / 256 / 384 / 512 in the tables) | bf16 |
| TriangleAttention (`triangle_attention`) | token_pair | (d_pair, d_hidden, heads): 128 / 128 / 4, 64 / 64 / 4, 64 / 128 / 4, 64 / 64 / 2, 256 / 256 / 8, 384 / 384 / 12; 32-channel heads | bf16 |
| Transition (`transition`) | token_pair, msa_token, token_single | n = 4: pair 64 / 128 / 256 / 384 (512 in tables), MSA 64 / 128; n = 2: pair 64 / 128 / 256, single 384 / 768 | bf16 |
| OuterProductMean (`outer_product`) | msa_token | (d_msa, d_pair, d_hidden): 64 / 128 / 32, 128 / 256 / 32, 128 / 384 / 32; MSA depth `+n_msa=1024` (also 4096) | bf16 |
| MSAPairWeightedAveraging (`msa_pair_weighted_averaging`) | msa_token | (d_msa, d_pair, heads): 64 / 128 / 8, 128 / 256 / 8, 128 / 384 / 8; `+n_msa=1024` | bf16 |
| AttentionPairBias (`attention_pair_bias`) | token_single | d_single 384, d_pair 128, 8 heads (`+apb_n_head=16` for the confidence head) | bf16 |
| token DiT block (`dit`) | token_single | d_single 768, d_cond 384, d_pair 128, heads 16 x 48 (24 x 32 / 12 x 64 at 768 via `+tdit_n_head`, 16 x 64 at `+d_single_token=1024`) | bf16 and fp32, A = 5 / 48 |
| bias-only token DiT (`bias_only_dit`) | token_single | as the token DiT, no q / k: attention weights come from the pair bias only (16 x 48, 24 x 32, 12 x 64, 16 x 64 via `n_head` / `d_head`) | bf16 and fp32, A = 5 / 48 |
| AF3 atom DiT (`local_dit`) | atom_single + trunked atom_pair | d_single = d_cond = 128, d_pair 16, 4 heads x 32, hidden 256; block-local attention: 32 queries x 128 keys per window, pair bias `[nwin, 32, 128, 16]` | bf16 and fp32, A = 5 / 48 (and 1) |
| SWA atom DiT (`swa_dit`, ESMFold2 style) | atom_single | d = 128, 4 heads x 32, sliding window \|i - j\| <= 64, SwiGLU hidden 256, adaLN-Zero gates, 3-D RoPE; no pair | bf16 and fp32, A = 5 / 1 inference, 48 training |
| support ops (`adaptive_layernorm`, `conditioned_transition`, `augmented_attention_*`, `swa_atom_attention`, `layernorm`, `gated_linear`, `swiglu_ffn`, ...) | see the CSV | see the CSV | see the CSV |

Block definitions (what the PyTorch reference computes; read the module for the exact code):

- **TriMul** (AF3 Alg. 12 / 13): LayerNorm(z) -> two gated projections a, b -> contraction over the third index (L^3 x d_hidden
  work) -> LayerNorm -> output projection -> gate -> (row-wise dropout) -> residual. `modules/triangle_multiplication`.
- **TriangleAttention** (Alg. 14 / 15): per-row attention over the pair tensor with a pair-derived bias and a gate; starting /
  ending node. `modules/triangle_attention`.
- **Transition**: LayerNorm -> expand (n x width) -> SwiGLU -> squeeze -> residual. `modules/transition`.
- **OPM / PWA**: MSA <-> pair exchange: outer-product mean over MSA rows into the pair (Alg. 9); pair-weighted averaging of
  MSA values with softmax weights from the pair (Alg. 10). `modules/outer_product`, `modules/msa_pair_weighted_averaging`.
- **AttentionPairBias** (Alg. 24 style): single representation attention with a bias from LayerNorm(pair) -> Linear -> heads,
  gate, output projection. `modules/attention_pair_bias`.
- **Token DiT** (Alg. 23): adaLN(cond) -> q|k|v|g projection (optional QK-norm) -> attention with the pair bias (the bias
  LN + Linear of a sampling pair tensor is computed once and hoisted) -> gated out-projection + residual -> adaLN +
  SwiGLU conditioned transition + residual. `modules/dit`.
- **AF3 atom DiT** (Alg. 23 with Alg. 24 trunking): the same block at atom widths with window attention and per-window
  pair bias; self mode (shared adaLN) and cross mode (separate key / value adaLN). `modules/local_dit`.
- **SWA atom DiT**: RMS-adaLN -> q|k|v|g with head RMS norm + RoPE -> sliding-window attention -> gated out-projection +
  residual -> RMS-adaLN + SwiGLU + residual; the conditioning repeats over the augments, so the adaLN modulation is computed
  once per batch element (`forward_hoisted`). `modules/swa_dit`.

Module names are also the benchmark target names; the kernel-level targets are the families of `registry.csv`.

## 4. Baselines every table is compared against

| column | what it is | where it applies |
|---|---|---|
| PyTorch compiled | the module with `implementation=pytorch` under `torch.compile` (the strongest PyTorch); eager only where compile cannot trace the row (state it) | everywhere; training rows of blocks whose compiled backward fails the repeatability check are timed compiled **without** a graph (`cudagraph=disabled`) and marked |
| cuEquivariance | `cuequivariance_torch` triangle ops | TriMul, TriAttn, APB (read `execution_path` in the CSV for what actually ran; its B200 TriMul kernel route is inference-only) |
| Anthropic | the uplifting-biomolecular-modeling release's kernels composed into the same block over the same weights (`MINIWORLD_ANTHROPIC_ROOT`) | inference only; TriMul, TriAttn, OPM, PWA, token DiT, local DiT, SWA DiT |
| engine (Triton / per-op) | what the engine ran before your CUDA path (`+swa_dit_fused=false`, `MINIWORLD_*=0` switches, `implementation=triton`) | the "vs the previous path" column |
| ours | `implementation=miniworld`: your path through the module, dispatch included | |

The speed-up column is **ours / the fastest of the other columns** at that row.

## 5. The benchmark harness

One entry point: `benchmarks/runners/bench.py`, driven by Hydra overrides on a target's `configs/bench.yaml`.

```bash
python benchmarks/runners/bench.py target=<name> level=module mode=<inference|training> \
    min_seq_len=384 max_seq_len=384 seq_len_step=128 'implementations=[pytorch,miniworld]'
```

Each result is one stdout line `<target> seq_len=<L> d_pair=<D> implementation=<impl> time=<ms> ms` and one row of a CSV under
`benchmarks/modules/<target>/artifacts/<device>/` (columns: value, implementation, precision, input_dtype,
parameter_dtype, mode, compiled, cudagraph, execution_path, status, accuracy columns against the fp32 reference).
A failed row prints `failed: <reason>` (an `unsupported` row is an explicit, valid answer, e.g. Anthropic training).

**Launch style matters.** Call the engine environment's `python` directly. A `pixi run ... bash -c` wrapper around the
harness made `torch.compile` produce a PyTorch row 2x slower on the SWA block (28.9 against 13.2 ms), which inflated every
speed-up computed from it. Do not mix launch styles inside one table, and note which one you used.

### 5.1 Options you will use

| option | meaning |
|---|---|
| `target=`, `level=module\|kernel` | what to run; the two levels are separate namespaces |
| `mode=inference\|training` | training times forward + backward of every input and parameter |
| `precision=bf16-mixed\|32` | `bf16-mixed`: fp32 master parameters + bf16 activations under autocast (default policy, section 2); `32`: fp32 (TF32 matmul unless `allow_tf32=false`) |
| `+param_master=false` | bf16 parameters (module cast to bf16); only for the "bf16 parameters" exception |
| `implementations=[pytorch,triton,miniworld,cuequivariance,anthropic]` | rows to run; `anthropic` is inference-only (training is an explicit `unsupported` row); run it as its own invocation with `compile=false` |
| `compile=true\|false` | `torch.compile` the module; `false` for paths that cannot be traced (SWA DiT, local DiT) |
| `cudagraph=auto\|manual\|disabled\|graphed` | `auto` = manual capture for inference; use `manual` for training tables; `graphed` refuses training |
| `min_seq_len`, `max_seq_len`, `seq_len_step` | token length sweep (atom targets multiply by 8); `sweep_axis=d_pair` with `min_d_pair`... sweeps the width |
| `d_pair=`, `d_single=` | widths of the pair-stack targets |
| `n_augment=` | A (default `auto`: 5 inference, 48 training) |
| `mask_prob=` | fraction of padded rows (ragged `seqused` / key masks); keep > 0 in correctness runs |
| `+shared_cond=true` | one conditioning for all samples (`dit`, `local_dit`; for `swa_dit` this is the model's contract and the only realistic setting) |
| `+n_msa=1024` | MSA depth of OPM / PWA |
| `+tri_n_head=<h> +tri_d_hidden=<d>` | TriangleAttention heads / hidden width |
| `+apb_n_head=8\|16` | AttentionPairBias heads |
| `+tdit_n_head=`, `+d_single_token=`, `+dit_qk_norm=true` | token DiT layout and QK-norm |
| `n_head=`, `+d_head=` | bias-only DiT layout |
| `+swa_dit_fused=false` | SWA per-op path (what the fused block replaces) |
| `dropout=<p>` | training dropout (row-wise for TriMul / TriAttn) |
| `name_suffix=_tag` | disambiguates artifact files of parallel runs |

Keys that are not in a target's yaml need the Hydra `+` prefix. Non-numeric lists are quoted for the shell.

### 5.2 Regimes and their traps

- `cudagraph=manual` is a capture of the step after one warm-up; the harness checks that the replay reproduces the eager
  outputs (and, for stochastic modules, that a seeded replay matches). It refuses a run that is neither compiled nor graphed
  unless `allow_eager=true`.
- Training with a CUDA graph requires deterministic backward: a compiled PyTorch reference whose gradient differs between
  identical steps (atomics) is refused (`timed execution differs for output[gradients]...`). Time that row with
  `cudagraph=disabled` and state it.
- Inference runs under `static_weights()` (weights do not change between replays, packed weight forms are cached across the
  capture). Training must not: a weight update between replays must be seen. Learn the contract in `kernels/_capture.py`.
- Timing is median of replays in CUDA events on an idle card. Under a power cap the clock settles and run-to-run spread is
  +-3-5 %; a difference below that is not a result. Re-run cells that move by more than 8 % against a committed value.
- Other processes on the card contaminate timings silently. Check `nvidia-smi --query-compute-apps=pid,process_name,used_memory
  --format=csv -i <gpu>` before and after.
- Kernels you build are JIT-compiled on first use into `MINIWORLD_ENGINE_JIT_ROOT`; give parallel runs separate roots or a
  warmed shared one (a concurrent build of the same cubin fails or stalls).
- Do not run two harness processes on one GPU.

### 5.3 The reference sweep of the main figure

Every module at L = 384 tokens (atom blocks 4096 atoms = `seq_len=512`), both modes, bf16-mixed and, for the diffusion
blocks, fp32. `d_pair=128` for the pair modules. One invocation per (module, mode); Anthropic separately:

| key | target | extra overrides | seq_len |
|---|---|---|---|
| trimul1 | `triangle_multiplication` | `d_pair=128` | 384 |
| trimul2 | `triangle_multiplication_bidirectional` | `d_pair=128` | 384 |
| triattn | `triangle_attention` | `d_pair=128 +tri_d_hidden=128 +tri_n_head=4` | 384 |
| trans | `transition` | `d_pair=128` | 384 |
| apb | `attention_pair_bias` | `sweep_axis=seq_len +apb_n_head=8 d_single=384` | 384 |
| opm | `outer_product` | `+n_msa=1024` | 384 |
| pwa | `msa_pair_weighted_averaging` | `+n_msa=1024` | 384 |
| tdit | `dit` | `+shared_cond=true` (inference: `mask_prob=0.2`) | 384 |
| bodit | `bias_only_dit` | | 384 |
| ldit | `local_dit` | | 512 |
| swadit3 | `swa_dit` | `compile=false +shared_cond=true` (PyTorch row: `compile=true`, `implementations=[pytorch]`) | 512 |

Training invocations add `cudagraph=manual`; fp32 invocations add `precision=32`. The driver and the plot script are
`benchmarks/runners/main_figure_campaign.py` (`PREC=32` for the fp32 runs) and `benchmarks/runners/main_figure_plot.py`.

## 6. Correctness checks

- **Reference**: the module's own `implementation=pytorch` path in fp32 with IEEE matmuls (`allow_tf32=False`) on the same
  weights and inputs. Never compare against another fast path alone.
- **Criterion**: relative Frobenius error of the output and of every gradient no worse than the existing engine path at that
  dtype (the B200 tests assert it per tensor), and below the per-dtype `rtol` of the kernel's `registry.csv` row. For fp32 TF32
  kernels the criterion is against an IEEE fp32 reference with the TF32 path's own error as the bound.
- **Weights must be non-trivial.** adaLN-Zero modules (`adaln_modulation`, output gates) are zero-initialised: with default
  weights the block is the identity and every accuracy column is vacuous. Randomise all parameters (`std = fan_in ** -0.5`
  for matrices, `1 + 0.1 n` for gammas) before comparing; also randomise before judging whether a gradient was rounded.
- **Ragged rows**: lengths that are not multiples of the tile (`mask_prob=0.2` style `seqused`), one row shorter than the
  window, N not a multiple of 128 (padded inside, the first N rows returned).
- **Every A**: A = 1, 2, 3 (small-tile builds), 5, 8, 48; per-sample conditioning and shared conditioning.
- **Determinism**: identical steps give identical gradients unless the reference itself is nondeterministic; document
  atomics-based accumulation.
- **Graph replay stress**: capture the forward (and the training step) and replay thousands of times with ragged inputs;
  a PDL chain that passes 800 replays has hung at 150 before (`tests/gpu/sm100/test_b200_swa_dit_graph_replay_gpu.py`
  runs each case in a subprocess with a timeout). Never call `griddepcontrol.launch_dependents` before the kernel's own
  `griddepcontrol.wait` unless the whole chain has been replay-stressed; the last kernel of a chain is not exempt.
- **fp32-master test**: gradients with fp32 parameters equal the bf16-parameter gradients up to the parameters' own rounding
  and are *not* bf16-rounded (`tests/gpu/sm100/test_b200_fp32_master_gpu.py`).

## 7. Developing a module (the order that worked)

1. Read the PyTorch module and the harness bench function for the target (`bench_module_<target>` in `bench.py`). Run the
   harness once on the existing path and keep the CSV: that is your "before".
2. Write the GPU test first: reference vs path, all registry shapes, all dtypes of section 2, ragged, A sweep, graph capture.
3. Build the path stage by stage with a profiled breakdown (`torch.profiler` kernel sums at the harness shapes); the
   per-kernel table of the page comes from it. Start from the largest stage; fuse neighbours that exchange an HBM
   round trip; launch-bound regions (small A, small L) are won by fewer kernels, not faster ones.
4. Integrate through `src/miniworld_engine/integrations/<module>_<gpu>.py`: arch gate (`serves()` / `refusal()` says why it declines,
   never raises), weight-pack caches keyed on pointer and `_version`, autograd Function that casts parameters inside,
   opaque ops + fakes, dispatch from the module with the old path as the fallback and an environment switch (`MINIWORLD_<NAME>=0`)
   to turn it off for A/B.
5. Add the registry rows, the layout / vocabulary tests that fail until the names agree, the benchmark yaml if a new target.
6. Measure with the harness (section 5), tables + figures with `python -m miniworld_engine.viz.measure_bars <page>`.
7. Write the page from `docs/gpus/template.md`, the completion tables, the CHANGELOG entry; `pixi run ci` and the GPU suite.

## 8. Engineering rules learned on the B200

- **Judge a kernel by energy under the power cap**, not by clock: floor time = max(bytes / HBM, FLOPs / tensor peak, energy /
  (cap - idle power)). A bandwidth- or tensor-bound kernel at the cap is at its ceiling when energy-SoL is ~100 %.
- **Record per-phase timestamps** (`clock64` in the tile loop) before optimising: the hidden cost was IEEE division /
  `sqrtf` in a row kernel (2860 of 5200 cycles per tile), `__fdividef` / `rsqrtf` fixed it.
- **Row kernels** reach copy bandwidth with 16 B per lane, half-warp rows, persistent 8 blocks per SM; a row kernel at a third
  of copy speed is structure, not the atomics.
- **TMA stores** are issued and waited on by a producer-warp lane, not the compute leader; staggering two warpgroups by half an
  iteration overlaps MUFU and row work.
- **Weight gradients**: one grouped split-K kernel for all dense weight gradients of a block (a launch per weight is what
  dominates small blocks); fp32 partials reduced in a fixed order for determinism.
- **Small A**: the modulation / conditioning work scales with A when conditioning is per sample; the model shares it over
  the augments, so the realistic shapes compute it once. Benchmark the realistic one and document the other.
- **Do not benchmark through a different environment than you develop in**; do not compare a number from one launch style
  with another.
- **Do not delete what you did not create**: other agents share the card and the checkout; never kill a process you did not
  start, never rebase or delete a branch that has a worktree attached. Work in your own detached worktree and commit locally.

## 9. Deliverables checklist

- [ ] `src/miniworld_engine/kernels/<family>/<backend>/` sources, built by the newest toolchain that knows the arch
- [ ] `src/miniworld_engine/integrations/<module>_<gpu>.py` with arch gate, caches, autograd Function, fakes
- [ ] registry rows, driver / check entries; `tests/cpu/layout`, `tests/cpu/registry` green
- [ ] GPU tests: numerics, ragged, A sweep, fp32-master, graph capture + replay stress, compile (no breaks)
- [ ] harness tables (bf16-mixed; fp32 for the diffusion blocks) for inference and training, commands in the page
- [ ] `docs/gpus/<gpu>/<module>/<module>.md` + figures, completion tables in `docs/gpus/<gpu>/<gpu>.md`
- [ ] CHANGELOG entry that says what was wrong and what changed; commit message that states the defect
- [ ] main-figure points (section 5.3) re-measured and the figure redrawn

## 10. Reference results (B200, 2026-10-11, fp32-master parameters)

Speed-up of ours over PyTorch (compiled where compile works) at L = 384 tokens / 4096 atoms; the targets other cards are
developed against, not promises. Produced by the sweep of section 5.3.

| module | inference, bf16-mixed | training, bf16-mixed | inference, fp32 | training, fp32 |
|---|---|---|---|---|
| TriMul (outgoing) | 0.102 ms, 3.8× (1.6× vs the fastest other) | 0.404 ms, 3.1× (2.4× vs the fastest other) | bf16 only | bf16 only |
| TriMul (bidirectional) | 0.151 ms, 4.6× (1.9× vs the fastest other) | 0.575 ms, 3.6× (2.7× vs the fastest other) | bf16 only | bf16 only |
| Triangle attention | 0.176 ms, 7.5× (1.8× vs the fastest other) | 0.668 ms, 3.3× | bf16 only | bf16 only |
| Transition (pair) | 0.070 ms, 3.0× | 0.287 ms, 2.1× | bf16 only | bf16 only |
| Attention pair bias | 0.047 ms, 1.5× (1.5× vs the fastest other) | 0.113 ms, 2.2× (1.8× vs the fastest other) | bf16 only | bf16 only |
| Outer product mean | 0.305 ms, 1.5× (1.5× vs the fastest other) | 0.956 ms, 1.4× | bf16 only | bf16 only |
| MSA pair-weighted avg. | 0.219 ms, 2.2× (2.2× vs the fastest other) | 0.736 ms, 1.9× | bf16 only | bf16 only |
| Token DiT | 0.076 ms, 2.6× (1.6× vs the fastest other) | 1.764 ms, 2.0× | 0.125 ms, 2.2× (1.2× vs the fastest other) | 2.832 ms, 1.6× |
| Bias-only token DiT | 0.053 ms, 2.8× | 1.073 ms, 1.3× | 0.080 ms, 2.5× | 2.016 ms, 1.2× |
| AF3 atom DiT (32×128) | 0.059 ms, 4.2× (4.0× vs the fastest other) | 1.588 ms, 2.8× | 0.117 ms, 2.6× (2.4× vs the fastest other) | 3.861 ms, 1.7× |
| SWA atom DiT (window 129) | 0.090 ms, 6.4× (1.9× vs the fastest other) | 1.226 ms, 11.0× | 0.119 ms, 17.8× (1.1× vs the fastest other) | 5.832 ms, 12.2× |

Each cell: ours latency, speed-up over PyTorch of the same precision, and (in brackets, where a cuEquivariance / Anthropic row
exists) over the fastest of all other rows. Triangle-attention training has no cuEquivariance row (it failed
under fp32-master autocast); Anthropic is inference only. Sources: the CSVs kept with the figure, produced by
`benchmarks/runners/main_figure_campaign.py`.

Where the B200 code lives (use it as the layout template):

| module | kernels | integration | page |
|---|---|---|---|
| TriMul | `kernels/trimul`, `kernels/triangle_multiplication` | `integrations/trimul_b200.py` | `docs/gpus/b200/trimul/` |
| TriangleAttention | `kernels/triangle_attention` | `integrations/triattn_b200.py` | `docs/gpus/b200/triattn/` |
| Transition | `kernels/transition/cuda` (`fused_sm100a`, `fused_wide_sm100a`) | module dispatch | `docs/gpus/b200/transition/` |
| OPM / PWA | `integrations/csrc/sm100/` | `integrations/opm_train.py`, `pwa_train.py` | `docs/gpus/b200/opm/`, `pwa/` |
| AttentionPairBias | `kernels/augmented_attention/cuda` | `integrations/attention_pair_bias_b200.py` | `docs/gpus/b200/attention_pair_bias/` |
| token DiT | `kernels/conditioned_transition`, `kernels/augmented_attention/cuda` | `integrations/token_dit.py`, `token_dit_train.py` | `docs/gpus/b200/token_dit/` |
| bias-only token DiT | `kernels/bias_only_dit/cuda` | `integrations/bias_only_dit*.py` | `docs/gpus/b200/bias_only_dit/` |
| AF3 atom DiT | `kernels/augmented_attention/cuda/sm100_atom`, `sm100_atom_local` | `integrations/local_dit.py` | `docs/gpus/b200/local_dit/` |
| SWA atom DiT | `kernels/swa_dit/cuda/sm100` | module dispatch (`kernels/swa_dit/dispatch.py`) | `docs/gpus/b200/swa_atom_dit/` |
| token pair init | `kernels/token_pair_init` | module | `docs/gpus/b200/token_pair_init/` |
