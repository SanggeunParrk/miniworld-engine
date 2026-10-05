# Changelog

Notable changes to the public API (`miniworld_engine.kernels`) are recorded
here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

The public surface is enforced by `tests/compile/test_public_api.py`.

## [Unreleased]

### Added

- fp32 master parameters (AMP `bf16-mixed`: fp32 parameters over bf16 activations) on every B200 training path, at the speed of bf16
  parameters: TriMul, Transition (D 64-512), AttentionPairBias, OuterProductMean, local and SWA atom DiT, token pair init (PWA,
  TriangleAttention and the token / bias-only DiT already were). The integrations cast the parameters for the kernels inside their
  autograd functions and return the kernels fp32 weight-gradient accumulators unrounded, in each parameters dtype; the SWA DiT and
  token pair init no longer refuse fp32 weights over bf16 activations (token pair init also takes bf16 `left` / `right`). Tests:
  `tests/integrations/test_b200_fp32_master_gpu.py`; page: `docs/gpus/b200/b200.md`.
- B200 TriMul with a hidden width twice the pair width, one direction, at D64 / D128 (`TriangleMultiplication(d_pair, d_hidden=2 *
  d_pair)`: AF3 / Protenix template blocks, pair 64, hidden 128): served by the D64 / D128 kernels unchanged -- per token it has the
  bidirectional block's shapes (4 D planes, LN_out over 2 D), only the cuBLAS contraction between the kernels pairs the planes in one
  direction over all 2 D channels. Inference and training, B <= 8 samples at D64, dropout, CUDA graphs (`b200_infer.hidden_ok`).
  Tests: `tests/integrations/test_trimul_b200_gpu.py` (`-k hidden`); page: `docs/gpus/b200/trimul/trimul.md`.
- B200 (sm_100a) block-local AF3 atom transformer block `modules/local_dit.LocalDiTBlock` (ops `local_dit_block_fwd` / `local_dit_block_bwd`):
  AF3 Alg. 23 with the 32 x 128 trunked attention (atom `i` sees the atoms `[32 (i // 32) - 48, 32 (i // 32) + 80)`, per-window pair bias
  from the trunked atom pair `[B, nwin, 32, 128, d_pair]`, `to_windows` makes it from a dense pair), hand-CUDA `mma.sync` attention and
  pair-bias kernels (`kernels/augmented_attention/cuda/sm100_atom_local`, forward and backward) on the atom DiT's row kernels. Inference and
  training, any N, optional key mask, eager / `torch.compile` / CUDA graph; A48 N4096 training 1.90 ms (7.4x the PyTorch path). Page:
  `docs/gpus/b200/local_dit/local_dit.md`; tests: `tests/integrations/test_b200_local_dit_gpu.py`, `tests/numerics/test_local_dit_reference.py`.
- B200 atom DiT block (`integrations/atom_dit`) is now two opaque ops under an autograd Function: it is served under `torch.compile` and
  in CUDA graphs (it used to refuse inside a compiled graph and take the module path); the weight pack is reused per parameter version and
  rebuilt inside a graph capture. Tests: `tests/integrations/test_b200_atom_dit_gpu.py`.
- Complete Anthropic provider loader (all 34 registered packages), JAX/Pallas serving
  and original backward entry points; A100 module adapters for bias-only DiT,
  bias-only/bidirectional triangle attention and OPM normalization after projection.
  Pairformer blocks compose the connected children. Optional JAX and ESM dependencies
  stay in isolated environments; original architecture and gradient contracts remain.
  See `docs/kernels/anthropic-integration.md` for qualification and limitations.

- A100 Anthropic inference module dispatch for norms, Transition, conditioned transition,
  pair-bias attention, dense token/atom DiT, OPM/PWA and SWA/gather compositions, plus
  locally built native sm80 TriMul in both directions. Explicit MSA requests enter the
  upstream adapter before the engine's own kernels. Unsupported architectures, shapes and
  training modes still refuse by name. See `docs/kernels/anthropic-integration.md`.

- A100 Anthropic native TriangleAttention rebuild registration: verify the pinned source tree
  and compiler record, then create an ABI-local checksum manifest without modifying the
  upstream release manifest. The original sm80 output vectors pass bitwise; module/reference
  and CUDA Graph checks cover both directions and absent, sparse, and fully masked keys.
  Explicit block rows now propagate upstream refusal instead of silently benchmarking
  `flash_triattn` as `triattn_native`. See `docs/kernels/anthropic-integration.md`.

- B200 (sm_100a) token-pair initialisation `kernels/token_pair_init` (ops `token_pair_init_fwd` / `token_pair_init_bwd`): the input
  feature embedder's `left[i] + right[j] + Linear(relative-position one-hot) + Linear(bond one-hot)` as one kernel pair that never builds
  the 139-wide fp32 one-hot (84 MB at 384 tokens), in exact fp32, any B and ragged L, CUDA-graph capturable. 5.6-16.9x the dense
  reference forward and 4.0-6.2x forward + backward at L128-768. Page: `docs/gpus/b200/token_pair_init/token_pair_init.md`; tests:
  `tests/integrations/test_token_pair_init_gpu.py`.
- `modules.swa_atom_attention`: the FlashAttention-4 forward saves its output and lse and the backward reads them (`flash_window_fa4`,
  `settings.swa_flash_saves_lse`, default on) instead of re-running the flash forward and cleaning q / k / v / dq / dk / dv with about 11
  eager launches per block; same numbers (`tests/integrations/test_swa_fa4_attention_gpu.py`). The legacy op stays behind the setting.
- B200 SWA atom DiT block: a global-attention variant (`kernels/swa_dit.interface.is_global`, a window `< 0` or `>= 65536`) that runs
  FlashAttention-4 between the sm_100a stages (bf16, B200). Slower than the per-op path at A = 1, so callers opt in. Tests:
  `tests/integrations/test_b200_swa_dit_global_gpu.py`.
- B200 (sm_100a) AttentionPairBias (the Pairformer single track): `AttentionPairBias.forward` runs hand-written CUDA and
  cuBLAS only through `integrations/attention_pair_bias_b200.py` (`kernels/augmented_attention/cuda/apb/`: pair LayerNorm +
  projection and its backward on TMA rings, the sm_100a attention cores, CUDA row kernels) for implementation MINIWORLD, bf16,
  B = 1, no QK-norm, (heads, d_single) in 8 x 48 / 12 x 32 / 16 x 24 / 24 x 16 at 384 and 16 x 32 at 512, d_pair 128;
  inference at L % 16 == 0, training at L % 128 == 0. `bench.py` (+`apb_n_head`, `apb_anthropic_core`): inference 1.17-1.66x
  the fastest other row, training with CUDA graphs 1.38-1.90x. Page: `docs/gpus/b200/attention_pair_bias/attention_pair_bias.md`;
  tests: `tests/integrations/test_b200_apb_gpu.py`, `tests/numerics/test_apb_b200_gpu.py`.
- B200 token DiT head layouts: the sm_100a attention cores and the fused token DiT paths also serve 24 x 32 and 12 x 64 at
  d768 and 16 x 64 at d1024 in bf16 (16 x 48 stays the only fp32 layout); `bench.py +tdit_n_head`. Tests:
  `tests/integrations/test_b200_token_dit_layouts_gpu.py`, `tests/numerics/test_tdit_heads_sm100_gpu.py`.
- B200 (sm_100a) SWA atom DiT block: the fused block of `kernels/swa_dit` (what `SWADiTBlock` takes) runs every stage on
  hand-written kernels on B200 in bf16 (`kernels/swa_dit/cuda/sm100/`: qkvg forward / backward, window attention forward and
  dQ / dK dV, out-projection backward, out-projection + FFN forward, FFN backward; the weight gradients on cuBLAS, dWqkv | dWg
  as one GEMM), and the hoisted adaLN modulation on `mod_fwd` / `mod_bwd`. The atom count must be a multiple of 128 there
  (callers pad; `seqused` masks the padding) -- the block raises `ValueError` otherwise. `MINIWORLD_SWA_DIT_SM100=0` keeps the
  Triton path. Sources from the research capsule `experiments/swaatom_sm100`; tests:
  `tests/integrations/test_b200_swa_dit_gpu.py` (the Triton tests in `tests/numerics/test_swa_dit_fused_gpu.py` pin that path).
- B200 atom DiT (`integrations/atom_dit.py`): the sm_100a path now serves a [B, N] key mask and any atom count N, so MiniWorld's
  atom transformer (which always passes the structure's `atom_mask`) takes it. N is padded to a multiple of 128 inside the call
  (the padded keys masked, the pair tensor read in place); `pair_bias.cu` folds the mask and the padding into the pair bias
  (masked keys -1e4: zero softmax weight in any row with a valid key) and drops dbias on masked keys, so the attention kernels
  are unchanged. Tests: masked and padded training / inference / CUDA-graph cases in `tests/integrations/test_b200_atom_dit_gpu.py`.

- B200 (sm_100a) AF3-style atom DiT block: `DiTBlock` at atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair 16,
  transition n = 2) runs the whole block, inference and training, on hand-written kernels (`kernels/augmented_attention/cuda/sm100_atom/`:
  conditioning projections, AdaLN + q / k / v / gate, the pair bias in both layouts, attention forward / dK dV / dQ / dbias, the
  post-attention + transition forward, and their backwards; the weight gradients on cuBLAS) through `integrations/atom_dit.py`.
  Gate: B200, bf16 inputs, B = 1, N a multiple of 128, no key mask, no QK-norm; anything else keeps the Triton path;
  `MINIWORLD_ATOM_DIT_SM100=0` turns it off. The sources are the research capsule's (`experiments/atomdit_sm100`), built on
  first use by the newest nvcc that knows sm_100a. Page: `docs/gpus/b200/atom_dit/atom_dit.md`; tests:
  `tests/integrations/test_b200_atom_dit_gpu.py`.
- B200 (sm_100a) TriangleAttention at d_pair 64-512 (heads of 16 or 32 channels) now serves training as well as inference
  (`b200_triattn.WideTrain`: the module's dropout, bf16 or fp32 parameters). The backward runs a gate backward and a
  projection-dgrad + LayerNorm-backward kernel on tcgen05; dq / dk / dv / dg / db land in one buffer, so the parameter
  gradients are one cuBLAS GEMM (K = L^2) and a finish kernel. The wide forward's projection kernel runs as 2-CTA clusters
  (`cta_group::2`, each SM streaming half of the weights) and the output kernel adds the residual through the MMA. Module
  step (CUDA graph) vs the fastest of PyTorch compiled / cuEquivariance / Anthropic: inference 1.23-1.86x, training
  1.84-3.05x; vs the repository's Triton path 1.30-2.04x / 1.44-2.23x. `viz.measure_bars` draws a "Triton path" column in
  the Triton colour. Page: `docs/gpus/b200/triattn/triattn.md` (measurement tables regrouped by head layout, with charts).
- `implementation="anthropic"` for TriangleMultiplication on B200 (sm_100). The release ships no sm_100 binary, so
  `miniworld-engine dev build-anthropic-sm100a <payload>/build` compiles its sm_80 member (unmodified sources) for sm_100a with the
  release's own builder and manifest; `integrations.anthropic_trimul` registers cc 10.0 as `sm_100a` in the release's loader and
  assembles the member as `sm80_ops.serve_sm80` does (K1 -> torch.bmm -> K3; the bidirectional module as one unit at twice the
  hidden width). Served: one direction D64-D384, bidirectional D64 / D128; everything else refuses with the reason. The harness's
  `anthropic` row runs it through `ANTHROPIC_TRIMUL_BUILD_DIR`; `tests/integrations/test_anthropic_trimul_b200_gpu.py` (23 tests)
  checks it against the fp32 reference.
- `miniworld_engine.viz.measure_bars`: a length sweep and a dimension sweep bar chart under every measurement table of a GPU page
  (the rule is in docs/gpus/README.md). Tables keyed `(Length, MSA depth)` get an MSA-depth sweep instead of the dimension one;
  `(Length, Dimension)` pages chart as before.
- docs/gpus/b200/opm/opm.md, docs/gpus/b200/pwa/pwa.md: kernel tables, flow figures and harness measurements of the B200
  OuterProductMean / MSAPairWeightedAveraging paths (inference L128-768 x S1024 / 2048 / 4096, training S1024), and their
  completion tables in b200.md.
- `modules/bias_only_dit`: `BiasOnlyDiTBlock` / `BiasOnlyAttention`, the token DiT block whose attention logits are the pair
  bias alone (no query, no key; the `bias_only_v` bench ablation made a module). On B200, bf16 inference with no autograd
  runs `integrations/bias_only_dit.py` -> `kernels/bias_only_dit` (hand CUDA + cuBLAS): the attention weights
  softmax(pair bias) are made once per pair, and the per-step attention is the sm_100a `pv_gate_inf` GEMM (P in tensor
  memory, sigmoid(g) in its epilogue). `bench.py target=bias_only_dit`: 1.52-2.22x PyTorch compiled (L128-768, per-sample
  conditioning), 1.58-2.55x with a shared conditioning. Training (bf16, autograd on) runs
  `integrations/bias_only_dit_train.py`: one opaque forward / backward per block (`dpb_sm100` for the bias gradient, the pair
  LayerNorm and bias on mma.sync in both directions, two-warp-per-row CUDA rows, the expand GEMM + SwiGLU keeping a | b),
  1.20-1.27x PyTorch compiled at A = 48 (L384 / L768), every gradient within 0.96-0.99x PyTorch bf16's error against fp32.
  Both paths serve 24 heads x 32 (`n_head=24`), 12 x 64 (`n_head=12`) and 16 x 64 (`BiasOnlyDiTBlock(d_head=64)`, 1024
  attention channels) as well as 16 x 48: training 1.15-1.30x, inference 1.46-2.63x (`bench.py target=bias_only_dit
  +n_head=24 | +n_head=12 | +d_head=64`).
  Page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md`.
- B200 TriMul, D64 native path: B samples (B <= 8) in one call (tokens b-major, per-sample token mask and row-dropout scale, the plane bmm batched
  over d x B) instead of B calls; MiniWorld's template embedder runs its templates this way. 1.1-2.1x the B sequential calls at L128-384
  in inference, 1.2-2.6x in training (page: `docs/gpus/b200/trimul/trimul.md`); tests: `tests/integrations/test_b200_trimul_batch_gpu.py`.
  `integrations.trimul_b200.MAX_BATCH` (8) states the limit for callers that fold samples into the batch; on an engine without it a B > 1 call
  takes the slow path.

### Changed

- **Breaking -- every `x = x + f(x)` module returns `x + f(x)`.** `AugmentedAttentionPairBias`, `ConditionedTransition`,
  `TrianglePairAttention` and `BidirectionalTriangleAttention` now add their own input as the residual, like TriangleAttention,
  TriangleMultiplication, Transition, AttentionPairBias and MSAPairWeightedAveraging already did; `DiTBlock` only chains its two
  parts. A caller that wrote `x = x + module(x)` must write `x = module(x)` (the old form adds the residual twice, with no error).
  `AugmentedAttentionPairBias.delta` and `ConditionedTransition.delta` return the update alone through the same dispatch, for a
  caller that composes it some other way (a magnitude-preserving sum, a released checkpoint's own residual). Cross-tensor or
  externally gated updates are unchanged: OuterProduct / OuterProductMean (`residual=`), the SWA DiT's attention / FFN parts.
- B200 MSAPairWeightedAveraging: the contractions o = w · v and dv = wᵀ · do (and dw) run on cuBLAS `bmm` (5-15 % faster than
  the tcgen05 kernels at L256-1024, same bits), followed by one gate / out-projection / dropout / residual pass
  (`pwa_gate_out`, bf16x2 gate math); inference takes the same forward. Removed the unused sm_100a kernels (`pwa_ctr`,
  `pwa_plain`, `pwa_fwd`, `pwa_fwd2`, `pwa_glue`, `dgv_bwd`, `dgv_finish`, the ablation switches; `pwa_sm100.cu` ~3100 ->
  ~1950 lines) and the H100 `pair3` build on B200. The per-CTA partial sums of `pwa_glue2` / `dv_bwd` are combined by 8 split
  groups in a fixed order inside their finish kernels. `refusal` on B200 also requires S a multiple of 128 and N <= 1024.
  Harness speedup over the fastest other row: inference 2.00-2.44x, training (CUDA graph, dropout 0.15) 1.68-1.91x.
- B200 OuterProductMean: the prologue backward's column sums are finished by `opm_pbwd_finalize` (the ticket reduction
  `opm_pbwd_reduce` is removed). Harness speedup over the fastest other row: inference 1.16-1.57x, training (CUDA graph)
  1.24-1.57x.
- The forward fakes of the B200 OPM / PWA custom ops return the sm_100a saved-tensor shapes (compiled training on B200).
- `tests/integrations/test_{opm,pwa}_train_gpu.py` run on B200 as well; new: dropout in `pwa_glue2` matches a materialized
  gradient bit for bit, and the B200 PWA output, gradients and inference update stay within 1.15x of the module's own bf16
  statements' error against an fp32 copy (the out / dmsa tolerance against the Triton path is 8e-3 on B200).
- B200 TriMul D128 bidirectional training runs the shared D64 / D128 kernels (k1w -> k3g -> b1g at H = 256 -> b7g at
  eight plane chunks) instead of its own K1 / K3 / B1r / B7r extension (`b200_bidir.py`, removed). Its fp32
  LayerNorm-parameter gradients are now bit-identical across runs, and the 148-SM restriction is gone. The per-CTA
  LayerNorm-gradient rows of every D64 / D128 backward are summed by one fixed-order kernel (`lnpart_sum`) instead of two
  torch reductions. Harness training step (CUDA graph, dropout 0.25) 0.99-1.06x the retired path's 2026-09-29 numbers;
  a same-card A/B of the two paths varied from 0.93x to 1.08x between cards.
- H100 TriMul inference K1 reads row-major `W_l, W_lg, W_r, W_rg` in place (`K1ParamsQ`, four
  TMA maps) instead of a per-call packed `w1`: no weight-pack kernels on bidirectional D64 and
  single-direction D64–384; output bitwise-identical. The column-major D128 bidirectional storage
  keeps the pack (a transposed-B K1 was slower and was not kept).
- H100 D128 training B7 (`B7_WT_MN`, mode bit 64) reads row-major front weights as MN-major
  tiles, so no transposed copies are made when the weights arrive row-major; the default
  column-major storage is unchanged.
- `docs/gpus/` is one folder per GPU (`<gpu>/<gpu>.md` module-level completion tables,
  `<gpu>/<module>/<module>.md` kernel-level tables and flow figures); `h100-dispatch.md` is now
  `h100/dispatch.md`. Page format: `docs/gpus/README.md`, `docs/gpus/template.md`.
- H100 TriMul K1 (inference, the bidirectional wide front and the D128 training front) and B7
  read the token mask [L] and form m[i] & m[j] themselves: no [L, L] pair mask is built per call
  (inference; bidirectional D128 training). The bidirectional wide inference front reads the
  projection weights in place (no pack). Outputs and gradients bitwise-identical.
- B200 LayerNorm / RMSNorm with implementation MINIWORLD resolve to PyTorch ops that `torch.compile` fuses (known-best table,
  compute capability 10.0); other cards are unchanged. `settings.b200_engine_triton` (default off) keeps the engine's Triton kernels on a
  B200. A strict `engine_backend="triton"` process still wins. The PyTorch `RMSNorm` that stands in for the kernel returns the
  input dtype like the kernel does (torch's `rms_norm` returns fp32 under autocast, which gave the attention's Triton kernels an
  fp32 q next to a bf16 k). Tests: `tests/compile/test_b200_norm_dispatch.py`.
- The opaque ops of the fused token DiT paths lost their architecture in the name: `token_dit_h100_infer` -> `token_dit_infer`,
  `token_dit_train_sm100_fwd` / `_bwd` -> `token_dit_train_fwd` / `_bwd` (they serve A100 too). The fused token DiT runner takes the CUDA row
  kernels on A100 as well as B200 (`MINIWORLD_TOKEN_DIT_ROWS_CUDA=0` keeps the Triton rows), and `integrations/token_dit.py` serves sm_80
  (L % 128 == 0, 16 x 48) like sm_90.

### Fixed

- Triton augmented attention, memory-efficient path (`compute_efficient=False`): a head dim below 16 padded its dot lanes to
  `next_power_of_2(D)` (8 for D = 8), under `tl.dot`'s K >= 16, so the kernels did not compile; it now pads to at least 16 as the
  compute-efficient path does (the padded lanes are masked loads of zero). `tests/numerics/test_augmented_attention_small_head_gpu.py`
  also picked a 32 x 32 tile that the B200 config set does not have (all 12 cases failed there); it now takes the smallest w4 / s2 tile
  of the active set (32 x 32 where there is one). 12 passed on B200.
- B200 token DiT attention cores could hang under concurrent GPU work (seen as a CUDA graph with two concurrent branches -- a sampling
  run beside a training step -- stuck at ~240 W): in `attn_inf_tf32`, `attn_fwd_tf32` and `attn_inf` (P double-buffered) the softmax
  warpgroup could hand over P(G) and P(G + 1) before the MMA warp tested P(G) (QK(G + 1) is issued before that wait), so the single
  `p_full` barrier completed two phases and the wait on its parity never returned. `p_full` is now one barrier per P buffer (G & 1,
  parity (G >> 1) & 1); completing one twice would need P(G + 2), whose S(G + 2) is only issued after the MMA warp passed P(G). Kernels
  with a single P buffer (`attn_fwd2`, the atom `attn_fwd`, SWA `attn_dkv` / `attn_dkvq`) wait for PV(G - 1) before writing P(G) and
  could not run ahead. Found with cuda-gdb (stuck warps' PCs mapped to source lines); a Protenix step graph with the rollout on a second
  stream, which hung in 6 of 7 runs, completed 4 of 4 runs of 300 replays. Same results (token DiT tests unchanged).
- B200 weight-pack caches were wrong inside CUDA graphs whose weights change between replays (an optimizer step, or a sampling
  run inside a training graph): the token DiT (inference runner and its pair-bias cache; training pack), bias-only DiT (inference runner
  and its attention-weight cache; training pack), TriAttn wide-width pack (inference and training), AttentionPairBias inference packs and
  the TriAttn modules' fused-inference weight concatenation reused, during a capture, an entry built by the eager warm-up -- so the graph
  recorded no packing kernel and every replay computed with the weights of capture time (training: gradients of stale weights). The
  caches are now scoped to the capture (`kernels/_capture.py`: the capture sequence id from the driver): an eager entry serves only eager
  calls; a capture packs once, recorded, so each replay repacks the current weights, and reuses that pack for its other calls; entries of
  finished captures are dropped. The atom DiT and SWA DiT caches, which bypassed the cache during a capture, now also pack once per
  capture instead of at every call. Updates that bypass the version counter (torch's fused optimizers) still need
  `torch.autograd.graph.increment_version(params)` after the step. Tests: `tests/integrations/test_b200_pack_cache_capture_gpu.py`
  (seven of its nine replay cases fail before the change).
- A100 fused Transition (`kernels/transition/cuda/fused_sm80.py`) under `torch.compile`: `torch.compile` ignores `lru_cache`, so it
  traced the 524288-element integer computation of the pack tables into the graph and Inductor refused it (a value-range
  assertion) in inference and in training. The tables are built inside the opaque pack op now; regression test
  `test_the_compiled_module_matches_eager` in `tests/numerics/test_transition_fused_sm80_gpu.py`.
- B200 SWA atom DiT weight-pack cache (`kernels/swa_dit/cuda/sm100._cached`) keyed on (data_ptr, version, shape) served the old
  tensor's packed copy to a new weight placed at a freed one's address (a bf16 cast, the next test's weights). It is keyed on the tensor
  objects now (weak references plus version), and bypassed while a CUDA graph is captured (a hit would record no pack kernel, so every
  replay would read the packed weights of the capture-time step) and for inference tensors. Tests:
  `tests/integrations/test_swa_dit_pack_cache.py`.
- B200 OuterProductMean training was not deterministic from run to run at L >= 384 (whole rows of one group): the prologue
  backward released an input stage to the next TMA load while its generic shared-memory reads could still be in flight. A
  `fence.proxy.async.shared::cta` before the release fixes it; the same fence now precedes the stage releases of the PWA
  `pair_fwd` / `pair_bwd`. Both paths give identical results across runs at every tested L; no speed change.

### Added

- SWA atom DiT fused block moved from team-gm 14f2c73 ("research(swa): preserve opt-in fused atom
  transformer work"; copied from team-gm 4fafa83, `swa_fused_triton.py` + `swa_cuda/`) into the new
  kernel family `kernels/swa_dit`: the ESMFold2 `SWAAtomBlock` (RMSNorm + adaLN-Zero, QKVG
  projections + q/k RMSNorm + 3D RoPE, window-128 attention with `seqused` masking, sigmoid gate +
  out-projection, SwiGLU FFN) as 9 autotuned Triton kernels (3 forward, 6 backward) and 3 hand-CUDA
  sm_90a bf16-wgmma stages (qkvg forward, out-projection + FFN forward, FFN backward), bf16,
  d_atom 128 / 4 heads / SwiGLU hidden 256. Kernel bodies and CUDA sources are unchanged; the
  autotune key is `atom_key(S, ...)` instead of team-gm's log2(rows), `configs/default` holds exactly
  the configs team-gm's decorators enumerated, the CUDA stages JIT-build under
  `MINIWORLD_ENGINE_JIT_ROOT` and fall back to Triton, and team-gm's `SWA_*` environment switches are
  `settings.swa_dit_*` (same defaults). Public: `kernels.swa_dit_block`,
  `kernels.swa_dit_hoist_modulation` (and `kernels.swa_dit.interface.refusal`). `SWADiTBlock`
  (implementation other than pytorch) now runs the fused block where `refusal` accepts the call
  (`settings.swa_dit_fused=False` keeps the per-op path), and `SWADiTBlock.forward_hoisted` takes the
  augment-invariant conditioning [B, S, d_cond] to compute the modulation once per batch element.
  Registry: 12 rows (`swa_dit_*`), drivers, checks, fp32 reference. Tests:
  `test_swa_dit_fused_gpu.py`, `test_swa_dit_fused_dispatch.py`.
- `swa_dit` keys every Triton launch on the augment count as well: `atom_key(S, A=N // B, C[, NHID])`,
  A = 1 included (MiniWorld's input feature embedder calls with num_aug=1, diffusion with 48 in
  training and 5 samples at evaluation); the drivers tune A = 1 / 48 (and 5 for the inference
  forward). The bf16 qkvg backward's config sets gained SP = 1 tiles for A = 1.
- fp32 fused SWA atom DiT block: `swa_dit_block` dispatches on the activation dtype; all-fp32 calls
  (MiniWorld v1.3's fp32 atom transformer) run 5 new Triton kernels (`swa_dit_*_fp32_triton`,
  `triton/forward_fp32.py`, `backward_fp32.py`; Triton only) around the shared bf16-operand window
  attention, as the per-op path's FlashAttention-4 does. Residual stream, norms, modulation, RoPE,
  gates and SwiGLU are fp32; projections tf32 and the FFN GEMMs tf32x3 (measured: Triton's tf32
  truncates, and only the FFN's truncation showed). Accuracy against the fp32 reference matches the
  per-op fp32 path; fwd+bwd of 3 blocks at N=1, S=4096 (H100, CUDA graph) 0.87 ms against 1.83 ms.
  `refusal` accepts all-bf16 or all-fp32 and refuses mixed dtypes.
- B200 (sm_100a) hand-CUDA TriMul for both modules (bidirectional and one direction), D64 / D128 / D256 / D384 /
  D512, inference and training (`kernels/trimul_inproj/cuda/b200_{infer,train,bidir}.py`, `b200_sources/`: k1w front ->
  cuBLAS contractions -> k3g (D <= 128) or k3w (D >= 256, LayerNorms folded into the output GEMMs); training backward
  b1s / b1g -> contraction grads -> b7m / b7g at D64 / D128 one direction, a cuBLAS + memory-kernel composite at D >= 256,
  and B1r / B7r for D128 bidirectional), dispatched by `integrations/trimul_b200.py`. LayerNorm-parameter gradients
  are fixed-order sums (bit-identical across runs). With a CUDA graph it is 1.3-3.1x the fastest of PyTorch compiled /
  cuEquivariance 0.12 in inference and 1.2-3.0x in training (`docs/gpus/b200/trimul/trimul.md`).
- B200 (sm_100): hand-CUDA fused Transition for D128/n=4 bf16 (`kernels/transition/cuda/fused_sm100a.py`,
  one forward kernel with 2-CTA tcgen05 products, one backward kernel + a partial reduction), dispatched
  from `modules.Transition` and `ops.transition`; `MINIWORLD_TRANSITION_FUSED_SM100A=0` opts out. Module
  training step 2.5x / 2.3x the Triton residual path at L384 / L768. The kernels are built into cubins
  by the newest nvcc that knows sm_100a (13.1 on the B200 box; 12.9's ptxas is ~8 % slower on the
  backward) and launched through the driver API. Page: `docs/gpus/b200/transition/transition.md`.
- B200 (sm_100): hand-CUDA Transition for D64/256/384/512, n=4, bf16 (`kernels/transition/cuda/fused_wide_sm100a.py`:
  D64 one fused forward + one fused backward; D256 fused forward, split backward; D384/512 LN -> expand+SwiGLU -> squeeze,
  split backward from the saved a / b), same dispatch and opt-out as D128. Module L384 vs torch.compile: inference
  1.3-3.6x, training 1.08-2.5x. The D128 kernels drop two cluster-scope releases per launch (D128 L128 inference
  15.1 -> 13.5 us kernel time, bit-identical outputs). Kernel sources under `sm100/` are generated from the research
  capsule (`experiments/transition_fused_sm100/export_engine.py`).
- B200 (sm_100): hand-CUDA Transition at n=2 for D64/128/256/384/512/768, bf16 (`fused_wide_sm100a`, same kernels
  built with `-DHID=2D`; D768 runs the D512 chain with the squeeze and d_xn GEMMs as two 384-column launches), for the
  pair, MSA and single streams (any whole number of 128-row tiles). D384 (n=2 and 4) is now one fused forward kernel
  (`widths/tfwd_d384.cu`) and one fused backward kernel for the gate, d_xn and the LayerNorm backward + a fixed-order
  dgamma / dbeta reduction (`widths/tbwd_d384.cu`; dW stays cuBLAS); single-stream D384 calls of at most 16 tiles take a
  cluster forward that splits the hidden units (`widths/tsmall_w.cu`) and the item-scheduled backward gate. Module
  (bf16 parameters, CUDA graph) vs the faster of Triton / torch.compile: pair L384 n=2 inference 1.8-2.8x, training
  1.2-1.9x; diffusion conditioning [48, L, 384] n=2 training 174.2 / 296.8 us at L384 / L768 (was 220.7 / 328.9 with the
  three-kernel D384 chain; torch.compile 228.1 / 370.5); single stream [1, L, 384] inference on par with Triton, training
  1.5-1.6x slower. Page: `docs/gpus/b200/transition/transition.md`.
- B200 (sm_100a) hand-CUDA TriangleAttention, the whole module (`integrations/triattn_b200.py`,
  `kernels/triangle_attention/cuda/b200_triattn.py`, `b200_sources/`): d_pair 128 / 4 heads fused for inference and
  training (L a multiple of 128), d_pair 64-512 inference. `module._b200_cuda = False` keeps the Triton path.
  Page: `docs/gpus/b200/triattn/triattn.md`.
- B200 (sm_100a) OuterProductMean / PWA training: `integrations/opm_train.py` and `pwa_train.py` accept compute
  capability (10, 0) with the H100 kernels' signatures on tcgen05 / TMEM (`integrations/csrc/sm100/`).
- B200 (sm_100a) token DiT, inference and training with hand CUDA + cuBLAS only (`integrations/token_dit.py`,
  new `integrations/token_dit_train.py`, `kernels/augmented_attention/cuda/sm100/`,
  `kernels/conditioned_transition/cuda/`). Page: `docs/gpus/b200/token_dit/token_dit.md`.
- A100 (sm_80) hand-CUDA TriMul, inference and training (`kernels/trimul_inproj/cuda/sm80.py`, wired through
  `integrations/trimul_sm80.py`): D128, one direction or bidirectional, any B (planes run one by one), L % 16 == 0. Module-level on an
  A100 80GB PCIe (CUDA graph, L384 / L768): inference 1.11–1.32× the best Anthropic A100 row (branch record, torch 2.10),
  training 1.59–1.73× cuEquivariance 0.12 (`docs/gpus/a100/trimul/trimul.md`).
  Ported from the `experiments/a100_trimul_fwd` research (tag `archive/a100-sm80-branch-20260928`) with a
  one-launch weight pack (no weight cache: captured graphs repack after an optimizer step) and deterministic
  LN_in dγ / dβ partials. `MINIWORLD_TRIMUL_SM80=0` keeps the Triton path. The pack reads the bidirectional
  module's `[in, out]`-stored front matrices in place (no per-call copy), and the backward ends in one `finalize`
  launch (B7's partial sums, the K1 row order undone, the casts) that writes every gradient in its parameter's
  strides and dtype, so autograd takes it over instead of copying it. Tests: the residual contract, the pack
  and finalize against their torch definitions, fp32 master parameters.
- A100 (sm_80) hand-CUDA TriangleAttention, inference and training (`kernels/triangle_attention/cuda/sm80.py` and `sm80/`, wired
  through `integrations/triattn_sm80.py`): d_pair 128, 4 heads x 32, bf16, L % 128 == 0, any B, starting and ending node (the
  kernels read and write the transposed positions), the module's broadcast dropout. Forward: LayerNorm + q|k|v|g + bias in one
  persistent kernel (the LayerNorm affine folded into the bf16 weights), the attention core (online softmax, the bias tile shared by
  two pair rows), gate + output projection + dropout + residual. Backward: gate / output-projection gradients (and the attention
  backward's row term), the key side (dK, dV) and the query side (dQ, the bias gradient as bf16 partials over groups of four rows plus a
  fixed-order reduction) of the attention, the projection input gradient + LayerNorm backward + residual in one pass, and the
  weight / LayerNorm-parameter gradients from one cuBLAS GEMM over the tokens (`G = Dᵀ[x̂ | 1]`, the rest algebraic). Official
  bench (`bench.py`, CUDA graph, A100 80GB PCIe, L384 / L768, 10 % masked keys, dropout 0.25): inference 0.635 / 3.678 ms =
  2.04x / 1.95x cuEquivariance (the Triton path 0.948 / 5.096), training step 2.572 / 14.915 ms = 2.17x / 2.21x cuEquivariance
  (Triton 3.836 / 22.853), the attention kernels making their cp.async index arithmetic once per thread (3-7 % per kernel);
  Anthropic's block with its Triton `k2b` core: 0.844 / 4.874 ms inference (its sm_80 CUDA member is not measured, see below; it ships no
  backward). `MINIWORLD_TRIATTN_SM80=0` or
  `module._sm80_cuda = False` keeps the Triton path; inference with dropout active and every shape outside the gate runs it
  unchanged. Page: `docs/gpus/a100/triattn/triattn.md`; tests: `tests/numerics/test_triattn_sm80_gpu.py` (every kernel and the
  module's output and nine gradients against the fp32 module, bit-identical replays, `torch.compile` in inference and training).
- A100 (sm_80) token DiT block, inference and training (`kernels/augmented_attention/cuda/sm80.py` and `sm80/`: a hand-CUDA attention core with a
  pair bias shared by the samples, head dim 48, forward with a gate epilogue or fp32 o + lse, backward dq + bias-gradient partials and dk / dv;
  `integrations/token_dit.py` and `integrations/token_dit_train.py` now serve sm_80 with cuBLAS GEMMs and the CUDA row kernels, which already
  build for sm_80). 16 heads x 48, d_single 768, bf16, L % 128 == 0; training needs an even A and no QK-norm. Official bench (`bench.py`, CUDA graph,
  A100 80GB PCIe, L384 / L768): inference (A = 5) 0.286 / 0.561 ms = 1.21x / 1.15x Anthropic (the release's DiT composition with its `apb_attn` core,
  0.345 / 0.644; its kits' recipe core `fpf_apb`: 0.465 / 1.004) and 1.84x / 2.55x PyTorch compiled (the module path it replaces: 0.464 / 0.994),
  training (A = 48) 7.338 / 16.463 ms = 1.47x / 1.84x PyTorch compiled (module path 8.960 / 22.768; Anthropic ships no backward); cuEquivariance has no DiT block.
  Page: `docs/gpus/a100/token_dit/token_dit.md`; tests: `tests/integrations/test_a100_token_dit_gpu.py`,
  `tests/integrations/test_a100_token_dit_train_gpu.py`. `MINIWORLD_AUGATTN_SM80=0` takes the hand-CUDA attention out.
- A100 (sm_80) atom DiT attention and pair bias, inference and training: `AugmentedAttentionPairBias.delta` asks `integrations/augattn_sm80.py` first and, on an A100 with bf16
  operands, B = 1, no QK-norm, head dim 32 or 48 and a key mask `[1, L]` or none (any L: padded to a multiple of 128 inside), runs the hand-CUDA attention core of
  `kernels/augmented_attention/cuda/sm80.py` (`plain_forward` / `plain_backward`: bf16 o and dq / dk / dv, the bias gradient summed from bf16 partials in a fixed order; head dim 32
  next to the token DiT's 48) and, at the atom width (16 pair channels, 4 heads), the fused pair bias (`pair_bias_fwd_kernel` / `pair_bias_bwd_kernel`: `LN(z) Wb^T` head-major with the mask
  and the padding folded in, and its backward) as autograd Functions over opaque ops (`torch.compile` keeps them). A `DiTBlock` at atom widths takes it through its attention part (the AdaLN,
  the projections and the transition stay the module's Triton / cuBLAS composition). The module-level core keeps the pair bias in raw units (`bias · sqrt(head_dim)`) and adds it to the
  score tile through the tensor core (an identity `mma`; a masked key is a very negative finite bias, never -inf), and the kernels make their cp.async index arithmetic once per thread.
  `AugmentedAttentionPairBias.cache_pair_bias(pair, mask)` / `clear_pair_bias()` is an opt-in, explicit cache of the pair bias for no-grad calls with unchanged tensors (an inference
  block 1.49x / 1.66x faster at N = 3072 / 6144, bit-identical output; a CUDA-graph replay reads the module's buffer, which every `cache_pair_bias` call refreshes in place).
  `bench.py` (CUDA graph, A100 80GB PCIe, N = 3072 / 6144 atoms): `dit_atom` inference (A = 5)
  0.785 / 2.626 ms = 4.44x / 5.13x PyTorch compiled (the module path it replaces: 2.328 / 8.975), training (A = 48) 16.784 / 59.402 ms = 3.89x / 4.27x (module path 30.542 / 115.842; PyTorch
  compiled runs out of memory from N = 7168); `augmented_attention_atom` inference 0.685 / 2.386 ms = 4.87x / 5.47x, training 13.701 / 49.132 ms = 4.48x / 4.97x; cuEquivariance has no DiT
  block, Anthropic is `unsupported` at this block (its DiT composition refuses d_pair 16, its atom attention is the windowed 32 x 128 op). `MINIWORLD_AUGATTN_SM80=0` takes it out.
  Page: `docs/gpus/a100/atom_dit/atom_dit.md`; tests: `tests/integrations/test_a100_augattn_gpu.py`.
- A100 (sm_80) SWA atom DiT block (ESMFold2 style: window 64, 4 heads x 32, hidden 256), inference and training, hand CUDA (`mma.sync` / `ldmatrix` / `cp.async`) in `kernels/swa_dit/cuda/sm80/`,
  taken first by `kernels/swa_dit/dispatch.py` on an A100 with bf16 operands: the adaLN modulation (`swa_dit_hoist_modulation`: `mod_fwd` and its backward `mod_bwd` with the fp32 modulation gradient
  fed to the tensor cores as hi + lo bf16 terms and fixed-order partial sums for dWmod), the three forward stages (RMS-adaLN + q | k | v | g + head norm + RoPE, the window attention, the gated
  out-projection + FFN), the window-attention backward (dQ; dK, dV) and the backward row stages (FFN gate / input side, out-projection, qkvg: every stage writes its columns of the modulation
  gradient once, no atomics; a conditioning per sample, or one shared by a multiple of 16 samples with in-warp reduction and partial buffers). Every stage reproduces the rounding points of its
  Triton twin; the five weight-gradient GEMMs stay cuBLAS. A = 1 or a multiple of 16 for the backward row stages (other A: those four stages run the Triton kernels), any A otherwise; fp32 and other
  widths keep the Triton path. The extension's build runs once at trace time under `torch.compile` (`assume_constant_result`), so `torch.compile(fullgraph=True)` of the module traces without a break
  and matches eager bit for bit. `bench.py target=swa_dit` (CUDA graph, A100 80GB PCIe, atoms N = 3072 / 6144), a conditioning per sample: inference (A = 5) 0.207 / 0.349 ms = 6.13x / 11.83x
  PyTorch compiled (the Triton path it replaces 0.237 / 0.399), training (A = 48) 4.777 / 9.334 ms = 6.48x / 12.52x (Triton path 6.219 / 12.356); one conditioning shared by the samples
  (`+shared_cond=true`, `forward_hoisted`): inference 0.156 / 0.252 ms, training 3.156 / 6.134 ms (Triton path 0.191 / 0.299 and 3.855 / 7.603). At N = 1024 the block is as fast as the Triton
  path (the forward kernels are latency-bound there). cuEquivariance has no DiT block; Anthropic's SWA atom block is not measured on A100. `bench.py`: `bench_module_swa_dit` takes
  `+shared_cond=true`. `MINIWORLD_SWA_DIT_SM80=0` keeps the Triton stages. Page: `docs/gpus/a100/swa_atom_dit/swa_atom_dit.md`; tests: `tests/integrations/test_a100_swa_dit_gpu.py`.
- A100 (sm_80) `modules.swa_atom_attention.SWA3DRoPEAttention`: the sliding-window attention runs on the hand-CUDA kernels of the SWA atom DiT block (`kernels/swa_dit/interface.py`:
  `swa_dit_window_attention`, an `autograd.Function` over the opaque ops `swa_dit_window_attn_fwd_sm80` / `_bwd_sm80`) for implementation TRITON / MINIWORLD, 4 heads x 32, half window 64, floating-point q / k / v
  (cast to bf16, as the flash path does), valid atoms front-packed (the FA4 path's precondition), `MINIWORLD_SWA_DIT_SM80=0` keeps flash. The module's `miniworld` arm did not run on this A100 before: its FA2
  extension (`flash_attn_2_cuda`) fails to import against torch 2.13 and FA4 is sm90+. The window-attention kernels take q / k / v / dq / dk / dv with any element strides (sample, row, head) instead of
  head-major planes only, so the module's tensors and the slice of its fused qkv projection go in without a copy (the fused block's head-major call keeps its code: a compile-time row stride), and a small
  `attn_delta_kernel` makes the backward's row term D = rowsum(dO o) where the fused block gets it from its out-projection backward. `bench.py target=swa_atom_attention` (CUDA graph, A100 80GB PCIe, N = 3072 / 6144):
  inference (A = 5) 0.093 / 0.168 ms, training (A = 48) 2.248 / 4.374 ms, 12.2x / 23.2x and 12.6x / 25.5x PyTorch compiled (dense band-masked SDPA). Tests: `tests/integrations/test_a100_swa_attention_gpu.py`
  (kernels against a dense fp32 band attention, strided / contiguous / head-major bit-identical, the module's output and gradients against its PyTorch path, CUDA graph, `torch.compile(fullgraph=True)`).
- A100 (sm_80) bias-only token DiT inference (`BiasOnlyDiTBlock`, `integrations/bias_only_dit.py`): the B200 runner (`kernels/bias_only_dit/cuda/runner.py`: cuBLAS GEMMs + the family's CUDA row kernels, which already built for
  sm_80) now serves A100 with a new attention core, `kernels/bias_only_dit/cuda/sm80` (`mma.sync` / `ldmatrix` / `cp.async`: sigmoid(g) * (P v) per head and sample, P shared by a group of SG samples per CTA, head widths 32 / 48 / 64,
  any L multiple of 128), `core_supported` takes an A100 (any such L) next to the B200 (L <= 768). bf16, B = 1, key mask [1, L] or none, a conditioning per sample or shared, 16 x 48 / 24 x 32 / 12 x 64 / 16 x 64 heads;
  `MINIWORLD_BIAS_ONLY_DIT=0` / `MINIWORLD_BIAS_ONLY_DIT_CORE=0` keep the PyTorch composition. `bench.py target=bias_only_dit` (CUDA graph, A100 80GB PCIe, A = 5, 16 x 48), L384 / L768: a conditioning per sample 0.245 / 0.460 ms
  = 1.50x / 1.83x PyTorch compiled (0.368 / 0.843), one conditioning shared by the samples 0.204 / 0.377 ms = 1.82x / 2.22x; the other layouts 1.7-2.3x; training on A100 is still the PyTorch composition (the B200 training path
  needs sm_100a kernels). Page: `docs/gpus/a100/bias_only_dit/bias_only_dit.md`; tests: `tests/integrations/test_a100_bias_only_dit_gpu.py`.
- A100 (sm_80) AttentionPairBias (the Pairformer single track), inference and training, hand CUDA + cuBLAS: `AttentionPairBias.forward` takes `integrations/attention_pair_bias_sm80.py` after the B200 hooks for
  implementation MINIWORLD, bf16, B = 1, no QK-norm, (heads, d_single) in 8 x 48 / 12 x 32 / 16 x 24 at 384 and 16 x 32 at 512, d_pair 128, key mask [1, L] or none, any L (padded to 128 inside:
  the single with zero rows, the pair read in place); `MINIWORLD_APB_SM80=0` keeps the module path, a failed extension build warns once and keeps it. The step is the B200 one on sm_80 pieces:
  `kernels/augmented_attention/cuda/apb/apb_rows.cu` is built a second time with `-DAPB_SM80` (its sm_90+ pair kernels are guarded out; the row passes are shared) and takes the pair -> bias pass and its backward
  from the new `apb_pair_bias_sm80.cuh` (`cp.async` tile rings, the same `mma.sync` + tensor-core row statistics; 89-97 % of the pair read's byte floor at L768), plus bf16-input variants of the gate / qkv rows; the
  attention is `kernels/augmented_attention/cuda/sm80` (the gated inference core now also at head dim 32; `plain_forward` / `plain_backward` take the softmax scale, so the padded 16 x 24 heads keep 1 / sqrt(24)).
  `apb.ext(arch)` picks the build. `bench.py target=attention_pair_bias` (CUDA graph, A100 80GB PCIe, L384 / L768): inference 0.078 / 0.170 ms (8 x 48) = 2.19x / 2.54x cuEquivariance (0.170 / 0.431),
  training 0.249 / 0.570 ms = 1.84x / 2.07x (0.459 / 1.180); 12 x 32 and 16 x 24 2.15-2.70x and 1.92-2.16x; outputs and every gradient as accurate as the bf16 PyTorch module. Page:
  `docs/gpus/a100/attention_pair_bias/attention_pair_bias.md`; tests: `tests/integrations/test_a100_apb_gpu.py`.
- A100 (sm_80) `AugmentedAttentionPairBias` runs hand CUDA + cuBLAS from the AdaLN output to the residual (`integrations/augattn_sm80.py`, `kernels/augmented_attention/cuda/sm80/`: the core, a TF32 core, glue kernels, a streaming pair-bias kernel
  for other pair widths) in bf16 and fp32 (TF32), any A / B / L, shared or per-sample key masks, inference and training; with the AdaLN / ConditionedTransition paths below the atom and token DiT blocks launch no Triton kernel. Against PyTorch
  compiled: atom module 3.4-5.6x inference / 3.3-5.1x training, token module 1.2-2.6x from L = 256, `dit_atom` block 4.6-5.3x inference / 4.2-4.6x training (N = 3072-8192, measured on the integrated tree: the atom DiT block launches no Triton kernel). Switches `MINIWORLD_AUGATTN_SM80=0`
  (old path), `_FUSED`, `_PBGEN`, `_OPS`. The token DiT block's fp32 inference runs the TF32 hand-CUDA core. Pages `docs/gpus/a100/{atom_dit,token_dit}/`; tests `tests/integrations/test_a100_augattn_gpu.py`, `test_a100_augattn_cpu.py`.
- A100 (sm_80) SWA atom attention output gate (`integrations/sigmoid_gate_sm80.py`, `MINIWORLD_SIGMOID_GATE_SM80=0` keeps Triton) and the SWA DiT backward for any sample count; bias-only token DiT training (`integrations/bias_only_dit_train.py`, the `dpb`
  bias-gradient core; A = 48, L <= 768, B = 1) and the `kernels.bias_only_attention` CUDA door (`bias_only_attention(v, bias)`, `MINIWORLD_BIAS_ONLY_ATTN_SM80`; a training call at L = 128 keeps Triton); the `projected_attention` token_single rows have a CUDA
  path that the default dispatch (`MINIWORLD_AUGATTN_SM80_OPS=auto`) uses for training calls with A·B·H·L² >= 2e6 or head dim 24. `bench.py` gains kernel rows `projected_attention` and `bias_only_attention` (training rows need `compile=false`).
  Pages `docs/gpus/a100/{swa_atom_dit,bias_only_dit}/`; tests `tests/integrations/test_a100_{swa_attention,swa_dit,bias_only_dit_train,bias_only_attention,projected_attention}_gpu.py`.
- A100 (sm_80) TriangleMultiplication at every registry width: `integrations/trimul_sm80.py` also routes D64 / D256 / D384 (bf16) and D64-D384 in fp32 (TF32 tensor cores), one direction and bidirectional, inference and training, to LayerNorm-folded wide
  kernels (`kernels/trimul_inproj/cuda/sm80_wide.py`, extension `trimul_sm80_wide`; `MINIWORLD_TRIMUL_SM80_WIDE=0` turns them off). Against cuEquivariance (CUDA graph, L128-768): bf16 inference 1.31-2.89x, training 1.36-2.12x; fp32 against PyTorch compiled
  1.4-3.9x. Page `docs/gpus/a100/trimul/trimul.md`; tests `tests/integrations/test_a100_trimul_gpu.py`.
- A100 (sm_80) gated projections: `kernels/gated_projection/cuda/sm80.py` (extension `gated_sm80`) serves the registry's `gated_linear` rows (fused gate + GEMM below 384 output columns, one-pass gate + cuBLAS above), `sigmoid_gate_fused`, `gated_residual`,
  tm1 / tm2 (`kernels.tm1.interface.cuda_tm1`, `kernels.tm2.interface.cuda_tm2`) and the TriMul output gate; `MINIWORLD_GATED_SM80=0` keeps Triton; 1.0-2.0x PyTorch compiled. Kernel benches gain the arms `cuda_tm1` / `cuda_tm2` / `cuda_gate_elem_bwd`.
  Page `docs/gpus/a100/gated_projection/gated_projection.md`; tests `tests/integrations/test_a100_gated_gpu.py`.
- A100 (sm_80) TriangleAttention at every registered width (d_pair 64 hidden 64 / 128 with 4 x 16, 2 x 32, 4 x 32 heads, 128, 256 8 x 32, 384 12 x 32; L a multiple of 128; starting / ending node; inference and training) in hand CUDA
  (`kernels/triangle_attention/cuda/sm80_wide.py`, `sm80_core2.py`: a fused narrow engine, native 16-channel heads, a row-kernel + cuBLAS engine), the `projected_attention` token_pair leaf (`ops.augmented_attention_pair_bias` with A = L pair rows,
  `sm80_projected.py`) and `ops.layer_norm_linear` (`kernels/layernorm_linear/cuda/sm80.py`, forward + backward). Module level against cuEquivariance 1.3-3.2x inference, 1.4-2.6x training; the leaf 1.5-3.0x / 1.9-2.6x the Triton path,
  `layer_norm_linear` 1.5-4.0x / 1.8-3.3x PyTorch compiled. Pages `docs/gpus/a100/triattn/triattn.md`, `docs/gpus/a100/layernorm_linear/layernorm_linear.md`; tests `tests/integrations/test_a100_{triattn_widths,projected,lnlinear}_gpu.py`.
  `MINIWORLD_TRIATTN_SM80=0` / `MINIWORLD_LNLINEAR_SM80=0` keep the Triton paths.
- A100 (sm_80) OuterProductMean and MSAPairWeightedAveraging: hand-written CUDA around cuBLAS via `integrations/opm_sm80.py` / `pwa_sm80.py` (`kernels/outer_product_mean/cuda/sm80/`, `kernels/pair_weighted_averaging/cuda/sm80/`) for implementation MINIWORLD,
  bf16, B = 1, every registry row (OPM d_msa 64 / 128, d_hidden 32, d_pair 128 / 256 / 384, any L and MSA depth; PWA 8 heads of 8 / 16 / 32, d_msa 64 / 128, d_pair 128 / 256 / 384, L a multiple of 16 up to 1024), inference and training (PWA with fused row
  dropout); `MINIWORLD_OPM_SM80=0` / `MINIWORLD_PWA_SM80=0` keep the module path. `bench.py` (S = 1024, CUDA graph, L384 / L768) against PyTorch compiled: OPM 1.25-1.29x (inference) and 1.17-1.19x (training) at d_pair 128, PWA 1.96-2.07x and 1.46-1.51x;
  `bench.py` gains a `triton` arm for the two targets and `+msa_d_msa` / `+msa_d_hidden`. Pages `docs/gpus/a100/outer_product/outer_product.md`, `docs/gpus/a100/msa_pair_weighted_averaging/msa_pair_weighted_averaging.md`;
  tests `tests/integrations/test_a100_{opm,pwa}_gpu.py`, `test_a100_msa_gate_cpu.py`.
- A100 (sm_80) AdaptiveLayerNorm and ConditionedTransition (atom 128 / 128, token 768 / 384 and 768 / 768; bf16 and fp32 on TF32 tensor cores; inference and training, per-sample and shared conditioning): the modules call hand-CUDA paths
  (`kernels/adaln/cuda/sm80*`, `kernels/conditioned_transition/cuda/sm80*`; `integrations/adaln_sm80.py`, `conditioned_transition_sm80.py`): cuBLAS GEMMs + warp-per-row CUDA row kernels + a second-stream GEMM branch at every width, fused tensor-core
  kernels at the atom width. `MINIWORLD_ADALN_SM80=0` / `MINIWORLD_CONDTRANS_SM80=0` keep the Triton path. Against PyTorch compiled: 1.3-2.2x at the atom width in bf16 (fp32 inference 1.0-1.9x), parity to 1.17x at the token widths (GEMM-bound); fp32 AdaLN
  inference at N = 1024 stays on Triton (5 % rule). Pages `docs/gpus/a100/adaptive_layernorm/adaptive_layernorm.md`, `docs/gpus/a100/conditioned_transition/conditioned_transition.md`; tests `tests/integrations/test_a100_adaln_gpu.py`,
  `test_a100_conditioned_transition_gpu.py`.
- A100 (sm_80) fp32 master parameters (AMP `bf16-mixed`: fp32 parameters over bf16 activations) on the hand-CUDA training paths of TriMul, Transition, OuterProductMean, PWA, AdaLN, ConditionedTransition,
  AttentionPairBias and AugmentedAttentionPairBias, as on the B200: the integrations pass the parameters themselves into their autograd functions, cast them for the bf16 kernels outside autograd (no copy in
  the graph) and return the kernels' fp32 weight-gradient accumulators unrounded in each parameter's dtype; the same kernels as for bf16 parameters (the fp32-parameter AugmentedAttention is served instead of
  falling back). The A100 weight-pack caches are scoped to CUDA-graph captures through `kernels/_capture.py`. Tests: `tests/integrations/test_a100_fp32_master_gpu.py`. The `DiTBlock` asks the dedicated
  attention / transition paths before the general CUDA composition (`integrations/a100_families.py`, `kernels/cuda_native.py`), which now only takes what they decline (QK-norm, other head widths).
- A100 (sm_80) Transition at every registry width (D64 / 128 / 256 / 384 / 768, n = 2 and 4, pair / single / MSA streams; bf16, inference and training) and the SwiGLU FFN (`ops.swiglu_ffn`, atom / pair / single rows) in hand CUDA:
  a lean pipelined `mma.sync` dual-GEMM + SwiGLU, a residual-folded squeeze, gate and LayerNorm-backward kernels around cuBLAS GEMMs (`kernels/transition/cuda/fused_wide_sm80.py`), a one-kernel D64 / D128 forward and fused backward
  (`fused_fwd_sm80.py`, `fused_bwd_sm80.py`) above row thresholds, the existing `fused_sm80` at D128 / n = 4 from 8192 rows. `MINIWORLD_TRANSITION_FUSED_SM80=0` keeps the Triton path. Against PyTorch compiled (CUDA graph, L384): inference
  1.3-4.8x (D128 n4 3.54x), training 0.99-2.55x (D384 n4 at parity, GEMM-bound). `bench.py` gains `+transition_n`, `+transition_stream`, `+transition_ffn` and a `triton` row that pins the A100 switch off; kernel-level CUDA arms
  `transition_cuda`, `layernorm_linear_cuda`, `dual_gemm_gate_cuda` (the `gemm_epilogue` / `dual_gemm_epilogue` targets keep their Triton default: the CUDA arms are slower at d_pair 128). Page: `docs/gpus/a100/transition/transition.md`;
  tests: `tests/integrations/test_a100_transition_gpu.py`, `test_a100_gemm_epilogue_gpu.py`.
- A100 (sm_80) row kernels: LayerNorm (every registry width incl. the odd 267 / 451 / 831 / 833 and 2560; vector, scalar, staged-`cp.async` and wide-row kernels, forward and backward with a fixed-order partial reduction), RMSNorm, the fused Q/K
  RMSNorm + 3D RoPE and `rms_norm_modulation` run hand-written CUDA (`kernels/{layernorm,rmsnorm,rope}/cuda/sm80.py`; `layernorm_kernel`, `RMSNorm.forward`, the SWA atom attention and `ops.rms_norm_modulation` route to it on an A100);
  `MINIWORLD_NORMS_SM80=0` keeps the Triton kernels, `=force` also serves the tiny training steps (20K-100K elements) that are routed to Triton by measurement. Against PyTorch compiled: LayerNorm backward 1.26-2.22x, forward 0.97-1.16x
  (bandwidth-bound), training step 1.02-1.74x, `rms_norm_modulation` 1.8-2.2x / 1.5-1.8x. Page: `docs/gpus/a100/layernorm/layernorm.md`; tests: `tests/integrations/test_a100_norms_gpu.py`.
- A100 (sm_80) ProteinMPNN edge side: the edge tail (the fused projection -> GELU -> projection -> GELU -> projection -> dropout -> residual -> LayerNorm chain, forward and backward), the edge MLP, the edge LayerNorm's
  compressed-save backward and the edge dropout's bit-packed mask run hand-written CUDA (`kernels/mpnn_edge_tail/cuda/sm80/`, `integrations/mpnn_edge_sm80.py`) wherever the Triton policies `triton` / `triton_compute` /
  `triton_memory` / `memory` / `bitpack` (and the MLP's `auto` from 98,304 rows) were used; new backend literal `cuda` on the four families; `MINIWORLD_MPNN_EDGE_SM80=0` keeps the Triton kernels. Against PyTorch compiled
  at N = 2048 / 65,536 nodes: edge tail 2.7x / 2.8x in inference and 2.4x / 2.25x in training (CUDA graph; the Triton path 1.3x), the recompute policy 1.9x / 1.6x (Triton 0.5x), edge MLP 2.1-2.4x / 1.6-1.7x;
  full ProteinMPNN step -11 % (N = 2048) and -9 % (N = 8192). The dropout draw is a counter hash (not Philox): same kept share and independence, a different stream. `benchmarks/runners/mpnn_compare.py`: `cuda` columns and
  `--graph-training`. Page: `docs/gpus/a100/mpnn/mpnn_edge.md`; tests: `tests/integrations/test_a100_mpnn_edge_gpu.py`.
- A100 (sm_80) ProteinMPNN message side: the hidden-message reduction (`message_hidden_reduce`, forward and backward, no edge-sized tensor written), the encoder node message (`node_message_reduce`) and the relative-position
  embedding's backward run hand-written CUDA (`kernels/mpnn_{message,node_message,relative_position}/cuda/sm80/`, `integrations/mpnn_msg_sm80.py`, three extensions) for the policies `auto` / `triton` / `triton_compute` /
  `triton_memory`; `MINIWORLD_MPNN_MSG_SM80=0` keeps the Triton kernels (the registry driver / checks of `mpnn_message` stay pinned to Triton). Against PyTorch compiled at N = 16,384 nodes: hidden message 2.93x inference /
  2.01x training, encoder node message 2.52x / 1.66x, relative-position backward 24x (3.0-7.1x faster than the Triton reduction); every reduction deterministic (no atomics). Page: `docs/gpus/a100/mpnn/mpnn_msg.md`;
  tests: `tests/integrations/test_a100_mpnn_msg_gpu.py`.
- Anthropic on A100 measured through `bench.py`'s `anthropic` arms (`MINIWORLD_ANTHROPIC_ROOT` = a copy of the pinned release's `common/opt_core`, which is checked out on cssb at
  `/home/psk6950/practice/refs/uplifting-biomolecular-modeling`: the A100 pages had said it was not on the cluster); two new bench options, `+triattn_anthropic_row=block:k2b | block:flash |
  block:triattn_native` (TriangleAttention) and `+dit_anthropic_core=fpf_apb | apb_attn` (the DiT composition's attention core; the kits' recipe `fpf_apb` stays the default, `apb_attn` is the core the
  release's own A100 cells name and is 1.35-1.56x faster on this card). The sm_80 member of `triattn_native` (the release's A100 CUDA TriangleAttention core) is NOT measured: its prebuilt
  binaries match no torch +cu129 stack, a rebuild from the unmodified sources with the release's own builder is refused by the release's `SHA256SUMS` seal, and recording the hash was not done.
- `miniworld_engine.viz.kernel_flow`: kernel-flow SVG figures (one box per kernel, HBM reads and
  writes) from a JSON spec.
- H100 single-direction TriMul training in CUDA at D64 (`h100_uni_d64_training`, the
  bidirectional D64 kernels at hidden 64) and D256/384 (`h100_uni_wide_training`, the wide
  sources compiled for hidden D), L384/768. Measured on an H100 80GB HBM3 against the Triton path
  (CUDA graph, fwd+bwd, 2026-09-29):
  about 2x at D64, 1.27-1.33x at D256, 1.31-1.36x at D384; accuracy against an FP32 reference
  matches the Triton path. Tests: `test_trimul_uni_d64_training_gpu.py`,
  `test_trimul_uni_wide_training_gpu.py`.

## [2.2.0] - 2026-09-28

Two tiers per op: hand-written CUDA where it exists, a Triton fallback everywhere else, plus the
PyTorch reference and the comparison baselines (cuequivariance, Anthropic payload, dtv1).
GPU qualification of this release is pending.

### Runtime

- torch 2.13.0+cu129 (triton 3.7.1, CUDA toolkit 12.9 — the newest pair the cluster's 575
  driver runs), cuequivariance / cuequivariance-ops-torch-cu12 0.12.0, Transformer Engine 2.19.
  Tuned autotune caches are keyed on the torch version and need rebuilding.

### Removed

- `experiments/`: every capsule is either ported to `src/` (fastest variant only) or history.
  The tree is at tag `archive/experiments-20260928` (and tag `archive/wip-main-20260928` for the
  Sept 27–28 local-H100 research); see README "Research history". The Anthropic
  `NOTICE` and v5 source hashes moved to `licenses/`.
- Every CuTe DSL / nvidia-cutlass-dsl / quack kernel and the `cute` extra: `kernels/*/cute`
  (tm1, tm2, transition, layernorm, layernorm_linear, trimul_inproj), the `fused_ln_mask`
  family, `_quack_compat`, the autotune CuTe candidate spaces and compile paths, and the
  unused CUTLASS C++ trees `kernels/adaln/cutlass`, `kernels/conditioned_transition/cutlass`
  and `transition_b2b_sm100_kernel.cu`. Hand-CUDA kernels that include CUTLASS C++ headers
  (triangle-attention CUDA, transition b2b/expand-gate/gate-bwd) are kept.
- `ImplementationType.CUTE` / `KernelBackend.CUTE`, `modules.dispatch.trimul_out_layout`.
- Public kernels `cuda_transition` (never implemented), `cuda_transition_b2b`,
  `cute_transition_fused`; `layernorm_linear_fn` / `LayerNormLinearFn`; `tm2_cute`;
  `trimul_inproj_cute`; `adaln_inference_lnfold`.
- Intermediate module paths: TriMul CuTe training/inference and the `trimul_sm90_kernels`
  parity hooks; Transition's split / b2b-inference / CuTe routes (`_old_triton_forward`,
  `_inference_forward`, `_training_forward`).
- The previous wide/D64 H100 TriMul training port: `h100_width`, `h100_width_base`, `h100_gp`,
  the unused `h100_wide_forward` plan and their `h100_sources/wide`, `wide_base`, `wide_forward`
  sources.
- Settings: `trimul_out_layout`, `trimul_cute_dispatch`, `trimul_sm90_kernels`,
  `trimul_train_front_fused`, `lnl_ws`, `transition_force_split`,
  `transition_residual_fusion`, `transition_large_d_training`, `transition_cute_backward`,
  `transition_dab_lnbwd`.

### Changed

- Add portable Triton OuterProductMean and MSAPairWeightedAveraging (forward and
  backward; kernel families `outer_product_mean`, `pair_weighted_averaging`, 12
  autotuned kernels). `miniworld` now takes them on every GPU and width the
  native H100 paths do not serve, and under `engine_backend="triton"`; before,
  those calls ran the module's PyTorch statements. `implementation="triton"` on
  either module used to run the same statements silently and now runs the
  kernels or raises `NotImplementedError` naming the reason (CPU input, non-bf16,
  interchain masking, d_msa / d_hidden not a power of two, OPM d_pair not a
  multiple of 32). Their autotune grids ship at full ladder width; no cache is
  built yet, so a first call uses the bounded miss fallback.
- `build.launch_bind` reads a kernel's tuned axes from the shipped grid spec of
  its `configs_for("<op>")`, so axes not spelled `BLOCK_*` are recognised.
- A100 (sm_80): hand-CUDA fused Transition for D128/n=4 bf16 (`kernels/transition/cuda/fused_sm80.py`,
  forward + two-kernel backward, training step about 1.44x the Triton residual path on A100).
  The sm_80 CUDA TriMul and the other A100 research runners are not ported to `src/`; they are
  at tag `archive/a100-sm80-branch-20260928` (`experiments/a100_*`).
- Repository layout (no behaviour change): registry CSVs and evidence files in `kernels/registry/`;
  A/B config sets in `autotune/configs/ab/` (short names such as `blk16` still resolve); docs are
  `docs/gpus/` (per-GPU how-to and completion tables, judged by the maintainer) and `docs/kernels/`,
  plus `docs/CHANGELOG.md` and `docs/standards.md`; benchmark docs beside `benchmarks/`; agent rules in
  `.claude/CLAUDE.md`; `scripts/` folded into `miniworld-engine dev` commands (`tools/`) and
  `benchmarks/runners/`. Removed records, experiments and scripts are listed in README "Research history".
- Bidirectional TriMul H100 training at D256/384/512 (L384/768) is a new flattened hand-CUDA +
  cuBLASLt port of the qualified large-width research plans (`h100_wide_training`, 29 frozen
  kernels in `h100_sources/wide_train`): 1.44-1.62x the Triton path in paired CUDA-graph fwd+bwd
  replay on H100 (was 0.69-0.75x). Its retained activations are larger than the old port's
  (D512/L768 11.3 GiB) and equal to or below the Triton path's. cuBLASLt algorithms frozen under 12.8.4 are matched by configuration against the
  running cuBLASLt (`lt_selection.json`), falling back to the first heuristic with a warning.
  See docs/gpus/h100/dispatch.md.
- `trimul_h100_training_widths` defaults to `(128, 256, 384, 512)`; D64 bidirectional training
  runs on Triton (its CUDA port measured 0.61x of Triton in graph replay).
- The H100 training opaque ops are renamed (`trimul_h100_train_{fwd,bwd}_wide_port`,
  `trimul_h100_dropout_nograd_wide_port`) because the wide saved-tensor contract changed.
- `miniworld` TriMul resolves to the Triton family on every arch; the hand-CUDA H100 kernels
  are still tried first inside the modules. Both TriMul modules now run the Triton path plane
  by plane for B > 1 (the CuTe path used to be the only one that looped).
- `Transition` and `ops.transition` always take the residual path: hand-CUDA `fused_sm90a` /
  `fused_wide_sm90a` on sm_90, else Triton `transition_residual`.
- `layernorm_linear` is Triton on every arch; adaLN wide-d inference uses the fused
  GEMM+gate kernel on every arch; token DiT uses cuBLAS for all GEMMs.
- Triton TriMul front backward computes its weight gradient with cuBLAS directly.

### Fixed

- JIT build lock guard (`kernels._nvcc`): honour `load(build_directory=...)` when locating the
  lock, and do not trust an NFSv3 exclusive-create mtime (read back as 1981) when deciding a
  lock is stale -- use max(mtime, ctime) and never reclaim a pre-2000 stamp. Both hung
  multi-rank training jobs on the cluster's NFS home.

## [2.1.0] - 2026-09-27

- Follow production backend dispatch when building caches; force alternative
  implementations only with an explicit request.
- Ship compact per-kernel Triton defaults while preserving global config spaces,
  shared-memory prediction and opt-in multi-GPU global searches.
- Keep compatible measured global winners usable without global runtime searches.
- Restrict token training build lengths to 384/768 and preserve inference ladders.
- Expose native-only inference builds and register exact-shape tuning for fused
  Transition and packaged TriMul inference, including candidate output validation.
- Document fixed/manual CUDA schedules separately from integrated autotuners.
- GPU qualification of the new tuning integrations remains pending.

### Consolidated work since 2.0.0


- Fuse OPM/PWA training dropout and residual epilogues, retain live-weight graph
  replay, and record MSA comparisons with the standard depth of 1024.
- Reuse D128 TriMul preparation and expose optimizer-state layout migration for
  existing checkpoints; preserve source/measurement provenance and limitations.

- Reuse forward weight packing and FP32 masks in bidirectional TriMul backward
  at D64/256/384/512; remove duplicate D512 input normalization and fix wide
  training `x_n` metadata for compiled execution.

- Add native single-direction H100 TriMul training for D128/L384/768, both
  outgoing and incoming: Anthropic-derived K1/K3, fused output backward and
  producer-consumer input backward, including dropout/residual and graph replay.

- Connect packaged H100 TriMul inference and latest bidirectional CUDA training
  to default `auto` module dispatch. Training widths: D64/128/256/384/512,
  L384/768; width-specific performance tuning remains separate.
- Enable supported packaged OPM/PWA paths automatically and connect fused
  token DiT inference. Preserve live weights, saved-tensor ownership, dropout,
  residuals and `torch.compile` boundaries.
- Make explicit backend comparison policy consistent for Transition.
- Ship selected CUDA sources/includes and write compilation products only to
  the user cache. [Dispatch contracts](gpus/h100/dispatch.md).

## [2.0.0] - 2026-09-23

### Breaking changes and migration

- Release the accumulated public API removals listed below, including
  `kernels.triton_adaptive_layer_norm` and obsolete autotune budget settings.
- `SWADiTBlock` uses the ESMFold2 RMSNorm/adaLN-Zero contract; old AF3-style
  block checkpoints are incompatible. Recreate those blocks and retune caches.
- New H100 Transition dispatch is enabled for supported BF16 widths. Users
  requiring the Triton route can set `MINIWORLD_TRANSITION_FUSED_SM90A=0`.
  The new TriMul training implementation remains an explicit research entry.

### Anthropic-derived development

- Preserve upstream credit, licenses and provenance. MiniWorld inherits the
  stronger published inference implementation and extends it for training;
  see `licenses/THIRD_PARTY_NOTICES.md`.
- Integrate K1/K3 single/bidirectional TriMul inference, fused Transition
  forward/backward and its wide-width ports, MSA OPM/PWA training and
  inference integrations, and token DiT research histories.
- Preserve latest CUDA TriMul B1–B4 and single-launch B7–B12, saved input `x_n`,
  output-LN recomputation, dropout/residual/mask semantics, all eleven gradients,
  strict D128 LN-gradient fixes, and D64/128/256/384/512 validation records.
- Package a relocatable research capsule and its pinned benchmark engine so the
  latest work no longer depends on a private MiniWorld checkout. D128 is the
  performance winner; other widths remain slower than Triton and their further
  optimization is deferred. This is not a claim of SoL90 or complete cache coverage.
- Fix large packed TriMul gradient offset arithmetic using 64-bit K indices.
- Archive 27 stale runtime cache files unchanged with SHA-256 provenance; they
  require rebuilding and are not represented as valid 2.0.0 tuning records.
- Transition adds the parallel LN-gradient reduction and transposed dWs stores.

Release map, migration limits and evidence: 2.0.0 (`archive/records-20260928:docs/project/release-2.0.0.md`).


### Added

- `ops.gated_residual(x, gate, branch)`: fused linear residual gate with backward.
- Fused sm_90a Transition, forward and backward, for the AF3 pair width (`d_hidden`
  128, `n` 4, bf16, whole 128-row tiles). Two launches replace five: LayerNorm,
  expand-SwiGLU and squeeze-with-residual on the way forward, and the squeeze,
  SwiGLU and LayerNorm backward chain on the way back. Measured through the wired
  dispatch on an H100 SXM, `modules.Transition` forward plus backward goes from
  1074 to 559 us at L=384 (1.92x) and from 4025 to 2073 us at L=768 (1.94x).
  On by default where it applies, off with `transition_fused_sm90a=False` or
  `MINIWORLD_TRANSITION_FUSED_SM90A=0`; every other shape, dtype and architecture
  keeps the existing path. Development record in
  `experiments/transition_fused/`.

### Changed

- Merge the 24 active MiniWorld consumer patches: packed TriMul buffers, configured
  F567 forward and dual-dgrad/LayerNorm-residual backward fusions, strict Triton
  backend selection, and expanded resumable CuTe/CUDA tuning.
  See integration and validation (`archive/records-20260928:docs/records/local-patches-20260917.md`).
- Retire 37 incompatible H100/A5000/A6000 cache files from runtime selection;
  original measurements and checksums remain in the integration archive.

- FA2 sliding-window attention uses static-capacity packing in backward as well as
  forward, removing dynamic `nonzero`/host length reads from the AOT backward graph.
  SWA DiT compiled benchmarks now require `fullgraph=True`.

- `SWADiTBlock` now matches MiniWorld ESMFold2 adaLN-Zero. Engine backends use
  fused RMSNorm modulation, SwiGLU, and residual gates; the previous AF3-style
  block checkpoints and SWA DiT benchmark results are incompatible with this version.

### Fixed

- AdaLN GEMM alignment, large compute-efficient attention backward scratch usage,
  mixed-affine LayerNorm and compiled FA4 backward.
- Wide TriMul packed reduction keys (`KP=4096`), compact CuTe cache selection,
  and shape-only derivation of another architecture's external FlashAttention path.

### Removed
- **`kernels.triton_adaptive_layer_norm`, and twelve kernels no production path reached.** Two
  audits of the adaln and conditioned_transition families found that half their registry surface
  was unreachable from any module, and the build was tuning all of it.

  adaln dispatches to `adaln_train` / `adaln_inference` and to nothing else. `main.py`'s
  `TritonAdaptiveLayerNormFunction` and `fused3.py`'s `adaln_fused3` / `adaln_fused3_train` were
  reached only by the bench, the drivers and the checkers — six kernels between them. `main.py` is
  gone; `fused3.py` kept the one kernel `inference.py` imports and is now `ln_strided.py`, named
  for it, at 110 lines instead of 430.

  conditioned_transition had three training files and one live one. `train_12_345.py` could not run
  at all — neither of its Functions takes a `length`, so every inner launch hit the `shape_key=None`
  branch, which raises. `train_fused.py`'s fused backward was never selected, and its own H100
  measurement says why: it lost to cuBLAS+elementwise by 1.6-7.1x at every stage. Its forward pair
  survives as `fwd_saveact.py` because `training.py` calls it. Six more kernels gone, plus the
  duplicate `cond_transition_fwd_12_345`.

  `triton_adaptive_layer_norm` was the only one of these on the public kernel surface, so this is
  the semver-relevant part. The registry, `axes.csv`, the device manifests, the config sets, the
  shipped tuned caches and the drivers/checkers were pruned with them: 91 triton kernels -> 79.

### Removed
- **`--bench-budget` and `settings.bench_budget_*`.** The feature abandoned an autotune config
  once one timed launch exceeded `factor x` the best so far. Its safety argument — "a config
  that could still win runs FASTER than the current best and is therefore always inside the
  budget" — is true only if both numbers are the same quantity, and they were not.

  `best` comes from triton's `do_bench`, whose timed loop never synchronises, so the queue
  stays full and the event window measures device time alone. The probe drained the stream
  (warm launch + `synchronize`) and timed ONE launch, so its number carried the host launch
  latency as well. Measured here on an A6000, one kernel, BLOCK swept 256→16384 and warps
  1→16:

      probe (1 launch)   0.0256 – 0.0502 ms   (spread 0.025 — flat: it is not the kernel)
      do_bench median    0.0051 – 0.0645 ms   (spread 0.059 — it tracks the config)
      budget = best x 3  0.0154 ms            → 12 of 12 configs abandoned

  So the first config in grid order sets `best` under the 300 ms first-round cap and every
  other config is dropped. In the shipped cache this leaves a fingerprint: **349 of 1244
  entries across 51 ops have the first config in grid order as their winner**, 346 of them
  with a single ranked config, from grids of 750 to 15552. `adaln_gemm_gate_triton` records
  `{BLOCK_K:16, BLOCK_M1:32, BLOCK_N:32, GROUP_M:1} warps=1 stages=1` — grid position 1 of
  15552 — for three different shape keys.

  It breaks whenever `kernel time < launch floor / 3`, which is nearly every kernel here
  (0.005–0.03 ms). Removed rather than fixed: a corrected probe would have to bench a queued
  batch, which is a different implementation, and this one silently corrupts caches when
  enabled. The A5000/A6000 caches need rebuilding.

### Changed
- **Per-kernel numerical tolerance, derived from measurement.** `registry.csv` had an `rtol`
  column and 0 of 103 rows filled it, so every kernel was held to one global `5e-2` — 18x the
  median kernel's actual error, and 88 of 97 sat more than 10x inside it. Each row now declares
  a band computed as `4x` its worst measured relative error, from the two device manifests
  `run_all` writes. The margin is from the data, not chosen: the same kernel's error across the
  two cards varies by median 1.07x, p90 1.47x, **max 1.95x**, so 4x leaves two more doublings
  than anything observed. Calibration only tightens — the two kernels already measuring close to
  the global band (`adaln_fwd_saveact_triton` 2.77e-2, `triangle_attention_bwd_atomic_triton`
  1.44e-2) keep `5e-2` rather than being handed something looser. Median band is now `1.1e-2`,
  a 5x tightening. `tests/registry/test_declared_tolerance.py` fails if a band drops below what
  the kernel measured, or drifts far above it.
- **`settings.AutotuneKernel` is three names, and an unknown one now raises.** It declared
  seven; four of them (`tri_multi`, `layernorm`, `layer_norm_linear`, `augmented_attention`)
  had no call site, so naming one unlocked nothing and reported nothing. `autotune_kernels`
  is typed against the vocabulary and `configure` rejects a name outside it — previously
  `autotune_kernels={"triangle_attention"}`, the family's *current* name rather than this
  vocabulary's older `tri_attention`, was accepted and did nothing.

## [1.0.0] - 2026-08-25

### Breaking
- **The distribution and import name changed: `miniworld-kernels` →
  `miniworld-engine`, `import miniworld_kernels` → `import miniworld_engine`.**
  There is no compatibility alias: the old import raises `ModuleNotFoundError`.
  This is why the version is 1.0.0 rather than 0.2.0 — the previous release and
  this one are different packages, and `0.1.0` was published under both names,
  so no consumer could tell them apart by any declared field. Pin
  `miniworld-engine>=1.0.0` and rewrite the import; a submodule consumer must
  also update the path and URL.

### Added
- **`miniworld-engine dev audit`** — verifies the build system and, new, that every
  DECLARED `(op, dtype, shape-bucket)` is present in the shipped autotune cache.
  Declared means `registry.csv` crossed with each kernel's `level` and `dtypes`,
  so a hole is reported against the contract rather than against whatever the
  last build happened to measure. `build` already told users to run this; the
  subcommand did not exist and `build/audit.py` crashed on import.
- **`settings.autotune_miss_cap`** (default 24) — how many configs an autotune
  cache MISS may search before falling back to a heuristic subset.
- **`miniworld_engine.ops` — the whole-op consumer contract.** Complete,
  autograd-transparent model-layer ops (weights as arguments, backend dispatch +
  fwd/bwd inside), consumed as a single call: `triangle_multiplicative_update`,
  `triangle_attention`, `transition`, `conditioned_transition`,
  `augmented_attention_pair_bias`, `layer_norm_linear`, `layer_norm`. Lazy,
  side-effect-free import; pinned by `_OPS_CONTRACT` in `tests/compile/test_public_api.py`.
  Verified fwd+bwd vs the pytorch/cuequiv reference on B200 (≥0.9998).
- Public API contract test (`tests/compile/test_public_api.py`): freezes the
  `kernels` surface and asserts the package import is side-effect-free.
- Numerical correctness suite (`tests/numerics/test_numerical.py`): each op's fused
  MINIWORLD backend vs the PyTorch reference (forward + input gradient),
  GPU-gated, asserting the fused path is actually engaged (no silent
  dtype-degrade). Promotes the benchmark cosine checks into an enforced
  correctness gate.
- CI (`.github/workflows/ci.yml`): ruff + ty + the whole CPU suite on every
  push/PR (nested cutlass submodule skipped; CPU torch wheel installed so the
  type gate can see the stubs it is checking against). `ty`, not pyright, which
  cannot parse jaxtyping shape strings; the step gates, with no `|| true`. GPU
  numerical suite runs via `pixi run test-gpu` on an allocated node.

### Deprecated
- **`kernels.cuda_transition`** — it has never had an implementation. It deferred to a
  `transition/cuda` symbol that git has no record of, and calling it raises
  `NotImplementedError`; the module's `KernelBackend.CUDA` branch called it with a signature
  nothing here provides. Use `implementation='triton'` on `Transition`, or
  `kernels.cuda_transition_b2b` for the hand-CUDA LN-fused path. It stays in the surface and now
  emits `DeprecationWarning`; removal no earlier than two releases out, per the procedure in
  CONTRIBUTING.md.

### Changed
- **One vocabulary for benchmark targets, and a level to hold it.** `bench.py`'s
  targets lived in one flat dict, which forced the kernel-level ones to abbreviate
  around the module-level ones: `tri_attn`, `bias_attn`, `aug_attn`, `ln_mask`,
  `gate_bwd`, `gemm_epil`. `bench_kernel triangle_attention` — the family's own name —
  came back "unknown target". `BenchConfig.kernel` is now `target` + `level`
  (`kernel` | `module`), the two levels are separate namespaces, and every target is
  spelled the way the engine spells it: a kernel target names its family in
  `kernels/registry/registry.csv`, a module target names the module it constructs. So
  `triangle_attention` is now a legal name at both levels and means the right thing at
  each. Renamed: kernel `tri_attn`→`triangle_attention`, `bias_attn`→
  `bias_only_attention`, `aug_attn`→`augmented_attention`, `ln_mask`→`fused_ln_mask`,
  `gate_bwd`→`gemm_gate_bwd`, `gemm_epil[_bwd]`→`gemm_epilogue[_bwd]`,
  `dual_gemm_epil[_bwd]`→`dual_gemm_epilogue[_bwd]`, `cond_transition_tail`→
  `conditioned_transition_tail`; module `bias_only_attention`→`attention_pair_bias`
  (it benches `AttentionPairBias`, and the old name belongs to the kernel family);
  build cases `*_bidir`→`*_bidirectional`, `tm1_triton`/`tm2_triton`→`tm1`/`tm2`;
  implementation labels `triton_tri_attn*`/`triton_bias_attn`/`triton_aug_attn`/
  `aug_attn_memory_efficient` spelled out. The 120 committed result tables that carried
  an old name in their `run_name`/`target`/`implementation` columns were rewritten in
  place; **no measured value changed** (checked cell by cell), and the 40 plots whose
  drawn title named the old target were re-rendered from those same tables.
  `tests/layout/test_bench_target_vocabulary.py` now holds the four name spaces —
  bench.py's tables, the CLI's, `builder.CASE_NAMES`, and the directory tree — to
  each other.
- **Each bench target loads its own config.** `@hydra.main(config_path=...)` was the
  constant `../modules/triangle_multiplication/configs`, so every run — kernel, module,
  atom — loaded that one file and the other 25 `configs/bench.yaml` were read by
  nothing. They disagreed with what ran: `augmented_attention_atom` declares a 128–384
  ladder and was swept at 384–1024, while its own committed tables show 128/256/384.
  The path is now computed from the `target=`/`level=` overrides before hydra starts,
  every one of the 26 targets owns a `configs/bench.yaml` (the 17 kernel targets' are
  copies of the base they already loaded, so nothing they measure changed), and a
  target with no config is an error instead of a silent fall back to another target's
  ladders.
- **`bench_module all` means all of them.** The "all" group read a table that
  `triangle_multiplication_bidirectional` had never been added to, so it ran eight of
  the nine module targets. The bench-args table and the build-case table — keyed by the
  same names, maintained apart — are now one `MODULE_TARGETS`.
- **Coverage no longer guesses a target's directory.** `_report_coverage` rebuilt the
  path from `target in KERNEL_BUILD_CASES`, which was wrong for
  `augmented_attention_token`/`_atom`: they shared one directory named after neither, so
  the lookup missed and both targets' kernels were reported as never launched. Each
  target now owns exactly one directory and the path is derived from `level`.
- **`miniworld-engine build <typo>` fails immediately.** It used to resolve the config
  set and import every kernel — minutes of triton compilation — before saying "unknown
  case". `builder.CASE_NAMES` is a declared tuple (`test_case_names_are_declared` pins it
  to `cases()`), and the per-op name space is read straight from `registry.csv`, so both
  are checked before the first import.
- **Every model-level op is a folder.** `modules/__init__.py` has always opened with the
  rule; `attention_pair_bias.py`, `msa_pair_weighted_averaging.py` and
  `swa_atom_attention.py` were flat files. They are now packages like the other eight,
  and `modules/ops.py` — one level below `miniworld_engine.ops`, the public whole-op
  contract, and meaning the opposite thing — is `modules/functional.py`.
  `tests/layout/test_module_layout.py` holds the rule and the four shared modules that are
  legitimately flat.
- **The linter now checks what the code was already written against.** `select` was
  `["E", "F", "W"]` while the source carried ~700 `# noqa:` comments naming PLC0415,
  BLE001, ANN001, SLF001, S603, ARG005 and two dozen other rules that were not
  enabled — so none of them suppressed anything, and nothing said so. The set is now
  a deliberate one (import sorting, bugbear, pyupgrade, simplification, perf, pytest
  and logging idioms, RUF including RUF100) with every excluded family named and
  justified in `pyproject.toml`, and it runs clean. 264 dead directives were turned
  back into plain comments, keeping their reasons — ruff's own RUF100 fix deletes the
  whole comment, and the reason is the part worth having. Relative imports are banned
  outright (`ban-relative-imports = "all"`): 137 of them became absolute, which is the
  form that broke this session when a flat module became a package and `from .ops
  import sigmoid_gate` silently pointed somewhere else. `[tool.ruff] src = ["src"]`
  was missing, so ruff's own fix for those rewrote them to `src.miniworld_engine.*`.
  Two real defects surfaced: an `assert sig is None or True` that could never fail,
  and eleven late-bound loop variables captured by autotune lambdas in
  `autotune/build.py` (harmless today because each lambda is called in its own
  iteration, one refactor away from tuning every bucket against the last shape).
- **`tools/` is in both gates.** It was tracked code that `ruff check` and `ty check`
  never looked at, in the pixi tasks and in CI alike.
- **`ty` is a gate, not a report.** CI ran `ty check src tests || true` against a
  job that installed the package with `--no-deps`, so torch and triton were
  absent, every `import torch` was an unresolved import, and every torch
  attribute was `Unknown` — the step could not see the types it was checking. It
  now installs the CPU torch wheel plus `[dev]`, checks `src tests benchmarks`,
  and has no `|| true`. The `dev` extra names `ty` instead of `pyright`, which
  cannot parse jaxtyping shape strings and reported 144 parse errors and no real
  findings.
- **`pixi run test` had been dead since the checkout was renamed**
  (miniworld-kernels → miniworld-engine): the `pytest` launcher script carries an
  absolute shebang. The tasks use `python -m pytest`, and `pixi run ci` runs the
  type gate like the CI job does.
- Removed `autotune.elem_bucket_of`, a factory for the per-kernel `bucket_of`
  objects deleted in fcd3c7a. No caller, and the `_miniworld_keys` attribute it
  documented as "lets build.audit introspect the extractor" had no reader.
- **An autotune cache miss no longer sweeps the full grid.** Triton kernels now
  narrow a miss to `autotune_miss_cap` configs centred on `num_warps` in {4, 8}
  and `num_stages` in {2, 3, 4}; the shipped grid is 205,266 configs, so the old
  behaviour put a tuning sweep inside the first production forward on any GPU
  without a cache. A build (`run_autotune=True`) still gets the whole space.
  The warning text now names the fallback that actually happened instead of
  always claiming "the full autotune grid".
- **Autotune cache entries are keyed on the toolchain and the kernel source**,
  not only on the config grid: new `env_identity` (triton/torch/CUDA/ptxas) and
  `op_identity` (kernel source + `key=[...]`) fields. Entries written before
  these existed still read, so committed caches do not become permanent misses.
- **API framing**: `ops` is now the supported **consumer** surface; `kernels` is
  reframed as the **internal primitive** surface (per-GEMM/LN/gate/attention units)
  out of which the ops are built — still pinned (`_CONTRACT`) for internal stability
  but not intended for model code. `triangle_multiplicative_update` moved from
  `kernels` to `ops` accordingly.
- **Packaging**: runtime `[project.dependencies]` slimmed to the kernel core
  (`torch`, `triton`, `einops`, `jaxtyping`, `numpy`). Benchmark harness,
  comparison baselines, the CuTeDSL backend, and dev tooling moved to
  `[project.optional-dependencies]` extras (`bench`, `baselines`, `cute`,
  `dev`). Installing the package for `miniworld_engine.kernels` no longer pulls
  lightning / hydra / cuequivariance / scipy / cutlass.

- Moved 21 unreferenced dev/probe/experiment kernel files (perf probes,
  autotune/tile sweeps, micro-benchmarks, superseded variants) out of the
  importable `src/` package into `research/` (convention: `src/` ships only
  the canonical path). Verified 0 importers + contract & numerical suites green.

### Fixed
- **A `level=both` kernel keyed its cache on LENGTH, so its two sides shared a
  bucket.** A pair activation `(B, L, L, D)` at L=1024 and an atom activation
  `(B, A, D)` at A=1024 both have `shape[-2] == 1024`, so both landed in
  `shape_key=1024` — while the first launches 1,048,576 rows and the second
  launches 1,024. Visible in the shipped A6000 cache as two adjacent lines of one
  op: `shape_key=384 → 0.6103 ms` (pair, 147,456 rows) next to
  `shape_key=1024 → 0.0215 ms` (atom, 1,024 rows). Since the driver was corrected
  to build atom activations above L=512, the sweep measured only the atom side
  there and the module bench, which runs the pair side, was served those configs:
  transition on an A6000 at L=1024 in its committed baseline's exact configuration
  went from 5.498 ms to 9.504 ms while the pytorch reference reproduced within 3%.
  `both_key` now takes the ROW count (`shape_key.BOTH_ROWS`), a both-level kernel
  is two work lists with an explicit `side`, and `cache.KEY_SCHEME` invalidates the
  entries the change re-based — level-aware, so the 68 token/atom kernels whose
  buckets did not move keep theirs.
- **The `compiled` column of every benchmark CSV recorded the request, not what
  ran.** Four of the eight module benches guard `model.compile()` with
  `and conf.cudagraph == "disabled"`, so `compile=true cudagraph=manual` runs eager
  for those; `actual_compiled_flag` knew about `transition` by name and missed the
  other three. 330 committed tables say `compiled=True, cudagraph=manual`, and for
  triangle_multiplication, triangle_attention and bias_only_attention that is a
  measurement of eager code.
- **42 kernel-bench rows could not run at all** — the harness pre-flattened
  activations that seven entry points read their cache key off, so `triton_tm1`,
  `triton_tm2`, `triton_atomic/partial/persistent`, `triton_cond_transition` and
  `layernorm_linear_triton` came back `failed: shape ... is already flattened`.
- **The library needed an environment variable to work at all.** With
  `MINIWORLD_CONFIG_DIR` unset, every op registered an empty config list, triton
  substituted its own `Config({})`, and the first launch of every triton kernel
  died with `TypeError: dynamic_func() missing 2 required positional arguments`.
  Every sbatch script and bench entry point in the repo exports the variable, so
  the failure only ever showed up outside the scaffolding — measured on an A6000
  with it unset, `miniworld` was a `status=failed` row next to a healthy
  `pytorch` one; it is now 1.795 ms against pytorch's 7.084 ms.
  `autotune.configs.default_config_dir()` selects `grid` when nothing is set.
- **The `pre_hook` timer never ran.** `_install_launch_probes` assigned
  `Autotuner.pre_hook`, but triton sets that attribute per instance in
  `__init__` and the class carries no default, so the `hasattr` guard never
  opened and an instance would have shadowed the class attribute anyway. The
  build report printed `pre_hook 0 x -> 0s` for the whole life of the feature.
- **`bench_triangle_attention` crashed on the path that guards it.** Its
  unsupported-`old_triton` branch returned `BenchResult(status=..., error=...)`;
  `BenchResult` is a NamedTuple with neither field, so the guard raised
  `TypeError` instead of the NaN row every other bench returns.
- **A `str` appended to quack's `EXTRA_SOURCE_DIRS: list[Path]`.** The
  membership guard therefore never matched, and every import of
  `kernels/_quack_compat.py` added another copy of the same directory to the
  list the cute JIT cache key hashes.
- **`modules.primitives` imported scipy at module scope** for one weight
  initializer. scipy is in the `baselines` extra, not the core, so a core-only
  install could not import `miniworld_engine.modules.primitives` — or anything
  built on it.
- **`layernorm_linear_pytorch` declared `ln_bias: torch.Tensor`** while both of
  its callers pass `None` for a LayerNorm without beta, which `F.layer_norm`
  accepts.
- **`audit` failed the one kernel whose autotune key correctly carries no
  shape.** `transition_fold_triton` reads only the weights, so `N` and `K` are
  its whole shape and one cache bucket is right; the builder already knew and
  drove it at one length, while the audit reported `key is pinned` against a
  correct build. Both now answer through `builder._keys_on_shape`.
- **`kernels.cuda_transition` has never worked and now says so.** Its body
  deferred to `transition.cuda.cuda_transition`, a symbol with no history in
  this repo; the module binds only `cuda_transition_b2b` and
  `cuda_transition_expand_gate`. `Transition(implementation="cuda")` resolves to
  `KernelBackend.CUDA` and calls it in `forward`, so that path raised
  `ImportError` from four frames down. It now raises `NotImplementedError`
  naming what does exist. The name stays (the surface is frozen) but the
  CUDA backend for `Transition` should be considered unavailable.
- **`miniworld-engine build all` works as documented.** Its config-set argument
  defaulted to the string `"default"` and `configs/default` has never existed,
  so the command failed at argument parsing; `bench_module` hardcoded the same
  string. Both now default to the full search grid.
- **The per-op sweep never merged its shards.** `build all` ran to completion,
  wrote its shards, and left `data/` untouched — the build reported success and
  shipped nothing. It also refused to merge at all if any single unit failed,
  discarding every good measurement in the run.
- **`build all` never drove fp32.** The work list ignored `registry.csv`'s
  `dtypes` column and emitted bfloat16 for every unit, so the fp32 half of 66
  kernels was never tuned — and the coverage check counted `(op, bucket)` with
  no dtype axis, so it reported full coverage over a half-built cache.
- **`level=both` kernels built pair activations at atom shapes.** 18 kernels
  asked for `M = L*L` where production hands `M = A` — 67,108,864 rows at
  L=8192 — which is the whole explanation for the sweep's CUDA OOMs and one
  int32 offset overflow, and it poisoned those kernels' atom-bucket entries.
- **Shard writes are atomic.** A bare `write_text` let two workers interleave on
  one shard and produce unparseable JSON, which the merge then dropped silently,
  losing a whole unit's measurements with nothing said.
- Repo-wide int64 promotion of Triton `program_id` / loop-index pointer offsets
  (raw-indexing kernels), fixing an illegal-memory-access at large `L`
  (`augmented_attention` atom training, L≥768). `make_block_ptr` kernels keep
  int32 block offsets (Triton constraint); CUTE int32 (tmem addr, RNG seed)
  left as-is.

## [0.1.0]
- Initial consolidation of AF3-style op kernels (triangle multiplication,
  transition, triangle/bias/augmented attention, layernorm, adaLN) with
  Triton / CuTeDSL / CUDA backends.

### Checkpoint shape coverage and composite ops (2026-09-15)

- Add public `ops.gated_linear`, `ops.swiglu_ffn`, and `ops.rms_norm_modulation` over existing autograd kernels.
- Drive actual AF3, Protenix v1/v2, OpenDDE and ESMFold2 dimensions from the module build registry; include FP32 norm affine parameters, expansion ratios and projected attention head layouts.
- Handle absent LayerNorm affine tensors on the CUDA path without a mixed-dtype PyTorch fallback.
- Record constructor provenance and unsupported asymmetric TriMul/local attention in `docs/checkpoint-shapes-20260915.md`.
