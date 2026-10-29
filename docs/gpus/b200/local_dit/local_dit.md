# Local atom DiT (AF3 block-local attention) on B200 (sm100)

Kernel-level status of the AF3 atom transformer block with **block-local attention** (engine `modules/local_dit.LocalDiTBlock`: AF3
Alg. 23 with the 32 x 128 trunking of Alg. 24): AdaLN -> q | k | v | gate -> attention in which atom `i` of query window
`w = i // 32` sees only the 128 atoms `[32 w - 48, 32 w + 80)` (clipped to `[0, N)`), with the per-window pair bias `LN(z) Wb^T` taken from
the *trunked* atom pair `[nwin, 32, 128, 16]` (shared by the A samples) -> gated out-projection + residual -> AdaLN + SwiGLU transition
(hidden 256) + residual; d_single = d_cond = 128, d_pair = 16, 4 heads x 32. This is the attention of the AF3 atom transformer
(n_queries 32, n_keys 128; see Not covered for the one difference of its blocks). The module-level summary is in
[b200.md](../b200.md).

**Where the code is.** `modules/local_dit/module.py` (the block, the PyTorch reference path, `to_windows` that makes the trunked pair
from a dense one), `integrations/local_dit.py` (the dispatch: two opaque ops `local_dit_block_fwd` / `local_dit_block_bwd` under an
autograd Function, served eagerly, under `torch.compile` and in CUDA graphs), and the attention and pair-bias kernels in
`kernels/augmented_attention/cuda/sm100_atom_local/`. The row kernels (conditioning, input projections, post-attention, transition and
their backward) are in `kernels/augmented_attention/cuda/sm100_atom/`. The block
has the parameters of `DiTBlock` at atom widths, so both blocks share a checkpoint layout. `serves()` accepts: B200, implementation
MINIWORLD, bf16 single / cond / pair, the atom widths, B = 1, any N, a `[B, N]` bool key mask or none, no QK-norm; anything else
keeps the PyTorch path, and `settings.local_dit_sm100=0` turns it off. N is padded to a multiple of 128 for the row kernels (zero rows,
the padded atoms masked as keys, the pair padded with zero windows); the first N rows come back.

- The attention and pair-bias kernels are hand CUDA, the attention on tcgen05 / TMEM / TMA; the row kernels are tcgen05 / TMA; all dense
  weight gradients of a block are one grouped split-K kernel (`atom_dw`, bf16 and tf32). The only cuBLAS products are the cross mode's two
  modulation GEMMs (`cn Wm^T` forward, `dmkv Wm` backward). No Triton, no quack, no N x N tensor anywhere.
- Cross-attention mode (`cross_attention=True`, the AF3 / Protenix atom blocks): keys and values are projected from a second AdaLN of x1
  (`attention.ada_ln_kv`). The K / V AdaLN runs inside `pre_fwd` (from the registers that hold x1) and its backward inside `pre_bwd`
  (`-DCROSS=1` builds of the same sources, `sm100_atom.pre_fwd_kv` / `pre_bwd_kv`); the conditioning's LayerNorm and its backward are
  half-warp row kernels (`lcross.cu`). See the optimization log below.
- Attention geometry (all three kernels): a CTA owns 128 rows of one head and a range of samples; TMEM lane = row. 128 queries (four
  windows) see 224 keys, and 128 keys see 224 queries when the key chunk starts at `128 j - 112` (every TMEM lane quadrant's 32 keys then
  see the same four windows). Per sample: `S = Q K^T` and `dP = dO V^T` [128 x 224] are tcgen05 MMAs into TMEM; 16 warps (lane
  quadrant x band quarter) read their 32 x 32 band cells, the bias comes from shared memory (log2 domain, key mask folded in as -inf),
  and the softmax / gradient values are written back as packed bf16 over S -- the TMEM A operand of the next product (`umma_ts`).
  Outputs leave through shared memory with per-warp TMA stores (32 rows x 16 columns). TMA loads run three samples ahead.
- `lattn_fwd.cu`: query-centric; S in two TMEM buffers so the next sample's QK^T runs during this softmax; the quadrant's four warps
  exchange row maxima and sums through shared memory; O = P V; a 17th warp issues the TMA loads and the MMAs.
- `lattn_dq.cu`: query-centric; dS = P (dP - D) over S, dQ = dS K; a 17th warp issues the TMA loads and the MMAs.
- `lattn_dkv.cu`: key-centric; P^T and dS^T over S^T, dV = P^T dO and dK = dS^T Q; dbias is summed over the samples in registers (every
  dbias element belongs to one key, hence one thread; two sample halves add exactly, `0 + a + b`). TMEM is full (S^T 224 + dP^T 224 +
  dK / dV 64 columns), so dQ cannot be fused here (it needs dS with queries as rows plus 64 more columns).
- `lbias.cu`: the pair bias and its backward (dz, block partials of the gamma / Wb gradients reduced by a one-block kernel). The
  wrappers cache the TMA descriptors by address and geometry (encoding one costs ~5-10 us of host time).
- Dtypes: activations, weights, gradients bf16; pair bias, dbias, LSE and D fp32; every MMA accumulates in fp32.
- Keys outside `[0, N)` and keys with `mask == 0` get -inf; a query whose window has no valid key outputs 0 (and has zero gradient).

## Measurements

One B200 (shared box), bf16, `torch.set_float32_matmul_precision("medium")`, ms. "PyTorch" is the module's reference path
(`implementation=PYTORCH`, the unfused eager ops replayed as a graph, window gather by `unfold`), not a `torch.compile`d block; no
cuEquivariance row exists for this op. "previous" is the first version of these kernels (`mma.sync` attention, ce7fd93b). "engine" is
the current kernels (2026-10-10: the grouped weight-gradient kernel, the tile-kernel latency fixes below); the PyTorch and previous columns are
the earlier measurement of the same shapes.

Block (shared-AdaLN, `cross_attention=False`), one layer, CUDA-graph replay, all-valid key mask. Inference is given twice: as a plain graph
(every replay repacks the weights and remakes the conditioning tables and the pair bias, ~30 small kernels) and under
`kernels._capture.static_weights()` + `static_inputs()`, which is how the benchmark harness runs inference (a replay is `pre_fwd`, attention,
`post_fwd`):

| block | PyTorch | previous | engine | engine, static | x PyTorch |
|---|---|---|---|---|---|
| training A48 N4096 (forward + backward) | 14.184 | 1.904 | 1.378 | | 10.3 |
| training A48 N8192 | 27.029 | 3.510 | 2.594 | | 10.4 |
| inference A48 N4096 | 5.693 | 0.530 | 0.414 | 0.268 | 13.8 |
| inference A5 N4096 | 1.198 | 0.120 | 0.106 | 0.054 | 11.3 |
| inference A5 N2048 | 0.649 | 0.082 | 0.079 | 0.040 | 8.2 |
| inference A1 N4096 | 0.814 | 0.072 | 0.068 | 0.030 | 12.0 |

Cross-attention block (`cross_attention=True`, the AF3 / Protenix atom block), same protocol; fp32 is the TF32 path below. There is no
PyTorch column: the module path does not run these shapes in a comparable graph. The training cells and the static inference cells
were re-measured on 2026-10-11 through the harness (`bench.py target=local_dit level=module compile=false`, `precision=bf16-mixed` with
fp32 master parameters or `precision=32`; training `cudagraph=manual`); the plain-graph column is the earlier in-process measurement
with bf16 parameters. The block packs its weights once per version (and per capture), so fp32 master and bf16 parameters give the same graph and the
same time (A5 N4096 0.059 ms both ways through the harness; the replayed graph is `pre_fwd`, attention, `post_fwd` in both). The
harness cells are 5-20 % above the in-process static numbers of the earlier measurement (A1 N4096 0.035 against 0.030 ms, A5 N2048
0.047 against 0.040): a method difference, not the parameter dtype.

| block | bf16 | bf16, static | fp32 | fp32, static |
|---|---|---|---|---|
| training A48 N4096 | 1.586 | | 3.854 | |
| training A48 N8192 | 2.975 | | 7.437 | |
| inference A48 N4096 | 0.482 | 0.289 | | 0.692 |
| inference A5 N4096 | 0.126 | 0.061 | 0.161 | 0.119 |
| inference A5 N2048 | | 0.047 | | 0.072 |
| inference A1 N4096 | 0.079 | 0.035 | | 0.053 |

Training of the cross block started this round at 2.52 ms (bf16) / 4.33 ms (fp32) at A48 N4096.

The attention kernels alone (events over 20 eager launches, no graph; the small shapes are bounded by host launch time):

| (A, N) | bias fwd | attention fwd | attention bwd (dq + dkv) | bias bwd |
|---|---|---|---|---|
| (48, 4096) | 0.014 | 0.076 | 0.193 | 0.027 |
| (48, 8192) | 0.013 | 0.147 | 0.378 | 0.046 |
| (5, 4096) | 0.014 | 0.017 | 0.038 | 0.027 |
| (1, 2048) | 0.013 | 0.017 | 0.037 | 0.027 |

Speed of light (the atom DiT method: a CUDA graph of back-to-back calls replayed for >= 2 s, wall-clock time and NVML energy; floor =
max(bytes / HBM, FLOPs / 2.38 PF, exps / 4.65 T/s, dynamic energy / (P_max - P_idle)) with compulsory bytes; SoL = floor / time). The
large shapes run at the 1000 W cap, so further speed has to come from less energy, not more overlap.

| (A, N) | attention fwd | attention bwd | bias fwd | bias bwd |
|---|---|---|---|---|
| (48, 4096) | 72.7 us, 54 % | 190.6 us, 61 % | 6.5 us, 58 % | 18.3 us, 34 % |
| (48, 8192) | 144.2 us, 55 % | 379.8 us, 61 % | 10.5 us, 71 % | 32.2 us, 38 % |
| (5, 4096) | 12.3 us, 43 % | 31.0 us, 49 % | 6.5 us, 57 % | 18.3 us, 34 % |
| (1, 4096) | 6.5 us, 32 % | 16.4 us, 35 % | 6.5 us, 57 % | 18.3 us, 34 % |

Against Anthropic's window kernel (`opt_core/kernels/apb/fpf_apb/atom_triton.py` `atom_apb`: QK^T + pair bias + softmax + PV of the
same 32 x 128 windows, forward only, no LSE output), attention forward, graph replay, us (it has no backward, so training has no
Anthropic row):

| (A, N) | engine | Anthropic bf16 | Anthropic fp32 (its default) | x Anthropic bf16 |
|---|---|---|---|---|
| (48, 4096) | 72.3 | 187.7 | 482.7 | 2.60 |
| (48, 8192) | 143.8 | 370.2 | 959.2 | 2.57 |
| (5, 4096) | 12.4 | 20.7 | 48.6 | 1.67 |
| (5, 2048) | 9.5 | 12.1 | 25.3 | 1.28 |
| (1, 4096) | 6.5 | 6.1 | 12.1 | 0.94 |
| (1, 2048) | 6.3 | 4.0 | 6.7 | 0.63 |

At A = 1 the per-CTA fixed cost (TMEM allocation, the bias load, descriptor fetch, ~5 us) dominates; Anthropic's kernel is faster there.

Accuracy: `tests/gpu/sm100/test_b200_local_dit_gpu.py` checks the output and every gradient against an fp64 PyTorch block (no worse
than the bf16 module path), with and without a key mask, at N not a multiple of 32 or 128, eagerly, under `torch.compile` and in a CUDA
graph; the window definition is checked against a per-atom loop on CPU (`tests/cpu/kernels/test_local_dit_reference.py`). dQ, dK, dV and
dbias are bit-identical to the previous `mma.sync` kernels; the forward's row sum is now taken in fp32 (it was the sum of the bf16 P),
which moves O by 1.2e-3 relative (bf16 rounding level; relative error against fp64 1.77e-3 as before).

## Per-item hoist (inference)

The conditioning tables (AdaLN modulation; in cross mode the conditioning LayerNorm and the K/V modulation GEMM) depend only on the
conditioning tensor and the weights, and the windowed pair bias only on the pair tensor and its two weights. Across the steps of a
sampler they do not change, so an inference call (no gradients saved) computes them once per (tensor, in-place version, weights) and
later calls read the cache; Anthropic's `FastAtomStack` does the same through its `dit_hoist`. A cache entry holds a weak reference to its
tensor, so a freed-and-reused address never serves stale tables. With `kernels._capture.static_inputs()` (or
`settings.static_conditioning=1`) a CUDA-graph capture serves from the eager entries and a replay launches none of these kernels: one
inference call goes from 11 to 6 kernels (a graph of the whole call under `static_weights()` as well is three kernels). Off by default; training never uses it. Tests: `tests/integrations/test_b200_local_dit_gpu.py`
(`-k hoist or static_inputs`).

## fp32 path (TF32 tensor cores)

fp32 single / cond / pair with fp32 parameters (no autocast) are served on B200 by a separate set of kernels: every product on tcgen05
`kind::tf32` (fp32 operands read at tf32 precision, fp32 accumulation), everything else fp32 -- the residual stream, LayerNorms / AdaLNs,
gates, the softmax and P, every saved activation and every gradient. The weight gradients are the grouped `atom_dw` kernel on tf32 (fp32
partials, fp32 gradients). No Triton, none of the bf16 kernels. Dispatch: `integrations/local_dit.py`
(`_fp32_call`, `_LocalBlock32` / `_infer32`, opaque ops `local_dit_block_fwd_tf32` / `local_dit_block_bwd_tf32`); the kernels load
separately from the bf16 ones (a build / load failure warns once and keeps the module path for fp32 calls only);
`settings.local_dit_tf32=0` keeps fp32 calls on the module path. A bf16 model fed fp32 activations, or a call under autocast, keeps the
module path. The per-item hoist applies (fp32 tables under their own keys).

Kernels (`kernels/augmented_attention/cuda/sm100_atom_local/`, `.../sm100_atom/`; budgets in each file's header):

| kernel (build entry) | design | smem / TMEM / threads |
|---|---|---|
| `lattn_fwd_tf32.cu` `local_attn_fwd_tf32` (`KERNELS_TF32["fwd32"]`) | the bf16 geometry (128 queries x 224 keys, TMEM lane = query); q / k K-major SW128, v MN-major in the 32-B-atom swizzle; P fp32 written in place over S (each warp its own 32 band cells + one zeroed out-of-band block); bias in registers; 2 stages | 168.3 KB / 480 cols / 17 warps |
| `lattn_dq_tf32.cu` `local_attn_dq_tf32` (`"dq32"`) | S, dP into TMEM, dS fp32 in place over S, dQ = dS K with K read a second time MN-major; q, dO, k, v, LSE, D in 2 stages, the MN-major k in one slot (S(it+1) precedes dQ(it+1)) | 222.3 KB / 480 cols / 17 warps |
| `lattn_dkv_tf32.cu` `local_attn_dkv_tf32` (`"dkv32"`) | key-centric (keys `128 j - 112`); P^T over S^T and dS^T over dP^T in place; q and dO read K-major (one slot: TMEM holds one S^T anyway) and MN-major (two slots, each doubling as its sample's dK / dV staging); dbias in registers, summed over the samples | 204.3 KB / 512 cols / 17 warps |
| `lbias_tf32.cu` `local_bias_{fwd,bwd,fin}_f32` (`"pb_f32"`, `"pb_b32"`, `"pb_fin32"`) | the pair bias and its backward on an fp32 pair (CUDA cores, one row per thread) | static |
| `gemm_tf32.cu` `atom_gemm_tf32` (`sm100_atom.KERNELS_TF32["gemm"]`, `gemm32`) | every projection and activation-gradient product: persistent 128 x NT (128 / 256) tiles, 3-stage TMA ring of 32-wide K slices, double-buffered TMEM accumulators; epilogues: bias + sigmoid on chosen 128-column blocks (conditioning tables), gated residual (`a1 = s + so (gated Wo^T)`, `out = a1 + st (h Ws^T)`, + the raw product for the backward), SwiGLU over an interleaved `[Wa_0; Wb_0; Wa_1; Wb_1]` pack; per-warp 4 KB TMA-store stagings; 4 warps round each A slice to tf32 in shared memory before the MMA | 208.3 KB / 512 cols / 16 warps, <= 128 registers |
| `gemm_rg_tf32.cu` `atom_gemm_rg_tf32` (`"gemm_rg"`) | the gated-residual GEMM at N = 128: res and gate arrive through TMA into per-warp shared buffers one chunk ahead (per-thread loads were 2.1-2.5x the memory floor, now ~1.2x), `out = fma(gate, acc, res)` | 224.3 KB / 512 cols / 16 warps (512 threads) |
| `dw_atom.cu` `atom_dw` + `atom_dw_reduce` (`sm100_atom.atom_dw(device, f32)`) | every dense weight gradient of the block in one launch (A^T B over all rows, column views of the saved activations / gradients): split-K over the rows, TMA stages of 32 rows: 2 x 80 KB (tf32; bf16 5 x 40 KB), the whole [128 mb, nn] output of a slice accumulated in TMEM, fp32 partials in a workspace, a deterministic in-order slice reduce into each weight's own tensor | 160 KB (bf16 200 KB) / <= 512 cols / 8 warps |
| `rows_tf32.cu` `f32_*` (`"ln"`, `"adaln"`, `"gate"`, `"tail_b"`, `"swiglu_b"`, `"adaln_b"`, `"gate_b"`, `"ln_b"`, `"colsum"`) | the row-wise stages: conditioning LayerNorms, AdaLN and its backward (with the output gate fused), gates, SwiGLU backward, D = rowsum(dO O) per head, the conditioning LayerNorm backward with block-reduced v4 `red.add`, bias-gradient column sums (partial rows per block, one tiny sum) | warp per row |

Operand rounding. A `kind::tf32` MMA reads an fp32 operand by dropping its low 13 mantissa bits; that truncation is a one-sided error
that accumulates coherently. Without rounding, the block output came out 1.6e-3 relative against fp64, 5.6x the cuBLAS TF32 module path
(2.8e-4), independent of shape. So every MMA operand is rounded to nearest (`cvt.rna.tf32.f32`) before the MMA reads it: the weight packs
on the host (`sm100_atom.round_tf32`), the GEMMs' A slices in shared memory, q / k / v in the projection's epilogue, dO in `f32_gate_bwd`,
and P / dS before they are written to TMEM. The stored activations stay exact fp32 (except q / k / v and dO, which only feed MMAs), as do
the row sums and dbias.

What the fp32 operands change in the attention kernels: a 32-wide fp32 head row is 128 B (one 128-B swizzle atom, four K = 8 MMA steps);
a tf32 MMA takes an MN-major operand only in the 128-B swizzle with 32-B atoms, so k (dQ), q and dO (dK / dV) are loaded a second time in
that layout; P / dS are as wide as S in fp32, so they go over S in place instead of packed; the bias (66 KB as a shared copy) lives in
registers (in `lattn_dkv_tf32` it is read per sample from L1 / L2 instead: 32 more registers spilled there). The fp32 tensors are about twice the bytes, so the stages drop from 3 to 2 (fwd, dq) or split by lifetime (dkv).

Layouts of the path: the conditioning table `[M, 768] = [s1 | bi1 | s2 | bi2 | so | st]` (three GEMMs from LN_g1(c), LN_g2(c), c; scales and
gates stored as sigmoids), the projection `[M, 512] = q | k | v | g` (cross mode: `q | g` from x1 and `k | v` from the K / V AdaLN, two
GEMMs), the transition's `[a | b]` interleaved per 128-column half. Forward: LN x2 + 3 GEMMs (hoisted in inference), AdaLN, 1 GEMM, pair
bias (hoisted), attention, gate, 3 GEMMs with 1 AdaLN between: an inference call with the hoist is 8 launches (cross: 10). Backward: 8 row
kernels, 7 GEMMs (cross: 10), dq + dkv, pair-bias backward, the grouped weight-gradient kernel.

Tests: `tests/gpu/sm100/test_b200_local_dit_tf32_gpu.py` (output and every gradient against an fp64 block, no worse than the fp32
module path with TF32 on -- `ef < 1.5 em + 1e-3` -- and loosely bounded by the IEEE module path; cross / non-cross, masks, N not a multiple
of 32 / 128; eager, `torch.compile`, CUDA graph for inference and for a whole training step; the TF32 kernels ran and no Triton / bf16
kernel; no kernel spills (`lmem == 0`; <= 128 registers, `atom_post_tf32` <= 168); the fp32 hoist; fused vs unfused inference).

### Fused inference kernels (`sm100_atom/fused_tf32.cu`)

An inference call (no saves) runs the block as three kernels: `atom_pre_tf32` (AdaLN 1 and, in cross mode, the K / V AdaLN from the exact
x1, written rounded into swizzled shared-memory A tiles, then x1 [Wq; Wg] and xkv [Wk; Wv] into TMEM; epilogue: q bias, q / k / v
rounding, TMA store), `local_attn_fwd_tf32`, and `atom_post_tf32` (gate, Wo, a1 = s + so y, AdaLN 2, SwiGLU in two 128-column halves,
h Ws, out = a1 + st t; y stays in TMEM). Persistent 128-row units; the row warps keep 4 rows of loads in flight per phase and run the
LayerNorm reductions of those rows interleaved; sigmoids use `rcp.approx`. The units of the last, partial wave are split in two (pre: two
256-column halves, post: two 64-row halves) when that shortens the makespan (`sm100_atom._nfull`: all halves when 2 x tiles <= SMs, else
only the tail beyond whole waves); measured, a half unit costs 0.7-0.8 of a full one, so splitting every unit loses past one wave.
`atom_post_tf32` is capped at 168 registers (384 threads, one CTA per SM: the 128 cap kept its unit-loop state on the stack).
`settings.local_dit_tf32_fused=0` keeps the unfused kernels; training always uses them (they save the activations). Fused and unfused
agree to < 2e-6.

What it took, in order (per block, 5 samples, us): the weight pack rebuilt in every graph replay (~30 small kernels -> one flat buffer, 3
launches; none under `static_inputs()`), the conditioning tables in 2 launches instead of 5, PDL between the fp32 kernels, the fusion (first
slower: one serial global round trip per row; fixed by batching the row loads), the IEEE-division sigmoid (23 % of post's stall samples),
the serial LayerNorm shuffle chains, the `__ldg` bias loads, and the last-wave split at 4096 atoms (160 tiles on 148 SMs).

### Measurements (2026-10-06)

Against Anthropic's AF3 atom transformer (Protenix v1 kit `protenix_fpf_ditfast.atom_fast.FastAtomStack`, fp32 activations, TF32 cuBLAS,
its own fused attention / row kernels and per-item hoist), the same Protenix AtomTransformer weights, 3 cross-attention blocks, 5 samples,
CUDA-graph replay (min of 9 x 60), B200. "hoisted": Anthropic's `dit_hoist` on / ours under `kernels._capture.static_inputs()`.

| atoms | per call: Anthropic / ours (us) | x | hoisted: Anthropic / ours | x |
|---|---|---|---|---|
| 1024 | 402.4 / 203.0 | 1.98 | 185.4 / 136.5 | 1.36 |
| 2048 | 523.4 / 280.2 | 1.87 | 254.7 / 194.5 | 1.31 |
| 3072 | 624.4 / 329.8 | 1.89 | 313.3 / 225.2 | 1.39 |
| 4096 | 754.0 / 465.9 | 1.62 | 386.0 / 337.8 | 1.14 |
| 6144 | 1017.8 / 638.2 | 1.59 | 544.7 / 456.2 | 1.19 |

Accuracy: ours 1.36e-4 against the fp64 engine block (the engine's IEEE fp32 module path is 1.6e-7 from it); Protenix's AtomTransformer
in IEEE fp32 is 1.06e-3 from that block and Anthropic 1.09e-3 (1.9e-4 from Protenix fp32) -- a definition difference between Protenix's
block and the engine's, not precision (Protenix's own kernels switched off change nothing). Against the PyTorch compiled fp32 module path
(bench harness, TF32 on, one block, L 128-768): inference 1.5-2.1x, training 1.25-1.49x (the training numbers predate the work above).

### Optimization log: backward and cross mode (2026-10-10)

Cross-attention training block, 1 layer, 48 samples x 4096 atoms (196 608 rows), CUDA-graph total, one B200 GPU, ms:

| | start | now |
|---|---|---|
| bf16 (production, bf16-mixed) | 2.52 | 1.58 |
| fp32 (TF32) | 4.33 | 3.85 |

At 8192 atoms bf16 is 2.95 ms and fp32 7.43 ms (linear in rows). Inference (A = 5, one cross-mode block, bf16, CUDA graph under
`kernels._capture.static_weights()` + `static_inputs()` as the harness runs it): 57 us at 4096 atoms, 92 us at 8192 -- `pre_fwd`, attention, `post_fwd`, nothing
else. Without the static declarations every replay repacks the weights and remakes the conditioning tables (about 30 small kernels, 136 us).

What changed:

| kernel | design |
|---|---|
| `dw_atom.cu` `atom_dw` + `atom_dw_reduce` (`sm100_atom.atom_dw(device, f32)`) | every dense weight gradient of a block (A^T B over all rows, A / B column views of the saved activations and gradients) in **one** launch instead of one cuBLAS GEMM each: split-K over the rows with slices per problem proportional to its operand columns, TMA stages of 128-B-swizzle boxes (bf16: 32 rows x 5 stages; fp32 / tf32: 32 rows x 2 stages -- 8 rows x 8 stages was 2x slower, 16 x 5 1.17x, 24 x 3 1.03x, 40 x 2 not exact at 196 608 rows), fp32 partials in a workspace, a deterministic in-order slice reduce writing each 128-row block of A to its destination tensor in the parameter's dtype. fp32 weight gradients of six problems at 196 608 rows: 334 us against 441 us cuBLAS. |
| `rows_tf32.cu` `f32_colsum` | bias / conditioning-column gradients (fp32 path): a persistent grid writes G partial rows, a tiny sum finishes (replaces `sum(0)` reductions) |
| `lcross.cu` `local_cond_ln`, `local_cond_ln_bwd` (it also had `local_kv_fwd` / `local_kv_bwd` until they were fused into pre_fwd / pre_bwd) | a row per half-warp with 16 B per lane (a row is one 256-B request, the LayerNorm reductions are 4 shuffle steps), persistent grid of 8 blocks per SM, bias / gamma gradients reduced in the block and added with one atomic per channel per block (per-warp atomics onto the same 128 addresses had serialized in L2). 196 608 rows: cond_ln 33 -> 19 us, cond_ln_bwd 62 -> 35 (kv_bwd, before its fusion: 127 -> 60). More rows in flight per half-warp (U = 2, 4) cost registers, hence blocks per SM, and were slower; copy rate is 6.0 TB/s here |
| `pre_fwd.cu` `atom_pre_fwd_sm100` (`-DCROSS=1`, `sm100_atom.pre_fwd_kv`) | the K / V AdaLN inside the input stage: xkv = LN(x1) sigmoid(mkv_pre + bs) + mkv_shift from the registers that hold x1 (a second LayerNorm over the same 8 threads), staged as its own tile, the B operand of the K / V MMAs (Q, G read the x1 tile), saved for the backward. 40-KB input stages, two (three needed a two-deep xkv ring and ran 30 % slower). Replaces `kv_fwd` (45 us) and two cuBLAS GEMMs (30 us); the kernel grows 76 -> 130 us |
| `pre_bwd.cu` `atom_pre_bwd_sm100` (`-DCROSS=1`, `sm100_atom.pre_bwd_kv`) | the real Wk, Wv in W^T: projections 1 and 2 give dxkv = rn(dk Wk + dv Wv) (one fp32 sum, one rounding), whose AdaLN backward runs in the row stage on staged x1 and pre (dx1_kv, dpre, d bs), added to dx1 = rn(dg Wg + dq Wq) in bf16 before AdaLN 1's backward; dmkv = [dpre \| dxkv] and d bs are written there. 44-KB stages, two per warpgroup. Replaces the K = 256 GEMM, `kv_bwd` (57 us) and the zero-column `Wp^T` clone; the kernel grows 95 -> 145 us |
| `gemm_rg_tf32.cu` `atom_gemm_rg_tf32` | the gated-residual GEMM at N = 128 with res and gate loaded through TMA into per-warp shared buffers one chunk ahead (the per-thread loads were uncoalesced: 2.1-2.5x the memory floor, now ~1.2x) |

In cross mode the conditioning LayerNorm backward of the K / V modulation adds onto `dc` in place (no add kernel). The key mask is passed as a uint8 view of the bool mask (no per-call conversion launch). `tests/gpu/sm100/test_b200_atom_dw_gpu.py`
covers `atom_dw` (bf16 / tf32, strided column views, several destination blocks, run-to-run determinism) and the fp32 column sum against fp64.

### Optimization log: tile-kernel latency (2026-10-10)

Per-tile timestamps (`clock64` marks in the compute warpgroup's loop, one CTA) showed where the 16- / 32-row tile kernels spent their time; three
findings, each worth a few percent of the step (bf16 training 1.68 -> 1.58 ms at 4096 atoms):

| change | effect |
|---|---|
| IEEE `1.f / sqrtf` and `1.f / (1 + expf)` in the cross-mode AdaLN math -> `rsqrtf`, `__fdividef` (as every other kernel here) | the K / V AdaLN stage of `pre_fwd` took 2860 of its 5200 clk per tile; 970 now. `pre_fwd` 130 -> 94 us, `pre_bwd` 134 -> 121 |
| TMA stores of `post_fwd` / `tr_bwd_gate` / `post_bwd` / `pre_bwd` / `cond_bwd` issued and waited for by a lane of the producer warp (`stfull` ring of NSW barriers) instead of the warpgroup's leader thread | the leader no longer stalls its warpgroup on the stage's read-out: `post_fwd` 162 -> 136 us, `pre_bwd` 145 -> 134, the others within noise |
| `cond_fwd`: the two warpgroups staggered by half an iteration (warpgroup 1 runs the epilogue of tile T, then the LayerNorm of tile T + 2) | one's sigmoids (MUFU) overlap the other's row work: 125 -> 114 us |

Where the tile kernels stand against the HBM copy rate (6 TB/s measured with a plain copy here), 196 608 rows: `pre_fwd` 94 us (floor 92), `pre_bwd`
121 (117), `post_bwd` 174 (160), `post_fwd` 132 (117), `tr_bwd_gate` 120 (100), `cond_bwd` 109 (84), `cond_fwd` 114 (67; MUFU: 100 M sigmoids are 44 us of
exp / reciprocal alone, and about 1600 instructions per tile per SMSP), `atom_dw` 256 (260). Tried without effect: batching the TMEM reads of an
epilogue into one wait (`cond_fwd`; `post_fwd` 136 -> 132), three input stages in the cross `pre_fwd` (a two-deep xkv ring waits on the
previous tile's store: 167 us), staggering `pre_fwd`, mixing polynomial `ex2` into the sigmoids (MUFU and issue slots break even).

## AlphaFold 3's options: parallel K/V norm, slid key window (2026-10-29)

FoldForge builds its AF3-family atom blocks as `LocalDiTBlock(cross_attention=True, kv_norm=..., key_window=...)`; in its "fast" mode every
family takes AlphaFold 3's `kv_norm="parallel"` (the K / V AdaLN normalises the raw atoms `s`, next to AdaLN 1, instead of AdaLN 1's output)
and `key_window="slide"` (a window's 128 keys start at `S_w = min(max(32 w - 48, 0), max(n_real - 128, 0))`, shifted back inside the real
atoms, which come first). Both ran the PyTorch statements on B200 before; now:

| option | inference | training | how |
|---|---|---|---|
| `kv_norm="parallel"` | bf16, fp32 | bf16, fp32 | `pre_fwd.cu` / `pre_bwd.cu` `-DPARALLEL=1` (`sm100_atom.KERNELS["prexp"]`, `["prebxp"]`): xkv = LN(s) sigmoid(pre + bs) + shift from the statistics AdaLN 1 already has; in the backward the K / V AdaLN's LN backward reads the staged `s` (x1 is not loaded) and goes into d single. fp32: `atom_pre_tf32` with `cross = 2`, the unfused path `adaln32(s, ..)` / `adaln_bwd32(s, ..)` |
| `key_window="slide"` | bf16, fp32 | module path | the clipped-window attention kernel, then `sm100_atom_local/lslide.cu` `local_slide_edges(_f32)` recomputes the windows the slide moves -- 0, 1 and the last three real ones, picked on the device from the key mask's population count (no host read: CUDA-graph capturable, a replay with another mask is right), a window listed twice or not moved skipped -- over their slid keys (same per-window bias tensor: the trunked pair is laid out from the window's first key either way) and overwrites their rows of O |

The edge kernel is a CTA per (listed window, head, sample), 512 threads on CUDA cores (scores with the key row in registers, softmax by
warp shuffles, P V a thread per (query, 2 channels)). It costs a constant ~9-10 us per call (latency of five serial phases in one CTA; the
work is 5 x 32 queries); measured inference of one cross block, CUDA graph under `static_weights()` + `static_inputs()`, us:

| (A, N) | bf16 chained / pad | bf16 parallel / pad | bf16 parallel / slide | fp32 chained / pad | fp32 parallel / pad | fp32 parallel / slide |
|---|---|---|---|---|---|---|
| (5, 4096) | 48.2 | 48.0 | 57.8 | 115.4 | 112.9 | 123.0 |
| (5, 1024) | 27.3 | 27.1 | 36.1 | 44.0 | 41.6 | 51.8 |
| (1, 4096) | 24.2 | 23.8 | 32.7 | 42.5 | 40.0 | 50.4 |

A gradient through the slid window keeps the module path (the backward kernels know the clipped window only: the edge windows' dQ, dK, dV,
dbias would need the same treatment). Tests: `tests/gpu/sm100/test_b200_local_dit_af3_variant_gpu.py` (the three option pairs x eight
(A, N, real atoms) cases incl. FoldForge's A = 5, N 1024, under 128 real atoms and windows listed twice; bf16 and fp32 against the fp64
module, `ef < 1.5 em + 3e-3` (bf16) / `1e-3` (fp32, em = the module path on TF32), on the real atoms; fp32 parameters with bf16
activations; graph capture with a mask changed between replays; parallel-norm training, every gradient; a gradient through the slid
window is not served). Without the edge kernel the slide cases fail (checked).

## ESMFold2's block: bf16 inference (2026-10-29)

`LocalDiTBlock(**ESMFOLD2)` (FoldForge's ESMFold2 / ESMFold2-fast atom stack: AF3's parallel K/V norm and slid window at 192 keys with the
+/- 64 rank window, RMS / SiLU conditioning with raw gates, per-head q / k RMSNorm + rotary) ran the PyTorch statements on B200. bf16
activations (bf16 or fp32 parameters, with or without autocast), no gradient, are now served by:

| stage | kernel | what changes against the AF3 block |
|---|---|---|
| conditioning (hoisted) | `cond_fwd.cu -DRMS=1` (`sm100_atom.KERNELS_ESM["cond_rms"]`), grid y = 3 | every table from rn(silu(c)), no sigmoid (the scales' identity bias 1, gate biases 0); the third half writes the K / V AdaLN's [scale \| shift] (mkv) |
| AdaLN 1, K / V AdaLN, projections | `pre_fwd.cu -DCROSS=1 -DPARALLEL=1 -DRMS=1` (`"pre_rms"`) | RMSNorm statistics; the K / V scale raw (`pre + 1`) |
| q / k norm + rotary | `sm100_atom_local/lesm.cu` `local_esm_qkrope` | per head RMSNorm (weight [32], eps float32's) then the rotary with per-atom tables, a warp per row |
| pair bias (hoisted per pair and mask) | `lesm.cu` `local_esm_bias` | the bias re-indexed by position: bias2 [4, nwin, 32, 160] (log2 units, -inf where a position is not a key of the query: outside the real atoms, \|i - j\| > 64, or outside the window's slid 192-key subset); the real atom count from the mask on the device |
| attention | `sm100_atom_local/lattn_esm.cu` `local_attn_esm` (tcgen05) | `lattn_fwd.cu`'s frame with 256 key positions per 128 queries (a window sees 160: five 32-key cells, the outer two triangular, one warp takes both); S in two 256-column TMEM buffers, so O lives in columns 128..159 of its sample's buffer and the next S into that buffer is issued once O is drained; 2 TMA stages; no spills (96 registers, 544 threads) |
| gate, Wo, AdaLN 2, SwiGLU, Ws | `post_fwd.cu -DRMS=1` (`"post_rms"`) | RMSNorm statistics (the raw gates are what the RMS tables hold) |

Measured, one block, CUDA graph under `static_weights()` + `static_inputs()`, conditioning shared by the samples (an expanded view), us:

| (A, N) | B200 kernels | module path (B200) | x |
|---|---|---|---|
| (5, 4096) | 59.2 | 2107 | 35.6 |
| (5, 1024) | 30.5 | 678 | 22.2 |
| (1, 4096) | 26.8 | 1335 | 49.8 |

At (5, 4096): post_fwd 22.5, attention 21.7, pre_fwd 15.8, q / k norm + rotary 6.4 us. Accuracy against the fp64 module (relative Frobenius,
real atoms): 5.2-5.3e-3, the module path's own bf16 run 4.8-5.0e-3, at all eight test shapes. Tests:
`tests/gpu/sm100/test_b200_local_dit_esmfold2_gpu.py` (eight (A, N, real atoms) shapes incl. N not a multiple of 128, under 192 real atoms
and FoldForge's A = 5 with a shared conditioning; fp32 parameters under autocast and bf16 parameters; `ef < 2 em + 1e-3`; bitwise
repeatable; graph capture replayed with another mask and inputs; a gradient, fp32 activations or a missing rotary keep the module path).
Not served yet: fp32 activations and training (module path).

## Not covered

B > 1, QK-norm outside ESMFold2's block, ESMFold2's block in fp32 or with a gradient, training through AF3's slid key window, fp16, other GPUs; fp32 activations with bf16 parameters or under autocast (module path: serving them would change the precision rules -- autocast keeps the residual stream fp32, the bf16 kernels round it to bf16 per block). The cross-attention mode of the AF3 / Protenix atom blocks (`cross_attention=True`: keys and
values projected from a second AdaLN `attention.ada_ln_kv` of the already normalised atoms) is served, but its extra work is the K / V AdaLN
inside `pre_fwd` / `pre_bwd` (above), the conditioning LayerNorm and its backward in `lcross.cu` and cuBLAS GEMMs for the modulation tables (one `cn Wm^T` forward, one `dmkv Wm` backward).
