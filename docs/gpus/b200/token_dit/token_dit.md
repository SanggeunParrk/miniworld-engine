# Token DiT on B200 (sm100)

Kernel-level status of the token DiT (AF3 Alg. 23 block: AdaLN + AugmentedAttention with pair bias + conditioned
SwiGLU transition; d_single 768, d_cond 384, d_pair 128, 16 heads x 48) on B200; the module-level summary is in
[b200.md](../b200.md). Columns are (Length, Dimension, dtype) from the shape registry (`token_single`, d_hidden 768).

**Where the code is.** The B200 kernels are **not in this repo's dispatch yet**: they live on branch
`b200/token-dit` (d0f14e7e, local worktree `~/practice/mw-engine-b200-dit`) under
`experiments/token_dit_fused/` (step runner, training block, Triton row kernels) and `experiments/augattn_sm100/`
(sm_100a attention cores, `build.sh` -> cubins). On the box they are at
`/NHNHOME/WORKSPACE/26mohw002_A/psk6950/mw-dit`. In the tables, **CUDA†** = sm_100a hand CUDA from that branch,
**Triton†** = Triton from that branch; what this repo's dispatch runs today is Triton for every shape (see b200.md).
"미검증" = the kernel accepts the shape (L % 128 == 0) but it has not been run there. fp32 has no B200 path (the
sm_100a cores are bf16 only).
cache build: nothing on the branch uses this repo's persisted autotune cache (`miniworld-engine build`). ✓ where
nothing needs one: the cores are cubins built by `build.sh`, the training row kernels have fixed configs. ✗ for the
inference row kernels: `@triton.autotune`, re-tuned on the first call of every process. The GEMM choices are also
timed on first call and kept only in-process: cuBLAS vs quack per (M, N, K) and the quack SwiGLU tile config in the
inference step (`FusedTokenDiT._mm`, `GATED_CFGS`), the quack gated forward / backward configs in training
(`tdit/qgemm.py` `_race`).

Two paths, both bf16 GEMM operands with fp32 accumulation and an fp32 residual stream:

- **Inference**: one sampling step of 24 blocks, S = 5 samples. Every block's pair bias is computed once per sample
  (one LayerNorm + one GEMM for all blocks, head-major), the AdaLN / gate conditioning once per step on L rows (shared by
  the samples); per block: q|k|v|g GEMM, attention core, Wo GEMM, residual + gate + AdaLN row kernel, expand GEMM with
  SwiGLU in its epilogue (quack), squeeze GEMM, residual + gate + next AdaLN row kernel.
- **Training**: forward + backward of a block stack, A = 48 augments, qk-norm, fp32 parameters. Pair bias hoisted for
  the whole stack (one LN + one GEMM forward, one backward); cond LayerNorm once per stack; conditioning as two GEMMs
  (cond-LN weights folded); q|k|v|g one GEMM; SwiGLU forward / backward in quack sm100 GEMM epilogues; every
  elementwise step one Triton row kernel; bf16 weight pack cached across steps.

## Inference

### I1 · attention core `attn_inf` (softmax(qk^T + bias) v, gated by sigmoid(g), written over q)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA† 미검증 | CUDA† 미검증 | CUDA† | CUDA† 미검증 | CUDA† 미검증 | CUDA† |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I2 · `resgate_adaln_rows` (x += sigmoid(gate) y in fp32, then the next AdaLN) · I3 `adaln_rows`

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | Triton† 미검증 | Triton† 미검증 | Triton† | Triton† 미검증 | Triton† 미검증 | Triton† |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

GEMMs (cuBLAS; expand + SwiGLU through quack `gemm_act`) and the hoisted pair-bias GEMM are not kernel rows.

## Training

### T1 · attention forward `attn_fwd2` · T2 · backward dQ + dbias `attn_dqb` · T3 · backward dK + dV `attn_dkv`

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA† 미검증 | CUDA† 미검증 | CUDA† | CUDA† 미검증 | CUDA† 미검증 | CUDA† |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T4 · Triton row kernels (`tdit/train_kernels.py`, 17 kernels)

Forward: `_cond_prep`, `_adaln_a`, `_qknorm`, `_gate_o`, `_res_adaln_b`, `_res_c`, `_ln_rows` (pair). Backward:
`_res_c_bwd`, `_res_adaln_b_bwd`, `_gate_o_bwd` (also the core-backward prep dO, D), `_qknorm_bwd`, `_adaln_a_bwd`,
`_cond_bwd`, `_unfold_lnw`, `_ln_rows_bwd` (pair); bias / qk-norm / gate gradient sums accumulate atomically inside
them.

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | Triton† 미검증 | Triton† 미검증 | Triton† | Triton† 미검증 | Triton† 미검증 | Triton† |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements (2026-09-29)

B200 (148 SMs, 1000 W power cap), torch 2.10.0+cu130 (the old uv venv of the box, not the v2.2.0 pixi env), GPU 6
through `gpuq` with nothing else on the GPU (checked before every step). Per block, us, median.
Reproduce on the box: `gpuq run -g 6 -- bash measure_b200.sh` in `experiments/token_dit_fused` of the branch; logs
and step JSONs in its `results/b200/2026-09-29_{nopp,pp}/`.

**Not the `benchmarks/runners/bench.py` protocol**: inference is CUDA-graph replay of the 24-block step (`bench.py`
of the branch); training is eager with CUDA events over a 4-block stack (`train_block.py --stack 4`), no torch.compile
baseline; PyTorch = the engine's `PYTORCH` implementation. No cuEquivariance column (it has no token DiT block).

### Inference (S = 5, 24-block step, CUDA graph)

| (Length, Dimension, dtype) | PyTorch (engine `MINIWORLD`) | Anthropic parts, pair bias hoisted | ours | × | rel_rms vs fp32 (ours / engine) |
|---|---|---|---|---|---|
| (384, 768, bf16) | 184.7 | 111.7 | **57.6** | 1.94 | 4.4e-3 / 1.1e-2 |
| (768, 768, bf16) | 371.6 | 192.7 | **101.4** | 1.90 | 4.4e-3 / 1.1e-2 |

Once per sample (all 24 blocks' pair bias): ours 51.8 / 179.0 us, Anthropic 24 x `ln_proj.pair_bias` 390.9 / 1314.6 us.
Energy SOL (`sol_b200.py`): 21.8 / 45.8 us. For reference, the same schedule on H100: 82.3 / 170.5 us.

### Training (A = 48, qk-norm, fp32 params, fwd + bwd, 4-block stack)

| (Length, Dimension, dtype) | PyTorch fp32 (TF32) | engine bf16 + our core | ours | × | max grad rel vs PyTorch fp32 |
|---|---|---|---|---|---|
| (384, 768, bf16) | 7281.9 | 4390.0 | **1623.0** | 2.70 | 8.7e-3 |
| (768, 768, bf16) | 17389.7 | 9029.9 | **3562.7** | 2.53 | 1.0e-2 |

Forward alone: ours 519.7 / 1048.2 us, PyTorch 2693.6 / 6890.2 us. Peak memory at L768: ours 17.3 GB, PyTorch 25.2 GB.
Energy SOL: 697.1 / 1548.5 us (both paths at 38-45 % of it). The max grad error is the bf16-operand floor: many
parameters sit at the same 8.6-8.7e-3 (L384) / 9.5-10e-3 (L768); the engine path is at 4.8-5.6e-3.

### Attention core alone (`augattn_sm100/bench_train.py`, A = 48)

| (Length, Dimension, dtype) | engine Triton | H100 sm_90a (other box, reference) | ours | × vs engine Triton |
|---|---|---|---|---|
| inference (384, 768, bf16) | 152.5 | 107 | **65.7** | 2.32 |
| training (384, 768, bf16) | 706.0 | 558 | **283.5** | 2.49 |
| inference (768, 768, bf16) | 517.4 | 334 | **210.0** | 2.46 |
| training (768, 768, bf16) | 2895.1 | 1841 | **962.6** | 3.01 |

Engine Triton measured on this box on 2026-09-27 (augattn round v1). Training = forward + glue + dkv + dqb; split at L384 / L768: 65.7 / 210.0, 27.1 / 50.6, 107.1 / 385.0, 84.4 / 327.2 us.
Errors against fp64: O 1.6e-3, dq 3.0e-3, dk 2.9e-3, dv 2.9e-3, dbias 2.4e-3.

## Where the time goes (training, L384, per block ~1570 us of kernel time)

| group | us | share | state |
|---|---:|---:|---|
| GEMMs (cuBLAS nvjet, quack gated fwd / bwd, split-K reduce) | ~738 | 47 % | 913 GFLOP at ~1.24 PFLOP/s: the power-capped cuBLAS ceiling (~1.30 PF) |
| Triton row kernels | ~480 | 31 % | each at ~65-85 % of 7 TB/s |
| attention cores | ~267 | 17 % | 30 % at L768 (1072 of ~3500 us); the largest gap to its own ceiling |

## Development record

Training block, per block, L384 fwd + bwd:

| round | commit (branch `b200/token-dit`) | us | change |
|---|---|---:|---|
| T1 | ded9b0b1 | 2598 | the fused block: two conditioning GEMMs, one q\|k\|v\|g GEMM, sm_100a core, row kernels |
| T2 | 565da489 | 2068 | bias reductions inside the backward row kernels, weight-pack cache, TF32 pair-bias dots; 89 -> 45 launches |
| T3 | e25ace6a | 1926 | SwiGLU in quack sm100 GEMM epilogues (forward and backward) |
| T4 | dc8c4aa8 | 1664 (4-block stack) | pair bias hoisted for the stack, tuned row kernels |
| PP | d5577279, 2b3665c6 | 1623 (4-block stack) | exponential ping-pong in the cores (below) |
| — | d0f14e7e | within noise | `_unfold_lnw` 16 rows per program (was 3072 programs x 384 atomics onto 768 addresses: 22-26 -> < 11 us); cond LayerNorm once per stack (~13 / ~25 us at L384 / L768) |

Inference core: `attn_inf` (95776774) = `attn_fwd2` adapted to the step (q|k|v|g column views, logits pre-scaled into
exp2 units, odd S by a clamped partner sample, g by TMA, sigmoid(g) o written over q by TMA store): 16.1 -> 13.3 us
(L384) and 41.8 -> 35.1 us (L768) against the Triton core, before PP.

**Exponential ping-pong (PP).** The two softmax (forward) / dS (backward) warpgroups take turns on the exponentials
through named barriers, so one's S load, bias, max and P store run under the other's MUFU phase (the forward had 25.5 %
of warp samples stalled on ex2 while MUFU averaged 48 %). Sweep (`sweep_fa4.sh`, us):

| kernel | L | PP = 0 | **PP** | PP + 1/4 poly exp | PP + 2/4 poly | PP + 3/8 poly |
|---|---|---:|---:|---:|---:|---:|
| attn_fwd2 | 384 | 69.2 | **65.9** | 67.7 | 70.5 | 68.3 |
| attn_dqb | 384 | 95.4 | **93.4** | 100.3 | 105.7 | 103.8 |
| attn_dkv | 384 | 100.8 | **99.7** | 102.1 | 106.7 | 103.1 |
| attn_fwd2 | 768 | 234.1 | **210.1** | 217.2 | 232.8 | 221.5 |
| attn_dqb | 768 | 342.5 | **332.0** | 353.6 | 375.1 | 366.3 |
| attn_dkv | 768 | 343.5 | **337.4** | 348.2 | 366.3 | 352.6 |

PP output is identical in the forward and within 2e-7 in the backward; polynomial exp2 shares lose everywhere (+4-6e-4
error). End to end: inference 58.8 / 104.7 -> 57.6 / 101.4 us, training 1662 / 3669 -> 1623 / 3563 us. The PP = 0
run also confirmed the T4 / inference numbers first taken while another job shared the GPUs (1664 / 3605 then).

Measured and not taken:

| change | result |
|---|---|
| backward row kernels walking row chunks per program, atomics once at the end | slower (`_res_adaln_b_bwd` 83 -> 113 us): the atomics were not the cost, parallelism was |
| cores read v in place from q\|k\|v\|g (strided TMA rows), no v copy | `_qknorm` -7 / -10 us per block, cores possibly 1-2 % slower at L768; not resolvable, reverted |
| inference: double-buffered P, `attn_inf2` (keys split over the two warpgroups) | flat at S = 5: the core is bound by one item's latency |

**Measurement noise on this card.** Under the 1000 W cap the same code's kernel-time total moves by +-4 % between
runs seconds apart (L768 13983-15177 us; `attn_dkv` 1545-1725 us) and eager wall time by +-3-5 %: 1-3 % changes need
per-kernel times normalised by an unchanged kernel of the same run, or energy per call.

## Next

- Wire the paths into this repo (kernels under `kernels/<family>/cuda/`, dispatch on capability 10.0, the
  `benchmarks/runners/bench.py` protocol) — the branch predates v2.2.0's layout.
- Attention forward writing sigmoid(g) o directly (as `attn_inf` does) to drop `_gate_o`, if D = rowsum(dO o) can be
  taken from the gated output without losing accuracy.
- The attention backward cores (30 % of the block at L768).
- Key mask in the training path; QK-norm in the inference path.
