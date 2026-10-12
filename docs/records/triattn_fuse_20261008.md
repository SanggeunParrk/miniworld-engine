# H100 TriangleAttention: fused front, max-free core (2026-10-08)

Source: working tree on `h100/cuda-sweep` (uncommitted), torch 2.13 / CUDA 12.9, H100 80GB, `normal_h100`.
All timings are CUDA-graph replay through `benchmarks/runners/bench.py` (module level, bf16-mixed, B1, starting node,
`n_augment` 5 for inference and 48 for training, dropout 0.25 in training), measured in a private source snapshot so that
concurrent edits to the shared checkout cannot invalidate a run. Baselines (PyTorch, cuEquivariance, Anthropic
`triattn_native` rebuilt for this toolchain) were measured in the same session with the same harness.

## What changed

| Part | Change | Files |
|---|---|---|
| Front (LN + `[Wq;Wk;Wv;Wg;Wb]` projection + head bias) | Two one-kernel implementations. `front_fused2`: 64-row tile, two warpgroups split the weight columns, 2 CTAs per SM at C<=256. `front_fused3`: 128-row tile, both warpgroups share one TMA weight ring. Bias is transposed in shared memory so each head writes contiguous keys; Q/K/V/G leave through TMA stores. | `h100_sources/front_fused2.cu`, `front_fused3.cu` |
| Core | Max-free softmax: per-row shift is the first key tile's row maximum (a real logit, so `p_max >= 1`); no running max, no accumulator rescale. A row whose sum leaves `[0.5, 1e30]` sets a flag and a second launch of the original running-max kernel recomputes only flagged tiles (no host sync, CUDA-graph safe). `lse = shift + log2(sum)` keeps the backward definition. | `h100_sources/training_forward.cu` (`MW_TRIATTN_MAXFREE`, default 1) |
| Dispatch | Front: C128 -> `fused2`, C256 -> `fused3`, other widths keep the cuBLAS front. Fused tail (existing kernel) for inference at C128/256 for every L (was four cells). Environment overrides unchanged. | `h100_triattn.py` |

The original kernels stay in place: `MINIWORLD_TRIATTN_H100_FRONTEND={grouped,fused,fused_reuse,fused2,fused3}`,
`MINIWORLD_TRIATTN_H100_TAIL={grouped,fused,fused3}`, `-DMW_TRIATTN_MAXFREE=0` (through `MW_F2_DEFS`, see below).

## Correctness

- `tests/integrations/test_triattn_h100_b200style_gpu.py`: 104 tests pass (shape cells, edge cases, graph replay,
  forward/backward against the independent PyTorch implementation), with the default dispatch.
- Front kernels vs an fp32 PyTorch reference: same error as the existing fused front (bf16 rounding), masks and LN
  statistics exact.
- Max-free core vs fp32 attention on adversarial inputs (later keys 40x larger than the first tile; first 64 keys masked
  for half the queries): no non-finite values, `lse` error <= 1e-4, output error at bf16 level; the safe pass is taken
  where needed. Measured max output error 0.0154 against 0.0087 for the running-max kernel (the shift is not an integer, so
  the bf16 rounding of P differs).

## Results

Ratios are baseline ms / ours ms (above 1 favours this branch). "start" is the engine at the beginning of the session.

### Inference

| C | L | cuEquivariance | start | final | start/final | cuEquivariance/final | Anthropic/final |
|---|---|---|---|---|---|---|---|
| 128 | 128 | 0.073 | 0.051 | 0.044 | 1.15 | 1.64 | 1.17 |
| 128 | 256 | 0.260 | 0.166 | 0.146 | 1.14 | 1.78 | 1.16 |
| 128 | 384 | 0.708 | 0.395 | 0.331 | 1.19 | 2.14 | 1.21 |
| 128 | 512 | 1.411 | 0.688 | 0.628 | 1.10 | 2.25 | 1.15 |
| 128 | 640 | 2.453 | 1.162 | 1.056 | 1.10 | 2.32 | 1.10 |
| 128 | 768 | 3.991 | 1.982 | 1.658 | 1.20 | 2.41 | 1.05 |
| 256 | 128 | 0.121 | 0.093 | 0.088 | 1.06 | 1.38 | 0.86 |
| 256 | 256 | 0.535 | 0.328 | 0.325 | 1.01 | 1.65 | 0.95 |
| 256 | 384 | 1.383 | 0.763 | 0.743 | 1.03 | 1.86 | 1.02 |
| 256 | 512 | 2.766 | 1.406 | 1.407 | 1.00 | 1.97 | 0.99 |
| 256 | 640 | 4.839 | 2.422 | 2.399 | 1.01 | 2.02 | 0.97 |
| 256 | 768 | 7.885 | 4.050 | 3.690 | 1.10 | 2.14 | 0.94 |

C64, C384 and C512 keep the previous paths (0.95-1.06x of start; C64 L128 is 2 us slower because the core now makes two
launches). Anthropic's native path does not serve C64, C384 or C512 (see `anthropic_protocol_20261005.md`).
Against cuEquivariance the branch is 1.17-2.41x faster at every inference cell.

### Training (forward + backward, dropout 0.25)

Within +-3% of start at every cell (C128: +1-3%). Against cuEquivariance: 1.14-2.33x at every cell (C128 1.20-2.12x,
C256 1.14-2.22x). Anthropic ships no training path.

## Where the remaining gap is (C256 vs Anthropic)

Per-kernel time of the default path (profiler, C256): L128 front 27.7 us, tail 24.5 us, core 21+3 us; L768 core 1929+57 us
(56%), tail 658 us, cuBLAS projection 580 us, LN 204 us. Ablations of the core at C256 L768 (2290 us base): removing the
bias add -289, max tracking -424, ex2 -200, row sum -174, all three of bias/max/sum -659.

## Negative results (kept for the record)

- Weight re-streaming from L2 is not the front's limit: halving it (128-row tiles) changed C256 L768 only 1057 -> 1042 us.
- Draining the TMA store late (`tma_store_wait` removed entirely) changed nothing.
- Front ablation at C256 L768 (993 us): epilogue -390, LN -223, MMA -132, skeleton (weight TMA only) 227; the parts add up
  almost exactly, i.e. with one CTA per SM nothing overlaps. Large single-CTA tiles lose to small high-occupancy kernels.
- `tail_fused3` (128-row tile, shared weight ring, in-place residual): bit-exact at C<=128, 1-ulp differences at C>=256, but
  1.1-1.8x slower than the existing tail (130 vs 72 us at C128 L384; 1001 vs 651 us at C256 L768) for the same reason.
  Kept as `TAIL=fused3`, not selected by default.
- Row sums on the tensor core (P x ones, N=8 RS WGMMA): no gain (+1-2% slower); the extra WGMMA and its dependency cancel
  the removed FADD. Reverted.
- ptxas reports `C7515` (WGMMA serialization) only for the running-max core instantiations, caused by `out *= alpha` writing
  the accumulator; the max-free instantiations do not have it.

## Open

- C256: L128/256/640/768 are still 3-14% behind Anthropic. Candidates: bias folded into the QK accumulator (fp32 fragment
  staging, 13% of the core in the ablation), software-pipelining QK/softmax/PV across tiles, a small-footprint
  (occupancy-friendly) front and tail rather than the large tiles tried here.
- The safe second launch costs 3% of the core at C256 L768 (60 us) and 2 us at small L; folding the fallback into the same
  CTA would remove it.
- Fullgraph `torch.compile` qualification, master-weight (fp32) timing parity, `docs/gpus/h100/h100.md` status rows and a
  commit are not done.

## Reproducing

Scripts and logs: `runs/experiments/triattn_remeasure_20261007/` (`bench_a.sh`, `gate.sh`, `kprof.py`, `core_time.py`,
`f2time.py`, `tail_test.py`; a source snapshot is taken into `snap/`). Experiment hooks in `h100_triattn.py`:
`MW_F2_DEFS="-D..."` adds compile flags to `front_fused2/3` and `training_forward` builds (variant name is hashed into the
extension name) and `MW_F2_PTXAS=1` prints `ptxas -v`. Ablation macros: `MW_F3_ABL_{NOLN,NOMMA,NOEPI,NOSTOREWAIT}`,
`MW_CORE_ABL_{NOBIASADD,NOMAX,NOEXP,NOSUM}` (outputs are wrong by design).
