# TriMul H100 D64 — batched samples (B samples in one call), 2026-10-19

Port of the B200 batched-samples D64 path (e1e272cd) to H100, so MiniWorld's template embedder runs its templates as one batch.
Branch `h100/trimul-batched-d64` off main ea7e8717. H100 80GB HBM3, torch 2.13.0+cu129, bf16 pair, FP32 parameters, CUDA-graph
timing (median of 5 x 20 replays after warm-up), one job, two GPUs (timing on one, tests on the other). Scratch harnesses:
`~/miniworld-engine-scratch/wt-batch-logs/` (`bench_stack.py`, `bench_b1.py`, `sweep_batch.py`, `prof_b1.py`).

## Design (what changed per kernel)

| stage | change |
|---|---|
| K1 (packaged inference K1, also the training front) | `K1Params.rows` = B L (0 = one sample, in the struct's alignment padding): the slow token axis covers every sample; the token mask reads `m[i] & m[s L + j]`; planes `[c, B L, L]` |
| contraction | planes folded to `[c B, L, L]` (a view): one cuBLAS strided-batched call per product, batch c x B |
| K3 (inference) | `K3Params.rows`: row bound and z / x / out maps over B L rows |
| `save_k3_d64` (training forward) | `RecomputeParams.rows`; the row-dropout scale `[B L, 64]` is read at row `s L + j` (base hoisted once per tile) |
| B1 (`d64_b1w_*`, `uni64_b1w_*`) | the dropout-scale TMA row is `s L + j` (`j0 = t % L + t / L^2 * L`) |
| B2, fin | none: row-wise maps over M = B L^2; the FP32 pair mask is `[B L, L]`; the per-CTA weight partials sum every sample in a fixed order |
| preparation (`mw_aux_d64_training_prepare`, `mw_aux_train_mask`) | `samples`: the FP32 pair mask per sample from a token mask `[B, L]` or pair mask `[B, L, L]` |
| integration | `MAX_BATCH = 48`; `serves` / `serves_inference` take B > 1 at D64 only (other widths keep one sample); `[B, L]` masks, `[B L, 64]` scales |

## Batch limit sweep (correctness)

B in {1, 2, 4, 8, 16, 32, 48} x L in {128, 256, 384, 768} x {bidirectional, outgoing, incoming} x {inference, training}: 168 / 168
cells pass, a mask and a dropout scale drawn per sample. In every cell y (and dx in training) are bitwise equal to B calls of one
sample and bitwise equal run to run; the weight gradients differ from the sum of B per-sample gradients only by the FP32 summation
order (largest: 2.1e-4 relative, training L768 B48). Training peak memory at L768 B48: 68 GB (bidirectional), 54 GB (single).
No kernel limit was hit: the only index bound is the int32 token index (B L^2 < 2^31, e.g. B < 2048 at L1024). `MAX_BATCH = 48`
is the largest batch validated.

## Template-stack block, B = 4 (ms)

MiniPairformer block = `BidirectionalTriangleMultiplication(64)` + `Transition(64)` (row dropout on in training; training = forward +
backward of x and every parameter). "PyTorch" = one call of 4 samples with the native batch off, which is what H100 ran for B > 1
before this change (TriMul's PyTorch statements; Transition was already CUDA on the flattened rows: `wide_d64_*` kernels).

| L | mode | one call | 4 calls | PyTorch | × vs 4 calls | × vs PyTorch |
|---|---|---|---|---|---|---|
| 128 | inference | 0.079 | 0.135 | 0.436 | 1.72 | 5.54 |
| 128 | training | 0.309 | 0.681 | 1.255 | 2.21 | 4.07 |
| 384 | inference | 0.601 | 0.695 | 3.832 | 1.16 | 6.37 |
| 384 | training | 2.130 | 2.922 | 10.228 | 1.37 | 4.80 |
| 768 | inference | 2.467 | 2.776 | 28.452 | 1.13 | 11.53 |
| 768 | training | 9.045 | 10.712 | 60.315 | 1.18 | 6.67 |

TriMul alone (`BidirectionalTriangleMultiplication(64)`), B = 4 (ms):

| L | mode | one call | 4 calls | PyTorch | × vs 4 calls | × vs PyTorch |
|---|---|---|---|---|---|---|
| 128 | inference | 0.056 | 0.101 | 0.415 | 1.82 | 7.45 |
| 128 | training | 0.203 | 0.504 | 1.155 | 2.48 | 5.69 |
| 384 | inference | 0.432 | 0.517 | 3.660 | 1.20 | 8.47 |
| 384 | training | 1.336 | 2.021 | 9.467 | 1.51 | 7.09 |
| 768 | inference | 1.786 | 2.046 | 27.866 | 1.15 | 15.60 |
| 768 | training | 5.794 | 7.463 | 57.353 | 1.29 | 9.90 |

## B = 1 against main (ms, two interleaved rounds, worktree / main)

| direction | L | inference | training |
|---|---|---|---|
| bidirectional | 128 | 0.0244-0.0245 / 0.0244-0.0247 | 0.1036-0.1039 / 0.1034-0.1043 |
| bidirectional | 384 | 0.1169-0.1170 / 0.1172-0.1173 | 0.3895-0.3913 / 0.3892-0.3913 |
| bidirectional | 768 | 0.4566-0.4568 / 0.4551-0.4566 | 1.4753-1.4765 / 1.4738-1.4790 |
| outgoing | 128 | 0.0181-0.0188 / 0.0182-0.0187 | 0.0711-0.0717 / 0.0712-0.0717 |
| outgoing | 384 | 0.0780-0.0792 / 0.0788-0.0801 | 0.2511-0.2541 / 0.2518-0.2539 |
| outgoing | 768 | 0.2919-0.2923 / 0.2928-0.2937 | 0.9015-0.9035 / 0.9021-0.9076 |
| incoming | 128 | 0.0182-0.0184 / 0.0178-0.0180 | 0.0715-0.0725 / 0.0714-0.0724 |
| incoming | 384 | 0.0810-0.0813 / 0.0811-0.0813 | 0.2569-0.2579 / 0.2573-0.2574 |
| incoming | 768 | 0.3017-0.3022 / 0.3013-0.3014 | 0.9021-0.9060 / 0.9020-0.9100 |

An intermediate version read the dropout-scale row as `(iw - iw % N + j)` per element in `save_k3_d64`'s epilogue: `save_k3` went
from 141 to 199 us (bidirectional L768) and B = 1 training was 4-6 % slower. Hoisting the sample base to once per tile (zero for one
sample) brought it to 138 us; the token-mask and preparation sample offsets are likewise skipped for one sample.

## Tests

`tests/gpu/sm90/test_h100_trimul_batch_gpu.py` 37 / 37; existing H100 TriMul tests (b200style 73, d64_b200gate 22, small_buckets 15,
single_h100 8, d64_training 23 + 2 skipped, uni_d64_training 30 + 4 skipped, hot_path 3, nograd_dropout 2, wide_inference 137,
uni_wide_inference 5, h100_module_wiring 34) pass. (A run with eleven test processes on one GPU OOMed in seven files next to a
59.6 GB process; rerun with the files spread over two GPUs, all pass.)
