# TriangleAttention H100 — batched samples (B samples in one call) and memory chunking, 2026-10-20

Makes the H100 TriangleAttention CUDA paths a deliberate B-sample call (templates, structures or augments folded into the batch):
up to `max_batch` samples per launch, deterministic chunks beyond it or beyond memory, and the kernel scheduling that a long launch
needs. Branch `h100/triattn-batched` (off 545b6850, brought onto 765e3ac2). H100 80GB HBM3 (`total_memory` 85 018 279 936 B), torch
2.13 + cu129, bf16 pair, CUDA-graph timing (best of 3 x up to 50 replays after warm-up). One GPU per Slurm job. Scratch harnesses:
`~/miniworld-engine-scratch/wt-tabatch-logs/probes/` (`p1_launch_mem.py` memory / OOM, `p2_bench.py` timings, `p3_kernels.py` per-kernel
profile and the one-launch OOM check, `p4`-`p11` the scheduling and out-projection experiments below).

## What B > 1 did before

| variant | B > 1 before | now |
|---|---|---|
| fused path (d_pair 128, 4 x 32), starting / ending, inference / training | one launch per kernel over the B L^2 rows / (sample, head, tile) tasks, per-sample mask [B, L] and dropout scale [B, L, C]; no memory bound (OOM at e.g. B48 L768 training); the attention core's static round robin and the query-side backward got slower per sample with the launch size | the same kernels, `max_batch` = 48 per launch, chunks beyond (or beyond memory), dynamic core schedule, query-side backward in one-wave launches |
| width-generic path (d_pair 64-512, heads of 16 / 32) | the same, plus cuBLAS GEMMs over all B L^2 rows; the ending node's inference out-projection is one cublasLt GEMM per sample (strided batch over i: the layout has no single batch stride over (b, i)) | the same; the per-sample GEMMs kept (one GEMM + the transposed residual kernel measured slower: 1.04-1.10x, except d_pair 64 L128 0.93x) |
| b200style path (`integrations/triattn_h100.py`) | serves B = 1 only (`pair.shape[0] != 1` refused); its attention core loops over samples in Python. Reached only with `module._h100_cuda = False` (the sm90 path serves every shape it serves) | unchanged (benchmark baseline) |
| prebuilt `.so` composed paths (`qg_projection_attention`, `ln_backward`, `bias_backward`, ...) | `can_use` requires `pair.shape[0] == 1`; reached only when neither CUDA path serves | unchanged |
| bias-only variant, `BidirectionalTriangleAttention`, QK-norm, heads of 64+ channels | the PyTorch statements (batched torch ops), no CUDA path on H100 | unchanged |

Kernel launches per call, B = 1 vs B = 4 (torch.profiler, L256): the same kernel list on both paths (the generic path's cuBLAS picks a
split-K variant at some M). The batched inference output was bitwise equal to B one-sample calls (fused and generic, both nodes).

## Calls that ran out of memory in one launch (before)

`_forward_one` (the unchunked body), H100 80GB: d_pair 128 training L768 B48, d_pair 64 (4 x 16) training L768 B48, d_pair 256
inference L768 B48, d_pair 512 inference L768 B16, d_pair 256 training L768 B16 — all OOM. Now chunked (peak, chunk): 64.9 GiB (4
per launch), 64.8 GiB (6), 63.5 GiB (17), 54.4 GiB (10), 60.4 GiB (5), each matching B one-sample calls (y bitwise, dx <= 1e-5,
parameter gradients 2.4e-3 relative to the fp32 sum of the per-sample ones).

## Memory model and chunk rule

Peak of one launch (torch `max_memory_allocated` above the parameters, pair and dy included), bytes per pair token, linear in B
(B 1 / 2 / 4, L256 / 384):

| width | inference | training (dropout 0.25) |
|---|---|---|
| d_pair 64, 4 x 16 / 4 x 32 | 1544 | 4057-4114 |
| d_pair 64, 2 x 32 | 900 | 2497-2550 |
| d_pair 128 (fused) | 1800 | 3704-4056 |
| d_pair 128, 8 x 32 | 3368-3384 | 8086-8195 |
| d_pair 256 | 4136-4152 | 9891-10095 |
| d_pair 384 | 6208 | 14845-15132 |
| d_pair 512 | 8264 | 19773-20177 |

`_bytes_per_token` counts the tensors each path allocates (fused: q | k | v | g, o, sbias, lse, the backward's do, dg, dq, dk, dv,
delta, dbias and its slab sum; generic: y, statistics, q | k | v | g | bias, sbias, o, lse, u, the backward's ddrop, du, dall, do,
dyln, delta, dbias); against the table: fused +0 / +5 %, generic +0 / +25 % (cublasLt's tuning copy of the output on a shape's first
call is counted). `plan(kind, B, L, d_pair, heads, training, total_memory)`, budget = 0.85 x total - 1 GiB:

- inference: one launch when B (x + y + working set) fits, else chunks of (budget - B (x + y)) / working set;
- training: one launch when B (x, y, dy, dx + saved + backward) fits; else chunks that keep their saved tensors when every chunk's
  saved tensors fit next to one chunk's backward, else chunks whose forward is rerun in the backward (`checkpoint`);
- at most `max_batch` = 48 samples per launch (validated), and B Lp^2 < 2^31 (the int32 TMA row coordinate).

The chunk is a function of the shapes, widths, mode and the device's total memory only (never live free memory): the same split
under torch.compile and in a CUDA graph. Chunks run one after another through the same ops; inference chunks write their rows of
the output in place (the fused tail / the cublasLt out-projection take the rows as their output; a padded length, dropout or a traced
call copies), training is one autograd node (`_Chunked`) that copies each chunk's y and dx into the batch's y / dx and adds the
parameter gradients in fp32 (the chunks get the parameters as an exact fp32 cast, so their gradients come back in fp32): summing 24
bf16 chunk gradients in bf16 had been 1.05e-2 off the fp32 per-sample sum, now 2.4e-3, the same as one launch.

## Scheduling: what a long launch needed

**Attention core forward.** The persistent grid walked tasks in a static round robin (task = (b, h, 128-query tile) x 2 pair rows,
the query tile fastest so that the CTAs running together read the same K / V rows through L2). In a long launch the CTAs drift
apart and the sharing degrades: 48 samples at L768 in launches of G samples, us per sample: G 1-4 = 870-880, G 8-48 = 940-1010; L640
from G 16. A dynamic schedule (`TA_DYN`, default): the producer thread takes each next task from a global counter (zeroed per
call) and hands it to the consumers through a shared-memory slot written before the Q tiles' arrive (read after its completion; -1
ends the CTA). Same bits as the static schedule (out and LSE, B = 1 and B = 48, run to run), and flat in G:

| L | schedule | G1 | G4 | G8 | G16 | G48 |
|---|---|---|---|---|---|---|
| 768 | static | 895-902 | 870-876 | 885-895 | 940-954 | 940-947 |
| 768 | dynamic | 847-852 | 839-842 | 851-886 | 842-845 | 841-843 |
| 384 | static | 133-138 | 126-131 | 130-131 | 129-130 | 130-133 |
| 384 | dynamic | 133-134 | 124 | 123-124 | 122-123 | 123-124 |
| 128 | static | 12.5 | 8.5 | 7.8 | 7.2 | 7.0 |
| 128 | dynamic | 12.5-12.9 | 8.7-9.0 | 7.7 | 7.3 | 6.9 |

A launch with fewer than 16 tasks per CTA keeps the static order (the counter's memset costs more than it saves at L128 B4).

**Query-side backward.** Task = (b, h, query tile) x slab, each a whole tile's pair rows (its dbias partials reduce-added in order:
deterministic). Per sample it was 1.07-1.55x slower in one launch of 48 samples than in launches of a few, at every head count:
after the first wave, a tile's task starts out of step with the other query tiles of its (b, h), which then read the same K / V rows
from DRAM at different times. More slabs (8 tasks per CTA) were slower (L384 B48 379 vs 394 us at P1, L768 2944 vs 2883). Launches
of g = max(1, 96 // (H x L / 128)) samples (one wave of ~96 tasks; one sample = one launch as before), median of 7, us per sample:

| H | L | B | one launch | groups of g | ratio |
|---|---|---|---|---|---|
| 4 | 128 | 48 | 16 | 15 (g 24) | 1.07 |
| 4 | 256 | 48 | 106 | 84 (g 12) | 1.26 |
| 4 | 384 | 48 | 401 | 277 (g 8) | 1.45 |
| 4 | 512 | 48 | 909 | 716 (g 6) | 1.27 |
| 4 | 768 | 48 | 2882 | 2440 (g 4) | 1.18 |
| 8 | 384 | 48 | 834 | 558 (g 4) | 1.50 |
| 8 | 768 | 24 | 5911 | 4953 (g 2) | 1.19 |
| 16 | 384 | 48 | 1737 | 1118 (g 2) | 1.55 |
| 16 | 768 | 12 | 11802 | 9978 (g 1) | 1.18 |

The key-side backward was flat in the launch size (1217-1291 us at L768, 161-180 at L384).

## Before / after at B = 1 (CUDA graph, ms)

"Before" = the unchunked body on the static-schedule core (the code before this branch; same bits).

| d_pair, mode | L384 before | L384 after | L768 before | L768 after |
|---|---|---|---|---|
| 128 inference | 0.266 | 0.263 | 1.373 | 1.337 |
| 128 training | 1.093 | 1.090 | 6.096 | 6.060 |
| 256 inference | 0.608 | 0.610 | 3.072 | 3.058 |
| 256 training | 2.498 | 2.499 | 13.864 | 13.828 |

## B samples: one call vs B calls vs cuEquivariance vs PyTorch (CUDA graph, ms)

Starting node, key mask (20 %), training = forward + backward with dropout 0.25 (dx and every parameter gradient). "B calls" = B
replays of the one-sample graph (in training each a forward + backward: a model's graph cannot do that at large B, it holds every
sample's saved tensors). PyTorch = the module's PyTorch statements under `torch.compile`, captured; cuEquivariance = its
`triangle_attention` in the module. Chunk = samples per launch (`plan`), `ck` = forward rerun in the backward.

| d_pair, mode | L | B | chunk | one call | B calls | cuEquivariance | PyTorch | vs B calls | vs best other |
|---|---|---|---|---|---|---|---|---|---|
| 128 inference | 128 | 4 | 4 | 0.096 | 0.120 | 0.434 | 0.326 | 1.25x | 3.38x |
| 128 inference | 128 | 48 | 48 | 0.975 | 1.455 | 4.866 | 3.829 | 1.49x | 3.93x |
| 128 inference | 384 | 4 | 4 | 0.987 | 1.046 | 4.469 | 6.903 | 1.06x | 4.53x |
| 128 inference | 384 | 48 | 48 | 11.66 | 12.67 | 52.80 | 82.20 | 1.09x | 4.53x |
| 128 inference | 768 | 4 | 4 | 5.281 | 5.379 | 22.54 | 49.84 | 1.02x | 4.27x |
| 128 inference | 768 | 48 | 48 | 65.38 | 66.10 | 271.2 | OOM | 1.01x | 4.15x |
| 128 training | 128 | 4 | 4 | 0.415 | 0.688 | 1.615 | 0.953 | 1.66x | 2.29x |
| 128 training | 128 | 48 | 48 | 3.680 | 8.186 | 15.02 | 10.09 | 2.22x | 2.74x |
| 128 training | 384 | 4 | 4 | 3.850 | 4.359 | 16.10 | 15.57 | 1.13x | 4.05x |
| 128 training | 384 | 48 | 48 | 45.82 | 52.38 | 189.6 | OOM | 1.14x | 4.14x |
| 128 training | 768 | 4 | 4 | 24.26 | 24.42 | 91.96 | 105.4 | 1.01x | 3.79x |
| 128 training | 768 | 48 | 4 | 299.0 | 293.4 | OOM | OOM | 0.98x | - |
| 64 (4 x 16) inference | 128 | 4 | 4 | 0.090 | 0.116 | 0.510 | 0.252 | 1.29x | 2.80x |
| 64 (4 x 16) inference | 128 | 48 | 48 | 0.858 | 1.434 | 5.666 | 2.866 | 1.67x | 3.34x |
| 64 (4 x 16) inference | 384 | 4 | 4 | 0.899 | 1.005 | 3.419 | 6.203 | 1.12x | 3.81x |
| 64 (4 x 16) inference | 384 | 48 | 48 | 10.40 | 12.11 | 40.38 | 73.74 | 1.17x | 3.88x |
| 64 (4 x 16) inference | 768 | 4 | 4 | 4.896 | 4.991 | 18.72 | 46.94 | 1.02x | 3.82x |
| 64 (4 x 16) inference | 768 | 48 | 48 | 59.56 | 61.05 | 224.3 | OOM | 1.03x | 3.77x |
| 64 (4 x 16) training | 128 | 4 | 4 | 0.479 | 0.778 | 1.438 | 0.722 | 1.62x | 1.51x |
| 64 (4 x 16) training | 128 | 48 | 48 | 4.275 | 9.282 | 12.66 | 7.304 | 2.17x | 1.71x |
| 64 (4 x 16) training | 384 | 4 | 4 | 4.194 | 4.762 | 12.47 | 13.52 | 1.14x | 2.97x |
| 64 (4 x 16) training | 384 | 48 | 48 | 50.05 | 57.33 | 145.1 | OOM | 1.15x | 2.90x |
| 64 (4 x 16) training | 768 | 4 | 4 | 25.52 | 25.71 | 76.37 | 97.31 | 1.01x | 2.99x |
| 64 (4 x 16) training | 768 | 48 | 6 | 311.9 | 307.9 | OOM | OOM | 0.99x | - |
| 256 inference | 128 | 4 | 4 | 0.230 | 0.245 | 0.787 | 0.692 | 1.07x | 3.00x |
| 256 inference | 128 | 48 | 48 | 2.476 | 2.941 | 8.673 | 7.592 | 1.19x | 3.07x |
| 256 inference | 384 | 4 | 4 | 2.339 | 2.466 | 8.196 | 13.87 | 1.05x | 3.50x |
| 256 inference | 384 | 48 | 48 | 29.10 | 30.30 | 96.42 | 166.3 | 1.04x | 3.31x |
| 256 inference | 768 | 4 | 4 | 12.40 | 12.44 | 42.02 | 100.0 | 1.00x | 3.39x |
| 256 inference | 768 | 48 | 17 | 151.1 | 152.8 | OOM | OOM | 1.01x | - |
| 256 training | 128 | 4 | 4 | 0.944 | 1.345 | 2.819 | 1.884 | 1.43x | 2.00x |
| 256 training | 128 | 48 | 48 | 9.812 | 16.13 | 28.28 | 20.04 | 1.64x | 2.04x |
| 256 training | 384 | 4 | 4 | 9.333 | 10.02 | 30.90 | 31.07 | 1.07x | 3.31x |
| 256 training | 384 | 48 | 48 | 116.6 | 121.3 | OOM | OOM | 1.04x | - |
| 256 training | 768 | 4 | 4 | 55.87 | 55.75 | 186.5 | OOM | 1.00x | 3.34x |
| 256 training | 768 | 48 | 2 ck | 852.3 | 671.6 | OOM | OOM | 0.79x | - |
| 512 inference | 768 | 16 | 10 | 117.5 | 117.8 | OOM | - | 1.00x | - |

One call is at least as fast as B calls wherever the batch's saved tensors fit (0.98-2.22x); the chunked training cells at L768 B48
pay the chunk copies of y and dx (0.98-0.99x) or, where the saved tensors of 48 samples do not fit next to x / y / dy / dx (d_pair
256: 58 GB of caller tensors), the rerun forward (0.79x, what a model's graph would pay at that size). Before the scheduling changes
(same harness): d_pair 128 inference L768 B48 one call 69.2 ms (B calls 66.4), d_pair 64 68.1 (62.9), d_pair 256 training L384 B48 130.6
(121.2), d_pair 128 training L384 B48 50.9; chunked inference with copies: d_pair 256 L768 B48 162.7 ms.

Ending node, inference, one call vs B calls: d_pair 128 L768 B48 65.49 vs 66.35 ms; d_pair 64 L128 B48 1.017 vs 1.439, L768 B48 59.87
vs 61.43 (the per-sample out-projection GEMMs cost the ending node 18 % over the starting node at d_pair 64 L128 B48, 0-5 % elsewhere).

## Correctness

`tests/gpu/sm90/test_triattn_h100_sm90_batched_gpu.py` (315 cases, 252 in the grid): B in {2, 4, 8, 16, 32, 48} x L in {128, 384, 768} x d_pair 64
(4 x 16, 4 x 32, 2 x 32), 128, 256, 384, 512 x inference / training (245 run, 7 not: the test's own x / dy / y / dx above 40 GB),
every sample its own key mask and dropout scale, against B calls of one sample: y bitwise equal in every case, dx <= 3e-5 relative,
parameter gradients <= 2.8e-3 relative to the fp32 sum of the per-sample gradients; against the fp32 PyTorch statements (all samples
where the [B, H, L, L, L] logits fit, then also every parameter gradient; else the first and last sample) within 1.25x the bf16
statements' own error. Ending node (B 4 / 48, L128 / 384), padded L208, forced chunks (saved / rerun / inference, L256 and padded
L208: y bitwise to one launch), the five former OOM calls, bitwise run-to-run determinism of y and every gradient (B8 L384, d_pair
64 / 128 / 256), CUDA-graph capture and torch.compile with one launch and with chunks. The existing H100 TriAttn tests (82) pass;
`tests/cpu/kernels/test_triattn_h100_batch_plan.py` checks `plan` / `max_batch` without a GPU.
