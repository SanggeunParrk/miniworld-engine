# TriangleAttention B200 — batched samples, any length, memory chunking, deterministic gradients, 2026-10-20

Brings the B200 (sm_100a) TriangleAttention paths to the contract H100 got in 59d910cf: B samples in one call by design, deterministic
chunks beyond `max_batch` or beyond memory, any length, and the scheduling a long launch needs. Worktree detached at 59d910cf. B200
(148 SMs, 191 503 138 816 B total memory), GPU 5 of the shared node (a ~2.3 GB utilisation dummy resident on every GPU), torch
2.13.0+cu129, cuEquivariance 0.12.0, sm_100a kernels built by nvcc 13.1. Probe scripts live outside the checkout
(`$W/triattn_scratch/triattn_probes/`: `p_b1_dump.py` / `p_cmp.py` bitwise and determinism dumps, `p_b1_time.py` B = 1 timing,
`p_mem.py` memory model, `p_sched.py` / `p_dyn.py` / `p_gsweep.py` core scheduling, `p_group.py` module-level grouping ablation,
`p_kprof.py` per-kernel profile, `p_bench.py` one call vs B calls vs cuEquivariance / PyTorch, `p_bcalls.py` gap check).

## What B > 1 did before

Both CUDA paths (fused d_pair 128 / 4 x 32 and the wide d_pair 64-512 path) already launched every kernel once over the B samples
(the front / tail / gate / head / weight-gradient kernels walk B L^2 rows, the core walks (sample, head, tile) tasks, mask [B, L] and
dropout [B, L, C] per sample), but: L had to be a multiple of 128 (other L ran the PyTorch statements), there was no memory bound
(one launch at any B), and the fused path's parameter gradients came from fp32 atomics (not the same bits run to run; main's own
gradients differ between two runs, e.g. d_pair 128 L384 dW_k 1e-5 relative).

## Integration (`integrations/triattn_b200.py`)

One `forward` for both paths; the launches are opaque custom ops (`triattn_b200_infer` / `_fwd` / `_bwd`) so the module compiles to
one graph (torch.compile fullgraph tested); `_TriAttnFn` (one op forward, one backward) and `_Chunked` (training chunks) follow the
H100 integration. `kind_of`, `max_batch(kind, length)` = 48 (B Lp^2 < 2^31), `plan(kind, B, L, d_pair, heads, training,
total_memory, ending)`. Any L >= 16: zero-padded to a multiple of 128 with the pad keys masked (padded training is exact: pad tokens
have zero gradient rows). The ending node runs the starting-node kernels on the transposed pair (one copy in, one out), as before.

## Memory model

Peak of one launch above the caller's tensors, bytes per pair token (`p_mem.py`, B 1 / 2 / 4, L256 / 384, starting node):

| width | inference | training (dropout 0.25) | model inference / training |
|---|---|---|---|
| d_pair 64, 4 x 16 | 1416 | 3649-3655 | 1416 / 3664 |
| d_pair 64, 2 x 32 | 772 | 2213-2227 | 772 / 2220 |
| d_pair 128 (fused) | 1544-1547 | 3146-3153 | 1544 / 3144 |
| d_pair 128, 8 x 32 | 2832 | 7036-7064 | 2832 / 7064 |
| d_pair 256 | 3088 | 8064-8115 | 3088 / 8088 |
| d_pair 384 | 4632 | 11974-12080 | 4632 / 12000 |
| d_pair 512 | 6176-6177 | 15876-16072 | 6176 / 15912 |

A padded length or the ending node adds 2c per token in inference and 4c in training (measured; both together the same as one).
Model error -1.0 / +2.3 % (the -1 % at d_pair 512 B = 1 L256: fixed allocations, under the 1 GiB slack). Budget 0.85 x total - 1 GiB.

## Scheduling of long launches

Core forward, per sample, isolated (profiler, `p_dyn.py`), static round robin vs a dynamic task counter (each CTA claiming its next
task from a global counter one task ahead, the slot handed to the consumers through a 4-deep shared ring; same bits):

| heads, L | G1 static / dyn | G4 | G8 | G16 | G48 |
|---|---|---|---|---|---|
| 4, 512 | 296 / 307 | 201 / 209 | 193 / 202 | 192 / 201 | 193 / 205 |
| 4, 640 | 391 / 406 | 370 / 388 | 367 / 386 | 367 / 385 | 399 / 392 |
| 4, 768 | 654 / 684 | 631 / 661 | 627 / 659 | 630 / 658 | 692 / 670 |
| 8, 768 | 1280 / 1334 | 1254 / 1317 | 1266 / 1327 | 1331 / 1334 | - |
| 4, 1536 | 4864 / 5128 | 4845 / 5111 | 4862 / 5111 | - | - |

The dynamic schedule costs 4-5 % per sample and wins only at G48 (drift); a first version that also put the handshake on the static
path made every launch 3-4 % slower (L512 B1 212.7 -> 221.8 us). Not kept. Instead the forward runs long launches (four pair rows
a task, L > 384) in groups of ~36 864 tasks (`fwd_group`). Sustained (`p_gsweep.py`, CUDA graph of the B-sample forward), us per
sample: 4 heads L768 G1 656 / G8 647 / G48 703; 8 heads L768 G2 1296 / G48 1480; 16 heads L768 G1 2607 / G16 2924. Module
(`p_group.py`, CUDA graph, bf16-mixed), grouped forward vs one launch: inference d_pair 128 L768 B48 41.6 vs 43.9 ms, L640 B48 25.8
vs 26.7, L512 B48 14.46 vs 14.61, d_pair 256 L768 B16 29.0 vs 29.5, d_pair 64 L768 B48 39.3 vs 40.3, d_pair 128 L1536 B8 46.0 vs
46.8; training +0.0-1.7 %. Grouping the two backward kernels the same way looked 6-10 % faster isolated (profiler with idle gaps)
and changed the module's training step by -0.5 to +0.1 %: not kept.

## Deterministic parameter gradients (fused path)

`tri_gate_bwd` (dW_o) and `tri_wgrad` (M_t = xhat^T d_t, column sums, dbias sums) store one fp32 partial per CTA (plain stores / TMA
stores instead of `red.global.add` / TMA reduce-add / atomicAdd); `tri_slot_sum` adds them in CTA order (32 float4 columns x 8 slot
lanes a block, lane sums added in lane order). Run to run (`p_b1_dump.py` x 2, 96 cases): y, dW_q|k|v|g|o bitwise on both paths; dx,
dgamma, dbeta, dW_b carry the query-side kernel's dbias reduce-add order (dx <= 1.1e-5 relative). Kernel time, d_pair 128 training
B = 1: L128 114.2 -> 115.7 us (slot sum 6.1, gate backward 16.6 -> 12.6, weight grads 16.9 -> 16.4), L384 665.4 -> 669.8 us (slot
sum 7.4). Fewer weight-gradient CTAs (2-8 tiles each, fewer partials) were slower. ptxas: `tri_wgrad` 104 B spill (main 120 B),
`tri_gate_bwd` / `tri_slot_sum` none.

The custom op's outputs may not alias each other: the backward ops return the stacked gradient tensors and `split_grads` takes them
apart outside the op (eight per-parameter clones had cost 1.6-14 us).

## B = 1 against main

y (and the forward's saved tensors) the same bits as main in all 96 cases at B = 1 / 2. CUDA graph, bf16-mixed, best of 2
interleaved runs (`p_b1_time.py`), every cell d_pair 64 (4 x 16) / 128 / 256 x L128-768 x inference / training x both nodes: within
1 % except d_pair 128 training at L128 / L256 (+1-4 %, the slot sum; graph timings at L128 move ~6 % run to run, kernel time +1.3 %)
and three single cells whose main runs themselves differed by 3-7 % (d_pair 64 L128 training ending, d_pair 256 L256 inference
ending).

## One call vs B calls vs cuEquivariance / PyTorch

Table and protocol in docs/gpus/b200/triattn/triattn.md (Batched samples · measurements). The cells where one call is slower than B
one-sample graph replays (d_pair 256 inference L768, 0.93-0.98x; chunked training 0.93-0.96x) are not kernel time: at L768 B48 the
kernels take 1806 us per sample against 1751 for one sample, and the same 48 one-sample calls captured in one graph (no idle gaps
between replays, so no clock recovery under the 1000 W cap) take 89.0 ms against the one call's 88.4 (`p_bcalls.py`).

## Correctness

`tests/gpu/sm100/test_triattn_b200_batched_gpu.py`: 529 passed, 3 not run (test tensors above 64 GB). Existing
`tests/gpu/sm100/test_triattn_b200_gpu.py` 57 passed (the shape-rejection case now expects L200 served), CPU
`tests/cpu/kernels/test_triattn_b200_batch_plan.py` 23 passed.

## Not done

- The ending node still transposes in and out (40-49 % over the starting node at d_pair 128 inference, 15-20 % in training at B = 1);
  H100 reads it through transposed tensor maps.
- Chunked training copies each chunk's y and dx into the batch's tensors (the main cost of the 0.93-0.96x cells).
- dbias stays an fp32 reduce-add in L2 order: dx / dgamma / dbeta / dW_b are not bitwise run to run (as on H100).
