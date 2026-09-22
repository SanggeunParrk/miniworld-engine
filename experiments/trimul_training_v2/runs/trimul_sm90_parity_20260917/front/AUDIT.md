# Bidirectional TriMul F2: SM90 parity implementation

## Reuse decision

Retain Quack `GemmSm90`'s TMA descriptors, explicit TMA global/shared copies,
WGMMA shared/shared mainloop, producer/consumer pipeline, and gated output stores.
Replace the old `MaskedGatedSm90` epilogue and configuration adapter.

The old epilogue computes `BF16(sigmoid(g_fp32)*p_fp32*mask_fp32)`.
Current Triton computes `BF16(BF16(sigmoid(g_fp32)*p_fp32)*mask_bf16)`.
These differ for fractional masks. New `ParityFrontSm90` preserves Triton's
intermediate rounding and stores the untouched raw gate/projection values in
BF16 for existing backward, including all interleaved channels.

The old adapter passes a two-dimensional tile and no explicit stage count to
Quack. New adapter supplies the real `(BLOCK_M1, 2*BLOCK_K_H2, BLOCK_K_D)`
physical GEMM tile, validates the actual CTA thread count, and overrides the
TMA pipeline stage count before shared-memory layouts/pipelines are built.
There is no relabeling of consumer warps as total CTA warps.

## Config domain and explicit infeasibility

The declared domain is read from the same `trimul_gemm_gate_mmajor_triton.csv`:
M=32/64/128, channel tile=16/32/64, K=16/32/64, warps=1/2/4/8,
stages=1/2/3/4/5/6/8/10 (864 tuples).

The implementation uses 8 physical warps at both M64 and M128: one consumer
warpgroup processes one or two M64 atoms, respectively. This replaces Quack's
default M128 two-consumer arrangement (12 total warps) without changing the tile.
M32 needs an alternate masked-atom implementation and is not supported here.
Other warp-count requests receive explicit rejection reasons; tiles and warp
counts are never silently changed. Before shared-memory checks there are 144
executable candidate tuples. This is an implementation-specific subset, not a
claim that all other tuples are intrinsically impossible on the hardware.

TMA row strides also require M and K multiples of 8 BF16 elements, and H2 even.
Unaligned shapes are explicitly rejected and should retain the existing Triton
path. Aligned non-tile-divisible shapes are supported (e.g. M144,K96,H2=40).

## Storage and scheduling

The GEMM issues all four projections through one interleaved packed weight
matrix, just as Triton. A 2D persistent tile schedule replaces Triton's explicit
per-CTA left/right channel loops; this is internal GEMM tiling, not a different
fusion boundary. Gate/mask and raw saved-value stores remain in that same kernel.

Outputs have the same logical `(1,H2,L,L)` shapes and contiguous strides. Left
and right now occupy disjoint halves of one allocation `(2H2,L*L)`; no `cat` or
copy is executed. Preactivations retain `(4H2,L*L)`. Views are made outside the
opaque boundary, which returns only packed output and preactivations.

## GPU validation

See `quick.log`, `validate.log`, and JSON results. First case M144/K96/H2=40,
fractional masks, stage2, K32/channel32 is bitwise equal to same-config Triton
for both output and raw saved preactivations.

### Single-stage mainloop correction

The inherited Quack loop keeps one WGMMA group in flight. With one shared-memory
stage it waits for the next tile before releasing the previous one, causing a
producer/consumer deadlock. `ParityFrontSm90.mma` adds the proper stage1 path:
wait TMA, issue WGMMA, wait completion, release the buffer, advance. The multi-stage
path remains inherited. All six GPU cases passed after this fix, including stage1;
all tested outputs and raw saved preactivations were bitwise identical to the
same-config Triton kernel. This was an implementation bug, not a hardware limit.

## Validation and initial component timing

`validate-m128.log` passes eight cases (M64/M128, K16/32/64, stages1/2/3/4,
non-tile-divisible M144/K96/H2=40, no/binary/fractional masks, saved tensors on/off).
All front outputs and saved preactivations are bitwise equal to the corresponding
Triton kernel. The maximum FP32-PyTorch reference relative error is below1e-4.

The first M64-only sweep measured71 valid configurations and rejected one for
shared-memory overflow (K64/channel64/stage10). On L384,K128,H2=256 it found
0.332338ms at M64/K64/channel64/stage6. The fixed Triton comparator
(M64/K64/channel64/8warps/stage4) measured0.266595ms; this first CuTe variant is
therefore slower. This compares against one explicit Triton config, not a claim
about its fully tuned winner. M128 tuning is recorded separately.

Static fullgraph compile and manual CUDA graph capture/replay passed both the
explicit-config opaque front and the cache-selected front.

### Final M128 and safety evidence

M128 uses **256 physical threads (8warps)** and passes the same rounding/saved
contract. Across M64+M128, 144 candidate tuples were attempted: **140 passed
correctness and timing, 4 exceeded shared-memory capacity**. No measured candidate
changed the raw saved values or output relative to the corresponding Triton test.
M128's best (K64/channel64/stage2) was **0.283832ms**, versus fixed Triton
**0.266530ms**: **6.5% slower** on this component workload. This is experimental,
not grounds to replace the faster production path.

`front/memcheck.log`: portable regression suite **6passed**, Compute Sanitizer
**ERROR SUMMARY:0errors**. Includes M64/M128, stage1, fractional masks, aligned
non-tile-divisible extents, no mask and inference without saved preactivations.

Generated SM90a PTX in `front/ptx/` contains actual
`cp.async.bulk.tensor.3d.shared::cluster.global` input loads,
`cp.async.bulk.tensor.3d.global.shared::cta` output stores,
`wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`, and
`.reqntid256,1,1`. See `front/evidence.json` for file hashes and config results.
