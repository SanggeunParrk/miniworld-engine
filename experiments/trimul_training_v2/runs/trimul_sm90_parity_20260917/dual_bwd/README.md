# TriMul B9+B10: SM90 parity implementation

Source: `engine/src/miniworld_engine/kernels/trimul_inproj/cute/parity_dual_bwd.py`.

## Contract

The two reductions are independent, as in Triton:

```
gate_grad = bf16(fp32_reduce(G @ W))
out = bf16(fp32_reduce(F @ V) + fp32(gate_grad))
```

Inputs retain production layouts: G row-major; F=dconc.T column-major;
W=Wg.T column-major; V=W_stack row-major. Metadata-only transpose views form
TMA B operands. No contiguous transpose, cat, or output-sized temporary.
The original autograd function continues to own backward.

## Hardware implementation

- Explicit `CopyBulkTensorTileG2SOp` TMA loads.
- Explicit `warpgroup` WGMMA, K/K major for gate GEMM and MN/MN for front GEMM.
- Separate per-stage mbarriers and circular K buffers. Completed slots refill
  while later slots compute. `num_stages` controls actual slots and prefetch.
- Gate and front GEMMs reuse the same shared-memory allocation only after
  the gate WGMMA has completed. Their accumulator fragments remain separate.
- Grouped CTA order matches Triton's last-partial-group calculation.
- Masked scalar output stores cover arbitrary M/N tails.

## Configuration contract

The native resolver imports the exact corresponding Triton CSV domain.
Physical implementation constraints are explicit: minimum M=64 and device shared-memory capacity. Four/eight warps are
independent of M64/M128: groups split M or N; one group may execute multiple
M64 atoms. Unsupported
candidates get reasons, never a renamed/substituted tile. No claim that all
Triton candidates can physically use a Hopper WGMMA instruction.

## Measurements

`bench.json`: current circular-buffer implementation, 12 tested configs per
shape, BF16, CUDA graph kernel-only timing; D=128, KG=256, KP=1024. The Triton
baseline used the runtime heuristic 24 candidates because its full cache was
missing. Compilation and autotuning are excluded. This is not a full-grid
winner comparison.

| L | Triton ms | Best of 12 CuTe ms | Triton / CuTe |
|---|---:|---:|---:|
|128|0.019580|0.023948|0.818|
|384|0.180554|0.182084|0.992|
|768|0.715165|0.652220|1.097|

All 36 measured outputs matched the selected Triton output exactly.
`batched-bench.json` is the slower initial version retained for comparison;
`batched_v0.py` is its source. Do not combine those timings with current source.

`check.py` exercises five representative configs: M64/128, N32/64/128,
K32/64/128, GROUP1/2/4/8, warps4/8, stages2/3/4, pitched/transposed operands
and M/N/K tails. Relative-L2 tolerance is 1e-4 (0.01%).

## Final evidence

- `memcheck.log`: ten selected cases pass, **ERROR SUMMARY: 0 errors**.
- `warp-check.log`: four additional cases for M64/eight warps and M128/four
  warps also pass memcheck with **0 errors**, exactly matching Triton.
  Includes M523, N48, KG40, KP72 with pitched/transposed inputs. This is not
  sanitizer coverage of the entire candidate grid.
- `ptx/`: generated SM90a PTX contains 15 `cp.async.bulk.tensor` instructions
  and 8 `wgmma.mma_async` instructions.
- `narrow-bench.json`: 21 additional configs at L384 (N32/64 and K128).
  All outputs exact vs Triton. Best 0.180688ms vs fresh Triton 0.179045ms,
  again near parity; not a meaningful measured win at that shape.
- `summary.json`: concise machine-readable evidence and source SHA-256.

Initial shell invocation lacked compute-sanitizer in PATH; `memcheck.log`
uses `/usr/local/cuda/bin/compute-sanitizer` explicitly and is the completed
successful memory-check run.


Portable GPU regressions: `engine/tests/numerics/test_trimul_sm90_dual_bwd_gpu.py` (14 cases; no GPU work during collection).

`warp-bench.json`: all twelve extra M64/eight-warp and M128/four-warp timings
passed exact output comparison. These configs did not improve the best earlier
candidates. Overall 69 measured shape/config outputs matched Triton exactly
(36 initial + 21 extra L384 + 12 independent warp configurations).
