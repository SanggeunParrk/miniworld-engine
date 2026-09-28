# Wide Transition fusion on local H100

Local checkout is authoritative. Vast was terminated by the user. All jobs use
`h100 / cssb / normal_h100`, one H100, eight CPUs and 64 GB host memory. See
[`local-h100.md`](../../docs/operations/local-h100.md). No rented resources are used.

## Selected experiment

`selected.TransitionCandidate` is an explicit experimental module. It preserves
the current native forward and FP32-output weight-gradient GEMMs.
D256 retains its existing CUDA backward. D384 and D512 replace the scalar tail
after the cuBLAS input-gradient GEMM with `ln_residual._ln_residual_persistent`:

1. Read dXn, x, saved statistics and FP32 gamma.
2. Compute LayerNorm backward and add the residual gradient in the same kernel.
3. Accumulate FP32 dgamma/dbeta partials across multiple row tiles in each CTA.
4. Reduce the compact partials once, without global atomic accumulation.

The D384/D512 BF16 rounding points are preserved: dXn is BF16, the normalized
gradient rounds to BF16 before residual addition, then the final output rounds
again. All five parameter gradients are returned in their original dtype.

The removed intermediate is the pre-residual dX buffer. Logical traffic saved
is `4*M*D` bytes (one BF16 read and write), before accounting for partials/cache.
At L768 this is 864 MiB for D384 and 1152 MiB for D512. These are analytical
byte counts, not hardware-counter measurements. Partial allocations are fixed
at 1.55 MiB (D384) / 4.125 MiB (D512) on a 132-SM H100, independent of length.
Both selected L384 builds use 128 registers/thread, zero reported spills and
4096 bytes shared memory. No activation h is retained between F+B by default.

Production dispatch is unchanged. Import this class only from this experiment
using the wrapper's PYTHONPATH. Do not relabel it as an installed engine gain.

## Other fusion algorithms tested

| Candidate | Lifetime / traffic change | Observed outcome |
| --- | --- | --- |
| `gate_dx.py` | Own rows, keep dX accumulator across hidden tiles; consume dAB for dX on chip | D256 spills; F+B over 2x slower |
| `gate_dw.py` | Own hidden slice plus row partition; persistent dWa/dWb/dWs accumulators; no h materialization; dAB only for dX | D256 spills; F+B about 3x slower |
| `dx_ln.py` | dX GEMM + LN + residual, tile-sized affine partials | D512 spills despite small row tiles; slower |
| `native_dx_ln.py` | CUDA column split, bounded x prefetch, reused register lifetimes | Zero local memory, but D384/D512 throughput loses to cuBLAS + tail |
| `ln_residual.py` | Preserve GEMM; fuse scalar consumers and compact affine reduction | Selected only at D384/D512; D256 replacement regresses |

For `dx_ln.py`, the recorded Triton IR maps eight warps as `[8,1]` for WGMMA
and uses an `[16,256,16]` instruction tile: increasing warp count did not split
the wide output columns as intended. Resource measurements, not an assumption
that fusion must be faster, rejected these builds.

The native D384 TMA-staged output variant failed numerical validation (dX
relative error about 1.41) and is explicitly blocked by the wrapper. Its
non-staged variants are correct but slower. Shared-memory-infeasible native
configs are rejected before compilation. The historical CUDA source is copied
here for experiments; no production kernel was modified by this work.

## Reproduction and evidence

```sh
mkdir -p .bench/transition-wide-local
sbatch --partition=h100 --account=cssb --qos=normal_h100 \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:25:00 \
  --output=.bench/transition-wide-local/job-%j.log \
  experiments/transition_wide_fusion/run.sh qualify_all
```

Replace `qualify_all` with `profile_stages`, `sweep_gate_dx`, `sweep_gate_dw`,
`sweep_dx_ln`, `sweep_ln_residual`, `sweep_native_dx_ln` or `sanitize_all`.
Use `qualify_d128` to reproduce the D128 comparison rows at L384/L768 with
the unchanged native D128 module and the same full F+B measurement protocol.
Shape-specific scripts accept `--width` and `--length`; persistent LN sweeps
also need `--persistent`. Run these sequentially on an allocated GPU. The
wrapper refuses login-node execution and isolates all build/tuning caches.

`qualify.py` measures the actual modules under torch.compile and CUDA Graphs:
fresh forward plus input and all five parameter gradients. The comparison is
the current native engine and compiled PyTorch, not historical Vast timing.
Each result records source hashes, Slurm job/QoS/node, all event samples,
gradient errors, changed-state replay errors and actual CUDA kernel names.

Full-module memcheck runs at L768 for D384/D512. Racecheck and synccheck cover
the newly changed epilogue kernels in isolation at L768, including changed
inputs and graph replay. They do not qualify every unchanged vendor/native
kernel for full-module racecheck. Read `VALIDATION.json` for completed gates.

Results and retained failure logs: `.bench/transition-wide-local/`.
The initial qualification harness failures were a missing custom-op type
annotation, an old autograd graph retained across capture streams, and an
incorrect comparison of eager PyTorch against compiled PyTorch, and two affine
gradient views sharing storage across a custom-op output boundary. The corrected
harness releases autograd graphs and compares replay to a fresh call of the
same compiled program, and returns independently owned affine outputs.
Numerical thresholds were not loosened.

At D512/L768, the first sanitizer harness retained its CUDA Graph pool while
allocating another full F+B reference, causing OOM after successful replay.
The final harness computes initial and changed-state reference outputs before
capture and holds them on the CPU. It then restores the exact original
inputs/weights, captures, and replays after applying the identical mutation.
Full shape, all gradients, and changed-state replay
checks are retained. Allocation/capture failure logs remain alongside the
final sanitizer logs with their job IDs in the filenames.
