# TriMul: save all versus full activation recomputation

Node02 H10080GB; BF16; batch1; C128; outgoing/incoming hidden128 each; shared output LN; mask; dropout25%; residual.
Both routes execute identical B1-B12 CUDA/cuBLAS kernels, all eleven gradients, and live weight-layout conversions. Each timing is a single CUDA Graph replay; 3 blocks x200 interleaved samples per route. Excludes optimizer, RNG generation, compilation and CPU/autograd dispatch.

**Scope:** this is whole-module activation checkpointing. The no-save forward retains no intermediate activation. Backward first re-executes the saved forward, materializing all intermediates in HBM, then runs the unchanged backward. It is NOT an implementation that fuses recomputation into backward registers or eliminates backward HBM writes.

| L | Route | Forward ms | Backward including recompute ms | Joint forward+backward ms |
|---|---|---:|---:|---:|
| 384 | saved | 0.440 | 0.724 | 1.166 |
| 384 | recompute | 0.285 | 1.165 | 1.440 |

L384: forward 1.544x faster; joint training time increases 23.51%.

L384 saved intermediate activations: 719.6 MB ->0. Forward peak incremental allocation including output: 757.3 MB ->264.2 MB. This is NOT whole-training peak memory.

| 768 | saved | 1.741 | 2.886 | 4.659 |
| 768 | recompute | 1.095 | 4.641 | 5.774 |

L768: forward 1.590x faster; joint training time increases 23.93%.

L768 saved intermediate activations: 2878.3 MB ->0. Forward peak incremental allocation including output: 3029.3 MB ->1057.0 MB. This is NOT whole-training peak memory.

## Correctness and interpretation

- Forward output and all eleven gradients are bit-identical between save-all and recompute. CUDA Graph replay also passed after changing input, two weights, dy and dropout scale in place.
- New no-save K3 passed compute-sanitizer memcheck at L384 (0 errors) and racecheck at L768 (0 errors, 0 warnings). Each ran dropout25, zero dropout-scale, and changed-mask/input/weight cases. Kernel filter: regex=infer_k3; this is not a sanitizer audit of unchanged backward kernels.
- Forward K1 is the unmodified Anthropic inference body. No-save K3 preserves its fused input/output LN and TMA/WGMMA pipeline, with the training rounding/dropout/residual epilogue. Upstream revision f4f62fa6592ae4938d49b1757bea0cfeff9f468e; Apache-2.0.
- Saved forward uses the selected per-shape K1 and previously tuned K3. No-save K1/K3 use the shipped H256 tile shapes; this experiment is not exhaustive retuning.
- Recompute pays for the complete forward twice, including both triangular cuBLAS contractions. The second pass still writes intermediates needed by the existing backward. This is why eliminating retention does not eliminate total training HBM traffic.
- Recomputed output y from the second pass is unused; this baseline re-executes the full saved forward, not an early-stop or register-local rematerialization kernel.
- Memory measurements are actual PyTorch allocated bytes in eager forward after warmup, excluding shared inputs/weights and allocator reserve. No claim is made about full-model or backward peak memory.
- Speed-only winner here: save-all. Full checkpointing trades more compute for less forward-retained memory. A fused backward-recomputation kernel or a selective-save policy is a separate experiment.
- The experimental adapter is local; production dispatch is unchanged.

## Reproduction

```bash
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_all_recompute_20260920/bench.py --length 384
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_all_recompute_20260920/bench.py --length 768
```
Run inside an allocated node02 H100 job, one GPU per process.
