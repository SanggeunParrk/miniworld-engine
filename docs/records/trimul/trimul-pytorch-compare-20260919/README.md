# Bidirectional TriMul: PyTorch / current Triton / H100

2026-09-19, node02 H100 80GB, B1, D=hidden=128, BF16 activations/linear weights and FP32 norm affine. Official bidirectional module benchmark, identical nonzero parameters, input and token mask. All arms use static compile and manual CUDA Graph. No optimizer, loading or communication. This is a module comparison, not whole-model training.

The H100 arm explicitly selects front/f567/dual_bwd/out_ln_bwd over the Triton algorithm, matching the weekly closeout. It is not the untouched auto default. The experimental mapped B4 is excluded. In inference only forward-reachable overrides run; backward options do not add work.

## training

Training uses dropout=0.25 with fresh production RNG on every timed replay. Inference uses eval/dropout=0.

| L | PyTorch ms | Triton ms | H100 mix ms | Triton/PyTorch speedup | H100/PyTorch speedup | H100/Triton speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 4.012088 | 1.591376 | 1.538008 | 2.521x | 2.609x | 1.035x |
| 768 | 35.339760 | 6.143504 | 5.916752 | 5.752x | 5.973x | 1.038x |

## inference

Training uses dropout=0.25 with fresh production RNG on every timed replay. Inference uses eval/dropout=0.

| L | PyTorch ms | Triton ms | H100 mix ms | Triton/PyTorch speedup | H100/PyTorch speedup | H100/Triton speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 1.302688 | 0.425152 | 0.404768 | 3.064x | 3.218x | 1.050x |
| 768 | 11.659520 | 1.629008 | 1.512752 | 7.157x | 7.707x | 1.077x |

## Measurement and validation

- Training: two independent processes/captures per shape, 12 rotated backend rounds in each, median of 24 graph timing samples. Reverse capture order in the second process. Inference: two independent captures with reversed backend order.
- PyTorch is also compiled BF16, not an eager or FP32 baseline. FP32 norm parameters are retained across all backends. Every observed compiled forward has one graph.
- Training fixture input/upstream-gradient/mask/parameter hashes match across backends and repetitions. The official paired-dropout FP32-reference checks run before timing; all output/input-gradient relative errors are below 0.02.
- All 11 gradients are present and finite. RNG reset reproduces each backend output/gradients; leaving RNG advancing changes output and dx. Fresh-gradient overwrite and stable graph buffers are checked. Timed execution never resets RNG or substitutes a fixed mask.
- Cross-backend graph RNG masks need not match for PyTorch versus custom ops. Their numerical reference comparison uses the official paired-dropout check; graph reset checks test each backend independently. H100 versus Triton additionally passes paired output/all-gradient comparison.
- Measured F2/F567/B9+B10 and native B4 configs are pinned to the prior closeout manifests. Other cache misses retain repository bounded tuning; this is not an exhaustive config search. Source hashes and exact runners are included.
- Some old Inductor cache entries could not be loaded and were recompiled during warmup. All accepted rows have positive compile evidence; compilation is outside graph replay timings.

Raw data, scripts and logs: /home/psk6950/MiniWorld/runs/trimul_pytorch_compare_20260919.
