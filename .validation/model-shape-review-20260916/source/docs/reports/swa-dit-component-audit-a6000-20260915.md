# SWA DiT component speed audit — A6000

## Findings

- The claim that every engine operation is faster is unsupported. Q/K norm+RoPE and SwiGLU improve both isolated training and whole-block time in this grid. Modulation is faster in isolation but slows the 3072-atom block and slightly improves the 4096-atom block.
- Residual gate is essentially tied in isolation and slows the full block. Attention sigmoid gate has only a small isolated advantage and also slows the block. The final combined result reflects these offsetting effects; isolated speedups must not be added.
- These conclusions concern BF16 training on this A6000 and this shape grid, not universal kernel rankings.

Memory follow-up: [isolated and block replacement peak-memory measurements](swa-dit-component-memory-a6000-20260915.md).

## Scope and controls

- Job `1698699`, `NVIDIA RTX A6000`, `GPU-20dbc787-efdc-b977-e3f0-c30466924996`, host `gpu01`; all measurements on one GPU allocation.
- Stable source hash: `ccc77e767aba44de90a687fd4c76e2b187fae1a5d5e26617d1f52a91b80cb94f`.
- Standard `benchmarks/runners/bench.py` timer, fullgraph compilation and CSV provenance. 33 separate benchmark processes, 132 successful rows. Each process pairs PyTorch and engine; 3 repetitions per case, with component order reversed in repetition 2.
- BF16 inputs and weights, TF32 disabled, N=A=48 (B=1), atom lengths3072/4096, D128, H4, head dimension32, SwiGLU hidden256. One block, half-window64, front-packed mask probability0.125, no dropout.
- Forward **plus backward** time, including input and parameter gradients and fresh gradients per step; no optimizer update. CUDA Graph OFF. Every measured forward uses one fullgraph-compiled executable with AOT backward.
- The five names denote complete differentiable operations, not individual low-level Triton launch timings. The shared FA2 attention core, dense QKV/output/gate projections and unchanged operations are held fixed.
- Modulation weights are initialized with normal std0.01 (active gates). The paired implementations receive identical initial weights, inputs and upstream gradients. These are active-gate results, separate from the earlier default-zero-gate report.
- Each row checks compiled output and all applicable gradients against the reference before timing. Full-block reference is the original production PyTorch SWADiTBlock, not the audit subclass. BF16 output tolerance: atol/rtol0.04; each gradient relative Frobenius error below0.03; finite gradients required.
- Largest observed relative output/gradient errors: 0.004160 / 0.008226.
- Q/K isolated inputs preserve their production noncontiguous views into packed QKV. The isolated modulation includes SiLU conditioning and one branch of RMSNorm/scale/shift/gate; its PyTorch reference has three projections. In the full block, the PyTorch reference combines six projections into one GEMM, so isolated and block effects need not agree.
- Engine cases use the shipped runtime dispatch/autotune policy (bounded fallback tuning on stale cache entries during warmup), not a newly rebuilt full tuned cache. No production dispatch change was made.
- Existing SWA benchmark contracts:25 passed. Ruff and project-environment type checks passed. H100 and other dtypes/shapes were not measured by this audit.

## Isolated operations

Speedup is the median of the three within-process PyTorch/engine time ratios; above1 means faster. Times are process medians in ms.

| Operation | Atoms | PyTorch ms | Engine ms | Speedup |
|---|---:|---:|---:|---:|
| RMSNorm + modulation | 3072 | 1.8724 | 1.7321 | 1.082x |
| RMSNorm + modulation | 4096 | 2.4730 | 1.9261 | 1.284x |
| Q/K norm + RoPE | 3072 | 0.7506 | 0.5796 | 1.295x |
| Q/K norm + RoPE | 4096 | 0.9974 | 0.7690 | 1.297x |
| SwiGLU FFN | 3072 | 2.2446 | 1.7654 | 1.273x |
| SwiGLU FFN | 4096 | 2.9932 | 2.3828 | 1.256x |
| Residual gate | 3072 | 0.5806 | 0.5816 | 0.998x |
| Residual gate | 4096 | 0.7711 | 0.7721 | 0.999x |
| Attention sigmoid gate | 3072 | 0.4270 | 0.4198 | 1.017x |
| Attention sigmoid gate | 4096 | 0.5652 | 0.5601 | 1.009x |

## Whole-block substitutions

Each row starts from the full PyTorch+FA2 block and replaces only the named operation (both branches for modulation/residual). Positive reduction means faster. Effects cannot be added: replacing a region changes which surrounding operations Inductor can fuse.

| Replacement | Atoms | PyTorch ms | Modified block ms | Latency reduction | Paired range |
|---|---:|---:|---:|---:|---:|
| RMSNorm + modulation | 3072 | 14.8081 | 15.2023 | -2.69% | -3.04…-2.66% |
| RMSNorm + modulation | 4096 | 19.4406 | 19.2481 | +0.98% | +0.96…+1.17% |
| Q/K norm + RoPE | 3072 | 14.7953 | 14.4737 | +2.16% | +2.12…+2.27% |
| Q/K norm + RoPE | 4096 | 19.4519 | 19.0740 | +1.94% | +1.89…+1.94% |
| SwiGLU FFN | 3072 | 14.7860 | 14.3432 | +2.98% | +2.95…+3.00% |
| SwiGLU FFN | 4096 | 19.4857 | 18.8232 | +3.37% | +3.36…+3.43% |
| Residual gate | 3072 | 14.7891 | 15.1649 | -2.51% | -2.54…-2.40% |
| Residual gate | 4096 | 19.4478 | 19.9757 | -2.71% | -2.75…-2.69% |
| Attention sigmoid gate | 3072 | 14.7845 | 14.8966 | -0.81% | -0.85…-0.68% |
| Attention sigmoid gate | 4096 | 19.4458 | 19.5779 | -0.68% | -0.68…-0.65% |
| All five | 3072 | 14.7994 | 14.7773 | +0.15% | -0.03…+0.30% |
| All five | 4096 | 19.5082 | 18.7832 | +3.72% | +3.61…+3.72% |

![Component audit](../../benchmarks/modules/swa_dit/artifacts/component_audit_20260915/component_audit.svg)

[All raw rows](../../benchmarks/modules/swa_dit/artifacts/component_audit_20260915/measurements.csv) · [Aggregates](../../benchmarks/modules/swa_dit/artifacts/component_audit_20260915/summary.csv) · [Paired comparisons](../../benchmarks/modules/swa_dit/artifacts/component_audit_20260915/comparisons.csv) · [Original CSV manifest](../../benchmarks/modules/swa_dit/artifacts/component_audit_20260915/raw_files.json)

## Reproduce

```bash
python benchmarks/runners/bench.py target=swa_dit level=module mode=training metric=time \
  precision=bf16-mixed compile=true cudagraph=disabled allow_tf32=false \
  mask_prob=0.125 n_layers=1 n_augment=48 min_seq_len=384 max_seq_len=512 seq_len_step=128 \
  'implementations=[pytorch,miniworld]' +swa_active_gates=true \
  +swa_component=rope '+swa_kernels=[rope]' name_suffix=unique_process_id
```

Use `+swa_component=block` for the single-substitution block measurement; choose from `modulation`, `rope`, `swiglu`, `residual`, `sigmoid_gate` for the component/list. Use all five in the list for the full engine replacement. Run each case in three fresh processes on one Slurm GPU allocation.
