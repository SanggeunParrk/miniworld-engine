# TriangleAttention matched comparison — 2026-09-27

Current production CUDA path is 1.61–1.69x faster than the composed Triton path, 2.57–3.07x faster than the cuEquivariance adapter, and 3.12–4.08x faster than the repository PyTorch reference in these measurements. No production implementation was changed.

## Conditions

H100 80GB; B1, D128, 4 heads, head dimension 32; BF16 activation/linear parameters, FP32 LayerNorm; training dropout 0.25; 10% token mask. Identical nonzero model state, input and output gradient across arms. Full module forward plus backward for input and all parameters, torch.compile(fullgraph=True), manual CUDA Graph, live RNG. Each shape/direction uses 90 interleaved event samples per backend on one GPU; medians below. GPU0 L384, GPU1 L768.

| L | Direction | Current CUDA ms | Triton ms | PyTorch ms | cuEquivariance ms | CUDA speedup over Triton |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 384 | starting | 1.250 | 2.015 | 3.939 | 3.239 | 1.61x |
| 384 | ending | 1.269 | 2.038 | 3.958 | 3.256 | 1.61x |
| 768 | starting | 6.359 | 10.716 | 25.944 | 19.501 | 1.69x |
| 768 | ending | 6.455 | 10.872 | 26.132 | 19.637 | 1.68x |

## Validation and scope

All 16 shape/direction/backend cases completed. Dropout-zero output and all-gradient comparisons against current engine had maximum relative L2 0.006875 (0.688%); every training capture had finite outputs/gradients and output-changing RNG replay. This is comparison validation, not new kernel promotion qualification.

Activity traces distinguish native qg_attention_fused, grouped_dkdv<8>, dq_tma from Triton _attn_fwd/_attn_bwd_dkdv/_attn_bwd_dq. cuEquivariance 0.10.0 executed cuDNN FMHA forward/backward. Local attention source hashes matched the recorded remote source after execution.

Triton comparison disables native training forward and five native front/backward fusions; common normalization/gate helpers remain. Default autotuning/cache policy is preserved (miss cap 24), not a new exhaustive tuning search. PyTorch is the repository einsum/softmax reference, not a best-possible SDPA/FlashAttention claim. No hardware counters or SOL measured.

Runtime: PyTorch 2.10.0+cu128, Triton 3.6.0, cuEquivariance ops 0.10.0. Earlier standalone absolute measurements differ slightly; all ratios here use this matched interleaved run.

## Reproduction and artifacts

- Harness: `experiments/triattn_compare_20260927/bench.py`; invocation `scripts/vast-sync.sh run 0 python experiments/triattn_compare_20260927/bench.py --length 384` and GPU1/L768.
- Completed managed execution handles: 63474 (GPU0), 18908 (GPU1), both exit 0.
- Local raw results: `.bench/vast-20260927/results/triattn-compare-20260927/compare-L{384,768}.json`. Contains all samples, script/source hashes, GPU UUID, correctness errors and kernel counts.
- Activity traces: same directory, `trace-L*-{True,False}-{engine,triton,pytorch,cueq}.json`.
- Remote logs: `/workspace/vast-results/triattn-compare-L{384,768}.log`.
- Summary: `experiments/triattn_compare_20260927/summary.csv`.
