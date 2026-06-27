# Bidirectional trimul (fwd+bwd) — roofline vs H100 ceiling

H100 80GB HBM3: 989.4 TFLOPS bf16 dense peak, 3.35 TB/s. B=1, bf16, h=d_pair.
FLOPs(fwd+bwd) = 66·L²·d² + 12·d·L³ (contraction part = 12·d·L³, ~59% of FLOPs).
`contraction floor` = the 6 batched bmm (2 fwd + 4 bwd) any impl must run, timed in isolation.
Bench `cute/roofline_bench.py` (event-timed, torch.compile).

| d | L | ours ms | TFLOPS | %peak | contr floor ms | bmm TFLOPS | bmm %peak | contr/ours |
|---|---|---|---|---|---|---|---|---|
| 128 | 512  | 4.304  | 114 | 11.5% | 0.451 | 457 | 46.2% | 10% |
| 128 | 768  | 9.440  | 141 | 14.3% | 1.145 | 608 | 61.4% | 12% |
| 128 | 1024 | 16.961 | 164 | 16.6% | 2.679 | 616 | 62.2% | 16% |
| 256 | 512  | 7.882  | 196 | 19.8% | 0.882 | 468 | 47.3% | 11% |
| 256 | 768  | 17.797 | 222 | 22.4% | 2.629 | 529 | 53.5% | 15% |
| 256 | 1024 | 32.222 | 243 | 24.6% | 5.516 | 598 | 60.4% | 17% |
| 512 | 384  | 9.704  | 299 | 30.2% | 0.942 | 369 | 37.3% | 10% |
| 512 | 512  | 17.203 | 312 | 31.5% | 1.753 | 470 | 47.5% | 10% |

## Verdict
- **Memory/launch-bound, NOT compute-bound.** The contraction is ~59% of FLOPs but only
  **10–17% of runtime**; the other 83–90% is the front/back GEMMs + LN + gate + elementwise
  + their backward + kernel-launch + compile glue. Matmul is not the bottleneck.
- **vs real kernels (dtv1/cuequiv): SOTA** — ours wins L≥384 (1.27–1.28x dtv1, ~3x cuequiv at
  d=512) and achieves higher utilization than dtv1 at the same point. Done relative to the field.
- **vs hardware peak: 12–31%, headroom exists but NOT in the contraction** — the contraction
  alone caps at 37–62% peak (batched-bmm shape limit) and is only ~13% of time. The remaining
  2–3x is in fusing the surrounding memory-bound glue, which is exactly the small-L regime
  where ours still trails dtv1 (launch-overhead-bound). Next optimization target.

![apples-to-apples](bidir_train_latency.png)
