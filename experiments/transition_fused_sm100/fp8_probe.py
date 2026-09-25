"""Power-capped ceiling for fp8 (e4m3, per-tensor scaled) vs bf16 dense GEMM on this card: cuBLAS via torch._scaled_mm, graph
replay, bench (20 x 5) and sustained (200 x 5) regimes."""
import json, subprocess, torch
from common import graph_time
torch.backends.cuda.matmul.allow_tf32 = False
res = {}
one = torch.ones((), device="cuda")
for (m, n, k) in [(8192, 8192, 8192), (147456, 1024, 128), (147456, 128, 1024), (1024, 128, 147456)]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
    for tag, fn in (("bf16", lambda: torch.mm(a, b.t())),
                    ("fp8", lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))):
        r = {}
        for reg, reps in (("bench", 20), ("sustained", 200)):
            us = graph_time(fn, reps=reps, rounds=5)
            r[reg] = round(2 * m * n * k / us / 1e6, 1)
        res[f"{m}x{n}x{k} {tag}"] = r
        print(f"{m}x{n}x{k} {tag}: TFLOPS {r}", flush=True)
