"""Attainable dense-bf16 ceiling on this card under its 1000 W power cap: cuBLAS in the benchmark's timing regime (graph replay,
20 x 5) and sustained (200 x 5), plus the SM clock nvidia-smi reports while sustained."""
import json, subprocess, torch
from common import graph_time
torch.backends.cuda.matmul.allow_tf32 = False
res = {}
for (m, n, k) in [(8192, 8192, 8192), (16384, 16384, 4096), (147456, 1024, 128), (147456, 128, 512), (589824, 1024, 128)]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.mm(a, b.t(), out=c)
    r = {}
    for tag, reps in (("bench", 20), ("sustained", 200)):
        us = graph_time(fn, reps=reps, rounds=5)
        r[tag] = {"us": us, "tflops": 2 * m * n * k / us / 1e6}
    r["clock_after"] = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader", "-i", "7"], capture_output=True, text=True).stdout.strip()
    res[f"{m}x{n}x{k}"] = r
print(json.dumps(res, indent=1))
