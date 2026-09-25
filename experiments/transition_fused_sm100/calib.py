"""Ceilings for the SoL accounting on this card: dense bf16 GEMM (cuBLAS) and >L2 streaming copy, CUDA-graph timed."""
import json, torch
from common import graph_time

torch.backends.cuda.matmul.allow_tf32 = False
res = {"device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(0).multi_processor_count}
for (m, n, k) in [(8192, 8192, 8192), (16384, 16384, 4096), (147456, 1024, 128), (147456, 128, 512), (589824, 1024, 128)]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    us = graph_time(lambda: torch.mm(a, b.t(), out=c), reps=10)
    res[f"gemm_{m}x{n}x{k}"] = {"us": us, "tflops": 2 * m * n * k / us / 1e6}
for mb in (512, 2048):
    n = mb * 2 ** 20 // 2
    s = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_(); d = torch.empty_like(s)
    us = graph_time(lambda: d.copy_(s), reps=10)
    res[f"copy_{mb}MB"] = {"us": us, "TBps": 2 * n * 2 / us / 1e6}
print(json.dumps(res, indent=1))
