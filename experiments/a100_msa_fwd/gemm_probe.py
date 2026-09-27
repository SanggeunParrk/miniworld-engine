"""GEMM1 of the OPM (O = a^T b, a, b [S, 32 L]) under each operand layout cuBLAS could see, CUDA-graph median, with SM clock."""
import statistics
import subprocess
import sys

import torch

S = 1024
for L in (384, 768):
    n = 32 * L
    a = torch.randn(S, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(S, n, device="cuda", dtype=torch.bfloat16)
    a[torch.rand(S, n, device="cuda") < .1] = 0
    at, bt = a.t().contiguous(), b.t().contiguous()
    o = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    cases = {"mm(a.t(), b)  [S,n] inputs": lambda: torch.mm(a.t(), b, out=o),
             "mm(at, bt.t()) [n,S] inputs": lambda: torch.mm(at, bt.t(), out=o),
             "mm(b.t(), a) -> O^T": lambda: torch.mm(b.t(), a, out=o),
             "mm(bt, at.t()) -> O^T": lambda: torch.mm(bt, at.t(), out=o)}
    for name, fn in cases.items():
        for _ in range(3):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(int(300 / 5)):
            g.replay()
        r = []
        for _ in range(7):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(20):
                g.replay()
            e.record()
            clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            e.synchronize()
            r.append(s.elapsed_time(e) / 20)
        ms = statistics.median(r)
        print(f"L{L} {name:30s} {ms*1000:8.1f} us  {2*n*n*S/ms/1e9:6.1f} TF  clk,W {clk}", flush=True)
    sys.stdout.flush()
