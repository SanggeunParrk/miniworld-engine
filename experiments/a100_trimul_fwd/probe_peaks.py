"""Achievable ceilings of this A100 for the SoL denominators: streaming DRAM bandwidth (read / write / copy) and bf16
tensor throughput (cuBLAS large GEMM and the TriMul contraction shapes), with the SM clock and power sampled under load."""
import json
import statistics
import sys
import threading
import time

import pynvml
import torch

out = sys.argv[1] if len(sys.argv) > 1 else "records/peaks.json"
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
rec = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
           max_sm_clock=pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
           max_mem_clock=pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM),
           power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000)


def sampled(fn, seconds=2.0):
    """Run fn back-to-back for `seconds` while sampling clock/power; return (median ms per call, clock/power stats)."""
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append((pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), pynvml.nvmlDeviceGetPowerUsage(h) / 1000))
            time.sleep(0.02)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t0 = time.time()
    th = threading.Thread(target=sampler)
    th.start()
    times = []
    while time.time() - t0 < seconds:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 10)
    stop.set()
    th.join()
    tail = samples[len(samples) // 3:]
    return statistics.median(times[len(times) // 3:]), dict(sm_clock_median=statistics.median(s[0] for s in tail),
                                                            power_median_w=statistics.median(s[1] for s in tail))


n = 1 << 30                                    # 2 GiB of bf16
x = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
y = torch.empty_like(x)
ms, st = sampled(lambda: y.copy_(x))
rec["copy"] = dict(ms=ms, GBps=2 * 2 * n / ms / 1e6, **st)
ms, st = sampled(lambda: y.fill_(1.0))
rec["write"] = dict(ms=ms, GBps=2 * n / ms / 1e6, **st)
acc = torch.empty(1 << 15, dtype=torch.float32, device="cuda")
ms, st = sampled(lambda: torch.sum(x.view(-1, 1 << 15), dim=0, dtype=torch.float32, out=acc))
rec["read"] = dict(ms=ms, GBps=2 * n / ms / 1e6, **st)
del x, y

for M in (8192, 16384):
    a = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    ms, st = sampled(lambda: torch.matmul(a, b, out=c), 3.0)
    rec[f"gemm_{M}"] = dict(ms=ms, TFLOPs=2 * M ** 3 / ms / 1e9, **st)
    del a, b, c
for L, C in ((384, 256), (768, 256), (384, 128), (768, 128)):
    a = torch.randn(C, L, L, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(C, L, L, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    ms, st = sampled(lambda: torch.bmm(a.transpose(1, 2), b, out=c), 1.5)
    rec[f"contract_L{L}_C{C}"] = dict(ms=ms, TFLOPs=2 * C * L ** 3 / ms / 1e9, GBps=3 * C * L * L * 2 / ms / 1e6, **st)
print(json.dumps(rec, indent=1))
open(out, "w").write(json.dumps(rec, indent=1))
