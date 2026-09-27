"""SM clock / power while the kernel replays back-to-back: python clock_probe.py 384 [-DFOO=1 ...]   (L < 0: -L rows directly)"""
import statistics
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.transition import Transition  # noqa: E402

L = int(sys.argv[1])
ext = TA.build(extra=sys.argv[2:])
mod, x = TA.fixture(L)
M = x.shape[0]
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
with torch.no_grad():
    pk = TA.pack(mod, TA.chunk(sys.argv[2:]))
    out = torch.empty(M, 128, device="cuda", dtype=torch.bfloat16)
    fn = lambda: TA.forward(ext, x, pk, out)  # noqa: E731
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append((pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), pynvml.nvmlDeviceGetPowerUsage(h) / 1000))
        time.sleep(0.01)


th = threading.Thread(target=sampler)
th.start()
t0, n = time.time(), 0
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
a.record()
while time.time() - t0 < 3.0:
    g.replay()
    n += 10
b.record()
b.synchronize()
stop.set()
th.join()
tail = samples[len(samples) // 3:]
us = a.elapsed_time(b) / n * 1000
clk = statistics.median(s[0] for s in tail)
print(f"L{L} M{M} {us:8.1f} us  SM clock {clk:.0f} MHz  power {statistics.median(s[1] for s in tail):.0f} W  "
      f"-> {6 * M * 128 * 512 / us / 1e6:.1f} TFLOP/s = {100 * 6 * M * 128 * 512 / us / 1e-6 / (108 * 2048 * clk * 1e6):.1f}% of HMMA peak at that clock")
