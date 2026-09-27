"""SM clock / power while each stage of the op (and the whole op) replays back-to-back: python clock_probe.py bidir 768"""
import statistics
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402

var, L = sys.argv[1], int(sys.argv[2])
ext = TA.build()
mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH) if var == "bidir"
       else TriangleMultiplication(128, implementation=I.PYTORCH)).cuda().bfloat16().eval()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
mask = torch.rand(1, L, device="cuda") > .1
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
with torch.no_grad():
    pk = TA.pack(mod)
    ch, T = pk["ch"], L * L
    zf = z.reshape(T, 128)
    ab = torch.empty(2 * ch, T, device="cuda", dtype=torch.bfloat16)
    x = torch.empty(ch, L, L, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(zf)
    m = mask.reshape(L).to(torch.uint8)
    stages = {
        "k1": lambda: ext.k1(zf, m, pk["w1"], pk["g_in"], pk["b_in"], ab, L, pk["eps_in"], 0, torch.empty(0, device="cuda")),
        "contraction": lambda: TA.contract(ab[:ch].view(ch, L, L), ab[ch:].view(ch, L, L), x, pk),
        "k3": lambda: ext.k3(x.view(ch, T), zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, pk["eps_out"], 0, torch.empty(0, device="cuda"), torch.empty(0, device="cuda"), L),
    }
    stages["op"] = lambda: (stages["k1"](), stages["contraction"](), stages["k3"]())
    for name, fn in stages.items():
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
        t0 = time.time()
        n = 0
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        while time.time() - t0 < 2.0:
            g.replay()
            n += 10
        b.record()
        b.synchronize()
        stop.set()
        th.join()
        tail = samples[len(samples) // 3:]
        print(f"{name:12s} {a.elapsed_time(b) / n * 1000:8.1f} us  SM clock {statistics.median(s[0] for s in tail):.0f} MHz  "
              f"power {statistics.median(s[1] for s in tail):.0f} W", flush=True)
