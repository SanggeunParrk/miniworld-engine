"""Sustained energy per call (NVML counter, physical GPU 7) and average clock of the attention kernels (graph of 20 calls, ~3 s)."""
import ctypes, time, sys, torch
from common import make, H, D
from ops import Fwd2, Dqb, Dkv

nvml = ctypes.CDLL("libnvidia-ml.so.1"); nvml.nvmlInit_v2()
h = ctypes.c_void_p(); nvml.nvmlDeviceGetHandleByIndex_v2(7, ctypes.byref(h))
def energy_mj():
    e = ctypes.c_ulonglong(); nvml.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e)); return e.value
def clock():
    c = ctypes.c_uint(); nvml.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(c)); return c.value   # NVML_CLOCK_SM

def measure(fn, secs=3.0, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    for _ in range(20): g.replay()
    torch.cuda.synchronize()
    n = 0; e0 = energy_mj(); t0 = time.time(); clk = []
    while time.time() - t0 < secs:
        for _ in range(10): g.replay()
        clk.append(clock())
        n += 10 * reps
        torch.cuda.synchronize()
    dt = time.time() - t0; de = (energy_mj() - e0) / 1000.0
    return dt / n * 1e6, de / n * 1e3, de / dt, sum(clk) / len(clk)

for L in [int(x) for x in (sys.argv[1:] or ["384", "768"])]:
    A = 48
    q, k, v, bias = make(A, L)
    do = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
    frun, O, LSE = Fwd2("build/attn_fwd2.cubin").bind(q, k, v, bias); frun()
    Dd = torch.randn(A, H, L, device="cuda") * 0.1
    rq, *_ = Dqb("build/attn_dqb.cubin").bind(q, k, v, do, bias, LSE, Dd)
    rk, *_ = Dkv("build/attn_dkv.cubin").bind(q, k, v, do, bias.transpose(1, 2).contiguous(), LSE, Dd)
    for name, fn in (("fwd2", frun), ("dqb", rq), ("dkv", rk)):
        us, mj, w, c = measure(fn)
        print(f"L{L} {name:5s}: {us:7.1f} us/call  {mj:6.2f} mJ/call  {w:6.1f} W  SM clock {c:6.0f} MHz", flush=True)
