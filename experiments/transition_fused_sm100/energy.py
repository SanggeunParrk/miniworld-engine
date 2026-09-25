"""Energy per call (NVML total-energy counter, physical GPU 7) for the fused kernels and for cuBLAS doing dense bf16 work, in the
benchmark's regime (CUDA graph of 20 back-to-back calls, replayed for ~3 s). Reports J/call, W, and pJ per tensor FLOP."""
import ctypes, time, json, torch
from common import make_inputs
from fwd_op import FusedFwd
from bwd_op import FusedTrain

nvml = ctypes.CDLL("libnvidia-ml.so.1")
nvml.nvmlInit_v2()
h = ctypes.c_void_p()
nvml.nvmlDeviceGetHandleByIndex_v2(7, ctypes.byref(h))
def energy_mj():
    e = ctypes.c_ulonglong(); nvml.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e)); return e.value

def measure(fn, flop, secs=3.0, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    for _ in range(20): g.replay()          # settle the power state
    torch.cuda.synchronize()
    n = 0; e0 = energy_mj(); t0 = time.time()
    while time.time() - t0 < secs:
        for _ in range(10): g.replay()
        n += 10 * reps
        torch.cuda.synchronize()
    dt = time.time() - t0; de = (energy_mj() - e0) / 1000.0
    return {"us_per_call": dt / n * 1e6, "J_per_call": de / n, "W": de / dt, "pJ_per_flop": de / n / flop * 1e12, "TFLOPS": flop * n / dt / 1e12}

torch.backends.cuda.matmul.allow_tf32 = False
res = {}
import sys
ONLY_FWD = '--fwd' in sys.argv
L = 384; M = L * L; D, H = 128, 512
x, wa, wb, ws, g, b = make_inputs(L)
f = FusedFwd(); f.set_weights(wa, wb, ws)
run_i, *_ = f.bind(x, g, b, save=False)
res["fused_fwd_inference_L384"] = measure(run_i, 6 * M * D * H)
from fwd_op import FusedFwd2
for LL in (384, 768):
    xx, wa2, wb2, ws2, g2, b2 = make_inputs(LL)
    for tag, ff in (("v1", FusedFwd()), ("v2", FusedFwd2())):
        ff.set_weights(wa2, wb2, ws2)
        for save in (False, True):
            rr, *_ = ff.bind(xx, g2, b2, save=save)
            res[f"fwd_{tag}_{'train' if save else 'infer'}_L{LL}"] = measure(rr, 6 * LL * LL * D * H)
if ONLY_FWD:
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()}, indent=1)); raise SystemExit
step = FusedTrain(f).bind(x, g, b, torch.randn_like(x) * 0.1)
res["fused_training_step_L384"] = measure(step, 6 * M * D * H + 22 * M * D * H)
for (m, n, k) in [(8192, 8192, 8192), (147456, 1024, 128)]:
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); Bm = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    res[f"cublas_{m}x{n}x{k}"] = measure(lambda: torch.mm(A, Bm.t(), out=C), 2 * m * n * k, reps=5 if m == 8192 else 20)
e0 = energy_mj(); time.sleep(2.0); res["idle_W"] = (energy_mj() - e0) / 1000.0 / 2.0
print(json.dumps(res, indent=1))
