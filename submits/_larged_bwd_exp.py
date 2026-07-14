"""Large-d BACKWARD isolation (lean + instrumented). split bwd vs triton fused stacked bwd,
manual cuda.Event timing, per-phase prints so a hang is visible. Few shapes."""
import time
import torch

from miniworld_kernels.modules import Transition, ImplementationType as IT
from miniworld_kernels.modules import dispatch as D

dev = "cuda"; bf16 = torch.bfloat16
_o90p, _o90 = D.is_sm90plus, D.is_sm90
def log(*a): print(*a, flush=True)


def build(d):
    mw = Transition(d, implementation=IT.MINIWORLD).to(dev).to(bf16)
    with torch.no_grad():
        mw.squeeze.weight.normal_(0, 0.02)
    ref = Transition(d, implementation=IT.PYTORCH).to(dev).to(bf16)
    ref.load_state_dict(mw.state_dict())
    return mw, ref


def timed(mod, x4, g, n=10):
    # warmup (pays autotune/compile once)
    for _ in range(3):
        mod.zero_grad(set_to_none=True); x4.grad = None
        mod(x4).backward(g)
    torch.cuda.synchronize()
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    # fwd-only (grad on)
    s, e = ev(), ev(); torch.cuda.synchronize(); s.record()
    for _ in range(n): mod(x4)
    e.record(); torch.cuda.synchronize(); ms_f = s.elapsed_time(e)/n
    # full step
    s, e = ev(), ev(); torch.cuda.synchronize(); s.record()
    for _ in range(n):
        mod.zero_grad(set_to_none=True); x4.grad = None
        mod(x4).backward(g)
    e.record(); torch.cuda.synchronize(); ms_full = s.elapsed_time(e)/n
    return ms_f, ms_full, max(ms_full-ms_f, 0.0)


def gradcos(mod, ref, x4, g):
    mod.zero_grad(set_to_none=True); x4.grad = None
    mod(x4).backward(g)
    xg, wg = x4.grad.detach().clone(), mod.squeeze.weight.grad.detach().clone()
    xr = x4.detach().clone().requires_grad_(True)
    ref.zero_grad(set_to_none=True); ref(xr).backward(g)
    c = lambda a, b: torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()  # noqa: E731
    return c(xg, xr.grad), c(wg, ref.squeeze.weight.grad)


SHAPES = [(256, 512), (512, 512)]
log(f"{'shape':<13}{'route':<14}{'fwd':>9}{'bwd':>9}{'full':>9}{'dXcos':>8}{'dWcos':>8}")
log("-"*70)
for d, L in SHAPES:
    for route in ("split", "fused-triton"):
        t0 = time.time()
        if route == "split":
            D.is_sm90plus, D.is_sm90 = _o90p, _o90
        else:
            D.is_sm90plus = lambda dev: True; D.is_sm90 = lambda dev: False
        log(f"  [start d={d} L={L} {route}] building...")
        mw, ref = build(d)
        x4 = (torch.randn(1, L, L, d, device=dev, dtype=bf16)*0.1).requires_grad_(True)
        g = torch.randn(1, L, L, d, device=dev, dtype=bf16)
        log(f"  [d={d} L={L} {route}] warmup+autotune (may take mins)...")
        try:
            dxc, dwc = gradcos(mw, ref, x4, g)
            log(f"  [d={d} L={L} {route}] grad done (dX={dxc:.4f}), timing...")
            mf, mfull, mb = timed(mw, x4, g)
            log(f"d={d} L={L:<7}{route:<14}{mf:>9.3f}{mb:>9.3f}{mfull:>9.3f}{dxc:>8.4f}{dwc:>8.4f}  [{time.time()-t0:.0f}s]")
        except Exception as ex:  # noqa: BLE001
            log(f"d={d} L={L} {route}  ERROR {type(ex).__name__}: {str(ex)[:80]}")
        D.is_sm90plus, D.is_sm90 = _o90p, _o90
log("LARGED BWD EXP DONE")
