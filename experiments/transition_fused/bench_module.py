"""What the fused backward is worth at the module level.

Times `miniworld_engine.modules.Transition` (d_hidden = 128, n = 4, bf16, training) forward + backward as the engine runs it
today, then again with the backward replaced by this kernel, and checks the two agree. The forward is untouched: the patched
path calls the same LayerNorm, expand-SwiGLU and squeeze-residual kernels the engine's own training path does, so the only
difference measured is the backward.

  python bench_module.py --length 384 [--iters 50] [--save records/module-L384.json]

Run on a compute node; needs the engine importable (PYTHONPATH=<repo>/src), torch with CUDA and `cuda.bindings`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--d-hidden", type=int, default=128)
p.add_argument("--iters", type=int, default=50)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--dw-repl", type=int, default=8)
p.add_argument("--ctas", type=int, default=132)
p.add_argument("--cubin", default="")
p.add_argument("--cubin-fwd", default="")
p.add_argument("--save", default="")
a = p.parse_args()

from miniworld_engine.modules.exceptions import ImplementationType  # noqa: E402
from miniworld_engine.modules.transition import Transition  # noqa: E402
from miniworld_engine.kernels.transition.triton import residual as resmod  # noqa: E402

D, N = a.d_hidden, 4
H = N * D
M = a.length * a.length
dev = "cuda"
torch.manual_seed(2319)
mod = Transition(D, N, implementation=ImplementationType.MINIWORLD).to(dev).train()
# The module zero-initialises the squeeze weight (AF3 practice), and with W_s = 0 the backward is degenerate: dh = 0, so
# dW_a, dW_b, dgamma and dbeta are all exactly zero and any agreement check between two backends compares zeros. Give it a
# trained-looking value instead. The kernels do not branch on data, so this does not change what is timed.
with torch.no_grad():
    mod.squeeze.weight.normal_(0, (N * D) ** -0.5)
for prm in mod.parameters():
    if prm.dtype == torch.float32 and prm.dim() > 1:
        prm.data = prm.data.to(torch.bfloat16)
x0 = torch.randn(1, a.length, a.length, D, device=dev, dtype=torch.bfloat16)   # the pair activation's real rank: rows_of needs it unflattened
dy = torch.randn_like(x0)
print(f"Transition(d={D}, n={N}) backend {mod._backend} | M {M} = {a.length}^2 | fused path: {resmod.transition_residual.__module__}", flush=True)

# ---------------------------------------------------------------- the patched path: same forward kernels, our backward
NDW, NDX = 8 * a.dw_repl, a.ctas - 8 * a.dw_repl
cubin = Path(a.cubin) if a.cubin else HERE / "build" / f"transition_bwd_r{a.dw_repl}.cubin"
SMEM = 231424
kern = drv.Kernel(str(cubin), "transition_bwd_fused", SMEM)
kred = drv.Kernel(str(cubin), "reduce_partials", 0)
cubin_f = Path(a.cubin_fwd) if a.cubin_fwd else HERE / "build" / "transition_fwd.cubin"
kfwd = drv.Kernel(str(cubin_f), "transition_fwd_fused", SMEM) if cubin_f.is_file() else None
_ws = {}
_fw = {}


def _fused_forward(flat, gamma, beta, wa, wb, ws, eps):
    """The fused forward: one kernel for LayerNorm + expand + SwiGLU + squeeze + residual, emitting xn / rstd / c1 too."""
    m = flat.shape[0]
    st = _fw.get("st")
    if st is None or st["m"] != m:
        tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
        wst = ws.t().contiguous()
        st = dict(m=m, wst=wst,
                  out=torch.empty_like(flat), xn=torch.empty_like(flat),
                  rstd=torch.empty(m, device=flat.device, dtype=torch.float32),
                  c1=torch.empty(m, device=flat.device, dtype=torch.float32))
        st["maps"] = (tm(flat, [D, m], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]),
                      tm(wb, [D, H], D * 2, [64, 64]), tm(wst, [D, H], D * 2, [64, 64]),
                      tm(st["out"], [D, m], D * 2, [64, 64]))
        _fw["st"] = st
    kfwd((a.ctas, 1, 1), (256, 1, 1), *st["maps"], _f32(gamma), _f32(beta),
         st["xn"], st["out"], st["rstd"], st["c1"], int(m), int(m // 128), float(eps))
    return st["out"], st["xn"], st["rstd"], st["c1"]


_cast = {}


def _f32(t):
    """gamma as fp32, cached: a per-call cast is an allocation and a launch a real wiring would not repeat."""
    k = t.data_ptr()
    v = _cast.get(k)
    if v is None:
        v = t.float().contiguous()
        _cast[k] = v
    return v


def _fused_backward(x, xn, rstd, c1, gamma, wa, wb, ws, dy_flat):
    m = x.shape[0]
    key = (m, x.data_ptr())
    w = _ws.get("buf")
    if w is None or w[0].shape[0] != m:
        w = (torch.empty_like(x), torch.zeros(D, device=dev), torch.zeros(D, device=dev),
             torch.empty(NDW * 3 * 64 * D, device=dev, dtype=torch.float32),
             torch.empty(NDX * 8 * 256, device=dev, dtype=torch.float32),
             torch.empty_like(wa), torch.empty_like(wb), torch.empty_like(ws))
        _ws["buf"] = w
    dx, dg, db, partw, dgbw, dWa, dWb, dWs = w
    maps = _ws.get("maps")
    if maps is None or _ws.get("m") != m:
        tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
        maps = (tm(dy_flat, [D, m], D * 2, [64, 64]), tm(xn, [D, m], D * 2, [64, 64]), tm(x, [D, m], D * 2, [64, 64]),
                tm(ws, [H, D], H * 2, [64, 128]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]))
        _ws["maps"], _ws["m"] = maps, m
    kern((a.ctas, 1, 1), (256, 1, 1), *maps, rstd, c1, gamma, dx, dg, db, partw, dgbw, int(m), int(m // 128))
    kred(((3 * 8 * 64 * D + 256 + 255) // 256, 1, 1), (256, 1, 1), partw, dWa, dWb, dWs, dgbw, dg, db)
    return dx, dg, db, dWa, dWb, dWs


FUSE_FWD = [False]


class _FusedTransition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        from miniworld_engine.autotune.shape_key import both_key, rows_of
        from miniworld_engine.kernels.layernorm.triton.main import _ln_fwd
        from miniworld_engine.kernels.transition.triton.main import _expand_swiglu
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        if FUSE_FWD[0]:
            out, xn, rstd, c1 = _fused_forward(flat, gamma, beta, wa, wb, ws, eps)
        else:
            key = both_key(rows_of(shape))
            xn, mean, rstd = _ln_fwd(flat, gamma, beta, None, eps, False, key)  # the engine's own LayerNorm forward
            h = _expand_swiglu(xn, wa, wb, key)                                 # ... its expand + SwiGLU
            out = resmod.squeeze_residual(h, ws, flat, key)                     # ... its squeeze + residual
            c1 = (mean * rstd).contiguous()
        ctx.save_for_backward(flat, xn, rstd, c1, gamma, wa, wb, ws)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy_in):
        flat, xn, rstd, c1, gamma, wa, wb, ws = ctx.saved_tensors
        dyf = dy_in.reshape(-1, dy_in.shape[-1])
        if not dyf.is_contiguous():
            dyf = dyf.contiguous()
        dx, dg, db, dWa, dWb, dWs = _fused_backward(flat, xn, rstd, c1, _f32(gamma), wa, wb, ws, dyf)
        return dx.reshape(ctx.shape), dg, db, dWa, dWb, dWs, None


CALLS = [0]


def patched(x, gamma, beta, wa, wb, ws, eps):
    CALLS[0] += 1
    return _FusedTransition.apply(x, gamma, beta, wa, wb, ws, eps)


# ---------------------------------------------------------------- agreement, then timing
def step():
    x = x0.detach().clone().requires_grad_(True)
    mod.zero_grad(set_to_none=True)
    mod(x).backward(dy)
    return [x.grad] + [prm.grad for prm in mod.parameters()]


base_name = resmod.transition_residual
names = ["dx"] + [n for n, _ in mod.named_parameters()]
g_base = [t.detach().clone() for t in step()]
g_ctrl = [t.detach().clone() for t in step()]        # control: the engine path against itself, so its own run-to-run noise is visible
resmod.transition_residual = patched
g_new = [t.detach().clone() for t in step()]
g_both = None
if kfwd is not None:
    FUSE_FWD[0] = True
    g_both = [t.detach().clone() for t in step()]
    FUSE_FWD[0] = False
resmod.transition_residual = base_name
rec = {"length": a.length, "d_hidden": D, "M": M, "agreement": {}, "engine_self": {}, "patched_calls": None}
print("  patched forward invoked", CALLS[0], "time(s) during the agreement check", flush=True)
rel = lambda u, v: float((u.float() - v.float()).abs().norm() / v.float().norm().clamp_min(1e-20))
for n, u, c, v in zip(names, g_base, g_ctrl, g_new):
    rec["agreement"][n] = dict(rel_rms=rel(u, v), max_abs=float((u.float() - v.float()).abs().max()))
    rec["engine_self"][n] = rel(u, c)
    if g_both is not None:
        rec.setdefault("agreement_both", {})[n] = rel(dict(zip(names, g_both))[n], v)
    extra = "" if g_both is None else f"   both-fused vs fused-bwd {rec['agreement_both'][n]:.3e}"
    print(f"  {n:>18s}: fused vs engine {rec['agreement'][n]['rel_rms']:.3e}   (engine vs itself {rec['engine_self'][n]:.3e}){extra}", flush=True)


def time_ms(iters):
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        step()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


tags = ["engine", "fused-backward"] + (["fused-both"] if kfwd is not None else [])
runs = {t: [] for t in tags}
for r in range(a.rounds):
    for tag in (tags if r % 2 == 0 else tags[::-1]):
        resmod.transition_residual = base_name if tag == "engine" else patched
        FUSE_FWD[0] = tag == "fused-both"
        runs[tag].append(time_ms(a.iters))
resmod.transition_residual = base_name
FUSE_FWD[0] = False
rec["ms"] = {k: dict(median=statistics.median(v), min=min(v), samples=v) for k, v in runs.items()}
b = rec["ms"]["engine"]["median"]
rec["speedup"] = {k: b / v["median"] for k, v in rec["ms"].items() if k != "engine"}
for k, v in rec["ms"].items():
    print(f"RESULT {k:>16s}: {v['median'] * 1000:8.1f} us per fwd+bwd (min {v['min'] * 1000:.1f})", flush=True)
print("module speed-up " + "  ".join(f"{k} {v:.2f}x" for k, v in rec["speedup"].items()), flush=True)
if a.save:
    out = Path(a.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("saved", out, flush=True)
