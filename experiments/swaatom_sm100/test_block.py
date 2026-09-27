"""The whole SWA atom block: H100 fused on its Triton path vs the same block with the sm_100a kernels installed (b200_block): output and
gradients against fp64, and inference / training latency."""
import argparse, copy, torch
from common import make, hoist_mod, block_ref, rel, graph_time, event_time, HW
import h100_fused as SF

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384, 768]); p.add_argument("--notime", action="store_true")
a = p.parse_args()
W = ("wqkv", "wg", "wo", "wu", "wd")


def grads(fn, q, cb, cos, sin, su, w, dy, dtype=None):
    q = q.detach().clone().requires_grad_(); cb = cb.detach().clone().requires_grad_()
    ws = {k: w[k].detach().clone().requires_grad_() for k in w}
    out = fn(q, cb, cos, sin, su, ws)
    out.backward(dy)
    return out.detach(), q.grad, cb.grad, {k: ws[k].grad for k in ws}


def fused(q, cb, cos, sin, su, ws):
    return SF.swa_block(q, SF.hoist_mod(cb, ws["wmod"]), cos, sin, su, *(ws[k] for k in W), 1, HW)


def ref64(q, cb, cos, sin, su, ws):
    mod = hoist_mod(cb.double(), ws["wmod"].double())
    return block_ref(q.double(), mod, cos, sin, su, *(ws[k].double() for k in W))


import b200_block
# ---- accuracy (A8, S1024): both paths vs fp64
q, cb, cos, sin, su, w = make(8, 1024)
dy = torch.randn_like(q)
wd64 = {k: v.double() for k, v in w.items()}
o64, gq64, gc64, gw64 = grads(ref64, q.double(), cb.double(), cos, sin, su, wd64, dy.double())
saved = dict(SF._CUDA)
for name in ("h100 fused (Triton)", "sm100 kernels"):
    if name == "sm100 kernels":
        b200_block.install()
    o, gq, gc, gw = grads(fused, q, cb, cos, sin, su, w, dy)
    worst = max((rel(gw[k], gw64[k]), k) for k in gw)
    print(f"{name:20s}: out {rel(o, o64):.2e}  dq {rel(gq, gq64):.2e}  dc {rel(gc, gc64):.2e}  worst dW {worst[1]} {worst[0]:.2e}", flush=True)
SF._CUDA.clear(); SF._CUDA.update(saved)
if a.notime:
    raise SystemExit
for L in a.lengths:
    S = 8 * L
    for name in ("h100fused", "sm100"):
        SF._CUDA.clear(); SF._CUDA.update(saved)
        SF._qkvg_fwd_cuda_ok = SF._ffn_fwd_cuda_ok = SF._ffn_bwd_cuda_ok = (lambda *x: False)
        SF.hoist_mod = hoist_mod
        if name == "sm100":
            b200_block.install()
        for mode in ("inference", "training"):
            A = 5 if mode == "inference" else 48
            q, cb, cos, sin, su, w = make(A, S)
            if mode == "inference":
                def step():
                    with torch.no_grad():
                        return fused(q, cb, cos, sin, su, w)
                t = graph_time(step)
                modc = SF.hoist_mod(cb, w["wmod"])                         # the modulation precomputed (hoisted per fold, as Anthropic's)
                def step2():
                    with torch.no_grad():
                        return SF.swa_block(q, modc, cos, sin, su, *(w[k] for k in W), 1, HW)
                print(f"L{L} (S={S}) {name:10s} {'inf-hoisted':9s} {graph_time(step2):10.1f} us", flush=True)
            else:
                q.requires_grad_(); cb.requires_grad_(); [w[k].requires_grad_() for k in w]
                dy = torch.randn_like(q)
                t = event_time(lambda: fused(q, cb, cos, sin, su, w).backward(dy))
            print(f"L{L} (S={S}) {name:10s} {mode:9s} {t:10.1f} us", flush=True)
