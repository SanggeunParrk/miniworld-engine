"""Verify bidirectional trimul TRAINING (BidirV6TriMul): all grads vs fp32 ref + timing."""

from __future__ import annotations

import copy
import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src = _here
while _src.name != "src" and _src.parent != _src:
    _src = _src.parent
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import torch
import torch.nn as nn

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def bench(fn, mod, p, g):
    def step():
        p.grad = None
        for pr in mod.parameters():
            pr.grad = None
        fn(p).backward(g)
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(50):
        step()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 50


def main():
    assert torch.cuda.is_available()
    print(f"bidir-training test on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16
    for d_pair in (128, 256):
        h = d_pair
        torch.manual_seed(0)
        base = BidirectionalTriangleMultiplication(
            d_pair=d_pair, d_hidden=h, implementation=ImplementationType.PYTORCH).cuda()
        for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                    base.to_gate, base.to_out):
            nn.init.normal_(lin.weight, std=d_pair**-0.5)
        base_fp = copy.deepcopy(base).float()
        mod = BidirV6TriMul(base.to(dt))

        L = 512
        pair = torch.randn(1, L, L, d_pair, device="cuda")
        dy = torch.randn_like(pair)
        base_fp.zero_grad(set_to_none=True)
        xr = pair.float().clone().requires_grad_(True)
        base_fp(xr).backward(dy)
        mod.zero_grad(set_to_none=True)
        xo = pair.to(dt).clone().requires_grad_(True)
        yo = mod(xo)
        yo.backward(dy.to(dt))
        c = dict(y=cos(base_fp(xr), yo), dx=cos(xr.grad, xo.grad),
                 dWL=cos(base_fp.to_left.weight.grad.t(), mod.WL.grad),
                 dWR=cos(base_fp.to_right.weight.grad.t(), mod.WR.grad),
                 dWp=cos(base_fp.to_out.weight.grad, mod.Wp_nn.grad),
                 dWg=cos(base_fp.to_gate.weight.grad.t(), mod.Wg.grad),
                 dLNin=cos(base_fp.ln_pair.weight.grad, mod.ln_in_w.grad),
                 dLNout=cos(base_fp.ln_out.weight.grad, mod.ln_out_w.grad))
        ok = min(c.values()) > 0.99
        print(f"  d_pair={d_pair} h={h} (back K={2*h}) correctness: "
              + " ".join(f"{k}={v:.5f}" for k, v in c.items())
              + f" -> {'PASS' if ok else 'FAIL'}", flush=True)

        # timing: ours eager/compiled  vs  compiled pytorch ref (no eager baseline per HARD RULE)
        p = torch.randn(1, L, L, d_pair, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(p)
        te = bench(lambda x: mod(x), mod, p, g)
        tc = bench(lambda x: torch.compile(mod)(x), mod, p, g)
        pyb = base.to(dt)
        pyc = torch.compile(pyb)
        tp = bench(lambda x: pyc(x), pyb, p, g)
        print(f"  d_pair={d_pair} L={L} fwd+bwd ms:  ours eager={te:.3f} compiled={tc:.3f}  |  "
              f"pytorch compiled={tp:.3f}  (ours speedup {tp/tc:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
