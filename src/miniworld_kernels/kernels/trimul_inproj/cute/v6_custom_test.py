"""Verify the v6 custom op: grads vs fp32 ref, graph-break count under compile, timing."""

from __future__ import annotations

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
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand
from miniworld_kernels.kernels.trimul_inproj.cute.v6_custom_op import v6_custom
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

EPS = 1e-5


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


class V6CustomMod(nn.Module):
    def __init__(self, base, direction="out"):
        super().__init__()
        b = base
        self.WL = nn.Parameter(b.to_left.weight.t().contiguous())
        self.WLg = nn.Parameter(b.to_left_gate.weight.t().contiguous())
        self.WR = nn.Parameter(b.to_right.weight.t().contiguous())
        self.WRg = nn.Parameter(b.to_right_gate.weight.t().contiguous())
        self.Wg = nn.Parameter(b.to_gate.weight.t().contiguous())
        self.Wp = nn.Parameter(b.to_out.weight.detach().clone())
        self.lin_w = nn.Parameter(b.ln_pair.weight.detach().clone())
        self.lin_b = nn.Parameter(b.ln_pair.bias.detach().clone())
        self.lout_w = nn.Parameter(b.ln_out.weight.detach().clone())
        self.lout_b = nn.Parameter(b.ln_out.bias.detach().clone())
        self.eps, self.direction = b.ln_pair.eps, direction

    def forward(self, pair):
        b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)
        return v6_custom(pair, self.WL, self.WLg, self.WR, self.WRg, self.Wg, self.Wp,
                         self.lin_w, self.lin_b, self.lout_w, self.lout_b, b_lr, self.eps,
                         self.direction)


def main():
    assert torch.cuda.is_available()
    print(f"v6 custom-op test on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16
    for D in (128, 256):
        torch.manual_seed(0)
        base = TriangleMultiplication(d_pair=D, outgoing=True,
                                      implementation=ImplementationType.PYTORCH).cuda()
        for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                    base.to_gate, base.to_out):
            nn.init.normal_(lin.weight, std=D**-0.5)
        import copy
        base_fp = copy.deepcopy(base).float()
        mod = V6CustomMod(base.to(dt), "out")

        L = 512
        pair = torch.randn(1, L, L, D, device="cuda")
        dy = torch.randn_like(pair)
        # fp32 ref grads (full backward)
        base_fp.zero_grad(set_to_none=True)
        xr = pair.float().clone().requires_grad_(True)
        base_fp(xr).backward(dy)
        # custom op grads
        mod.zero_grad(set_to_none=True)
        xo = pair.to(dt).clone().requires_grad_(True)
        yo = mod(xo)
        yo.backward(dy.to(dt))
        c = dict(y=cos(base_fp(xr), yo), dx=cos(xr.grad, xo.grad),
                 dWL=cos(base_fp.to_left.weight.grad.t(), mod.WL.grad),
                 dWp=cos(base_fp.to_out.weight.grad, mod.Wp.grad),
                 dLNin=cos(base_fp.ln_pair.weight.grad, mod.lin_w.grad),
                 dLNout=cos(base_fp.ln_out.weight.grad, mod.lout_w.grad))
        ok = min(c.values()) > 0.99
        print(f"  D={D} correctness: " + " ".join(f"{k}={v:.5f}" for k, v in c.items())
              + f" -> {'PASS' if ok else 'FAIL'}", flush=True)

        # graph-break count under compile
        try:
            from torch._dynamo import explain
            exp = explain(lambda p: mod(p))(pair.to(dt).requires_grad_(True))
            print(f"  D={D} dynamo graph breaks = {exp.graph_break_count}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  D={D} explain n/a: {type(e).__name__}", flush=True)

        # timing: eager vs compiled fwd+bwd
        def bench(fn, p, g):
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
        p = torch.randn(1, L, L, D, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(p)
        te = bench(lambda x: mod(x), p, g)
        comp = torch.compile(mod)
        tc = bench(lambda x: comp(x), p, g)
        # head-to-head: V6TriMul (@torch.compiler.disable autograd.Function) in SAME harness
        from miniworld_kernels.kernels.trimul_inproj.cute.v6_training import V6TriMul
        dmod = V6TriMul(base.to(dt), "out")

        def dbench(fn):
            def step():
                p.grad = None
                for pr in dmod.parameters():
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
        td_e = dbench(lambda x: dmod(x))
        dcomp = torch.compile(dmod)
        td_c = dbench(lambda x: dcomp(x))
        print(f"  D={D} L={L} fwd+bwd ms:  custom_op eager={te:.3f} compiled={tc:.3f}  |  "
              f"disable eager={td_e:.3f} compiled={td_c:.3f}", flush=True)


if __name__ == "__main__":
    main()
