"""Diagnose torch.compile on the trimul forward: WHERE it graph-breaks and
WHETHER reduce-overhead's cudagraph actually engages (vs skips) per region.

Goal: see if torch.compile alone (no manual capture) can cudagraph the whole
cute pipeline. Run with TORCH_LOGS=graph_breaks,cudagraphs. COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication,
    _load_cute_fns,
)


def _bench(fn, *, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"compile diag on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    tm1_cute_forward, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(pair.reshape(b * l1 * l2, d), mod.ln_pair.weight,
                                 mod.ln_pair.bias, eps=mod.ln_pair.eps, layout="nd->nd")
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def _ln_out(tri, b, l1, l2, d):
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
                                  eps=mod.ln_out.eps, layout="dbn->bnd")
        return (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)

    def v2_raw(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, None,
            bdll_direct=True, compute_gate=False)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out(tri, b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    L = 512
    pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)

    # --- 1. WHERE does Dynamo break? ---
    print("\n=== torch._dynamo.explain (v2_raw, L=512) ===", flush=True)
    exp = torch._dynamo.explain(v2_raw)(pair)
    print(f"graphs={exp.graph_count}  graph_breaks={exp.graph_break_count}", flush=True)
    for i, r in enumerate(exp.break_reasons):
        reason = getattr(r, "reason", str(r))
        stack = getattr(r, "user_stack", None)
        where = ""
        if stack:
            fr = stack[-1]
            where = f"{getattr(fr, 'name', '')} @ {getattr(fr, 'filename', '')}:{getattr(fr, 'lineno', '')}"
        print(f"  break {i}: {reason}  [{where}]", flush=True)
    torch._dynamo.reset()

    # --- 2. Does cudagraph engage? (watch TORCH_LOGS=cudagraphs output below) ---
    print("\n=== compile + run (cudagraph engage/skip logs above/below) ===", flush=True)
    cfn = torch.compile(v2_raw, mode="reduce-overhead")
    with torch.no_grad():
        for _ in range(6):
            cfn(pair)
        t_eager = _bench(lambda: v2_raw(pair))
        t_comp = _bench(lambda: cfn(pair))
    print(f"\nL={L}  v2 eager={t_eager:.3f}ms  compile-RO={t_comp:.3f}ms  "
          f"speedup={t_eager / t_comp:.2f}x", flush=True)


if __name__ == "__main__":
    main()
