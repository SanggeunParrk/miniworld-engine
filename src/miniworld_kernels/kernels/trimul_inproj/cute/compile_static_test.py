"""Does marking the input static close the compile-RO vs manual-graph gap?

In a fully torch.compiled model, this op's input is an intermediate activation
already in the cudagraph pool (no copy). Isolating the op forces cudagraph_trees
to copy the external input each call. mark_static_address simulates the
"input already static" case. If compile-RO(static) ~= manual graph, the gap was
purely the input copy. Fwd-only, B=1, D=128, bf16. COMPUTE NODE only.
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


def _make_graphed(fn, pair, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn(pair)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(pair)
    return g


def main():
    assert torch.cuda.is_available()
    print(f"compile static-input test on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    _t1, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

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

    print(f"\n{'L':>5} | {'eager':>7} | {'compile-RO':>10} | {'compile-static':>14} | "
          f"{'manual-graph':>12}")
    print("-" * 62)
    for L in (512, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        with torch.no_grad():
            t_e = _bench(lambda: v2_raw(pair))

            torch._dynamo.reset()
            c1 = torch.compile(v2_raw, mode="reduce-overhead")
            for _ in range(6):
                c1(pair)
            t_c1 = _bench(lambda: c1(pair))

            torch._dynamo.reset()
            pair_s = pair.clone()
            torch._dynamo.mark_static_address(pair_s)
            c2 = torch.compile(v2_raw, mode="reduce-overhead")
            for _ in range(6):
                c2(pair_s)
            t_c2 = _bench(lambda: c2(pair_s))

            g = _make_graphed(v2_raw, pair)
            t_g = _bench(lambda: g.replay())
        print(f"{L:>5} | {t_e:>7.3f} | {t_c1:>10.3f} | {t_c2:>14.3f} | {t_g:>12.3f}",
              flush=True)


if __name__ == "__main__":
    main()
