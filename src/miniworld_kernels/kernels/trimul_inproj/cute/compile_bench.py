"""eager vs torch.compile vs manual CUDA graph, trimul forward (current vs v2).

torch.compile(mode="reduce-overhead") applies cudagraphs only to the segments
Dynamo can compile; it graph-breaks at the cute kernels (tm1/tm2/layer_norm_
transpose are plain calls into cute, not torch.library ops), so those launches
stay OUTSIDE a cudagraph. Manual stream-level capture grabs the WHOLE pipeline
(cute kernels included). This measures how much that matters. Fwd-only, B=1,
D=128, bf16. COMPUTE NODE only.
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
    print(f"trimul eager/compile/graph on {torch.cuda.get_device_name(0)}")
    _bdll_patch.apply()

    dtype = torch.bfloat16
    D = 128
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

    def current_raw(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right = tm1_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, out_layout="bdll_direct")
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out(tri, b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

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

    def timed_compile(fn, pair):
        cfn = torch.compile(fn, mode="reduce-overhead")
        for _ in range(5):  # trigger compile + cudagraph warmup outside timing
            cfn(pair)
        return _bench(lambda: cfn(pair))

    variants = (("current", current_raw), ("v2", v2_raw))
    print(f"\n{'L':>5} | {'variant':>8} | {'eager':>8} | {'compile-RO':>10} | "
          f"{'graph':>8} | {'e/compile':>9} | {'e/graph':>8}")
    print("-" * 74)
    for L in (384, 512, 768, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        for name, fn in variants:
            with torch.no_grad():
                t_e = _bench(lambda: fn(pair))
                try:
                    t_c = timed_compile(fn, pair)
                    ec = f"{t_e / t_c:.2f}x"
                except Exception as ex:  # noqa: BLE001
                    t_c, ec = float("nan"), f"FAIL:{type(ex).__name__}"
                try:
                    g = _make_graphed(fn, pair)
                    t_g = _bench(lambda: g.replay())
                    eg = f"{t_e / t_g:.2f}x"
                except Exception as ex:  # noqa: BLE001
                    t_g, eg = float("nan"), f"FAIL:{type(ex).__name__}"
            print(f"{L:>5} | {name:>8} | {t_e:>8.3f} | {t_c:>10.3f} | {t_g:>8.3f} | "
                  f"{ec:>9} | {eg:>8}")
        torch._dynamo.reset()  # avoid cudagraph pool buildup across shapes


if __name__ == "__main__":
    main()
