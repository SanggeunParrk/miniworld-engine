"""Is FUSING the two trimul directions actually faster than running them SEPARATELY,
as the pairformer block actually does it?

team-gm pairformer (src/team_gm/modules/blocks/pairformer.py) applies trimul as TWO
SEQUENTIAL residual blocks, with rowwise dropout on each update:

    pair = pair + drop_row(tri_multi_outgoing(pair))
    pair = pair + drop_row(tri_multi_incoming(pair))   # incoming sees the UPDATED pair

So the two directions are sequentially dependent — incoming reads the outgoing-updated
pair. Computing both from the same `pair` and summing is WRONG. This bench models the
faithful sequential-residual-with-dropout block, and compares it against the FUSED
bidirectional block (both directions from one shared input, ONE residual):

    fused:  pair = pair + drop_row(bidirectional(pair))

NOTE the semantic gap: bidirectional CANNOT see the outgoing-updated pair (both come
from the same input) — it is a different model, not a drop-in. This bench answers the
SPEED question only; equal per-direction hidden width h = d_pair = 128.

HARD RULE: pytorch = torch.compile (no eager); ours = manual CUDA-graph. Timed in
TRAIN mode (dropout active, p=0.25 — the training-forward regime). cos checked in
EVAL mode (dropout off) so ours matches its pytorch counterpart. B=1, bf16, no mask.
COMPUTE NODE only, fresh QUACK_CACHE_DIR.
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

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.back_split import trimul_back_split
from miniworld_kernels.kernels.trimul_inproj.cute.bidirectional import bidirectional_trimul_ours
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

LS = [256, 512, 1024]
COLS = ["pytorch_sep", "pytorch_bidir", "ours_sep", "ours_bidir"]
D_PAIR = 128
P_DROP = 0.25


def drop_row(x, training):
    """AF3-style rowwise dropout: drops whole rows of the pair (mask shared over cols)."""
    if not training or P_DROP == 0:
        return x
    B, I, _, _ = x.shape
    keep = (torch.rand(B, I, 1, 1, device=x.device) > P_DROP).to(x.dtype) / (1 - P_DROP)
    return x * keep


def _bench(fn, *, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"   bench fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_compiled(fn, pair):
    try:
        for _ in range(10):
            fn(pair)
        torch.cuda.synchronize()
        return _bench(lambda: fn(pair))
    except Exception as e:  # noqa: BLE001
        print(f"   compiled fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_cudagraph(fn, pair):
    try:
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    fn(pair)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(pair)
        return _bench(g.replay)
    except Exception as e:  # noqa: BLE001
        print(f"   cudagraph fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


class SepBlock(nn.Module):
    """Faithful pairformer separate: sequential residuals, incoming sees updated pair."""

    def __init__(self, out_mod, in_mod):
        super().__init__()
        self.out_mod, self.in_mod = out_mod, in_mod

    def forward(self, pair):
        pair = pair + drop_row(self.out_mod(pair), self.training)
        pair = pair + drop_row(self.in_mod(pair), self.training)
        return pair


class BidirBlock(nn.Module):
    """Fused: one residual on the bidirectional update."""

    def __init__(self, bidir_mod):
        super().__init__()
        self.bidir_mod = bidir_mod

    def forward(self, pair):
        return pair + drop_row(self.bidir_mod(pair), self.training)


def _pack(mod):
    WL = mod.to_left.weight.T.contiguous()
    WLg = mod.to_left_gate.weight.T.contiguous()
    WR = mod.to_right.weight.T.contiguous()
    WRg = mod.to_right_gate.weight.T.contiguous()
    return dict(
        WL=WL, WLg=WLg, WR=WR, WRg=WRg,
        Wg=mod.to_gate.weight.T.contiguous(), Wp_nn=mod.to_out.weight.contiguous(),
        ln_in_w=mod.ln_pair.weight, ln_in_b=mod.ln_pair.bias,
        ln_out_w=mod.ln_out.weight, ln_out_b=mod.ln_out.bias,
        eps=mod.ln_pair.eps, b_lr=prepack_lr_operand(WL, WLg, WR, WRg),
    )


def single_dir_ours(pair, p, direction):
    B, L, _, d = pair.shape
    xn = triton_layernorm(pair.reshape(B * L * L, d), p["ln_in_w"], p["ln_in_b"],
                          p["eps"]).view(B, L, L, d)
    left, right, _ = trimul_inproj_cute_forward(
        xn, p["WL"], p["WLg"], p["WR"], p["WRg"], None,
        bdll_direct=True, compute_gate=False, b_lr=p["b_lr"])
    if direction == "out":
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
    else:
        tri = torch.einsum("bdki,bdkj->bdij", left, right)
    return trimul_back_split(tri, xn, p["Wp_nn"], p["Wg"], p["ln_out_w"], p["ln_out_b"], p["eps"])


def _seed(mod, seed):
    torch.manual_seed(seed)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D_PAIR**-0.5)
    return mod


def main():
    assert torch.cuda.is_available()
    print(f"bidir-vs-separate (faithful pairformer block) on {torch.cuda.get_device_name(0)} "
          f"| d_pair={D_PAIR} h={D_PAIR} dropout={P_DROP}", flush=True)
    print("separate = sequential residuals (incoming sees outgoing-updated pair) + drop_row",
          flush=True)
    print("regime: pytorch=torch.compile; ours=CUDA-graph; TIMED in train mode (dropout on)",
          flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16
    PY = ImplementationType.PYTORCH

    out_mod = _seed(TriangleMultiplication(d_pair=D_PAIR, outgoing=True, implementation=PY).cuda(), 0).to(dt)
    in_mod = _seed(TriangleMultiplication(d_pair=D_PAIR, outgoing=False, implementation=PY).cuda(), 1).to(dt)
    bidir_mod = BidirectionalTriangleMultiplication(d_pair=D_PAIR, implementation=PY).cuda()
    _seed(bidir_mod, 2)
    bidir_mod = bidir_mod.to(dt)

    p_out, p_in = _pack(out_mod), _pack(in_mod)
    h = D_PAIR

    bWL = bidir_mod.to_left.weight.T.contiguous()
    bWLg = bidir_mod.to_left_gate.weight.T.contiguous()
    bWR = bidir_mod.to_right.weight.T.contiguous()
    bWRg = bidir_mod.to_right_gate.weight.T.contiguous()
    bWg = bidir_mod.to_gate.weight.T.contiguous()
    bWp = bidir_mod.to_out.weight.contiguous()
    b_lr_bidir = prepack_lr_operand(bWL, bWLg, bWR, bWRg)

    sep_block = SepBlock(out_mod, in_mod).cuda()
    bidir_block = BidirBlock(bidir_mod).cuda()
    pyt_sep_c = torch.compile(sep_block, mode="reduce-overhead")
    pyt_bidir_c = torch.compile(bidir_block, mode="reduce-overhead")

    def ours_sep(pair, training):
        pair = pair + drop_row(single_dir_ours(pair, p_out, "out"), training)
        pair = pair + drop_row(single_dir_ours(pair, p_in, "in"), training)
        return pair

    def ours_bidir(pair, training):
        upd = bidirectional_trimul_ours(
            pair, bWL, bWLg, bWR, bWRg, bWg, bWp,
            bidir_mod.ln_pair.weight, bidir_mod.ln_pair.bias,
            bidir_mod.ln_out.weight, bidir_mod.ln_out.bias, bidir_mod.ln_pair.eps, b_lr_bidir, h)
        return pair + drop_row(upd, training)

    rows = {}
    for L in LS:
        pair = torch.randn(1, L, L, D_PAIR, device="cuda", dtype=dt)
        # correctness in EVAL mode (dropout off) — ours vs its pytorch counterpart
        sep_block.eval()
        bidir_block.eval()
        with torch.no_grad():
            cs = cos(ours_sep(pair, False), sep_block(pair))
            cb = cos(ours_bidir(pair, False), bidir_block(pair))
        print(f"   [L={L}] cos(eval): ours_sep={cs:.5f}  ours_bidir={cb:.5f}", flush=True)
        # timing in TRAIN mode (dropout active)
        sep_block.train()
        bidir_block.train()
        t = {}
        t["pytorch_sep"] = bench_compiled(pyt_sep_c, pair)
        t["pytorch_bidir"] = bench_compiled(pyt_bidir_c, pair)
        with torch.no_grad():
            t["ours_sep"] = bench_cudagraph(lambda p: ours_sep(p, True), pair)
            t["ours_bidir"] = bench_cudagraph(lambda p: ours_bidir(p, True), pair)
        rows[L] = t
        print(f"[L={L}] " + " ".join(f"{c}={t[c]:.3f}" for c in COLS), flush=True)
        del pair
        torch.cuda.empty_cache()

    print("\n=== fused vs separate (faithful block), ms/layer; fuse↑ = separate/fused ===")
    print(f"{'L':>5} | {'py_sep':>8} {'py_bidir':>9} {'fuse↑':>6} | "
          f"{'ours_sep':>9} {'ours_bidir':>11} {'fuse↑':>6}")
    print("-" * 70)
    for L in LS:
        r = rows[L]
        pf = r["pytorch_sep"] / r["pytorch_bidir"]
        of = r["ours_sep"] / r["ours_bidir"]
        print(f"{L:>5} | {r['pytorch_sep']:>8.3f} {r['pytorch_bidir']:>9.3f} {pf:>5.2f}x | "
              f"{r['ours_sep']:>9.3f} {r['ours_bidir']:>11.3f} {of:>5.2f}x")
    print("DATA " + ";".join(
        f"{L}=" + ",".join(f"{rows[L][c]:.4f}" for c in COLS) for L in LS), flush=True)


if __name__ == "__main__":
    main()
