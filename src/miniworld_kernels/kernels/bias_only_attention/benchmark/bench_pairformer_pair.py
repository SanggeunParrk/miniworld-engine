"""Pairformer PAIR-track: old (separate) vs new (bidirectional) blocks.

team-gm PairformerBlock pair track (src/team_gm/modules/blocks/pairformer.py),
residual, no dropout in the active path:
    pair += tri_multi_outgoing(pair); pair += tri_multi_incoming(pair)
    pair += tri_atten_starting(pair); pair += tri_atten_ending(pair)
    pair += transition(pair)

NEW collapses the two trimul into one BidirectionalTriangleMultiplication and the
two (bias-only) triangle attentions into one BidirectionalTriangleAttention:
    pair += bidir_trimul(pair); pair += bidir_attn(pair); pair += transition(pair)

Configs:
  old_py  : all PYTORCH (team-gm baseline)
  old_opt : trimul/attn/transition all TRITON (optimized old)
  new_py  : bidir blocks, all PYTORCH (isolates the architectural fusion)
  new_opt : bidir_trimul PYTORCH (no module fast-path; the cute bidir-trimul KERNEL
            exists but isn't module-wired) + bidir_attn TRITON + transition TRITON

bias-only triangle attention (use_self_attention=False). Pair track only (the
single track / AttentionPairBias is unchanged by these ops). inference (eval,
no_grad) + fwd+bwd (train). B=1, bf16. REPO env (quack) on a compute node.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import triton

from miniworld_kernels.modules import Transition, TriangleAttention, TriangleMultiplication
from miniworld_kernels.modules.exceptions import ImplementationType as IT
from miniworld_kernels.modules.triangle_attention import BidirectionalTriangleAttention
from miniworld_kernels.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul

DEVICE = torch.device("cuda")


def bench(fn, gtn=None):
    return triton.testing.do_bench(fn, warmup=10, rep=50, quantiles=[0.5, 0.2, 0.8],
                                   grad_to_none=gtn or [])[0]


def make_old(d_pair, nh, impl, dtype):
    mk_tm = lambda og: TriangleMultiplication(d_pair=d_pair, outgoing=og, implementation=impl)
    mk_ta = lambda st: TriangleAttention(d_pair, nh, starting=st, use_self_attention=False,
                                         implementation=impl)
    mods = torch.nn.ModuleList([mk_tm(True), mk_tm(False), mk_ta(True), mk_ta(False),
                                Transition(d_pair, implementation=impl)]).to(DEVICE).to(dtype)

    def track(pair, mask):
        to, ti, a_s, a_e, tr = mods
        pair = pair + to(pair, mask)
        pair = pair + ti(pair, mask)
        pair = pair + a_s(pair, mask)
        pair = pair + a_e(pair, mask)
        return pair + tr(pair)

    return mods, track


def make_new(d_pair, nh, tm_impl, attn_impl, tr_impl, dtype):
    bt = BidirectionalTriangleMultiplication(d_pair=d_pair, implementation=tm_impl)
    ba = BidirectionalTriangleAttention(d_pair, nh, implementation=attn_impl)
    tr = Transition(d_pair, implementation=tr_impl)
    mods = torch.nn.ModuleList([bt, ba, tr]).to(DEVICE).to(dtype)

    def track(pair, mask):
        bt, ba, tr = mods
        pair = pair + bt(pair, mask)
        pair = pair + ba(pair, mask)
        return pair + tr(pair)

    return mods, track


def make_new_cute(d_pair, nh, dtype):
    """Fully-optimized new: cute bidir trimul (BidirV6TriMul) + triton bidir attn +
    triton/cute transition. The cute bidir trimul takes no mask (speed-only)."""
    base = BidirectionalTriangleMultiplication(d_pair=d_pair, implementation=IT.PYTORCH)
    base = base.to(DEVICE).to(dtype)
    bt = BidirV6TriMul(base).to(DEVICE).to(dtype)
    ba = BidirectionalTriangleAttention(d_pair, nh, implementation=IT.TRITON).to(DEVICE).to(dtype)
    tr = Transition(d_pair, implementation=IT.TRITON).to(DEVICE).to(dtype)
    mods = torch.nn.ModuleList([bt, ba, tr])

    def track(pair, mask):
        bt, ba, tr = mods
        pair = pair + bt(pair)  # cute bidir trimul (no mask arg)
        pair = pair + ba(pair, mask)
        return pair + tr(pair)

    return mods, track


def run(B, d_pair, nh, dtype, seq_lens):
    PY, TR = IT.PYTORCH, IT.TRITON
    print(f"# pairformer pair-track  B={B} d_pair={d_pair} H_tri={nh} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}  (pair track only; single track excluded)")
    print("# columns: L config infer_ms fwdbwd_ms")
    for L in seq_lens:
        pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = None  # speed-only; the cute bidir trimul takes no mask, keep all consistent
        dy = torch.randn_like(pair)
        configs = {
            "old_py": make_old(d_pair, nh, PY, dtype),
            "old_opt": make_old(d_pair, nh, TR, dtype),
            "new_py": make_new(d_pair, nh, PY, PY, PY, dtype),
            "new_opt": make_new(d_pair, nh, PY, TR, TR, dtype),
            "new_cute": make_new_cute(d_pair, nh, dtype),
        }
        res = {}
        for name, (mods, track) in configs.items():
            mods.eval()
            try:
                with torch.no_grad():
                    im = bench(lambda: track(pair, mask))
            except Exception as e:  # noqa: BLE001
                print(f"{L} {name} INFER_ERR {type(e).__name__}:{str(e)[:60]}"); im = float("nan")
            mods.train()
            p = pair.clone().requires_grad_(True)

            def fb():
                mods.zero_grad(set_to_none=True)
                track(p, mask).backward(dy)

            try:
                fb_ms = bench(fb, [p])
            except Exception as e:  # noqa: BLE001
                print(f"{L} {name} FB_ERR {type(e).__name__}:{str(e)[:60]}"); fb_ms = float("nan")
            res[name] = (im, fb_ms)
            print(f"{L} {name} {im:.3f} {fb_ms:.3f}", flush=True)
        # speedups vs old_py
        oi, ofb = res["old_py"]
        print(f"{L} SPEEDUP_vs_old_py  "
              + "  ".join(f"{n}: inf {oi / res[n][0]:.2f}x / fb {ofb / res[n][1]:.2f}x"
                          for n in ("old_opt", "new_py", "new_opt", "new_cute")), flush=True)
        print(flush=True)
        del configs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, nh=4, dtype=torch.bfloat16, seq_lens=[256, 512, 768])
