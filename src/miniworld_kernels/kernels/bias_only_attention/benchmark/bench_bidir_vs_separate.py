"""Is the FUSED bidirectional bias-only attention faster than running the two
single-directions SEPARATELY (as an AF3-style pairformer block does)?

separate (faithful block): two sequential residual updates with rowwise dropout,
  ending sees the starting-updated pair:
    pair = pair + drop_row(start(pair))
    pair = pair + drop_row(end(pair))
fused: one residual on the bidirectional block:
    pair = pair + drop_row(bidir(pair))

Semantic gap (same as bidir trimul): bidir can't see the start-updated pair (both
directions from one input) -- a different model, not a drop-in. SPEED question only.
Equal per-direction hidden h = d_pair. PYTORCH and TRITON impls; timed in TRAIN mode
(dropout on, fwd+bwd) and EVAL (inference, no_grad). B=1, bf16. Compute node only.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import triton

from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_attention import (
    BidirectionalTriangleAttention,
    TriangleAttention,
)

DEVICE = torch.device("cuda")
P_DROP = 0.25


def drop_row(x, training):
    if not training or P_DROP == 0:
        return x
    B, I = x.shape[0], x.shape[1]
    keep = (torch.rand(B, I, 1, 1, device=x.device) > P_DROP).to(x.dtype) / (1 - P_DROP)
    return x * keep


class SepBlock(nn.Module):
    def __init__(self, start_mod, end_mod):
        super().__init__()
        self.start_mod, self.end_mod = start_mod, end_mod

    def forward(self, pair, mask):
        pair = pair + drop_row(self.start_mod(pair, mask), self.training)
        pair = pair + drop_row(self.end_mod(pair, mask), self.training)
        return pair


class BidirBlock(nn.Module):
    def __init__(self, bidir_mod):
        super().__init__()
        self.bidir_mod = bidir_mod

    def forward(self, pair, mask):
        return pair + drop_row(self.bidir_mod(pair, mask), self.training)


def bench(fn, gtn=None):
    return triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
                                   grad_to_none=gtn or [])[0]


def make(impl, dtype, d_pair, n_head):
    start = TriangleAttention(d_pair, n_head, starting=True, use_self_attention=False,
                              implementation=impl)
    end = TriangleAttention(d_pair, n_head, starting=False, use_self_attention=False,
                            implementation=impl)
    bidir = BidirectionalTriangleAttention(d_pair, n_head, implementation=impl)
    sep = SepBlock(start, end).to(DEVICE).to(dtype)
    bid = BidirBlock(bidir).to(DEVICE).to(dtype)
    return sep, bid


def run(B, d_pair, n_head, dtype, seq_lens):
    print(f"# bidir vs separate (bias-only)  B={B} d_pair={d_pair} H={n_head} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}  separate=2 sequential residuals + drop_row")
    print("# columns: L impl sep_infer bidir_infer infer_fuse sep_fb bidir_fb fb_fuse")
    for L in seq_lens:
        pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = torch.rand(B, L, device=DEVICE) > 0.2
        dy = torch.randn_like(pair)
        for impl in (ImplementationType.PYTORCH, ImplementationType.TRITON):
            sep, bid = make(impl, dtype, d_pair, n_head)

            sep.eval(); bid.eval()
            with torch.no_grad():
                si = bench(lambda: sep(pair, mask))
                bi = bench(lambda: bid(pair, mask))

            sep.train(); bid.train()
            ps = pair.clone().requires_grad_(True)
            pb = pair.clone().requires_grad_(True)

            def sep_fb():
                sep.zero_grad(set_to_none=True)
                sep(ps, mask).backward(dy)

            def bid_fb():
                bid.zero_grad(set_to_none=True)
                bid(pb, mask).backward(dy)

            sfb = bench(sep_fb, [ps])
            bfb = bench(bid_fb, [pb])
            print(f"{L} {impl.value} {si:.3f} {bi:.3f} {si/bi:.2f} {sfb:.3f} {bfb:.3f} {sfb/bfb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16, seq_lens=[256, 512, 1024])
