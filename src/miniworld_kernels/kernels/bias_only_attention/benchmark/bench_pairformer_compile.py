"""Pairformer PAIR-track, three configs, EAGER vs full torch.compile.

  PT : all PYTORCH       (trimul PY + bias-only tri-attn PY + transition PY)
  CE : cuequiv trimul    (+ bias-only tri-attn PY + transition PY)
  MW : miniworld bidir    (cute bidir trimul + TRITON bidir attn + TRITON transition)

Residual pair-track, mask=None (cute bidir trimul takes no mask -> speed-only,
consistent). inference (eval/no_grad) + training (fwd+bwd). B=1, bf16. REPO env.

NOTE: MW's custom cute/triton kernels graph-break under torch.compile (they are
@compiler.disable), so MW-compile carries Dynamo overhead and is NOT its best mode
(eager/CUDA-graph is). Reported both so the comparison is honest.
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

from miniworld_kernels.modules import Transition, TriangleAttention, TriangleMultiplication
from miniworld_kernels.modules.exceptions import ImplementationType as IT
from miniworld_kernels.modules.triangle_attention import BidirectionalTriangleAttention
from miniworld_kernels.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul

DEVICE = torch.device("cuda")


class PTPair(nn.Module):
    def __init__(self, d, nh, trimul_impl):
        super().__init__()
        self.tmo = TriangleMultiplication(d_pair=d, outgoing=True, implementation=trimul_impl)
        self.tmi = TriangleMultiplication(d_pair=d, outgoing=False, implementation=trimul_impl)
        self.ats = TriangleAttention(d, nh, starting=True, use_self_attention=False,
                                     implementation=IT.PYTORCH)
        self.ate = TriangleAttention(d, nh, starting=False, use_self_attention=False,
                                     implementation=IT.PYTORCH)
        self.trn = Transition(d, implementation=IT.PYTORCH)

    def forward(self, p):
        p = p + self.tmo(p, None)
        p = p + self.tmi(p, None)
        p = p + self.ats(p, None)
        p = p + self.ate(p, None)
        return p + self.trn(p)


class MWPair(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        base = BidirectionalTriangleMultiplication(d_pair=d, implementation=IT.PYTORCH)
        self.bt = BidirV6TriMul(base)
        self.ba = BidirectionalTriangleAttention(d, nh, implementation=IT.TRITON)
        self.trn = Transition(d, implementation=IT.TRITON)

    def forward(self, p):
        p = p + self.bt(p)
        p = p + self.ba(p, None)
        return p + self.trn(p)


def bg(fn, g=None):
    return triton.testing.do_bench(fn, warmup=15, rep=60, quantiles=[0.5, 0.2, 0.8],
                                   grad_to_none=g or [])[0]


def time_model(model, pair, dy, compiled):
    fn = torch.compile(model) if compiled else model
    model.eval()
    with torch.no_grad():
        for _ in range(6):
            fn(pair)
        im = bg(lambda: fn(pair))
    model.train()
    p = pair.clone().requires_grad_(True)
    for _ in range(6):
        model.zero_grad(set_to_none=True)
        fn(p).backward(dy)
    tm = bg(lambda: (model.zero_grad(set_to_none=True), fn(p).backward(dy)), [p])
    return im, tm


def run(B, d, nh, dtype, L):
    print(f"# pairformer pair-track (eager vs full compile)  L={L} d={d} bf16")
    print(f"# device={torch.cuda.get_device_name()}")
    pair = torch.randn(B, L, L, d, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(pair)
    configs = {
        "PT (all pytorch)": PTPair(d, nh, IT.PYTORCH).to(DEVICE).to(dtype),
        "CE (cuequiv trimul)": PTPair(d, nh, IT.CUEQUIVARIANCE).to(DEVICE).to(dtype),
        "MW (miniworld bidir)": MWPair(d, nh).to(DEVICE).to(dtype),
    }
    print(f"{'config':>22} | {'eager inf':>10} {'comp inf':>9} | {'eager train':>12} {'comp train':>11}")
    print("-" * 74)
    rows = {}
    for name, m in configs.items():
        ei, et = time_model(m, pair, dy, compiled=False)
        try:
            ci, ct = time_model(m, pair, dy, compiled=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:>22} | compile FAIL: {type(e).__name__} {str(e)[:40]}")
            ci = ct = float("nan")
        rows[name] = (ei, ci, et, ct)
        print(f"{name:>22} | {ei:>10.3f} {ci:>9.3f} | {et:>12.3f} {ct:>11.3f}", flush=True)
    return rows


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d=128, nh=4, dtype=torch.bfloat16, L=384)
