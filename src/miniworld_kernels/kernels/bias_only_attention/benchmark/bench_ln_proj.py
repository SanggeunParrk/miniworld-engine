"""Does LayerNormLinear fusion beat (layernorm_kernel + 3 cuBLAS GEMMs)?

The bias-only module computes:  pln = LN(pair); value=Wv@pln; bias=Wb@pln; gate=Wg@pln.
All three projections read pln. Candidates for the LN+projection region:

  baseline   : layernorm_kernel(pair) -> 3x F.linear           (pln materialized, read 3x)
  ll_concat  : layernorm_linear(pair, [Wv|Wb|Wg]) -> split     (LN fused into 1 GEMM, no pln in HBM)
  ll_split   : layernorm_linear per projection (value, gate) + bias on materialized pln

inference uses `layernorm_linear`; training uses `layernorm_linear_te_fn` (autograd,
stride-transparent) and `layernorm_linear_fn` (cute autograd). Verifies + times.
Run via srun on a compute node.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn.functional as F
import triton

from miniworld_kernels import kernels
from miniworld_kernels.kernels.layernorm_linear import (
    layernorm_linear_te_fn,
    layernorm_linear_triton,
    layernorm_linear_triton_fn,
)

DEVICE = torch.device("cuda")


def bench(fn, grad_to_none=None):
    med, _, _ = triton.testing.do_bench(
        fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )
    return med


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def run(B, d_pair, n_head, dtype, seq_lens):
    d = d_pair
    Hh = n_head
    print(f"# LN+proj region  B={B} d={d} H={Hh} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant cos_v cos_g infer_ms fwdbwd_ms")
    for L in seq_lens:
        pair = torch.randn(B * L * L, d, device=DEVICE, dtype=dtype)  # 2D [M, d]
        lw = torch.randn(d, device=DEVICE, dtype=dtype)
        lb = torch.randn(d, device=DEVICE, dtype=dtype)
        Wv = (torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05)
        Wb = (torch.randn(Hh, d, device=DEVICE, dtype=dtype) * 0.05)
        Wg = (torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05)
        Wcat = torch.cat([Wv, Wb, Wg], dim=0).contiguous()  # [2d+H, d]
        eps = 1e-5
        splits = [d, Hh, d]

        # reference (baseline)
        def baseline_fwd(p):
            pln = kernels.layernorm_kernel(p, lw, lb, eps)
            return F.linear(pln, Wv), F.linear(pln, Wb), F.linear(pln, Wg)

        with torch.no_grad():
            ref_v, ref_b, ref_g = baseline_fwd(pair)

        def ll_concat_fwd(p, fn):
            proj = fn(p, lw, lb, Wcat, None, eps)
            v, b, g = proj.split(splits, dim=-1)
            return v, b, g

        # correctness vs baseline (te_fn path)
        with torch.no_grad():
            v2, b2, g2 = ll_concat_fwd(pair, layernorm_linear_te_fn)
        cv, cg = cos(v2, ref_v), cos(g2, ref_g)

        # ---- inference timing (no grad) ----
        def base_inf():
            with torch.no_grad():
                return baseline_fwd(pair)

        def ll_inf():
            with torch.no_grad():
                proj = layernorm_linear_triton(pair, lw, lb, Wcat, None, eps)
                return proj.split(splits, dim=-1)

        bi = bench(base_inf)
        li = bench(ll_inf)

        # ---- fwd+bwd timing ----
        def make_leaf():
            return pair.detach().clone().requires_grad_(True)

        dy_v = torch.randn(B * L * L, d, device=DEVICE, dtype=dtype)
        dy_g = torch.randn(B * L * L, d, device=DEVICE, dtype=dtype)

        pf = make_leaf()
        bfb = bench(lambda: _fb_base(pf, lw, lb, eps, Wv, Wg, dy_v, dy_g), grad_to_none=[pf])

        pf2 = make_leaf()
        lfb = bench(
            lambda: _fb_concat(pf2, lw, lb, Wcat, eps, splits, dy_v, dy_g, layernorm_linear_te_fn),
            grad_to_none=[pf2],
        )

        pf3 = make_leaf()
        lfb_t = bench(
            lambda: _fb_concat(pf3, lw, lb, Wcat, eps, splits, dy_v, dy_g, layernorm_linear_triton_fn),
            grad_to_none=[pf3],
        )

        print(f"{L} baseline {1.0:.5f} {1.0:.5f} {bi:.4f} {bfb:.4f}")
        print(f"{L} ll_concat_te {cv:.5f} {cg:.5f} {li:.4f} {lfb:.4f}")
        print(f"{L} ll_concat_triton - - {li:.4f} {lfb_t:.4f}")
        print(flush=True)


def _fb_base(p, lw, lb, eps, Wv, Wg, dy_v, dy_g):
    pln = kernels.layernorm_kernel(p, lw, lb, eps)
    v = F.linear(pln, Wv)
    g = F.linear(pln, Wg)
    ((v * dy_v).sum() + (g * dy_g).sum()).backward()


def _fb_concat(p, lw, lb, Wcat, eps, splits, dy_v, dy_g, fn):
    proj = fn(p, lw, lb, Wcat, None, eps)
    v, b, g = proj.split(splits, dim=-1)
    ((v * dy_v).sum() + (g * dy_g).sum()).backward()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16, seq_lens=[512, 1024])
