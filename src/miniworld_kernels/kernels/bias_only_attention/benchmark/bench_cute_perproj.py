"""Re-test cute the RIGHT way: clean per-projection d->d (where the layernorm_linear
bench shows cute #1), not the awkward concat I used before.

(A) single value projection (d->d): cute layernorm_linear_fn vs baseline
    (layernorm_kernel + cuBLAS) vs triton layernorm_linear -- reproduce the folder's
    cute-wins result.
(B) full bias-only front (value d->d, bias d->H, gate d->d sharing ONE LN):
    - baseline      : layernorm_kernel(pln) + 3x cuBLAS                 (LN once)
    - triton_concat : layernorm_linear_triton(pln,[Wv|Wb|Wg])           (current inference)
    - cute_perproj  : cute LN+Linear(value) + cute LN+Linear(gate) + cuBLAS bias
                      (cute's winning shape, but LN folded 2-3x)

inference + fwd+bwd, d=128. Run in the REPO env (.pixi/envs/default, has quack).
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
    layernorm_linear,           # cute inference dispatch (SM90, tuned configs)
    layernorm_linear_fn,        # cute autograd (SM90)
    layernorm_linear_triton,    # portable inference fwd
    layernorm_linear_triton_fn, # portable autograd
)

DEVICE = torch.device("cuda")


def bg(fn, gtn=None):
    return triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
                                   grad_to_none=gtn or [])[0]


def run(B, d, n_head, dtype, seq_lens):
    Hh = n_head
    print(f"# cute per-projection  B={B} d={d} H={Hh} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    for L in seq_lens:
        M = B * L * L
        pair = torch.randn(M, d, device=DEVICE, dtype=dtype)
        lw = torch.randn(d, device=DEVICE, dtype=dtype)
        lb = torch.randn(d, device=DEVICE, dtype=dtype)
        Wv = torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05
        Wb = torch.randn(Hh, d, device=DEVICE, dtype=dtype) * 0.05
        Wg = torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05
        Wcat = torch.cat([Wv, Wb, Wg], 0).contiguous()
        eps = 1e-5

        # (A) single value projection d->d
        def a_base():
            with torch.no_grad():
                return F.linear(kernels.layernorm_kernel(pair, lw, lb, eps), Wv)

        def a_triton():
            with torch.no_grad():
                return layernorm_linear_triton(pair, lw, lb, Wv, None, eps)

        def a_cute():
            with torch.no_grad():
                return layernorm_linear(pair, lw, lb, Wv, None, eps)

        print(f"L={L}  [A single d->d value proj, inference ms]  "
              f"base={bg(a_base):.4f}  triton={bg(a_triton):.4f}  cute={bg(a_cute):.4f}")

        # (B) full front inference
        def b_base():
            with torch.no_grad():
                p = kernels.layernorm_kernel(pair, lw, lb, eps)
                return F.linear(p, Wv), F.linear(p, Wb), F.linear(p, Wg)

        def b_triton():
            with torch.no_grad():
                return layernorm_linear_triton(pair, lw, lb, Wcat, None, eps).split([d, Hh, d], -1)

        def b_cute():
            with torch.no_grad():
                v = layernorm_linear(pair, lw, lb, Wv, None, eps)
                g = layernorm_linear(pair, lw, lb, Wg, None, eps)
                b = F.linear(kernels.layernorm_kernel(pair, lw, lb, eps), Wb)
                return v, b, g

        print(f"L={L}  [B full front, inference ms]  "
              f"base={bg(b_base):.4f}  triton_concat={bg(b_triton):.4f}  cute_perproj={bg(b_cute):.4f}")

        # (B) full front fwd+bwd
        dv = torch.randn(M, d, device=DEVICE, dtype=dtype)
        dgr = torch.randn(M, d, device=DEVICE, dtype=dtype)
        dbb = torch.randn(M, Hh, device=DEVICE, dtype=dtype)

        def leaves():
            return [t.clone().requires_grad_(True) for t in (pair, lw, lb, Wv, Wb, Wg)]

        def base_fb():
            p_, lw_, lb_, Wv_, Wb_, Wg_ = L0
            pln = kernels.layernorm_kernel(p_, lw_, lb_, eps)
            ((F.linear(pln, Wv_) * dv).sum() + (F.linear(pln, Wb_) * dbb).sum()
             + (F.linear(pln, Wg_) * dgr).sum()).backward()

        def cute_fb():
            p_, lw_, lb_, Wv_, Wb_, Wg_ = L1
            v = layernorm_linear_fn(p_, lw_, lb_, Wv_, None, eps)
            g = layernorm_linear_fn(p_, lw_, lb_, Wg_, None, eps)
            b = F.linear(kernels.layernorm_kernel(p_, lw_, lb_, eps), Wb_)
            ((v * dv).sum() + (b * dbb).sum() + (g * dgr).sum()).backward()

        L0 = leaves(); L1 = leaves()
        bfb = bg(base_fb, L0)
        cfb = bg(cute_fb, L1)
        print(f"L={L}  [B full front, fwd+bwd ms]  base={bfb:.4f}  cute_perproj={cfb:.4f}  "
              f"cute_x={bfb / cfb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d=128, n_head=4, dtype=torch.bfloat16, seq_lens=[512, 1024])
