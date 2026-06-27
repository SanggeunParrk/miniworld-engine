"""Does the CUTE LayerNormLinear (trimul-grade GemmSm90 + LN fold) beat the
torch baseline for the bias-only LN+projection region? (repo env only -- needs quack)

  baseline    : layernorm_kernel(pair) -> 3x F.linear           (current path)
  cute_concat : layernorm_linear[_fn](pair, [Wv|Wb|Wg]) -> split

inference uses `layernorm_linear` (SM90 cute dispatch); training uses
`layernorm_linear_fn` (cute autograd). Verifies correctness + times.

Run in the REPO env (.pixi/envs/default), on a compute node.
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
    layernorm_linear,
    layernorm_linear_fn,
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


def run(B, d, n_head, dtype, seq_lens):
    Hh = n_head
    print(f"# cute LN+proj  B={B} d={d} H={Hh} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L cos_v cos_g base_inf cute_inf base_fb cute_fb sp_inf sp_fb")
    for L in seq_lens:
        M = B * L * L
        pair = torch.randn(M, d, device=DEVICE, dtype=dtype)
        lw = torch.randn(d, device=DEVICE, dtype=dtype)
        lb = torch.randn(d, device=DEVICE, dtype=dtype)
        Wv = torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05
        Wb = torch.randn(Hh, d, device=DEVICE, dtype=dtype) * 0.05
        Wg = torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05
        # cute GemmSm90 needs the output width (and stride) divisible by 8; the
        # d->4 bias makes 2d+H=260 unaligned -> pad to the next multiple of 8.
        pad = (-(2 * d + Hh)) % 8
        Wcat = torch.cat(
            [Wv, Wb, Wg, torch.zeros(pad, d, device=DEVICE, dtype=dtype)], 0
        ).contiguous()
        eps = 1e-5
        splits = [d, Hh, d, pad] if pad else [d, Hh, d]
        dv = torch.randn(M, d, device=DEVICE, dtype=dtype)
        dg = torch.randn(M, d, device=DEVICE, dtype=dtype)
        db = torch.randn(M, Hh, device=DEVICE, dtype=dtype)

        # correctness (inference)
        with torch.no_grad():
            pln = kernels.layernorm_kernel(pair, lw, lb, eps)
            rv, rg = F.linear(pln, Wv), F.linear(pln, Wg)
            try:
                proj = layernorm_linear(pair, lw, lb, Wcat, None, eps)
                v, b, g = proj.split(splits, dim=-1)[:3]
                cv, cg = cos(v, rv), cos(g, rg)
            except Exception as e:  # noqa: BLE001
                print(f"{L} CUTE_INF_ERR {type(e).__name__}:{e}")
                continue

        def base_inf():
            with torch.no_grad():
                p = kernels.layernorm_kernel(pair, lw, lb, eps)
                return F.linear(p, Wv), F.linear(p, Wb), F.linear(p, Wg)

        def cute_inf():
            with torch.no_grad():
                return layernorm_linear(pair, lw, lb, Wcat, None, eps).split(splits, -1)[:3]

        # clean-width cute: value+gate concat (256, no pad) via cute LN-fold;
        # bias (d->4, odd) via the portable triton LN-fold. Avoids the N=260/264 poison.
        Wvg = torch.cat([Wv, Wg], 0).contiguous()

        def cvg_inf():
            with torch.no_grad():
                vg = layernorm_linear(pair, lw, lb, Wvg, None, eps)
                b = layernorm_linear_triton(pair, lw, lb, Wb, None, eps)
                v, g = vg.split([d, d], -1)
                return v, b, g

        bi = bench(base_inf)
        ci = bench(cute_inf)
        cvgi = bench(cvg_inf)

        # training
        Wv_p = Wv.clone().requires_grad_(True); Wb_p = Wb.clone().requires_grad_(True)
        Wg_p = Wg.clone().requires_grad_(True)
        lw_p = lw.clone().requires_grad_(True); lb_p = lb.clone().requires_grad_(True)
        lw_q = lw.clone().requires_grad_(True); lb_q = lb.clone().requires_grad_(True)
        Wv_q = Wv.clone().requires_grad_(True); Wb_q = Wb.clone().requires_grad_(True)
        Wg_q = Wg.clone().requires_grad_(True)
        pf = pair.clone().requires_grad_(True); pf2 = pair.clone().requires_grad_(True)

        def base_fb():
            p = kernels.layernorm_kernel(pf, lw_p, lb_p, eps)
            v = F.linear(p, Wv_p); b = F.linear(p, Wb_p); g = F.linear(p, Wg_p)
            ((v * dv).sum() + (b * db).sum() + (g * dg).sum()).backward()

        def cute_fb():
            wc = torch.cat(
                [Wv_q, Wb_q, Wg_q, torch.zeros(pad, d, device=DEVICE, dtype=dtype)], 0
            ) if pad else torch.cat([Wv_q, Wb_q, Wg_q], 0)
            proj = layernorm_linear_fn(pf2, lw_q, lb_q, wc, None, eps)
            v, b, g = proj.split(splits, -1)[:3]
            ((v * dv).sum() + (b * db).sum() + (g * dg).sum()).backward()

        pf3 = pair.clone().requires_grad_(True)
        lw_r = lw.clone().requires_grad_(True); lb_r = lb.clone().requires_grad_(True)
        Wv_r = Wv.clone().requires_grad_(True); Wb_r = Wb.clone().requires_grad_(True)
        Wg_r = Wg.clone().requires_grad_(True)

        def cvg_fb():
            vg = layernorm_linear_fn(pf3, lw_r, lb_r, torch.cat([Wv_r, Wg_r], 0), None, eps)
            b = layernorm_linear_triton_fn(pf3, lw_r, lb_r, Wb_r, None, eps)
            v, g = vg.split([d, d], -1)
            ((v * dv).sum() + (b * db).sum() + (g * dg).sum()).backward()

        try:
            bfb = bench(base_fb, grad_to_none=[pf, Wv_p, Wb_p, Wg_p, lw_p, lb_p])
            cfb = bench(cute_fb, grad_to_none=[pf2, Wv_q, Wb_q, Wg_q, lw_q, lb_q])
            cvgfb = bench(cvg_fb, grad_to_none=[pf3, Wv_r, Wb_r, Wg_r, lw_r, lb_r])
        except Exception as e:  # noqa: BLE001
            print(f"{L} {cv:.5f} {cg:.5f} {bi:.4f} {ci:.4f} CUTE_FB_ERR {type(e).__name__}:{e}")
            continue

        print(f"{L} concat264 {cv:.5f} {cg:.5f} {bi:.4f} {ci:.4f} {bfb:.4f} {cfb:.4f} {bi/ci:.2f} {bfb/cfb:.2f}")
        print(f"{L} cute_vg256 - - {bi:.4f} {cvgi:.4f} {bfb:.4f} {cvgfb:.4f} {bi/cvgi:.2f} {bfb/cvgfb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    # d-sweep: cute GemmSm90's win grows with K=d -- does cute LN+proj start to
    # beat the (layernorm_kernel + cuBLAS) baseline for training at d>=256?
    for d, Ls in [(256, [512, 768]), (512, [384, 512])]:
        run(B=1, d=d, n_head=4, dtype=torch.bfloat16, seq_lens=Ls)
