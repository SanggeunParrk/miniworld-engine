"""Does concatenating the value/bias/gate projections help TRAINING?

Both keep the strong repo LayerNorm (kernels.layernorm_kernel) -- isolating the
projection structure (the te_style fusion lost earlier because of its weaker LN
backward, not the concat). torch already lowers F.linear(pln, cat([Wv,Wb,Wg]))'s
backward to ONE wgrad + ONE dgrad GEMM (vs 3+3 for separate linears), so this
measures the trimul "_dconcat" idea directly.

  baseline : layernorm_kernel(pair) -> 3x F.linear            (6 GEMMs in bwd)
  concat   : layernorm_kernel(pair) -> F.linear(pln, Wcat) -> split   (2 GEMMs in bwd)

Reports inference (no_grad) and fwd+bwd. Run via srun on a compute node.
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

DEVICE = torch.device("cuda")


def bench(fn, grad_to_none=None):
    med, _, _ = triton.testing.do_bench(
        fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )
    return med


def run(B, d, n_head, dtype, seq_lens):
    Hh = n_head
    print(f"# concat-projection (training)  B={B} d={d} H={Hh} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant infer_ms fwdbwd_ms sp_infer sp_fb")
    for L in seq_lens:
        M = B * L * L
        pair = torch.randn(M, d, device=DEVICE, dtype=dtype)
        lw = torch.randn(d, device=DEVICE, dtype=dtype)
        lb = torch.randn(d, device=DEVICE, dtype=dtype)
        Wv = torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05
        Wb = torch.randn(Hh, d, device=DEVICE, dtype=dtype) * 0.05
        Wg = torch.randn(d, d, device=DEVICE, dtype=dtype) * 0.05
        eps = 1e-5
        splits = [d, Hh, d]
        dv = torch.randn(M, d, device=DEVICE, dtype=dtype)
        dg = torch.randn(M, d, device=DEVICE, dtype=dtype)
        db = torch.randn(M, Hh, device=DEVICE, dtype=dtype)

        Wv_p = Wv.clone().requires_grad_(True)
        Wb_p = Wb.clone().requires_grad_(True)
        Wg_p = Wg.clone().requires_grad_(True)
        lw_p = lw.clone().requires_grad_(True)
        lb_p = lb.clone().requires_grad_(True)

        def base_inf():
            with torch.no_grad():
                pln = kernels.layernorm_kernel(pair, lw, lb, eps)
                return F.linear(pln, Wv), F.linear(pln, Wb), F.linear(pln, Wg)

        def concat_inf():
            with torch.no_grad():
                pln = kernels.layernorm_kernel(pair, lw, lb, eps)
                proj = F.linear(pln, torch.cat([Wv, Wb, Wg], 0))
                return proj.split(splits, dim=-1)

        # clean-width design: value+gate concat (256, tensor-core-friendly) + bias separate
        Wvg = torch.cat([Wv, Wg], 0)

        def vg256_inf():
            with torch.no_grad():
                pln = kernels.layernorm_kernel(pair, lw, lb, eps)
                vg = F.linear(pln, Wvg)
                b = F.linear(pln, Wb)
                v, g = vg.split([d, d], dim=-1)
                return v, b, g

        def base_fb():
            p = pf
            pln = kernels.layernorm_kernel(p, lw_p, lb_p, eps)
            v = F.linear(pln, Wv_p); b = F.linear(pln, Wb_p); g = F.linear(pln, Wg_p)
            ((v * dv).sum() + (b * db).sum() + (g * dg).sum()).backward()

        def concat_fb():
            p = pf2
            pln = kernels.layernorm_kernel(p, lw_p, lb_p, eps)
            proj = F.linear(pln, torch.cat([Wv_p, Wb_p, Wg_p], 0))
            v, b, g = proj.split(splits, dim=-1)
            ((v * dv).sum() + (b * db).sum() + (g * dg).sum()).backward()

        def vg256_fb():
            pln = kernels.layernorm_kernel(pf3, lw_p, lb_p, eps)
            vg = F.linear(pln, torch.cat([Wv_p, Wg_p], 0))
            b = F.linear(pln, Wb_p)
            v, g = vg.split([d, d], dim=-1)
            ((v * dv).sum() + (b * db).sum() + (g * dg).sum()).backward()

        pf = pair.clone().requires_grad_(True)
        pf2 = pair.clone().requires_grad_(True)
        pf3 = pair.clone().requires_grad_(True)

        bi = bench(base_inf)
        ci = bench(concat_inf)
        vi = bench(vg256_inf)
        bfb = bench(base_fb, grad_to_none=[pf, Wv_p, Wb_p, Wg_p, lw_p, lb_p])
        cfb = bench(concat_fb, grad_to_none=[pf2, Wv_p, Wb_p, Wg_p, lw_p, lb_p])
        vfb = bench(vg256_fb, grad_to_none=[pf3, Wv_p, Wb_p, Wg_p, lw_p, lb_p])

        print(f"{L} baseline {bi:.4f} {bfb:.4f} 1.00 1.00")
        print(f"{L} concat260 {ci:.4f} {cfb:.4f} {bi/ci:.2f} {bfb/cfb:.2f}")
        print(f"{L} vg256+bias {vi:.4f} {vfb:.4f} {bi/vi:.2f} {bfb/vfb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d=128, n_head=4, dtype=torch.bfloat16, seq_lens=[384, 512, 768, 1024])
