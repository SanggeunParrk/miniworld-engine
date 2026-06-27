"""Is the fused_gate_out BACKWARD actually optimized, or lazy?

Current bwd: cuBLAS d_a=do@wo  ->  separate elementwise (d_r,d_g,a; 6x [M,DH] HBM)
             ->  cuBLAS d_wo=do^T@a.  The elementwise dominates (d_a materialized
             then re-read).

Improved bwd: ONE triton kernel does d_a=do@wo (GEMM) AND computes d_r,d_g,a in its
              epilogue (d_a never materialized; gate/out read once) -> cuBLAS d_wo.

Compares fwd+bwd of the two backward designs (forward identical). Run via srun.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import triton
import triton.language as tl

from miniworld_kernels.kernels.bias_only_attention.triton.gate_out import _fwd, fused_gate_out

DEVICE = torch.device("cuda")


# ---- improved bwd: fused dgrad-GEMM + gate-bwd epilogue ----
@triton.autotune(
    configs=[triton.Config({"BM": bm}, num_warps=w, num_stages=s)
             for bm in (32, 64, 128) for w in (4, 8) for s in (2, 3, 4)],
    key=["M", "N", "DH"],
)
@triton.jit
def _dgrad_epi(
    do_ptr, wo_ptr, gate_ptr, out_ptr,         # do[M,N] wo[N,DH] gate[M,DH] out[M,DH]
    dr_ptr, dg_ptr, a_ptr,                      # outputs [M,DH]
    M, N: tl.constexpr, DH: tl.constexpr,
    s_dom, s_don, s_won, s_woh, s_gm, s_gh, s_om, s_oh, s_rm, s_rh,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, N)
    rh = tl.arange(0, DH)
    mm = rm[:, None] < M
    # d_a[BM,DH] = do[BM,N] @ wo[N,DH]
    do = tl.load(do_ptr + rm[:, None] * s_dom + rn[None, :] * s_don, mask=mm, other=0.0)
    wo = tl.load(wo_ptr + rn[:, None] * s_won + rh[None, :] * s_woh)        # [N,DH]
    d_a = tl.dot(do, wo)                                                    # [BM,DH] fp32
    s = tl.sigmoid(tl.load(gate_ptr + rm[:, None] * s_gm + rh[None, :] * s_gh,
                           mask=mm, other=0.0).to(tl.float32))
    o = tl.load(out_ptr + rm[:, None] * s_om + rh[None, :] * s_oh, mask=mm, other=0.0).to(tl.float32)
    off = rm[:, None] * s_rm + rh[None, :] * s_rh
    tl.store(dr_ptr + off, (s * d_a).to(dr_ptr.dtype.element_ty), mask=mm)
    tl.store(dg_ptr + off, (d_a * o * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=mm)
    tl.store(a_ptr + off, (s * o).to(a_ptr.dtype.element_ty), mask=mm)


class _FusedGateOutV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, out, wo):
        shape = gate.shape; DH = shape[-1]
        g2 = gate.reshape(-1, DH).contiguous(); r2 = out.reshape(-1, DH).contiguous()
        out2 = _fwd(g2, r2, wo.contiguous())
        ctx.save_for_backward(g2, r2, wo); ctx.shape = shape; ctx.N = wo.shape[0]
        return out2.reshape(*shape[:-1], wo.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        g2, r2, wo = ctx.saved_tensors
        M, DH = g2.shape; N = ctx.N
        do2 = grad_out.reshape(M, N).contiguous()
        d_r = torch.empty_like(g2); d_g = torch.empty_like(g2); a = torch.empty_like(g2)
        grid = lambda meta: (triton.cdiv(M, meta["BM"]),)
        _dgrad_epi[grid](
            do2, wo, g2, r2, d_r, d_g, a, M, N, DH,
            do2.stride(0), do2.stride(1), wo.stride(0), wo.stride(1),
            g2.stride(0), g2.stride(1), r2.stride(0), r2.stride(1),
            d_r.stride(0), d_r.stride(1),
        )
        d_wo = do2.transpose(0, 1) @ a
        return d_g.reshape(ctx.shape), d_r.reshape(ctx.shape), d_wo


def fused_gate_out_v2(gate, out, wo):
    return _FusedGateOutV2.apply(gate, out, wo)


def benchg(fn, gtn):
    return triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
                                   grad_to_none=gtn)[0]


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


def run(B, d, dtype, seq_lens):
    DH = N = d
    print(f"# gate bwd  B={B} DH={DH} N={N} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant cos fwdbwd_ms speedup")
    for L in seq_lens:
        M = B * L * L
        gate = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        out = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        wo = torch.randn(N, DH, device=DEVICE, dtype=dtype) * 0.05
        dy = torch.randn(M, N, device=DEVICE, dtype=dtype)

        # correctness of v2 grads vs v1
        g1 = gate.clone().requires_grad_(True); o1 = out.clone().requires_grad_(True); w1 = wo.clone().requires_grad_(True)
        g2 = gate.clone().requires_grad_(True); o2 = out.clone().requires_grad_(True); w2 = wo.clone().requires_grad_(True)
        fused_gate_out(g1, o1, w1).backward(dy)
        fused_gate_out_v2(g2, o2, w2).backward(dy)
        c = min(cos(g2.grad, g1.grad), cos(o2.grad, o1.grad), cos(w2.grad, w1.grad))

        gv1 = gate.clone().requires_grad_(True); ov1 = out.clone().requires_grad_(True); wv1 = wo.clone().requires_grad_(True)
        gv2 = gate.clone().requires_grad_(True); ov2 = out.clone().requires_grad_(True); wv2 = wo.clone().requires_grad_(True)
        t1 = benchg(lambda: fused_gate_out(gv1, ov1, wv1).backward(dy), [gv1, ov1, wv1])
        t2 = benchg(lambda: fused_gate_out_v2(gv2, ov2, wv2).backward(dy), [gv2, ov2, wv2])
        print(f"{L} v1_current 1.00000 {t1:.4f} 1.00")
        print(f"{L} v2_fused_bwd {c:.5f} {t2:.4f} {t1/t2:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d=128, dtype=torch.bfloat16, seq_lens=[384, 512, 768, 1024])
