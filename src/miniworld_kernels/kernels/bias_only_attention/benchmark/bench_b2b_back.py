"""Back-to-back back kernel: fuse gate-GEMM + sigmoid*mul + to_out into ONE kernel?

Current back (after einsum) is 2 kernels:  glogit = pln@Wg (cuBLAS) ; then
fused_gate_out(glogit, out, Wo) = [sigmoid*mul + @Wo] (one triton GEMM). glogit is
materialized to HBM between them.

b2b: ONE kernel keeps glogit in SRAM:
    glogit = pln @ Wg            (GEMM1, in-reg)
    A      = sigmoid(glogit)*out (epilogue elementwise)
    y      = A @ Wo^T            (GEMM2)
Feasible only when the intermediate [BM, DH] fits regs/SRAM -> small DH (d=128 here:
D=DH=N=128, both GEMMs are a single tl.dot). Forward-only probe (answers "is one
kernel slower?"). Run via srun on a compute node.
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
import triton.language as tl

from miniworld_kernels.kernels.bias_only_attention.triton.gate_out import fused_gate_out

DEVICE = torch.device("cuda")


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
             for bm in (32, 64, 128) for w in (4, 8) for s in (2, 3, 4)],
    key=["M", "D", "DH", "N"],
)
@triton.jit
def _gate_b2b_fwd(
    pln_ptr, wg_ptr, out_ptr, wo_ptr, y_ptr,
    M, D: tl.constexpr, DH: tl.constexpr, N: tl.constexpr,
    s_pm, s_pd, s_wgd, s_wgh, s_om, s_oh, s_won, s_woh, s_ym, s_yn,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    rh = tl.arange(0, DH)
    rn = tl.arange(0, N)
    mmask = rm < M

    # GEMM1: glogit[BM,DH] = pln[BM,D] @ Wg[D,DH]
    pln = tl.load(pln_ptr + rm[:, None] * s_pm + rd[None, :] * s_pd, mask=mmask[:, None], other=0.0)
    wg = tl.load(wg_ptr + rd[:, None] * s_wgd + rh[None, :] * s_wgh)        # [D,DH]
    glogit = tl.dot(pln, wg)                                                # [BM,DH] fp32
    o = tl.load(out_ptr + rm[:, None] * s_om + rh[None, :] * s_oh, mask=mmask[:, None], other=0.0)
    a = (tl.sigmoid(glogit) * o.to(tl.float32)).to(wo_ptr.dtype.element_ty)  # [BM,DH]

    # GEMM2: y[BM,N] = A[BM,DH] @ Wo^T[DH,N]   (Wo is [N,DH])
    woT = tl.load(wo_ptr + rn[None, :] * s_won + rh[:, None] * s_woh)        # [DH,N]
    y = tl.dot(a, woT)                                                       # [BM,N]
    tl.store(y_ptr + rm[:, None] * s_ym + rn[None, :] * s_yn,
             y.to(y_ptr.dtype.element_ty), mask=mmask[:, None])


def gate_b2b(pln, Wg, out, Wo):
    """pln[M,D] Wg[D,DH] out[M,DH] Wo[N,DH] -> y[M,N]. One fused b2b kernel (fwd-only)."""
    M, D = pln.shape
    DH = Wg.shape[1]
    N = Wo.shape[0]
    y = torch.empty(M, N, device=pln.device, dtype=pln.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _gate_b2b_fwd[grid](
        pln, Wg, out, Wo, y, M, D, DH, N,
        pln.stride(0), pln.stride(1), Wg.stride(0), Wg.stride(1),
        out.stride(0), out.stride(1), Wo.stride(0), Wo.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ---- b2b with backward (training) ----
@triton.autotune(
    configs=[triton.Config({"BM": bm}, num_warps=w) for bm in (32, 64, 128) for w in (4, 8)],
    key=["M", "DH"],
)
@triton.jit
def _b2b_bwd_elem(dA_ptr, sig_ptr, out_ptr, dout_ptr, dglog_ptr, a_ptr, M, DH: tl.constexpr,
                  BM: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rh = tl.arange(0, DH)
    mm = rm[:, None] < M
    off = rm[:, None] * DH + rh[None, :]
    dA = tl.load(dA_ptr + off, mask=mm, other=0.0).to(tl.float32)
    s = tl.load(sig_ptr + off, mask=mm, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + off, mask=mm, other=0.0).to(tl.float32)
    tl.store(dout_ptr + off, (dA * s).to(dout_ptr.dtype.element_ty), mask=mm)
    tl.store(dglog_ptr + off, (dA * o * s * (1.0 - s)).to(dglog_ptr.dtype.element_ty), mask=mm)
    tl.store(a_ptr + off, (s * o).to(a_ptr.dtype.element_ty), mask=mm)


class _GateB2B(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, pln, Wg, out, Wo):
        M, D = pln.shape
        DH = Wg.shape[1]; N = Wo.shape[0]
        y = torch.empty(M, N, device=pln.device, dtype=pln.dtype)
        sig = torch.empty(M, DH, device=pln.device, dtype=pln.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _gate_b2b_fwd_sig[grid](
            pln, Wg, out, Wo, y, sig, M, D, DH, N,
            pln.stride(0), pln.stride(1), Wg.stride(0), Wg.stride(1),
            out.stride(0), out.stride(1), Wo.stride(0), Wo.stride(1),
            y.stride(0), y.stride(1), sig.stride(0), sig.stride(1),
        )
        ctx.save_for_backward(pln, Wg, out, Wo, sig)
        return y

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, dy):
        pln, Wg, out, Wo, sig = ctx.saved_tensors
        M, DH = sig.shape
        dA = dy @ Wo                       # [M,DH]  cuBLAS dgrad of GEMM2
        dout = torch.empty_like(out); dglog = torch.empty_like(sig); a = torch.empty_like(sig)
        grid = lambda meta: (triton.cdiv(M, meta["BM"]),)
        _b2b_bwd_elem[grid](dA.contiguous(), sig, out, dout, dglog, a, M, DH)
        dWo = dy.t() @ a                   # [N,DH]  wgrad GEMM2
        dpln = dglog @ Wg.t()              # [M,D]   gate dgrad
        dWg = pln.t() @ dglog              # [D,DH]  gate wgrad
        return dpln, dWg, dout, dWo


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
             for bm in (32, 64, 128) for w in (4, 8) for s in (2, 3, 4)],
    key=["M", "D", "DH", "N"],
)
@triton.jit
def _gate_b2b_fwd_sig(
    pln_ptr, wg_ptr, out_ptr, wo_ptr, y_ptr, sig_ptr,
    M, D: tl.constexpr, DH: tl.constexpr, N: tl.constexpr,
    s_pm, s_pd, s_wgd, s_wgh, s_om, s_oh, s_won, s_woh, s_ym, s_yn, s_sm, s_sh,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D); rh = tl.arange(0, DH); rn = tl.arange(0, N)
    mmask = rm < M
    pln = tl.load(pln_ptr + rm[:, None] * s_pm + rd[None, :] * s_pd, mask=mmask[:, None], other=0.0)
    wg = tl.load(wg_ptr + rd[:, None] * s_wgd + rh[None, :] * s_wgh)
    sig = tl.sigmoid(tl.dot(pln, wg))
    tl.store(sig_ptr + rm[:, None] * s_sm + rh[None, :] * s_sh,
             sig.to(sig_ptr.dtype.element_ty), mask=mmask[:, None])
    o = tl.load(out_ptr + rm[:, None] * s_om + rh[None, :] * s_oh, mask=mmask[:, None], other=0.0)
    a = (sig * o.to(tl.float32)).to(wo_ptr.dtype.element_ty)
    woT = tl.load(wo_ptr + rn[None, :] * s_won + rh[:, None] * s_woh)
    y = tl.dot(a, woT)
    tl.store(y_ptr + rm[:, None] * s_ym + rn[None, :] * s_yn,
             y.to(y_ptr.dtype.element_ty), mask=mmask[:, None])


def gate_b2b_fn(pln, Wg, out, Wo):
    return _GateB2B.apply(pln, Wg, out, Wo)


def bench(fn):
    med, _, _ = triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8])
    return med


def bench_g(fn, gtn):
    med, _, _ = triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
                                        grad_to_none=gtn)
    return med


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


def run(B, d, dtype, seq_lens):
    DH = N = D = d
    print(f"# b2b back  B={B} D={D} DH={DH} N={N} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant cos fwd_ms fwdbwd_ms sp_fwd sp_fb")
    for L in seq_lens:
        M = B * L * L
        pln = torch.randn(M, D, device=DEVICE, dtype=dtype)
        Wg = torch.randn(D, DH, device=DEVICE, dtype=dtype) * 0.05   # to_gate.weight.T
        out = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        Wo = torch.randn(N, DH, device=DEVICE, dtype=dtype) * 0.05   # to_out.weight
        dy = torch.randn(M, N, device=DEVICE, dtype=dtype)

        def separate():  # current: cuBLAS gate-GEMM + fused_gate_out
            return fused_gate_out(F.linear(pln, Wg.t()), out, Wo)

        def b2b():
            return gate_b2b(pln, Wg, out, Wo)

        # correctness (fwd + grads) of the b2b autograd vs the separate autograd
        p1 = pln.clone().requires_grad_(True); g1 = Wg.clone().requires_grad_(True)
        o1 = out.clone().requires_grad_(True); w1 = Wo.clone().requires_grad_(True)
        p2 = pln.clone().requires_grad_(True); g2 = Wg.clone().requires_grad_(True)
        o2 = out.clone().requires_grad_(True); w2 = Wo.clone().requires_grad_(True)
        ys = fused_gate_out(F.linear(p1, g1.t()), o1, w1); ys.backward(dy)
        yb = gate_b2b_fn(p2, g2, o2, w2); yb.backward(dy)
        c = min(cos(yb, ys), cos(p2.grad, p1.grad), cos(g2.grad, g1.grad),
                cos(o2.grad, o1.grad), cos(w2.grad, w1.grad))

        sep = bench(separate); bb = bench(b2b)

        pf = pln.clone().requires_grad_(True); Wg_p = Wg.clone().requires_grad_(True)
        out_p = out.clone().requires_grad_(True); Wo_p = Wo.clone().requires_grad_(True)
        pf2 = pln.clone().requires_grad_(True); Wg_q = Wg.clone().requires_grad_(True)
        out_q = out.clone().requires_grad_(True); Wo_q = Wo.clone().requires_grad_(True)

        def sep_fb():
            fused_gate_out(F.linear(pf, Wg_p.t()), out_p, Wo_p).backward(dy)

        def b2b_fb():
            gate_b2b_fn(pf2, Wg_q, out_q, Wo_q).backward(dy)

        sfb = bench_g(sep_fb, [pf, Wg_p, out_p, Wo_p])
        bfb = bench_g(b2b_fb, [pf2, Wg_q, out_q, Wo_q])

        print(f"{L} separate 1.00000 {sep:.4f} {sfb:.4f} 1.00 1.00")
        print(f"{L} b2b {c:.5f} {bb:.4f} {bfb:.4f} {sep/bb:.2f} {sfb/bfb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d=128, dtype=torch.bfloat16, seq_lens=[384, 512, 768, 1024])
