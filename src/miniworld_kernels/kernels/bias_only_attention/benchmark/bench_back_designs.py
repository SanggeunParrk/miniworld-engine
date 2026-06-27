"""Bias-only BACK region (gate + to_out), done properly: which design wins?

    y = (sigmoid(gate) * out) @ Wo^T          gate,out: [M,DH]; Wo: [N,DH]

Candidates (gate assumed already projected; the to_gate GEMM is the same cuBLAS
for all and excluded):
  torch    : sigmoid(gate)*out (torch 2-pass) then F.linear            (materialize + cuBLAS)
  split    : ONE-pass triton sigmoid*mul -> A, then F.linear(A, Wo)     (trimul GateElem style:
             cuBLAS GEMM + fused elementwise; A materialized)
  fused    : fused_gate_out (gate folded into the to_out GEMM prologue; A never in HBM)

trimul's GateElem deliberately uses the SPLIT (its fused full-N tl.dot overflowed SM90
shared at N=512). bias-only has N=128, so `fused` is viable -- this measures whether it
actually beats the split here. fwd + fwd+bwd, correctness. Run via srun on a compute node.
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
    configs=[triton.Config({"BLK": b}, num_warps=w) for b in (1024, 2048, 4096) for w in (4, 8)],
    key=["n"],
)
@triton.jit
def _sigmul_fwd(g_ptr, o_ptr, a_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    g = tl.sigmoid(tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32))
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(a_ptr + off, (g * o).to(a_ptr.dtype.element_ty), mask=m)


@triton.autotune(
    configs=[triton.Config({"BLK": b}, num_warps=w) for b in (1024, 2048, 4096) for w in (4, 8)],
    key=["n"],
)
@triton.jit
def _sigmul_bwd(da_ptr, g_ptr, o_ptr, dg_ptr, do_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    da = tl.load(da_ptr + off, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32))
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(do_ptr + off, (da * s).to(do_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + off, (da * o * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=m)


class _SigMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, out):
        a = torch.empty_like(gate)
        n = gate.numel()
        grid = lambda M: (triton.cdiv(n, M["BLK"]),)
        _sigmul_fwd[grid](gate, out, a, n)
        ctx.save_for_backward(gate, out)
        return a

    @staticmethod
    def backward(ctx, da):
        gate, out = ctx.saved_tensors
        dg = torch.empty_like(gate); do = torch.empty_like(out)
        n = gate.numel()
        grid = lambda M: (triton.cdiv(n, M["BLK"]),)
        _sigmul_bwd[grid](da.contiguous(), gate, out, dg, do, n)
        return dg, do


def split_gate_out(gate, out, wo):
    a = _SigMul.apply(gate, out)
    return F.linear(a, wo)


def torch_gate_out(gate, out, wo):
    return F.linear(torch.sigmoid(gate) * out, wo)


def bench(fn, gtn=None):
    med, _, _ = triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
                                        grad_to_none=gtn or [])
    return med


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


def run(B, d, dtype, seq_lens):
    DH = N = d
    print(f"# back region (gate+to_out)  B={B} DH={DH} N={N} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant cos fwd_ms fwdbwd_ms sp_fwd sp_fb")
    cands = {"torch": torch_gate_out, "split": split_gate_out, "fused": fused_gate_out}
    for L in seq_lens:
        M = B * L * L
        gate = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        out = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        wo = torch.randn(N, DH, device=DEVICE, dtype=dtype) * 0.05
        dy = torch.randn(M, N, device=DEVICE, dtype=dtype)
        ref = torch_gate_out(gate, out, wo)

        res = {}
        for name, fn in cands.items():
            with torch.no_grad():
                c = cos(fn(gate, out, wo), ref)
            ff = bench(lambda: fn(gate, out, wo))
            g1 = gate.clone().requires_grad_(True); o1 = out.clone().requires_grad_(True)
            w1 = wo.clone().requires_grad_(True)

            def full():
                fn(g1, o1, w1).backward(dy)

            fb = bench(full, gtn=[g1, o1, w1])
            res[name] = (c, ff, fb)

        bf, bfb = res["torch"][1], res["torch"][2]
        for name in cands:
            c, ff, fb = res[name]
            print(f"{L} {name} {c:.5f} {ff:.4f} {fb:.4f} {bf/ff:.2f} {bfb/fb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    # d-sweep: at larger d (=N), the fused tl.dot approaches the SM90 shared-mem
    # regime where trimul switched to the cuBLAS split -- does the winner flip?
    for d, Ls in [(128, [512, 1024]), (256, [512, 768]), (512, [384, 512])]:
        run(B=1, d=d, dtype=torch.bfloat16, seq_lens=Ls)
