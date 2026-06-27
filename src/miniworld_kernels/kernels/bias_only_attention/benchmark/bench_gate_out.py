"""Micro-bench: fused sigmoid-gate + to_out vs torch baseline.

baseline: (sigmoid(gate) * out_r) @ Wo^T   (elementwise kernel + cuBLAS GEMM)
fused   : single triton GEMM with the gating in the prologue

gate/out_r are [M, DH] with M = B*L*L. Verifies fwd+bwd grads, times both.
Run via srun on a compute node.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import triton

from miniworld_kernels.kernels.bias_only_attention.triton.gate_out import fused_gate_out

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


def torch_baseline(gate, out_r, wo):
    return torch.nn.functional.linear(torch.sigmoid(gate) * out_r, wo)


def run(B, d_pair, n_head, dtype, seq_lens):
    DH = d_pair
    N = d_pair
    print(f"# fused gate+out  B={B} DH={DH} N={N} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L out_cos dgate_cos doutr_cos dwo_cos base_fwd fused_fwd base_fb fused_fb sp_fwd sp_fb")
    for L in seq_lens:
        M = B * L * L
        gate = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        out_r = torch.randn(M, DH, device=DEVICE, dtype=dtype)
        wo = torch.randn(N, DH, device=DEVICE, dtype=dtype) * 0.02

        # correctness
        g1 = gate.clone().requires_grad_(True); r1 = out_r.clone().requires_grad_(True); w1 = wo.clone().requires_grad_(True)
        g2 = gate.clone().requires_grad_(True); r2 = out_r.clone().requires_grad_(True); w2 = wo.clone().requires_grad_(True)
        dy = torch.randn(M, N, device=DEVICE, dtype=dtype)
        yb = torch_baseline(g1, r1, w1); yb.backward(dy)
        yf = fused_gate_out(g2, r2, w2); yf.backward(dy)
        oc = cos(yf, yb); gc = cos(g2.grad, g1.grad); rc = cos(r2.grad, r1.grad); wc = cos(w2.grad, w1.grad)

        # timing
        bf = bench(lambda: torch_baseline(gate, out_r, wo))
        ff = bench(lambda: fused_gate_out(gate, out_r, wo))

        gg = gate.clone().requires_grad_(True); rr = out_r.clone().requires_grad_(True); ww = wo.clone().requires_grad_(True)

        def base_full():
            y = torch_baseline(gg, rr, ww); y.backward(dy)

        def fused_full():
            y = fused_gate_out(gg, rr, ww); y.backward(dy)

        bfb = bench(base_full, grad_to_none=[gg, rr, ww])
        ffb = bench(fused_full, grad_to_none=[gg, rr, ww])

        print(f"{L} {oc:.5f} {gc:.5f} {rc:.5f} {wc:.5f} "
              f"{bf:.4f} {ff:.4f} {bfb:.4f} {ffb:.4f} {bf/ff:.2f} {bfb/ffb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16,
        seq_lens=[256, 384, 512, 768, 1024])
