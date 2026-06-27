"""Micro-benchmark for the bias-only triangle-attention op.

Isolates the op being optimized:

    out[b,h,i,j,d] = sum_k softmax_k(bias[b,h,j,k]) * value[b,h,i,k,d]

Candidates
----------
- torch_einsum : current module path (softmax + einsum "bhjk,bhikd->bhijd")
- torch_bmm    : softmax once, then broadcast batched matmul A.unsqueeze(2) @ V
- triton_flash : vendored team-gm flash-style kernel (recomputes softmax per i)

For each: correctness (cosine vs torch_einsum reference, fp32), forward median ms,
and forward+backward median ms. Parseable stdout; writes .out -> render .md/.png
separately (team-gm harness style).

Run via srun on a compute node (see CLAUDE.md). Never on the login node.
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

from miniworld_kernels.kernels import triton_bias_only_attention
from miniworld_kernels.kernels.bias_only_attention.triton.fused import bias_only_fused_fwd

DEVICE = torch.device("cuda")


def torch_einsum(value, bias):
    attention = F.softmax(bias, dim=-1)
    return torch.einsum("bhjk,bhikd->bhijd", attention, value)


def torch_bmm(value, bias):
    attention = F.softmax(bias, dim=-1)  # [B,H,Lj,Lk]
    # [B,H,1,Lj,Lk] @ [B,H,Li,Lk,D] -> [B,H,Li,Lj,D]
    return torch.matmul(attention.unsqueeze(2), value)


def triton_flash(value, bias):
    return triton_bias_only_attention(value, bias)


def triton_fused(value, bias):
    return bias_only_fused_fwd(value, bias)


CANDIDATES = {
    "torch_einsum": torch_einsum,
    "triton_flash": triton_flash,
    "triton_fused": triton_fused,
}


def make_inputs(B, H, L, D, dtype):
    value = torch.randn(B, H, L, L, D, device=DEVICE, dtype=dtype)
    bias = torch.randn(B, H, L, L, device=DEVICE, dtype=dtype)
    return value, bias


def cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def bench_time(func, grad_to_none=None):
    med, p20, p80 = triton.testing.do_bench(
        func, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )
    return med


def run(B, H, D, dtypes, seq_lens):
    print(f"# bias_only micro-bench  B={B} H={H} D={D}")
    print(f"# device={torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("# columns: dtype L impl cosine fwd_ms fwdbwd_ms")
    for dtype in dtypes:
        dtype_name = {torch.float32: "fp32", torch.bfloat16: "bf16",
                      torch.float16: "fp16"}[dtype]
        for L in seq_lens:
            value, bias = make_inputs(B, H, L, D, dtype)
            ref = torch_einsum(value, bias).float()
            for name, fn in CANDIDATES.items():
                # correctness (fwd)
                try:
                    out = fn(value, bias)
                    cos = cosine(out, ref)
                except Exception as e:  # noqa: BLE001
                    print(f"{dtype_name} {L} {name} ERROR_fwd {e}")
                    continue

                # forward timing
                v = value.detach().clone()
                b = bias.detach().clone()
                try:
                    fwd_ms = bench_time(lambda: fn(v, b))
                except Exception as e:  # noqa: BLE001
                    print(f"{dtype_name} {L} {name} {cos:.5f} ERROR_fwd_time {e}")
                    continue

                # fwd+bwd timing
                vg = value.detach().clone().requires_grad_(True)
                bg = bias.detach().clone().requires_grad_(True)
                dy = torch.randn(B, H, L, L, D, device=DEVICE, dtype=dtype)

                def full():
                    o = fn(vg, bg)
                    o.backward(dy)

                try:
                    fb_ms = bench_time(full, grad_to_none=[vg, bg])
                except Exception as e:  # noqa: BLE001
                    print(f"{dtype_name} {L} {name} {cos:.5f} {fwd_ms:.4f} ERROR_bwd {e}")
                    continue

                print(f"{dtype_name} {L} {name} {cos:.5f} {fwd_ms:.4f} {fb_ms:.4f}")
            print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run(
        B=1,
        H=4,
        D=32,
        dtypes=[torch.bfloat16, torch.float32],
        seq_lens=[128, 256, 384, 512, 768, 1024],
    )
