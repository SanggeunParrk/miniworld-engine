"""Honest 3-way: ORIGINAL team-gm forward vs current-pytorch vs current-triton.

The module bench's "pytorch" already has the unconditional .contiguous() removal,
so triton-vs-pytorch understates the real win. This reconstructs the UNMODIFIED
team-gm forward (torch LayerNorm + .contiguous() rearranges + torch sigmoid_gate +
cuBLAS) and measures the current triton path against it. Reports absolute ms.

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
from einops import rearrange

from miniworld_kernels.modules import TriangleAttention
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.ops import sigmoid_gate

DEVICE = torch.device("cuda")


def bench(fn, gtn=None):
    med, _, _ = triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
                                        grad_to_none=gtn or [])
    return med


def original_forward(m, pair, mask):
    """Unmodified team-gm bias-only forward: torch LN, .contiguous() rearranges,
    torch sigmoid_gate, cuBLAS projections."""
    H = m.n_head
    pln = F.layer_norm(pair, (pair.shape[-1],), m.ln_pair.weight, m.ln_pair.bias, m.ln_pair.eps)
    value = m.to_value(pln)
    bias = m.to_bias(pln)
    value = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=H).contiguous()
    bias = rearrange(bias, "B L L2 H -> B H L L2").contiguous()
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
    attn = F.softmax(bias, dim=-1)
    out = torch.einsum("bhjk,bhikd->bhijd", attn, value)
    out = rearrange(out, "B H L L2 D -> B L L2 (H D)").contiguous()
    out = sigmoid_gate(m.to_gate(pln), out)
    return m.to_out(out)


def run(B, d_pair, n_head, dtype, seq_lens):
    print(f"# vs ORIGINAL  B={B} d_pair={d_pair} H={n_head} D={d_pair // n_head} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant infer_ms fwdbwd_ms infer_x fwdbwd_x  (x = vs original)")
    for L in seq_lens:
        m = TriangleAttention(d_pair, n_head, use_self_attention=False,
                              implementation=ImplementationType.TRITON).to(DEVICE)
        with torch.no_grad():
            m.to_out.weight.normal_(0, 0.02)
            m.to_gate.weight.normal_(0, 0.02)
        m = m.to(dtype)
        pair0 = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = torch.rand(B, L, device=DEVICE) > 0.2
        dy = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)

        def orig_inf():
            with torch.no_grad():
                return original_forward(m, pair0, mask)

        def cur_inf():
            with torch.no_grad():
                return m(pair0, mask)

        p1 = pair0.clone().requires_grad_(True)
        p2 = pair0.clone().requires_grad_(True)

        def orig_fb():
            m.zero_grad(set_to_none=True)
            original_forward(m, p1, mask).backward(dy)

        def cur_fb():
            m.zero_grad(set_to_none=True)
            m(p2, mask).backward(dy)

        oi = bench(orig_inf)
        ci = bench(cur_inf)
        ofb = bench(orig_fb, gtn=[p1])
        cfb = bench(cur_fb, gtn=[p2])
        print(f"{L} original {oi:.4f} {ofb:.4f} 1.00 1.00")
        print(f"{L} triton {ci:.4f} {cfb:.4f} {oi/ci:.2f} {ofb/cfb:.2f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16, seq_lens=[512, 1024])
    run(B=1, d_pair=256, n_head=4, dtype=torch.bfloat16, seq_lens=[512, 768])
