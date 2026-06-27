"""Stage breakdown of the bias-only TriangleAttention.forward (use_self_attention=False).

The attention einsum itself is already ~optimal (single big GEMM). This locates
where the rest of the module-level time goes: LN, projections, the .contiguous()
rearranges, the gate, and to_out — so we know what is actually worth fusing.

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
from miniworld_kernels.modules.ops import sigmoid_gate

DEVICE = torch.device("cuda")


def bench(fn, grad_to_none=None):
    med, _, _ = triton.testing.do_bench(
        fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )
    return med


def run(B, d_pair, n_head, dtype, seq_lens):
    H = n_head
    D = d_pair // n_head
    print(f"# module breakdown  B={B} d_pair={d_pair} H={H} D={D} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L stage ms")
    for L in seq_lens:
        m = TriangleAttention(d_pair, n_head, use_self_attention=False).to(DEVICE)
        if dtype != torch.float32:
            m = m.to(dtype)
        pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = torch.rand(B, L, device=DEVICE) > 0.2

        def full_fwd():
            return m(pair, mask)

        # full forward
        t_full = bench(full_fwd)

        # --- stage isolation (mirror module.forward) ---
        with torch.no_grad():
            pln = m.ln_pair(pair)
            value = m.to_value(pln)
            biasp = m.to_bias(pln)
            value_r = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=H).contiguous()
            bias_r = rearrange(biasp, "B L L2 H -> B H L L2").contiguous()
            bias_m = bias_r.masked_fill(~mask[:, None, None, :], float("-inf"))
            attn = F.softmax(bias_m, dim=-1)
            out_att = torch.einsum("bhjk,bhikd->bhijd", attn, value_r)
            out_r = rearrange(out_att, "B H L L2 D -> B L L2 (H D)").contiguous()
            gated = sigmoid_gate(m.to_gate(pln), out_r)

        def s_ln():
            return m.ln_pair(pair)

        def s_value():
            return m.to_value(pln)

        def s_bias():
            return m.to_bias(pln)

        def s_value_rearrange():
            return rearrange(value, "B L L2 (H D) -> B H L L2 D", H=H).contiguous()

        def s_bias_rearrange():
            r = rearrange(biasp, "B L L2 H -> B H L L2").contiguous()
            return r.masked_fill(~mask[:, None, None, :], float("-inf"))

        def s_attn():
            a = F.softmax(bias_m, dim=-1)
            return torch.einsum("bhjk,bhikd->bhijd", a, value_r)

        def s_out_rearrange():
            return rearrange(out_att, "B H L L2 D -> B L L2 (H D)").contiguous()

        def s_gate():
            return sigmoid_gate(m.to_gate(pln), out_r)

        def s_out():
            return m.to_out(gated)

        stages = {
            "FULL_fwd": t_full,
            "ln": bench(s_ln),
            "to_value": bench(s_value),
            "to_bias": bench(s_bias),
            "value_rearrange": bench(s_value_rearrange),
            "bias_rearrange+mask": bench(s_bias_rearrange),
            "attn(softmax+einsum)": bench(s_attn),
            "out_rearrange": bench(s_out_rearrange),
            "gate(to_gate+sigmoid)": bench(s_gate),
            "to_out": bench(s_out),
        }
        for name, ms in stages.items():
            print(f"{L} {name} {ms:.4f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16,
        seq_lens=[128, 256, 512, 1024])
