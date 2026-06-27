"""Quick what-if probes for the bias-only module forward:

  baseline      : module as-is (torch LN, .contiguous() rearranges)
  triton_ln     : swap ln_pair to the triton layernorm kernel
  no_contig     : drop the .contiguous() on the value/out rearranges (let einsum
                  consume strided views)
  triton_ln+nc  : both

forward-only, under grad (training forward) and no_grad (inference). Locates the
realizable win before committing to a fused kernel.
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

from miniworld_kernels import kernels
from miniworld_kernels.modules import TriangleAttention
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.ops import sigmoid_gate

DEVICE = torch.device("cuda")


def bench(fn):
    med, _, _ = triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8])
    return med


def variant_forward(m, pair, mask, *, triton_ln, contig, fused_proj=False):
    H = m.n_head
    if triton_ln:
        pln = kernels.triton_layernorm(pair, m.ln_pair.weight, m.ln_pair.bias, m.ln_pair.eps)
    else:
        pln = m.ln_pair(pair)
    if fused_proj:
        # to_value (d->d), to_bias (d->H), to_gate (d->d) all read pln: one GEMM.
        d = m.to_value.weight.shape[0]
        Hd = m.to_bias.weight.shape[0]
        w = torch.cat([m.to_value.weight, m.to_bias.weight, m.to_gate.weight], dim=0)
        proj = torch.nn.functional.linear(pln, w)
        value, biasp, gate_pre = proj.split([d, Hd, d], dim=-1)
    else:
        value = m.to_value(pln)
        biasp = m.to_bias(pln)
        gate_pre = None
    value_r = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=H)
    bias_r = rearrange(biasp, "B L L2 H -> B H L L2")
    if contig:
        value_r = value_r.contiguous()
        bias_r = bias_r.contiguous()
    bias_r = bias_r.masked_fill(~mask[:, None, None, :], float("-inf"))
    attn = F.softmax(bias_r, dim=-1)
    out_att = torch.einsum("bhjk,bhikd->bhijd", attn, value_r)
    out_r = rearrange(out_att, "B H L L2 D -> B L L2 (H D)")
    if contig:
        out_r = out_r.contiguous()
    gate = gate_pre if gate_pre is not None else m.to_gate(pln)
    gated = sigmoid_gate(gate, out_r)
    return m.to_out(gated)


def run(B, d_pair, n_head, dtype, seq_lens):
    print(f"# module variants  B={B} d_pair={d_pair} H={n_head} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L variant grad_ms nograd_ms cos_vs_baseline")
    for L in seq_lens:
        m = TriangleAttention(d_pair, n_head, use_self_attention=False).to(DEVICE)
        # to_out/to_gate are zero/gating-init -> output is all zeros at init, which
        # makes cosine degenerate. Randomize so correctness checks are meaningful.
        with torch.no_grad():
            m.to_out.weight.normal_(0, 0.02)
            m.to_gate.weight.normal_(0, 0.02)
        if dtype != torch.float32:
            m = m.to(dtype)
        pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = torch.rand(B, L, device=DEVICE) > 0.2

        with torch.no_grad():
            ref = m(pair, mask).float()

        variants = {
            "baseline":      dict(triton_ln=False, contig=True),
            "triton_ln":     dict(triton_ln=True,  contig=True),
            "no_contig":     dict(triton_ln=False, contig=False),
            "triton_ln+nc":  dict(triton_ln=True,  contig=False),
            "ln+nc+fproj":   dict(triton_ln=True,  contig=False, fused_proj=True),
        }
        for name, kw in variants.items():
            try:
                with torch.no_grad():
                    out = variant_forward(m, pair, mask, **kw).float()
                cos = torch.nn.functional.cosine_similarity(
                    out.flatten(), ref.flatten(), dim=0
                ).item()
            except Exception as e:  # noqa: BLE001
                print(f"{L} {name} ERR {e}")
                continue

            pair.requires_grad_(True)
            g = bench(lambda: variant_forward(m, pair, mask, **kw))
            pair.requires_grad_(False)
            with torch.no_grad():
                ng = bench(lambda: variant_forward(m, pair, mask, **kw))
            print(f"{L} {name} {g:.4f} {ng:.4f} {cos:.5f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16,
        seq_lens=[128, 256, 512, 1024])
