"""Per-operation timing of the OPTIMIZED bias-only path (TRITON).

For each op, reports:
  - inference : forward only, under torch.no_grad()
  - fwd+bwd   : a standalone forward+backward (leaf inputs require grad)

This locates the remaining cost (esp. LN + the three projections, the candidates
for LayerNormLinear fusion). Run via srun on a compute node.
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

DEVICE = torch.device("cuda")


def bench(fn, grad_to_none=None):
    med, _, _ = triton.testing.do_bench(
        fn, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grad_to_none or [],
    )
    return med


def infer(fn):
    with torch.no_grad():
        return bench(fn)


def fwdbwd(make_inputs, fn):
    """make_inputs() -> (leaf_list, args); time forward+backward."""
    leaves, args = make_inputs()
    dy_holder = {}

    def step():
        y = fn(*args)
        if "dy" not in dy_holder:
            dy_holder["dy"] = torch.randn_like(y)
        y.backward(dy_holder["dy"])

    return bench(step, grad_to_none=leaves)


def run(B, d_pair, n_head, dtype, seq_lens):
    H = n_head
    D = d_pair // n_head
    print(f"# op breakdown (optimized triton path)  B={B} d_pair={d_pair} H={H} D={D} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}")
    print("# columns: L op infer_ms fwdbwd_ms")
    for L in seq_lens:
        m = TriangleAttention(d_pair, n_head, use_self_attention=False,
                              implementation=ImplementationType.TRITON).to(DEVICE)
        with torch.no_grad():
            m.to_out.weight.normal_(0, 0.02)
            m.to_gate.weight.normal_(0, 0.02)
        m = m.to(dtype)
        Wv, Wb, Wg, Wo = m.to_value.weight, m.to_bias.weight, m.to_gate.weight, m.to_out.weight
        lw, lb, eps = m.ln_pair.weight, m.ln_pair.bias, m.ln_pair.eps

        pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = torch.rand(B, L, device=DEVICE) > 0.2
        # materialized intermediates (for op isolation)
        with torch.no_grad():
            pln = kernels.layernorm_kernel(pair, lw, lb, eps)
            value = F.linear(pln, Wv)
            value_r = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=H)
            biasp = rearrange(F.linear(pln, Wb), "B L L2 H -> B H L L2")
            biasp = biasp.masked_fill(~mask[:, None, None, :], float("-inf"))
            attn = F.softmax(biasp, dim=-1)
            out_att = torch.einsum("bhjk,bhikd->bhijd", attn, value_r)
            out_r = rearrange(out_att, "B H L L2 D -> B L L2 (H D)")
            gate = F.linear(pln, Wg)

        def leaf(t):
            return t.detach().clone().requires_grad_(True)

        ops = {
            "ln (layernorm_kernel)": (
                lambda: kernels.layernorm_kernel(pair, lw, lb, eps),
                lambda: ([p := leaf(pair)], (p, lw, lb, eps)),
                lambda *a: kernels.layernorm_kernel(*a),
            ),
            "to_value (Linear d->d)": (
                lambda: F.linear(pln, Wv),
                lambda: ([p := leaf(pln)], (p, Wv)),
                F.linear,
            ),
            "to_bias (Linear d->H)": (
                lambda: F.linear(pln, Wb),
                lambda: ([p := leaf(pln)], (p, Wb)),
                F.linear,
            ),
            "attention (softmax+einsum)": (
                lambda: torch.einsum("bhjk,bhikd->bhijd", F.softmax(biasp, -1), value_r),
                lambda: ([b := leaf(biasp), v := leaf(value_r)], (b, v)),
                lambda b, v: torch.einsum("bhjk,bhikd->bhijd", F.softmax(b, -1), v),
            ),
            "to_gate (Linear d->d)": (
                lambda: F.linear(pln, Wg),
                lambda: ([p := leaf(pln)], (p, Wg)),
                F.linear,
            ),
            "fused_gate_out": (
                lambda: kernels.fused_gate_out(gate, out_r, Wo),
                lambda: ([g := leaf(gate), o := leaf(out_r)], (g, o, Wo)),
                kernels.fused_gate_out,
            ),
        }

        for name, (inf_fn, mk, fb_fn) in ops.items():
            im = infer(inf_fn)
            fb = fwdbwd(mk, fb_fn)
            print(f"{L} | {name} | {im:.4f} | {fb:.4f}")
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16, seq_lens=[512, 1024])
