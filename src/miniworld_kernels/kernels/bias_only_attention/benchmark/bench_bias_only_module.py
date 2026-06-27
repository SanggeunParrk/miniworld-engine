"""Bias-only TriangleAttention: optimized (triton) vs pytorch baseline.

Verifies forward AND backward correctness (output + input/weight grads share
weights between the two modules) and times full forward and forward+backward.

This is the canonical bench for the bias-only optimization. Parseable stdout;
capture to .out, then render .md/.png separately. Run via srun on a compute node.
"""

import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import triton

from miniworld_kernels.modules import TriangleAttention
from miniworld_kernels.modules.exceptions import ImplementationType

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


def make_module(d_pair, n_head, impl, dtype):
    m = TriangleAttention(d_pair, n_head, use_self_attention=False, implementation=impl)
    # randomize the zero/gating-init output layers so grads are non-degenerate
    with torch.no_grad():
        m.to_out.weight.normal_(0, 0.02)
        m.to_gate.weight.normal_(0, 0.02)
    m = m.to(DEVICE)
    if dtype != torch.float32:
        m = m.to(dtype)
    return m


def run(B, d_pair, n_head, dtype, seq_lens):
    print(f"# bias-only module  B={B} d_pair={d_pair} H={n_head} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("# columns: L impl out_cos dpair_cos wgrad_cos infer_cos infer_ms fwdbwd_ms sp_infer sp_fb")
    for L in seq_lens:
        base = make_module(d_pair, n_head, ImplementationType.PYTORCH, dtype)
        opt = make_module(d_pair, n_head, ImplementationType.TRITON, dtype)
        opt.load_state_dict(base.state_dict())  # share weights

        pair0 = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)
        mask = torch.rand(B, L, device=DEVICE) > 0.2
        dy = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=dtype)

        results = {}
        grads = {}
        for name, m in [("pytorch", base), ("triton", opt)]:
            p = pair0.detach().clone().requires_grad_(True)
            m.zero_grad(set_to_none=True)
            y = m(p, mask)
            y.backward(dy)
            results[name] = y.detach()
            grads[name] = (p.grad.detach(), m.to_out.weight.grad.detach().clone())

        out_cos = cos(results["triton"], results["pytorch"])
        dpair_cos = cos(grads["triton"][0], grads["pytorch"][0])
        wgrad_cos = cos(grads["triton"][1], grads["pytorch"][1])

        # inference (no_grad) correctness -- exercises the fused inference path
        with torch.no_grad():
            yb_i = base(pair0, mask)
            yo_i = opt(pair0, mask)
        infer_cos = cos(yo_i, yb_i)

        timings = {}
        for name, m in [("pytorch", base), ("triton", opt)]:
            p = pair0.detach().clone().requires_grad_(True)

            def infer():
                with torch.no_grad():
                    return m(pair0, mask)

            def full():
                m.zero_grad(set_to_none=True)
                y = m(p, mask)
                y.backward(dy)

            timings[name] = (bench(infer), bench(full, grad_to_none=[p]))

        sp_infer = timings["pytorch"][0] / timings["triton"][0]
        sp_fb = timings["pytorch"][1] / timings["triton"][1]
        print(
            f"{L} pytorch 1.00000 1.00000 1.00000 1.00000 "
            f"{timings['pytorch'][0]:.4f} {timings['pytorch'][1]:.4f} 1.00 1.00"
        )
        print(
            f"{L} triton {out_cos:.5f} {dpair_cos:.5f} {wgrad_cos:.5f} {infer_cos:.5f} "
            f"{timings['triton'][0]:.4f} {timings['triton'][1]:.4f} {sp_infer:.2f} {sp_fb:.2f}"
        )
        print(flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(B=1, d_pair=128, n_head=4, dtype=torch.bfloat16,
        seq_lens=[128, 192, 256, 384, 512, 768, 1024])
    # larger d_hidden: exercises the split back path (DH>=256)
    run(B=1, d_pair=256, n_head=4, dtype=torch.bfloat16, seq_lens=[384, 512, 768])
    run(B=1, d_pair=512, n_head=4, dtype=torch.bfloat16, seq_lens=[384, 512])
