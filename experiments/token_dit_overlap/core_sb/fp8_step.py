"""The whole step with the hoisted bias in fp8: what it buys and what it costs, against the IEEE fp32 reference."""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                    # noqa: E402
from tdit import runner as R                                      # noqa: E402
from attn_fp8 import attention_fp8, quantize_bias                 # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
torch.backends.cuda.matmul.allow_tf32 = False
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType   # noqa: E402

torch.manual_seed(0)
ref_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                 for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in ref_blocks.parameters():
        if prm.ndim == 2:
            prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1:
            prm.add_(torch.randn_like(prm) * 0.1)
    for blk in ref_blocks:
        blk.attention.to_out.weight.mul_(0.25)
        blk.transition.squeeze.weight.mul_(0.25)
bf_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                for _ in range(NB)).to(dev)
bf_blocks.load_state_dict(ref_blocks.state_dict())
bf_blocks = bf_blocks.to(bf).eval()
single = torch.randn(S, 1, L, DS, device=dev)
cond = torch.randn(1, 1, L, DC, device=dev).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev)
s_bf, c_bf, z_bf = single.to(bf), cond.to(bf), pair.to(bf)
with torch.no_grad():
    x = single
    for blk in ref_blocks:
        x = blk(x, cond, pair)
    ref = x.reshape(S * L, DS)

f = FusedTokenDiT(bf_blocks, dtype=bf)
bias = f.hoist(z_bf)
qb, scale, qdesc = quantize_bias(bias)


def time_us(fn, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            g.replay()
        en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1e3 / reps)
    return statistics.median(out)


def rel(got):
    return float((got.float().reshape(S * L, DS) - ref).norm() / ref.norm())


with torch.no_grad():
    print(f"L={L} S={S} {NB} blocks bf16, per block; rel_rms vs IEEE fp32", flush=True)
    base_t, base_e = time_us(lambda: f.step(s_bf, c_bf, bias)) / NB, rel(f.step(s_bf, c_bf, bias))
    print(f"  bias bf16 (packaged)   {base_t:6.1f} us   {base_e:.2e}", flush=True)
    orig = R.attention_gated_in_place2
    R.attention_gated_in_place2 = lambda q, k, v, g, bd, b, prec: attention_fp8(q, k, v, g, qdesc, scale, b, prec)
    try:
        t8, e8 = time_us(lambda: f.step(s_bf, c_bf, bias)) / NB, rel(f.step(s_bf, c_bf, bias))
        print(f"  bias fp8 (e4m3)        {t8:6.1f} us   {e8:.2e}   {base_t - t8:+.1f} us, error x{e8 / base_e:.1f}",
              flush=True)
    finally:
        R.attention_gated_in_place2 = orig
