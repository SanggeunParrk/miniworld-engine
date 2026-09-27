"""fp8 go/no-go by emulation: fake-quantize (e4m3, quantize -> dequantize) the inputs of chosen GEMMs in the fused
step and measure rel_rms against the IEEE fp32 reference, per GEMM group. Scales: per row for activations, per output
channel for weights (amax / 448), the usual fp8 inference recipe. Emulates quantization error only -- not Hopper's
reduced-precision fp8 accumulation, which a real kernel would promote to fp32 every few k-steps."""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                    # noqa: E402
from tdit import runner as R                                      # noqa: E402
from tdit.cuda_core import attn_core                                     # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
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


PER_TENSOR = False


def fq(t, dim):
    """e4m3 quantize -> dequantize with one scale per slice along `dim` (amax / 448), or one for the whole tensor."""
    tf = t.float()
    s = (tf.abs().amax().clamp_min(1e-12) if PER_TENSOR else tf.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)) / 448.0
    return ((tf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).to(t.dtype)


f = FusedTokenDiT(bf_blocks, dtype=bf)
bias = f.hoist(z_bf)
MM, EXP = f._mm, f._expand_swiglu
GROUPS = {                                                  # (M, N, K) of each plain GEMM
    "q|k|v|g": (S * L, 4 * DS, DS), "Wo": (S * L, DS, DS), "squeeze": (S * L, DS, 2 * DS),
}


def run(which):
    def mm(A, W, out, b=None):
        name = next((n for n, k in GROUPS.items() if (A.shape[0], W.shape[0], A.shape[1]) == k), None)
        if name in which:
            A, W = fq(A, 1), fq(W, 1)
        return MM(A, W, out, b)

    def exp(xa, wab_i, h):
        if "expand" in which:
            xa, wab_i = fq(xa, 1), fq(wab_i, 1)
        return EXP(xa, wab_i, h)

    f._mm, f._expand_swiglu = mm, exp
    out = f.step(s_bf, c_bf, bias).float().reshape(S * L, DS)
    f._mm, f._expand_swiglu = MM, EXP
    return float((out - ref).norm() / ref.norm())


with torch.no_grad():
    f.step(s_bf, c_bf, bias)                                  # picks + builds before any wrapping
    base = run(())
    print(f"L={L}: bf16 as shipped rel_rms {base:.2e}   (engine bf16 path is 1.1e-2)", flush=True)
    for which in (("expand",), ("squeeze",), ("expand", "squeeze"), ("q|k|v|g",), ("Wo",),
                  ("q|k|v|g", "Wo"), ("q|k|v|g", "Wo", "expand", "squeeze")):
        e = run(which)
        print(f"  fp8 {'+'.join(which):<30} rel_rms {e:.2e}  (x{e / base:.2f})", flush=True)
    PER_TENSOR = True
    for which in (("q|k|v|g",), ("q|k|v|g", "Wo")):
        e = run(which)
        print(f"  fp8 per-tensor {'+'.join(which):<19} rel_rms {e:.2e}  (x{e / base:.2f})", flush=True)
