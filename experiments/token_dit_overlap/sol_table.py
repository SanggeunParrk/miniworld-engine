"""Per-kernel latency and SoL for the fused token DiT block, against roofs measured in this same process.

Roofs: a large bf16 cuBLAS GEMM (the compute roof the block's GEMMs are judged by), an HBM read+write stream
(the roof the row passes are judged by), and roof/l2_roof.cu's TMA read of L2-resident tiles (the attention
core's roof -- its working set is L2-resident at these shapes).
"""
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import us                                                  # noqa: E402
from tdit import FusedTokenDiT                                        # noqa: E402
from tdit import kernels as K                                         # noqa: E402
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType    # noqa: E402

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 4, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
M = S * L

torch.manual_seed(0)
blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                             for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in blocks.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1: prm.add_(torch.randn_like(prm) * 0.1)
    for blk in blocks:
        blk.attention.to_out.weight.mul_(0.25); blk.transition.squeeze.weight.mul_(0.25)
blocks = blocks.to(bf).eval()
f = FusedTokenDiT(blocks, dtype=bf)
single = torch.randn(S, 1, L, DS, device=dev, dtype=bf)
cond = torch.randn(1, 1, L, DC, device=dev, dtype=bf).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev, dtype=bf)

# ---------------------------------------------------------------- roofs
def gemm_roof():
    n = 8192
    A = torch.randn(n, n, device=dev, dtype=bf); B = torch.randn(n, n, device=dev, dtype=bf)
    C = torch.empty(n, n, device=dev, dtype=bf)
    t = us(lambda: torch.mm(A, B, out=C))
    return 2 * n ** 3 / t / 1e6                                       # TFLOP/s

def hbm_roof():
    n = 256 * 2 ** 20                                                 # 512 MB bf16, 1 GB moved
    src = torch.empty(n, device=dev, dtype=bf); dst = torch.empty_like(src)
    t = us(lambda: dst.copy_(src))
    return 2 * n * 2 / t / 1e6                                        # TB/s

def l2_roof():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    ensure_cuda_home()
    v5 = "/home/psk6950/miniworld-engine-dit2/src/miniworld_engine/kernels/transition/cuda/anthropic_v5"
    ext = load_extension(name="tdit_l2_roof", sources=[str(Path(__file__).resolve().parent / "roof/l2_roof.cu")],
                         extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{v5}",
                                            "--expt-relaxed-constexpr", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
                         extra_cflags=["-std=c++17"], verbose=False)
    best = 0.0
    for mb in (16, 32):
        buf = torch.randn(mb * 2 ** 20 // 128, 64, device=dev, dtype=bf)
        for ctas, iters in ((264, 200), (528, 100)):
            t = us(lambda: ext.roof(buf, iters, ctas))
            best = max(best, ctas * iters * 64 * 64 * 2 / t / 1e6)
    return best

# ---------------------------------------------------------------- one block's ops
rows = []
with torch.no_grad():
    bias = f.hoist(pair)
    for _ in range(3): f.step(single, cond, bias)                     # picks the GEMM configs, builds the core
    torch.cuda.synchronize()
    G_ROOF, H_ROOF, L2_ROOF = gemm_roof(), hbm_roof(), l2_roof()
    print(f"roofs: cuBLAS bf16 {G_ROOF:.0f} TFLOP/s   HBM stream {H_ROOF:.2f} TB/s   TMA/L2 {L2_ROOF:.2f} TB/s\n", flush=True)

    buf = f._buffers(S, L, dev)
    x, xa, qkvg, y, h = (buf[k] for k in ("x", "xa", "qkvg2", "y", "h"))
    g1, g2 = f._cond(cond, L, DS)
    b, p = 1, f.per[1]

    def gemm(name, fn, flops, byts):
        t = us(fn); rows.append((name, t, f"{flops / t / 1e6:.0f} TF/s", 100 * flops / t / 1e6 / G_ROOF, "cuBLAS roof"))

    def mem(name, fn, byts, roof, label):
        t = us(fn); rows.append((name, t, f"{byts / t / 1e6:.2f} TB/s", 100 * byts / t / 1e6 / roof, label))

    gemm("q|k|v|g  GEMM", lambda: f._mm(xa, p["wqkvg"], qkvg, p["bqkvg"]), 2 * M * 4 * DS * DS, 0)
    f._cuda_core(qkvg, bias, b, S, H)
    core_bytes = (qkvg.numel() * 2 + L * L * H * 2 * S) / 1e6         # q|k|v|g read+write + the bias tiles read per sample
    mem("attention core (CUDA)", lambda: f._cuda_core(qkvg, bias, b, S, H), core_bytes * 1e6, L2_ROOF, "TMA/L2 roof")
    gemm("Wo  GEMM", lambda: f._mm(qkvg[:, :DS], p["wo"], y), 2 * M * DS * DS, 0)
    rb = M * DS * (4 + 2 + 4 + 2)                                     # x fp32 in/out, y bf16 in, xa bf16 out
    mem("resgate+AdaLN rows", lambda: K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, f.eps),
        rb, H_ROOF, "HBM roof")
    gemm("expand+SwiGLU GEMM", lambda: f._expand_swiglu(xa, p["wab_i"], h), 2 * M * 4 * DS * DS, 0)
    gemm("squeeze  GEMM", lambda: f._mm(h, p["ws"], y), 2 * M * DS * 2 * DS, 0)

print(f"L={L}  S={S}  M={M}  bf16, standalone do_bench (L2 evicted)\n")
print(f"{'kernel':<24}{'us':>8}{'rate':>14}{'% of roof':>12}  roof")
for n, t, r, s, lab in rows: print(f"{n:<24}{t:8.2f}{r:>14}{s:11.0f}%  {lab}")
print(f"\nsum of the block's kernels: {sum(r[1] for r in rows):.1f} us")
