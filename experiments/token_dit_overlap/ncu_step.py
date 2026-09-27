"""One token DiT block's kernels, inside an NVTX range, for ncu --nvtx-include 'prof/'."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                        # noqa: E402
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType    # noqa: E402

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 2, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
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
with torch.no_grad():
    bias = f.hoist(pair)
    for _ in range(4): f.step(single, cond, bias)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("prof")
    f.step(single, cond, bias)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
print("done", flush=True)
