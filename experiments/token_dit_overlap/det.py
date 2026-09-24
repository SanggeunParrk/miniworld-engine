"""Is the step deterministic? Same inputs, repeated, nothing else touching the buffers.

The A/B harness found off-vs-off 1.6e-3 at L768, which is the size of the whole error budget, so this isolates
where it comes from: the CUDA core, the Triton core, or the block outside the core.
"""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                        # noqa: E402
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--blocks", type=int, default=24)
p.add_argument("--reps", type=int, default=6)
p.add_argument("--order", default="cuda,gated2", help="which core is exercised first")
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, a.blocks, "cuda", torch.bfloat16
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
    for core in a.order.split(","):
        f.core = core
        if core == "gated2":
            f._cuda_core = False                                      # pins the Triton core
        warm = f.step(single, cond, bias).float().clone()
        outs = [f.step(single, cond, bias).float().clone() for _ in range(a.reps)]
        ref = outs[-1]
        d = [float((o - ref).norm() / ref.norm()) for o in [warm] + outs[:-1]]
        print(f"L={L} blocks={NB} core={core:7s} vs the LAST run -- warm-up {d[0]:.1e}, "
              f"then {' '.join(f'{x:.1e}' for x in d[1:])}", flush=True)
