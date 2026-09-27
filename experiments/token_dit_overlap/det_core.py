"""The CUDA attention core alone, same input, many times, compared BITWISE. Isolates the core from the step."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from tdit.cuda_core import attn_core                                  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--reps", type=int, default=40)
p.add_argument("--nb", type=int, default=24)
a = p.parse_args()
L, S, H, DS, dev, bf = a.length, 5, 16, 768, "cuda", torch.bfloat16
torch.manual_seed(0)
src = (torch.randn(S * L, 4 * DS, device=dev, dtype=bf) * DS ** -0.5).contiguous()
bias = (torch.randn(a.nb * H, L, L, device=dev, dtype=bf) * 0.3).contiguous()
qkvg = torch.empty_like(src)
outs = []
for i in range(a.reps):
    qkvg.copy_(src)
    attn_core(qkvg, bias, 3, S, H)
    outs.append(qkvg[:, :DS].clone())
ref = outs[0]
bad = [(i, int((o != ref).sum())) for i, o in enumerate(outs) if not torch.equal(o, ref)]
print(f"L={L} nb={a.nb} reps={a.reps}: {len(bad)} of {a.reps} runs differ bitwise from run 0", flush=True)
if bad:
    i, n = bad[0]
    d = (outs[i].float() - ref.float())
    print(f"  first differing run {i}: {n} elements of {ref.numel()} "
          f"({100 * n / ref.numel():.4f} %), max |d| {float(d.abs().max()):.3e}, "
          f"rel {float(d.norm() / ref.float().norm()):.2e}", flush=True)
    rows = torch.nonzero((outs[i] != ref).any(1)).flatten()
    print(f"  rows touched: {rows.numel()} (first {rows[:8].tolist()}), samples "
          f"{sorted(set((rows // L).tolist()))[:8]}", flush=True)
