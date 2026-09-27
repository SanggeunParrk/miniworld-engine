"""Characterise the CUDA core's rare nondeterminism: which CTA, which rows, and what kind of error.

A CTA owns (m_tile, head, sample) = 128 rows x 48 columns. If a bad row's 48 values are all off by one common
factor, the row's denominator (or the whole score row) moved; if only some are off, it is the numerator (PV).
The fp32 reference says which run is right.
"""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from tdit.cuda_core import attn_core                                  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--reps", type=int, default=300)
p.add_argument("--block", type=int, default=3)
a = p.parse_args()
L, S, H, DS, DH, NB, dev, bf = a.length, 5, 16, 768, 48, 24, "cuda", torch.bfloat16
torch.manual_seed(0)
src = (torch.randn(S * L, 4 * DS, device=dev, dtype=bf) * DS ** -0.5).contiguous()
bias = (torch.randn(NB * H, L, L, device=dev, dtype=bf) * 0.3).contiguous()

# fp32 reference in the kernel's convention: logits already in the exp2 domain, bias added, no scale
q, k, v, g = (src[:, i * DS:(i + 1) * DS].float().view(S, L, H, DH).transpose(1, 2) for i in range(4))
bb = bias[a.block * H:(a.block + 1) * H].float()
sc = q @ k.transpose(-1, -2) + bb[None]
pr = torch.exp2(sc - sc.amax(-1, keepdim=True)); pr = pr / pr.sum(-1, keepdim=True)
ref = (torch.sigmoid(g) * (pr @ v)).transpose(1, 2).reshape(S * L, DS)

qkvg = torch.empty_like(src)
outs = []
for i in range(a.reps):
    qkvg.copy_(src)
    attn_core(qkvg, bias, a.block, S, H)
    outs.append(qkvg[:, :DS].clone())

# majority output = the one most runs agree with
base = outs[0]
same = sum(torch.equal(o, base) for o in outs)
if same < a.reps // 2:
    base = outs[1]
bad = [i for i, o in enumerate(outs) if not torch.equal(o, base)]
err = lambda o: float((o.float() - ref).norm() / ref.norm())
print(f"L={L}: {len(bad)}/{a.reps} runs differ from the majority; majority err vs fp32 {err(base):.3e}", flush=True)
for i in bad[:6]:
    o = outs[i]
    diff = (o != base)
    rows = torch.nonzero(diff.any(1)).flatten()
    cols = torch.nonzero(diff.any(0)).flatten()
    heads = sorted(set((cols // DH).tolist()))
    samples = sorted(set((rows // L).tolist()))
    tiles = sorted(set(((rows % L) // 128).tolist()))
    r0 = int(rows[0]); hc = slice(heads[0] * DH, heads[0] * DH + DH)
    ratio = (o[r0, hc].float() / base[r0, hc].float())
    print(f"  run {i}: {int(diff.sum())} elems, {rows.numel()} rows, heads {heads}, samples {samples}, "
          f"m_tiles {tiles}, rows-in-tile {sorted(set(((rows % L) % 128).tolist()))[:16]}", flush=True)
    print(f"     err vs fp32: this run {err(o):.3e} (majority {err(base):.3e});  row {r0} head {heads[0]} "
          f"ratio min/max {float(ratio.min()):.5f}/{float(ratio.max()):.5f}", flush=True)
