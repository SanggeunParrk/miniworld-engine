"""Stage the core: one key block, then several, then several samples."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_core import attn_core  # noqa: E402

H, DS, NB, dev, bf = 16, 768, 24, "cuda", torch.bfloat16
D = DS // H


def ref_attn(qkvg, bias, block, S, L, gate=True):
    x = qkvg.view(S, L, 4 * DS).float()
    q, k, v, g = (x[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)).permute(0, 2, 1, 3) for i in range(4))
    b = bias[block * H:(block + 1) * H].float()[None]
    s = q @ k.transpose(-1, -2) + b
    p = torch.softmax(s, -1)
    o = p @ v
    if gate:
        o = o * torch.sigmoid(g)
    return o.permute(0, 2, 1, 3).reshape(S * L, DS)


for (S, L, zero_bias) in ((1, 64, True), (1, 64, False), (1, 128, False), (1, 768, False), (5, 128, False), (5, 768, False)):
    torch.manual_seed(0)
    base = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
    bias = torch.zeros(NB * H, L, L, device=dev, dtype=bf) if zero_bias else (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
    block = 0
    ref = ref_attn(base, bias, block, S, L)
    got = base.clone()
    attn_core(got, bias, block, S, H)
    torch.cuda.synchronize()
    a = got[:, :DS].float()
    err = float((a - ref).norm() / ref.norm())
    # where does it differ: per sample, per head, per row block
    per_head = [(float((a.view(S, L, H, D)[:, :, h] - ref.view(S, L, H, D)[:, :, h]).norm()
                       / ref.view(S, L, H, D)[:, :, h].norm())) for h in range(min(H, 4))]
    per_samp = [float((a.view(S, L, DS)[s] - ref.view(S, L, DS)[s]).norm() / ref.view(S, L, DS)[s].norm()) for s in range(S)]
    print(f"S{S} L{L} bias={'0' if zero_bias else 'rand'}: rel {err:.2e}  heads0-3 {['%.1e' % e for e in per_head]}  "
          f"samples {['%.1e' % e for e in per_samp]}", flush=True)
