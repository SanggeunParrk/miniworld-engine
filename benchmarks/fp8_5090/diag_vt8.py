"""ln_vt8 against the default ln_v: where the largest element differences sit (diagnostic)."""
import torch

from foldjet_engine.kernels.integrations import msa_f8_sm120 as X
from foldjet_engine.kernels.pair_weighted_averaging.cuda import sm80

for d, c, nt in ((64, 8, 64), (128, 8, 64), (128, 16, 32)):
    s, n = 70, 208
    torch.manual_seed(1)
    m = torch.randn(s, n, d, device="cuda").bfloat16()
    lnw = 1.0 + 0.1 * torch.randn(d, device="cuda")
    lnb = 0.1 * torch.randn(d, device="cuda")
    wv = (torch.randn(8 * c, d, device="cuda") * d**-0.5).bfloat16()
    v, _ = sm80.ln_v(m, lnw, lnb, wv, 1e-5, False, 1)
    vinv = torch.tensor([4.0], device="cuda")
    amax = torch.zeros(1, device="cuda")
    vt = X.ln_vt8(m, lnw, lnb, wv, 1e-5, vinv, amax, nt, 2)
    ref = (v.float() * 4.0).transpose(1, 2)
    y = ((m.float() - m.float().mean(-1, keepdim=True)) * torch.rsqrt(m.float().var(-1, unbiased=False, keepdim=True) + 1e-5) * lnw + lnb).bfloat16()
    ref32 = (y.float() @ wv.float().t() * 4.0).view(s, n, 8, c).permute(2, 0, 3, 1).reshape(8, s * c, n)
    got = vt.float()
    for name, r in (("ln_v", ref), ("fp32", ref32)):
        rel = (got - r).abs() / r.abs().clamp_min(2.0**-6)
        i = int(rel.argmax())
        idx = torch.unravel_index(torch.tensor(i), rel.shape)
        print(d, c, nt, name, "max rel", float(rel.max()), "at", [int(t) for t in idx], "got", float(got.flatten()[i]), "ref", float(r.flatten()[i]),
              "rms", float((got - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()), "n>0.13", int((rel > 0.13).sum()), flush=True)
    print("  ln_v vs fp32 ref rms", float((ref - ref32).pow(2).mean().sqrt() / ref32.pow(2).mean().sqrt()), "max", float((ref - ref32).abs().max()))
