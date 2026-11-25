import sys, torch
from foldjet_engine.modules import triangle as T
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
L = 384
for starting in (True, False):
  for s in (1.0, 4.0, 10.0, 25.0):
    res = {}
    for qs in (False, True):
        K.QS_INFER = qs
        torch.manual_seed(5)
        m = T.TriangleAttention(d_pair=128, n_head=4, starting=starting, backend="cuda").cuda().float().eval()
        r = T.TriangleAttention(d_pair=128, n_head=4, starting=starting, backend="pytorch").cuda().float().eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.normal_(0, 0.5 if p.dim() == 2 else 0.2)
                if n == "to_bias.weight": p.mul_(s)
            r.load_state_dict(m.state_dict())
        x = (torch.randn(1, L, L, 128, device="cuda") * 2).bfloat16()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): y = m(x).float(); yr = r(x).float()
        res[qs] = (y, yr)
    y0, yr = res[False]; y1, _ = res[True]
    print(f"start={starting} bias x{s}: |y|max {y0.abs().max():.2f} base-vs-ref {(y0-yr).abs().max():.3f} QS-vs-ref {(y1-yr).abs().max():.3f} QS-vs-base {(y1-y0).abs().max():.3f} nan base {bool(torch.isnan(y0).any())} QS {bool(torch.isnan(y1).any())} ref {bool(torch.isnan(yr).any())}", flush=True)
