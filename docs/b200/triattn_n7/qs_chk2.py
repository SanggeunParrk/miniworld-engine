import sys, torch
from foldjet_engine.modules import triangle as T
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
torch.manual_seed(1)
cfgs = [dict(d_pair=256, n_head=8), dict(d_pair=64, n_head=2), dict(d_pair=64, n_head=4, d_hidden=128), dict(d_pair=384, n_head=12)]
for cfg in cfgs:
  for starting in (True, False):
    for L in (384, 768):
        m = T.TriangleAttention(**cfg, starting=starting, backend="cuda").cuda().float().eval()
        r = T.TriangleAttention(**cfg, starting=starting, backend="pytorch").cuda().float().eval()
        with torch.no_grad():
            for p in m.parameters(): p.normal_(0, 0.5 if p.dim() == 2 else 0.2)
            r.load_state_dict(m.state_dict())
        x = (torch.randn(1, L, L, cfg["d_pair"], device="cuda") * 2).bfloat16()
        def run(mod):
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return mod(x).float()
        K.QS_INFER = False; y0 = run(m)
        K.QS_INFER = True; y1 = run(m); y1b = run(m)
        yr = run(r); d = (y1 - y0).abs()
        print(f"{cfg} start={starting} L{L} |y|max {y0.abs().max():.2f} QS vs base max {d.max():.4f} mean {d.mean():.2e} | err base {(y0-yr).abs().max():.4f} err QS {(y1-yr).abs().max():.4f} rerun_equal {bool((y1==y1b).all())} nan {bool(torch.isnan(y1).any())}", flush=True)
