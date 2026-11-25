import sys, torch
from foldjet_engine.modules import triangle as T
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
torch.manual_seed(1)
for starting in (True, False):
  for L in (130, 640, 1030):
    m = T.TriangleAttention(d_pair=128, n_head=4, starting=starting, backend="cuda").cuda().float().eval()
    r = T.TriangleAttention(d_pair=128, n_head=4, starting=starting, backend="pytorch").cuda().float().eval()
    with torch.no_grad():
        for p in m.parameters(): p.normal_(0, 0.5 if p.dim() == 2 else 0.2) if p.dim() >= 1 else None
        r.load_state_dict(m.state_dict())
    x = (torch.randn(1, L, L, 128, device="cuda") * 2).bfloat16()
    mask = None
    def run(mod):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return mod(x).float()
    K.QS_INFER = False; y0 = run(m)
    K.QS_INFER = True; y1 = run(m); y1b = run(m)
    with torch.no_grad(): yr = r.double()(x.double()).float() if False else run(r)
    d = (y1 - y0).abs()
    print(f"start={starting} L{L} |y|max {y0.abs().max():.2f} QS vs base max {d.max():.4f} mean {d.mean():.2e} | err base {(y0-yr).abs().max():.4f} err QS {(y1-yr).abs().max():.4f} rerun_equal {bool((y1==y1b).all())} nan {bool(torch.isnan(y1).any())}", flush=True)
