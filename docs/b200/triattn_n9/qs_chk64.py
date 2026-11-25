# qs_chk64.py : head-dim-64 core (TriangleAttention d_pair 128, n_head 2, d_hidden 128), QS vs old vs the fp32 pytorch reference
import sys, torch
from foldjet_engine.modules import triangle as T
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
K.QS_MIN_L64 = int(sys.argv[1]) if len(sys.argv) > 1 else K.QS_MIN_L64
def build(starting, bias_mul=1.0, wqk_mul=1.0, seed=5):
    torch.manual_seed(seed)
    kw = dict(d_pair=128, n_head=2, d_hidden=128, starting=starting)
    m = T.TriangleAttention(**kw, backend="cuda").cuda().float().eval()
    r = T.TriangleAttention(**kw, backend="pytorch").cuda().float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.normal_(0, 0.5 if p.dim() == 2 else 0.2)
            if n == "to_bias.weight": p.mul_(bias_mul)
            if n in ("to_query.weight", "to_key.weight"): p.mul_(wqk_mul)
        r.load_state_dict(m.state_dict())
    return m, r
def run(mod, x, mask):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return mod(x, mask).float() if mask is not None else mod(x).float()
for tag, L, bm, wm, usemask in [("typ", 384, 1, 1, False), ("typ", 640, 1, 1, False), ("typ", 768, 1, 1, False), ("typ", 1030, 1, 1, False), ("mask", 700, 1, 1, True), ("wqk x3", 640, 1, 3, False), ("wqk x8", 640, 1, 8, False), ("bias x25", 640, 25, 1, False), ("bias x4", 640, 4, 1, False)]:
  for starting in (True, False):
    m, r = build(starting, bm, wm)
    x = (torch.randn(1, L, L, 128, device="cuda") * 2).bfloat16()
    mask = None
    if usemask:
        mask = torch.ones(1, L, dtype=torch.bool, device="cuda"); mask[0, L - 37:] = False
    K.QS_INFER = False; y0 = run(m, x, mask)
    K.QS_INFER = True; y1 = run(m, x, mask); y1b = run(m, x, mask)
    yr = run(r, x, mask)
    d = (y1 - y0).abs()
    print(f"{tag:8s} start={starting!s:5s} L{L} |y|max {y0.abs().max():.2f} QS-vs-old max {d.max():.4f} mean {d.mean():.1e} | old-vs-ref {(y0-yr).abs().max():.4f} QS-vs-ref {(y1-yr).abs().max():.4f} rerun_equal {bool((y1==y1b).all())} nan old {bool(torch.isnan(y0).any())} QS {bool(torch.isnan(y1).any())} ref {bool(torch.isnan(yr).any())}", flush=True)
