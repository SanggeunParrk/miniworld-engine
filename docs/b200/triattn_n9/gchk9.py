import torch
from foldjet_engine.modules import triangle as T
for cfg in (dict(d_pair=64, n_head=4), dict(d_pair=64, n_head=2), dict(d_pair=256, n_head=8)):
  for starting in (True, False):
    torch.manual_seed(3)
    m = T.TriangleAttention(**cfg, starting=starting, backend="cuda").cuda().float().eval()
    with torch.no_grad():
        for p in m.parameters(): p.normal_(0, 0.5 if p.dim() == 2 else 0.2)
    x = (torch.randn(1, 640, 640, cfg["d_pair"], device="cuda") * 2).bfloat16()
    def step():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return m(x)
    ye = step().clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): yg = step()
    g.replay(); torch.cuda.synchronize()
    print(cfg, starting, "graph == eager:", bool((yg == ye).all()), "nan", bool(torch.isnan(yg).any()), flush=True)
