import sys, json, torch
from miniworld_engine.modules import triangle as T
kw = json.loads(sys.argv[1]); L = int(sys.argv[2])
def mk(backend, mode):
    torch.manual_seed(0)
    mod = T.TriangleAttention(**kw, backend=backend).cuda().float().train()
    g_ = torch.Generator(device="cpu").manual_seed(7)
    for n, p in mod.named_parameters():
        r = torch.randn(p.shape, generator=g_) * (0.1 if p.dim() == 2 else 0.2) + (1.0 if n == "ln_pair.weight" else 0.0)
        if mode == "default": continue
        if mode == "ln_exact" and n.startswith("ln_pair"): continue   # keep default LN (1, 0), everything else random
        if mode.startswith("d_") and n == mode[2:]: continue   # that one parameter at its default init, the rest random
        p.data.copy_(r)
    return mod
def run(backend, mode):
    mod = mk(backend, mode); torch.manual_seed(5)
    x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    with torch.autocast("cuda", torch.bfloat16): y = mod(x)
    y.backward(dy)
    return mod.to_out.weight.grad.clone()
cos = lambda a, b: float((a.float().flatten() @ b.float().flatten()) / (a.float().norm() * b.float().norm() + 1e-30))
for mode in ("d_to_query.weight", "d_to_key.weight", "d_to_value.weight", "d_to_bias.weight"):
    a, b = run("pytorch", mode), run("cuda", mode)
    print(f"{mode:13s}: dWo |pt| {float(a.norm()):.1f} |cuda| {float(b.norm()):.1f} cos {cos(a, b):.4f}", flush=True)
m = mk("cuda", "default")
for n, p in m.named_parameters(): print(n, "mean %.4f std %.4f min %.3f max %.3f absmax-row-norm-ratio %.2f" % (p.mean(), p.std(), p.min(), p.max(), float(p.norm(dim=-1).max() / (p.norm(dim=-1).min() + 1e-9)) if p.dim() == 2 else 0))
