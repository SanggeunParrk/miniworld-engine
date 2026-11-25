# bitscan.py OUTFILE : inference outputs of TriangleAttention-family modules (random NON-ZERO weights, real shapes, key masks, ragged lengths,
# both orientations) of the tree on PYTHONPATH, saved for a bitwise comparison between trees (cmpscan.py).
import sys, torch, itertools
from foldjet_engine.modules import triangle as T

out = {}
torch.backends.cuda.matmul.allow_tf32 = True
cases = []
for cls, kw in [("TriangleAttention", dict(d_pair=128, n_head=4)), ("TriangleAttention", dict(d_pair=64, d_hidden=128, n_head=4)),
                ("TriangleAttention", dict(d_pair=256, n_head=8)), ("TriangleAttentionAF3", dict(d_pair=128, n_head=4)), ("TriangleAttentionRF3", dict(d_pair=128, n_head=4))]:
    for starting in (True, False):
        for L, B, usemask in [(256, 1, False), (384, 1, True), (512, 2, False), (300, 1, True), (640, 1, True), (130, 1, False)]:
            cases.append((cls, kw, starting, L, B, usemask))
for cls, kw, starting, L, B, usemask in cases:
    label = f"{cls}-{kw}-{'S' if starting else 'E'}-L{L}-B{B}-m{int(usemask)}"
    try:
        torch.manual_seed(1)
        m = getattr(T, cls)(**kw, starting=starting, p_drop=0.25, backend="cuda").cuda().float().eval()
        g = torch.Generator(device="cuda").manual_seed(7)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, device="cuda", generator=g) * (0.5 if p.ndim == 1 else 0.15) + (1.0 if p.ndim == 1 and "weight" in "" else 0.0))
        x = torch.randn(B, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, generator=g)
        mask = (torch.rand(B, L, device="cuda", generator=g) > 0.1) if usemask else None
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            y = m(x, mask) if mask is not None else m(x)
            y2 = m(x, mask) if mask is not None else m(x)
        out[label] = y.detach().float().cpu()
        out[label + "|repeat"] = torch.tensor(bool((y == y2).all()))
        print("ok", label, "repeat-bitwise", bool((y == y2).all()), flush=True)
    except Exception as e:
        print("SKIP", label, type(e).__name__, str(e)[:100], flush=True)
torch.save(out, sys.argv[1])
