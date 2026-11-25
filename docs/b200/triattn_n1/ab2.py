# ab.py FLAGSPEC L cfgjson... : module training step graph, variants interleaved (>=3 rounds); FLAGSPEC "name:KQ_MERGE=0,KQ_PDL=1;name2:..." set attrs of b200_triattn at capture
import sys, json, torch
from foldjet_engine.modules import triangle as T
from foldjet_engine.kernels.integrations import bidir_triattn_h100 as K
variants = [(v.split(":")[0], dict(kv.split("=") for kv in v.split(":")[1].split(",") if kv)) for v in sys.argv[1].split(";")]
L = int(sys.argv[2]); B = int(sys.argv[3])
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for cj in sys.argv[4:]:
    cls, kwj = cj.split("@", 1); kw = json.loads(kwj); D = kw["d_pair"]
    torch.manual_seed(0)
    mod = getattr(T, cls)(**kw, backend="cuda").cuda().float().train()
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    def step():
        for p in mod.parameters(): p.grad = None
        x.grad = None
        with torch.autocast("cuda", torch.bfloat16): y = mod(x)
        y.backward(dy)
    gs = {}; outs = {}
    for nm, fl in variants:
        for a, v in fl.items(): setattr(K, a, v not in ("0", ""))
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): step()
        g.replay(); torch.cuda.synchronize(); gs[nm] = g
        outs[nm] = [x.grad.clone()] + [p.grad.clone() for p in mod.parameters() if p.grad is not None]
    ref = variants[0][0]
    tt = {nm: [] for nm, _ in variants}
    for rd in range(4):
        for nm, _ in variants:
            for _ in range(15):
                flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); gs[nm].replay(); b.record(); torch.cuda.synchronize(); tt[nm].append(a.elapsed_time(b) * 1e3)
    m = {nm: sorted(v)[len(v) // 2] for nm, v in tt.items()}
    eq = {nm: all(torch.equal(a, b) for a, b in zip(outs[ref], outs[nm])) for nm, _ in variants[1:]}
    mx = {nm: max(float((a.float() - b.float()).abs().max()) for a, b in zip(outs[ref], outs[nm])) for nm, _ in variants[1:]}
    print(f"L{L} B{B} {json.dumps(kw)}: " + "  ".join(f"{nm} {m[nm]:.1f}" for nm, _ in variants) + f" | vs {ref}: bitwise {eq} maxdiff {mx}", flush=True)
