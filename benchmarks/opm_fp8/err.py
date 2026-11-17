"""Error of the 2.5.12 default and the FP8 routes against fp64 PyTorch (same module, same inputs), per case.
usage: err.py OUTNAME CASE...   CASE = cm:cz:L:S[:a] (random module + inputs) | row:R:L:S (the registry row's construction, measure.config)
Metrics vs y64 (module in float64 on the PyTorch backend): rms = rms(y - y64) / rms(y64), max = max|y - y64| / max|y64| (= the engine judge
metric; tol = max(err_pytorch, 8e-3) with err_pytorch the PyTorch backend's own judge metric under the bf16 master autocast)."""
import copy, dataclasses, json, math, os, sys, time
import torch
from miniworld_engine.kernels.integrations import opm_fp8 as F
from miniworld_engine.modules.msa import OuterProductMean

out_name, specs = sys.argv[1], sys.argv[2:]
OUT = f"/home/psk6950/miniworld-engine-scratch/fp8x-opm/res/{out_name}.jsonl"
DONE = {json.loads(l)["case"] for l in open(OUT)} if os.path.exists(OUT) else set()
DEADLINE = int(os.environ.get("DEADLINE", time.time() + 500)) - 40
dev = "cuda"

def init_params(m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))

def metr(y, y64):
    d = y.double() - y64
    return {"rms": float(d.pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()), "max": float(d.abs().max() / y64.abs().max()),
            "finite": bool(torch.isfinite(y).all())}

def setr(r):
    F.MIN_S = 0; F.GUARD = "sync"
    F.POLICY = {"def": None, "f8b": "bound", "f8bf": "bound", "f8c": "calib"}[r]
    F.FAST_ACCUM = r == "f8bf"
    F._CACHES.clear()

for spec in specs:
    if spec in DONE:
        continue
    if time.time() > DEADLINE:
        print("DEADLINE", flush=True); sys.exit(0)
    p = spec.split(":")
    torch.manual_seed(0)
    if p[0] == "row":
        from miniworld_engine.measure import config
        from miniworld_engine.modules.primitives import unpadded_masks
        R, L, S = map(int, p[1:])
        cfg = config.Config(modules=("OuterProductMean",), rows=(R,), lengths=(L,), depths="min", modes=("eval",), samples="min")
        case = dataclasses.replace(next(iter(config.cases(cfg))), depth=S)
        base = config.build(case, device=torch.device(dev))
        inputs = config.inputs(case, base, device=torch.device(dev))
        x, mask = inputs["msa"], inputs.get("mask")
        kw = {k: v for k, v in inputs.items() if k not in ("msa", "mask")}
        ac = config.autocast(case)
    else:
        cm, cz, L, S = map(int, p[:4]); after = len(p) > 4 and p[4] == "a"
        base = OuterProductMean(cm, d_hidden=32, d_pair=cz, normalize_before_proj=not after, master_weights=True, backend="pytorch").to(dev).eval()
        init_params(base, cm + cz)
        x = torch.randn(1, S, L, cm, device=dev).bfloat16()
        mask = torch.ones(1, S, L, dtype=torch.bool, device=dev)
        mask[:, S - S // 8:, : L // 2] = False
        kw = {}; ac = None
    with torch.no_grad():
        g = copy.deepcopy(base).double().set_backend("pytorch")
        y64 = g(x.double(), mask, **{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()})
        del g
        ctx = (lambda: torch.autocast("cuda", dtype=ac)) if ac is not None else (lambda: torch.autocast("cuda", enabled=False))
        rec = {"case": spec}
        mp = copy.deepcopy(base).set_backend("pytorch")
        with ctx():
            rec["pytorch"] = metr(mp(x, mask, **kw), y64)
        del mp
        mc = copy.deepcopy(base).set_backend("cuda")
        for r in ("def", "f8b", "f8bf", "f8c"):
            setr(r)
            F.overflow_peak()
            with ctx():
                y = mc(x, mask, **kw)
                if r == "f8c":
                    y = mc(x, mask, **kw)   # (the first call calibrates; the second is the static-scale call)
            rec[r] = metr(y, y64)
            rec[r]["cache"] = {k: v["s"].tolist() for k, v in F._CACHES.items()} if r != "def" else None
            rec[r]["guard"] = dict(F.GUARD_STATS)
        setr("def")
    rec["tol"] = max(rec["pytorch"]["max"], 8e-3)
    for r in ("def", "f8b", "f8bf", "f8c"):
        rec[r]["over_tol"] = rec[r]["max"] / rec["tol"]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    del y64, mc, base, x
    torch.cuda.empty_cache()
print("done", flush=True)
