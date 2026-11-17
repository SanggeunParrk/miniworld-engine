"""sha1 of the default route's outputs (FP8 off) per case, for the base (91a9241fb) vs the fp8x/opm tree, separate processes.
usage: bit.py LABEL CASE...   CASE = cm:cz:L:S[:a][:r][:o][:t]  (a after-projection order, r pair residual, o count_offset 1e-3, t training: + grads)"""
import hashlib, json, os, sys, time
import torch
from miniworld_engine.modules.msa import OuterProductMean

label, specs = sys.argv[1], sys.argv[2:]
OUT = f"/home/psk6950/miniworld-engine-scratch/fp8x-opm/res/bit_{label}.jsonl"
DONE = {json.loads(l)["case"] for l in open(OUT)} if os.path.exists(OUT) else set()
DEADLINE = int(os.environ.get("DEADLINE", time.time() + 500)) - 30
dev = "cuda"
assert not os.environ.get("MINIWORLD_OPM_FP8")

def sha(t):
    return hashlib.sha1(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]

for spec in specs:
    if spec in DONE:
        continue
    if time.time() > DEADLINE:
        print("DEADLINE", flush=True); sys.exit(0)
    p = spec.split(":"); fl = set(p[4:])
    cm, cz, L, S = map(int, p[:4])
    torch.manual_seed(0)
    m = OuterProductMean(cm, d_hidden=32, d_pair=cz, normalize_before_proj="a" not in fl, count_offset=1e-3 if "o" in fl else None,
                         master_weights=True, backend="cuda").to(dev)
    g = torch.Generator(device="cpu").manual_seed(1)
    with torch.no_grad():
        for n_, q in m.named_parameters():
            q.copy_(torch.randn(q.shape, generator=g) * (q.shape[-1] ** -0.5 if q.dim() > 1 else 0.1) + (1.0 if n_.endswith("weight") and q.dim() == 1 else 0.0))
    x = torch.randn(1, S, L, cm, device=dev).bfloat16()
    mask = torch.rand(1, S, L, device=dev) > 0.1
    res = torch.randn(1, L, L, cz, device=dev).bfloat16() if "r" in fl else None
    out = {}
    if "t" in fl:
        m.train(); x.requires_grad_(True)
        y = m(x, mask, residual=res)
        y.backward(torch.randn_like(y))
        out["y"] = sha(y); out["dx"] = sha(x.grad)
        for n_, q in m.named_parameters():
            out["d_" + n_] = sha(q.grad)
    else:
        m.eval()
        with torch.no_grad():
            y = m(x, mask, residual=res); y2 = m(x, mask, residual=res)
        out["y"] = sha(y); out["rerun_equal"] = bool(torch.equal(y, y2))
    rec = {"case": spec, "sha": out}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    del m, x, y
    torch.cuda.empty_cache()
print("done", flush=True)
