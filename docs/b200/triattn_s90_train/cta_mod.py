# cta_mod.py KWARGS L : per-CTA %globaltimer timeline of the heavy kernels of one module training step (D64 L128: tri_l128 train, core kv / q / db_slabs, whbwd, wgbwd, tallk)
import sys, json, torch, functools
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels import _nvcc
orig = _nvcc.load_extension
def patched(*a, **kw):
    kw["extra_cuda_cflags"] = [*kw.get("extra_cuda_cflags", []), "-DS90_TL=1", "-DTA_TL=1"]
    kw["name"] = kw["name"] + "_s90tl"
    return orig(*a, **kw)
K.load_extension = patched
from miniworld_engine.modules import triangle as T
kw, L = json.loads(sys.argv[1]), int(sys.argv[2])
torch.manual_seed(0)
mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
def step():
    for p in mod.parameters(): p.grad = None
    x.grad = None
    with torch.autocast("cuda", torch.bfloat16): y = mod(x)
    y.backward(dy)
for _ in range(4): step()
torch.cuda.synchronize()
mods = {"core": K._core, "wide": K._wide, "tallk": K._tallk, "l128": K._l128}
names = {0: "bwd_kv", 1: "bwd_q", 2: "db_slabs", 3: "whbwd", 4: "wgbwd", 5: "tallk", 6: "tri_l128(fwd train)"}
src = {0: "core", 1: "core", 2: "core", 3: "wide", 4: "wide", 5: "tallk", 6: "l128"}
ext = {"core": K._core(K._rows_per_task(K.padded_length(L), 1)), "wide": K._wide(), "tallk": K._tallk(), "l128": K._l128()}
for kid in [int(a) for a in sys.argv[3:]] or range(7):
    t = ext[src[kid]].s90_tl().tolist()[kid]
    rows = [x for x in t if x[0] > 0]
    if not rows: print(names[kid], "no stamps"); continue
    t0 = min(x[0] for x in rows)
    ent = sorted((x[0] - t0) / 1e3 for x in rows)
    pro = sorted((x[1] - x[0]) / 1e3 for x in rows if x[1] > x[0])
    body = sorted((x[2] - x[1]) / 1e3 for x in rows if x[2] > x[1] > 0)
    end = sorted((x[2] - t0) / 1e3 for x in rows if x[2] > 0)
    m = lambda v: v[len(v) // 2] if v else 0.0
    print(f"{names[kid]:22s} CTAs {len(rows):3d} | entry spread {ent[0]:.2f}..{ent[-1]:.2f} | entry->prologue done (incl. PDL wait) median {m(pro):.2f} max {pro[-1] if pro else 0:.2f} | body median {m(body):.2f} (min {body[0] if body else 0:.2f} max {body[-1] if body else 0:.2f}) | first CTA end {end[0]:.2f} median {m(end):.2f} last {end[-1]:.2f} us after first entry", flush=True)
