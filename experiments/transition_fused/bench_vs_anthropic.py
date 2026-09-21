"""D = 128 Transition forward: Anthropic's own kernels, the engine's paths and this one, in a single process with one timing method.

Anthropic's Transition provider has CUDA kernels only at c = 256 / hidden = 1024 (esm_t16, flash_sm90a); at MiniWorld's pair
width the rows it offers are Triton (v2, v1, pf, lnl, af3_fused). This measures those against the engine's paths and the
fused kernel, all as the same module call under CUDA-graph replay, all against one fp32 reference.

Needs the Anthropic integration: run it under runs/anthropic_adoption_20260919/env.sh (MINIWORLD_ANTHROPIC_ROOT and the
engine checkout that carries `implementation="anthropic"`), plus MINIWORLD_MATHDX_HOME for the engine's CUDA b2b path.

  python bench_vs_anthropic.py --length 384 --cubin build/transition_fwd_infer.cubin
"""
import argparse, copy, statistics, sys, os, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drv

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", default=str(Path(__file__).resolve().parent / "build" / "transition_fwd.cubin"))
p.add_argument("--rows", nargs="+", default=["v2", "v1", "pf", "af3_fused"])
a = p.parse_args()
D, H, L, NCTA = 128, 512, a.length, 132
M = L * L
dev = "cuda"
torch.manual_seed(2319)

from miniworld_engine import settings
from miniworld_engine.modules import Transition

x4 = torch.randn(1, L, L, D, device=dev, dtype=torch.bfloat16)
base = Transition(D, implementation="pytorch").to(dev).to(torch.bfloat16).eval()
with torch.no_grad():
    base.squeeze.weight.normal_(0, H ** -0.5)


def graph_time(fn, reps=20, rounds=3):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.synchronize()
    out = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        g.replay(); torch.cuda.synchronize()
        st.record()
        for _ in range(reps):
            g.replay()
        en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / reps)
    return statistics.median(out)


res, refs = {}, {}
with torch.no_grad():
    refs["fp32"] = copy.deepcopy(base).float()(x4.float())   # .float() is in-place on a Module: copy first


def add(tag, mod):
    try:
        m = mod.to(dev).eval()
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            y = m(x4)
        res[tag] = (graph_time(lambda: m(x4)), float((y.float() - refs["fp32"]).norm() / refs["fp32"].norm()))
    except Exception as exc:  # noqa: BLE001
        res[tag] = (None, repr(exc)[:90])


add("engine default (miniworld)", Transition(D, implementation="miniworld"))
settings.configure(transition_residual_fusion=False)
add("engine b2b (residual_fusion=0)", Transition(D, implementation="miniworld"))
settings.configure(transition_residual_fusion=True)
for row in a.rows:
    try:
        add(f"anthropic {row}", Transition(D, implementation="anthropic", anthropic_row=row))
    except Exception as exc:  # noqa: BLE001
        res[f"anthropic {row}"] = (None, repr(exc)[:90])

# the fused kernel, same graph timing
flat = x4.reshape(M, D).contiguous()
wa = base.expand_a.weight.contiguous(); wb = base.expand_b.weight.contiguous(); ws = base.squeeze.weight.contiguous()
wst = ws.t().contiguous()
gam = base.ln_in.weight.float().contiguous(); bet = base.ln_in.bias.float().contiguous()
k = drv.Kernel(a.cubin, "transition_fwd_fused", 231424)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
outk = torch.empty_like(flat); xnk = torch.empty_like(flat)
maps = (tm(flat, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]),
        tm(wb, [D, H], D * 2, [64, 64]), tm(wst, [D, H], D * 2, [64, 64]), tm(outk, [D, M], D * 2, [64, 64]))
rk = torch.empty(M, device=dev, dtype=torch.float32); ck = torch.empty(M, device=dev, dtype=torch.float32)
def fused():
    k((NCTA, 1, 1), (256, 1, 1), *maps, gam, bet, xnk, outk, rk, ck, int(M), int(M // 128), 1e-5)
with torch.no_grad():
    fused(); torch.cuda.synchronize()
    res["fused (this work, inference build)"] = (graph_time(fused),
        float((outk.reshape(1, L, L, D).float() - refs["fp32"]).norm() / refs["fp32"].norm()))

print(f"\n=== Transition forward, D={D}, L={L}, M={M}, bf16, inference (CUDA-graph replay median)")
ok = {t: v for t, v in res.items() if v[0] is not None}
bestname = min(ok, key=lambda t: ok[t][0]) if ok else None
for t, (us, err) in res.items():
    if us is None:
        print(f"  {t:<36s}  unavailable: {err}")
    else:
        print(f"  {t:<36s}  {us:8.1f} us   rel_rms vs fp32 {err:.3e}" + ("   <-- fastest" if t == bestname else ""))
