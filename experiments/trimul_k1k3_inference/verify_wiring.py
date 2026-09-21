"""The wired modules against their own fp32 reference, both directions, payload vs the engine's fallback.

    TRIMUL_NATIVE_BUILD_DIR=<payload>/build python verify_wiring.py --length 768 --output wiring-L768.json

Checks, for TriangleMultiplication (outgoing, incoming) and BidirectionalTriangleMultiplication:
  * implementation="anthropic" runs and matches the fp32 module reference in the payload's tolerance class,
  * implementation="miniworld" picks the payload up from the environment (same numbers as the explicit option),
  * the same module with the payload hidden falls back and still runs,
  * grad enabled falls back (the wiring must never take a training forward),
  * one-call CUDA graph time of each.
"""
import argparse, copy, json, math, os, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--output", required=True)
a = p.parse_args()

torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from miniworld_engine.integrations import anthropic_trimul as native  # noqa: E402
from miniworld_engine.modules import BidirectionalTriangleMultiplication, TriangleMultiplication  # noqa: E402


def init(m):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(torch.bfloat16); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if name.endswith("weight") else t.data.normal_(0, .05)
    return m


def err(y, r):
    d = y.float() - r.float()
    return dict(rel_rms=float(d.square().mean().sqrt() / (r.float().square().mean().sqrt() + 1e-12)),
                max_abs=float(d.abs().max()), finite=bool(torch.isfinite(y).all()))


def graph_us(fn):
    with torch.no_grad():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
        out = []
        for _ in range(3):
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(100):
                g.replay()
            en.record(); torch.cuda.synchronize()
            out.append(st.elapsed_time(en) * 1000 / 100)
    return statistics.median(out)


def like(ref_mod, make, impl):
    """A module of ``impl`` carrying ref_mod's weights AND its dtypes: load_state_dict keeps the DESTINATION dtype, which would
    leave the 2-D parameters fp32 and make the fused fallback paths refuse the bf16 pair."""
    m = make(impl).to("cuda").eval()
    m.load_state_dict(ref_mod.state_dict())
    for _n, prm in m.named_parameters():
        prm.data = prm.data.to(torch.bfloat16 if prm.ndim >= 2 else torch.float32)
    return m


x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
res = {}
cases = [("outgoing", lambda impl: TriangleMultiplication(a.width, outgoing=True, implementation=impl, p_drop=0.0)),
         ("incoming", lambda impl: TriangleMultiplication(a.width, outgoing=False, implementation=impl, p_drop=0.0)),
         ("bidirectional", lambda impl: BidirectionalTriangleMultiplication(a.width, implementation=impl, p_drop=0.0))]
for name, make in cases:
    ref_mod = init(make("pytorch"))
    with torch.no_grad():
        ref = copy.deepcopy(ref_mod).float()(x.float(), mask)
    r = {}
    for impl in ("anthropic", "miniworld"):
        m = like(ref_mod, make, impl)
        with torch.no_grad():
            y = m(x, mask)
        r[impl] = dict(err=err(y, ref), us=graph_us(lambda: m(x, mask)),
                       selection=getattr(m, "native_selection", None))
        if impl == "anthropic":
            y_anthropic = y
        else:
            r["auto_matches_explicit"] = bool(torch.equal(y, y_anthropic))
    # the payload hidden: the same module must fall back and still be correct
    keep = os.environ.pop(native.ENV)
    m = like(ref_mod, make, "miniworld")
    with torch.no_grad():
        y = m(x, mask)
    r["fallback"] = dict(err=err(y, ref), us=graph_us(lambda: m(x, mask)))
    os.environ[native.ENV] = keep
    # grad must never take the payload path
    m = like(ref_mod, make, "miniworld")
    xg = x.clone().requires_grad_()
    yg = m(xg, mask)
    r["grad_falls_back"] = bool(yg.requires_grad) and getattr(m, "native_selection", None) is None
    res[name] = r
    print("RESULT", name, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "selection"})
                                      for k, v in r.items()}), flush=True)
json.dump(dict(length=a.length, width=a.width, res=res), open(a.output, "w"), indent=1)
