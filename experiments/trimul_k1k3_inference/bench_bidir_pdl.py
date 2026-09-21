"""Programmatic dependent launch on / off for the BIDIRECTIONAL composition, in one process with alternating CUDA-graph replays
(same clock and thermal state): one op, and two bidirectional ops chained the way a pairformer stack runs them.

    TRIMUL_NATIVE_BUILD_DIR=<payload256>/build python bench_bidir_pdl.py --length 768 --output bidir-pdl-L768.json

A single op cannot show the gain: PDL overlaps a kernel's prologue with the tail of the kernel before it, and inside one op the
predecessor of K3 is the cuBLAS contraction, which never triggers dependents.  The chain is where K1 of the next op can start under
the K3 of this one.  Outputs of the two graphs are compared bitwise."""
import argparse, copy, importlib, json, math, os, statistics, sys
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--mask-dtype", default="bool", choices=["bf16", "fp32", "bool"])
p.add_argument("--rounds", type=int, default=8)
p.add_argument("--output", required=True)
a = p.parse_args()
MASK_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "bool": torch.bool}

torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from miniworld_engine.modules import BidirectionalTriangleMultiplication  # noqa: E402


def init(m, dt=torch.bfloat16):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(dt); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if name.endswith("weight") else t.data.normal_(0, .05)
    return m


def weights_of(m):
    return {k: t.detach().contiguous() for k, t in dict(
        ln_in_w=m.ln_pair.weight, ln_in_b=m.ln_pair.bias, w_ag=m.to_left_gate.weight, w_ap=m.to_left.weight,
        w_bg=m.to_right_gate.weight, w_bp=m.to_right.weight, ln_out_w=m.ln_out.weight, ln_out_b=m.ln_out.bias,
        w_o=m.to_out.weight, w_og=m.to_gate.weight).items()}


bd = os.environ.get("TRIMUL_NATIVE_BUILD_DIR") or sys.exit("set TRIMUL_NATIVE_BUILD_DIR=<payload>/build")
py = os.path.join(os.path.dirname(os.path.abspath(bd)), "python")
if py not in sys.path:
    sys.path.insert(0, py)
ops = importlib.import_module("trimul_native.ops")
KM = importlib.import_module("trimul_native.kernel")
if not hasattr(KM, "PDL"):
    raise SystemExit("this payload's host package has no PDL switch (build with the overlay of this experiment)")

m1 = init(BidirectionalTriangleMultiplication(a.width, d_hidden=a.hidden, implementation="pytorch", p_drop=0.0))
m2 = init(BidirectionalTriangleMultiplication(a.width, d_hidden=a.hidden, implementation="pytorch", p_drop=0.0))
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
tok = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
tok[:, ::7] = False
pairmask = (tok.unsqueeze(-1) & tok.unsqueeze(-2)).reshape(a.length, a.length).to(MASK_DTYPES[a.mask_dtype]).contiguous()
z3 = x[0].contiguous()
h, eps = a.hidden, m1.ln_pair.eps
packs = [ops.pack_weights(weights_of(m)) for m in (m1, m2)]
ops._check(z3, packs[0])
ch = packs[0]["ch"]
Np = ops.ceil16(a.length)
cache = {}


def op(zin, w, tag):
    ab = ops.planes(zin, pairmask, w, transpose=False, lnm=2, Np=Np, cfg=None, cache=cache, eps=eps)
    tri = ops._buf(cache, ("tri" + tag, Np, ch), (ch, Np, Np), torch.bfloat16, zin.device)
    aa, bb = ab[:ch], ab[ch:]
    torch.bmm(aa[:h], bb[:h].transpose(1, 2), out=tri[:h])
    torch.bmm(aa[h:].transpose(1, 2), bb[h:], out=tri[h:])
    return ops.epilogue(tri, zin, w, residual=True, lnm=1, cfg=None, cache=cache, eps=eps)


def single():
    return op(z3, packs[0], "a")


def chain():
    return op(single().contiguous(), packs[1], "b")


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    return g, out


def replay_us(g, warm=20, reps=100):
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / reps


res = {}
with torch.no_grad():
    for mode, fn in (("single", single), ("chain", chain)):
        graphs = {}
        for flag in (False, True):
            KM.PDL = flag
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graphs[flag] = capture(fn)
        times = {False: [], True: []}
        for rnd in range(a.rounds):
            for flag in ((False, True) if rnd % 2 == 0 else (True, False)):
                times[flag].append(replay_us(graphs[flag][0]))
        r = dict(off_us=statistics.median(times[False]), on_us=statistics.median(times[True]),
                 off_all=times[False], on_all=times[True], same_output=bool(torch.equal(graphs[False][1], graphs[True][1])))
        r["gain_us"] = r["off_us"] - r["on_us"]
        res[mode] = r
        print("RESULT", mode, json.dumps(dict(off=round(r["off_us"], 2), on=round(r["on_us"], 2), gain=round(r["gain_us"], 2),
                                              off_spread=round(max(times[False]) - min(times[False]), 2),
                                              on_spread=round(max(times[True]) - min(times[True]), 2),
                                              same_output=r["same_output"])), flush=True)
json.dump(dict(length=a.length, width=a.width, hidden_per_direction=a.hidden, c_hidden=ch, mask_dtype=a.mask_dtype, res=res),
          open(a.output, "w"), indent=1)
