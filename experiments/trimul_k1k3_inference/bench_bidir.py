"""Bidirectional TriMul through the native K1/K3 payload: K1 (c_hidden = 2h, natural layout) -> two half-channel contractions
(outgoing NT, incoming TN) -> K3 (shared output LayerNorm over 2h, residual fused).

    TRIMUL_NATIVE_BUILD_DIR=<payload>/build python bench_bidir.py --length 768 --output out.json [--configs baseline|sweep|@file]

Per-kernel CUPTI durations over eager calls, whole-op time as a one-call CUDA graph (3 x 100 replays after 20 warm), error against the
engine's fp32 PyTorch BidirectionalTriangleMultiplication.  Composed from trimul_native.ops primitives, so the face's test-vector gate is
not involved; the fp32 comparison here is the numerical check."""
import argparse, copy, importlib, json, math, os, statistics, sys
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128, help="c_z (pair channels)")
p.add_argument("--hidden", type=int, default=128, help="hidden channels PER DIRECTION; the unit is c_hidden = 2 x this")
p.add_argument("--configs", default="baseline")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--mask-dtype", default="bf16", choices=["bf16", "fp32", "bool"])
p.add_argument("--output", required=True)
p.add_argument("--save", default="", help="torch.save the bf16 output (bitwise comparison between payloads)")
a = p.parse_args()

MASK_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "bool": torch.bool}


def payload_ops():
    bd = os.environ.get("TRIMUL_NATIVE_BUILD_DIR") or sys.exit("set TRIMUL_NATIVE_BUILD_DIR=<payload>/build")
    py = os.path.join(os.path.dirname(os.path.abspath(bd)), "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    ops = importlib.import_module("trimul_native.ops")
    got = os.path.dirname(os.path.abspath(ops.__file__))
    if got != os.path.join(py, "trimul_native"):
        sys.exit("trimul_native imported from %s, not the payload %s" % (got, py))
    return ops


def init(m, dt=torch.bfloat16):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(dt); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if name.endswith("weight") else t.data.normal_(0, .05)
    return m


def error(y, r):
    d = y.float() - r.float()
    return dict(rel_rms=float(d.square().mean().sqrt() / (r.float().square().mean().sqrt() + 1e-12)),
                max_abs=float(d.abs().max()), finite=bool(torch.isfinite(y).all()))


def cupti_per_kernel(fn, iters):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per, names = {}, set()
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        n = e.name
        key = "k1" if "tmn_k1" in n else "k3" if "tmn_k3" in n else "cublas" if ("nvjet" in n or "gemm" in n.lower() or "cutlass" in n.lower()) else "other"
        per.setdefault(key, []).append(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
        if "tmn_" in n:
            names.add(n[:70])
    return {k: dict(count=len(v) / iters, sum_per_call_us=sum(v) / iters) for k, v in per.items()}, sorted(names)


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


torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

from miniworld_engine.modules import BidirectionalTriangleMultiplication  # noqa: E402

m = init(BidirectionalTriangleMultiplication(a.width, d_hidden=a.hidden, implementation="pytorch", p_drop=0.0))
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
tokmask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
tokmask[:, ::7] = False
with torch.no_grad():
    ref = copy.deepcopy(m).float()(x.float(), tokmask)
pairmask = (tokmask.unsqueeze(-1) & tokmask.unsqueeze(-2)).reshape(a.length, a.length).to(MASK_DTYPES[a.mask_dtype]).contiguous()
weights = {k: t.detach().contiguous() for k, t in dict(
    ln_in_w=m.ln_pair.weight, ln_in_b=m.ln_pair.bias, w_ag=m.to_left_gate.weight, w_ap=m.to_left.weight,
    w_bg=m.to_right_gate.weight, w_bp=m.to_right.weight, ln_out_w=m.ln_out.weight, ln_out_b=m.ln_out.bias,
    w_o=m.to_out.weight, w_og=m.to_gate.weight).items()}

ops = payload_ops()
print("ops", ops.__file__, "build", os.environ["TRIMUL_NATIVE_BUILD_DIR"], flush=True)
z3 = x[0].contiguous()
eps = m.ln_pair.eps
h = a.hidden                      # per direction; the unit's c_hidden is 2h
packed = ops.pack_weights(weights)
ops._check(z3, packed)
ch = packed["ch"]
assert ch == 2 * h, (ch, h)
Np = ops.ceil16(a.length)


def call(cfg, cache):
    k1cfg = (cfg or {}).get("k1_cfg")
    k3cfg = (cfg or {}).get("k3_cfg")
    ab = ops.planes(z3, pairmask, packed, transpose=False, lnm=2, Np=Np, cfg=k1cfg, cache=cache, eps=eps)
    tri = ops._buf(cache, ("tri", Np, ch), (ch, Np, Np), torch.bfloat16, z3.device)
    aa, bb = ab[:ch], ab[ch:]
    torch.bmm(aa[:h], bb[:h].transpose(1, 2), out=tri[:h])                     # outgoing half: NT
    torch.bmm(aa[h:].transpose(1, 2), bb[h:], out=tri[h:])                     # incoming half: TN
    return ops.epilogue(tri, z3, packed, residual=True, lnm=1, cfg=k3cfg, cache=cache, eps=eps)


def measure(cfg):
    cache = {}
    with torch.no_grad():
        y = call(cfg, cache)
        torch.cuda.synchronize()
        err = error(y.unsqueeze(0), ref)
        if a.save and not os.path.exists(a.save):
            torch.save(dict(y=y.cpu()), a.save)
        for _ in range(5):
            call(cfg, cache)
        torch.cuda.synchronize()
        kernels, names = cupti_per_kernel(lambda: call(cfg, cache), a.iters)
        g, yg = capture(lambda: call(cfg, cache))
        rounds = [replay_us(g) for _ in range(3)]
        err_replay = error(yg.unsqueeze(0), ref)
    return dict(config=cfg, error=err, error_after_replay=err_replay, kernels=kernels, kernel_names=names,
                op_us=dict(rounds=rounds, median=statistics.median(rounds)))


if a.configs == "baseline":
    cfgs = [None]
elif a.configs == "sweep":
    K1S = [(2, 64, 8, 2), (6, 32, 8, 2), (3, 64, 8, 2), (1, 128, 8, 2), (2, 64, 4, 2)]
    K3S = [(2, 64, 4, 1), (2, 64, 6, 1), (2, 64, 4, 2), (1, 128, 4, 1), (1, 64, 4, 1)]   # the set that fits at c_hidden 256
    cfgs = [None] + [{"k1_cfg": k} for k in K1S] + [{"k3_cfg": k} for k in K3S]
else:
    cfgs = json.loads(open(a.configs[1:]).read() if a.configs.startswith("@") else a.configs)

results = []
for cfg in cfgs:
    try:
        r = measure(cfg)
    except Exception as ex:
        r = dict(config=cfg, failed=repr(ex)[:400])
    results.append(r)
    ks = r.get("kernels", {})
    print("RESULT", json.dumps(dict(config=cfg, op_us=r.get("op_us", {}).get("median"), k1=ks.get("k1", {}).get("sum_per_call_us"),
                                    k3=ks.get("k3", {}).get("sum_per_call_us"), cublas=ks.get("cublas", {}).get("sum_per_call_us"),
                                    other=ks.get("other", {}).get("sum_per_call_us"), err=r.get("error"),
                                    err_replay=r.get("error_after_replay"), failed=r.get("failed"), names=r.get("kernel_names"))), flush=True)
    json.dump(dict(length=a.length, width=a.width, hidden_per_direction=a.hidden, c_hidden=ch, mask_dtype=a.mask_dtype,
                   iters=a.iters, build=os.environ["TRIMUL_NATIVE_BUILD_DIR"], results=results), open(a.output, "w"), indent=1)
