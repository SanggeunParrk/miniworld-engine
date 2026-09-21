"""The engine's own bidirectional TriMul inference path (implementation="miniworld" -> CuTe on sm90) on the same inputs as
bench_bidir.py, so the native payload and the current production path are comparable.

    python bench_engine_bidir.py --length 768 --output engine-L768.json [--impl miniworld|triton|pytorch]
"""
import argparse, copy, json, math, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--impl", default="miniworld")
p.add_argument("--output", required=True)
a = p.parse_args()

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


def error(y, r):
    d = y.float() - r.float()
    return dict(rel_rms=float(d.square().mean().sqrt() / (r.float().square().mean().sqrt() + 1e-12)),
                max_abs=float(d.abs().max()), finite=bool(torch.isfinite(y).all()))


m = init(BidirectionalTriangleMultiplication(a.width, d_hidden=a.hidden, implementation="pytorch", p_drop=0.0))
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
with torch.no_grad():
    ref = copy.deepcopy(m).float()(x.float(), mask)

e = BidirectionalTriangleMultiplication(a.width, d_hidden=a.hidden, implementation=a.impl, p_drop=0.0).to("cuda").eval()
e.load_state_dict(m.state_dict())
print("backend", getattr(e, "_backend", "?"), flush=True)

with torch.no_grad():
    y = e(x, mask)
    torch.cuda.synchronize()
    err = error(y, ref)
    print("ERR", json.dumps(err), flush=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            e(x, mask)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        yg = e(x, mask)
    rounds = []
    for _ in range(3):
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(100):
            g.replay()
        en.record(); torch.cuda.synchronize()
        rounds.append(st.elapsed_time(en) * 1000 / 100)
    err_replay = error(yg, ref)

out = dict(length=a.length, width=a.width, hidden_per_direction=a.hidden, impl=a.impl,
           backend=str(getattr(e, "_backend", "?")), error=err, error_after_replay=err_replay,
           op_us=dict(rounds=rounds, median=statistics.median(rounds)))
print("RESULT", json.dumps(out), flush=True)
json.dump(out, open(a.output, "w"), indent=1)
