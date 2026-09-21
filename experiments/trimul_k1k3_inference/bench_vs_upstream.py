"""Module-level time of the payload named by the environment, for each TriMul shape — run once per payload
(pristine upstream vs ours) and interleave the processes to compare them on one clock.

    TRIMUL_NATIVE_BUILD_DIR=<payload>/build python bench_vs_upstream.py --length 768 --output x.json
"""
import argparse, copy, json, math, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from miniworld_engine.modules import BidirectionalTriangleMultiplication, TriangleMultiplication  # noqa: E402


def init(m):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(torch.bfloat16); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if name.endswith("weight") else t.data.normal_(0, .05)
    return m


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


def err(y, r):
    d = y.float() - r.float()
    return float(d.square().mean().sqrt() / (r.float().square().mean().sqrt() + 1e-12))


x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
CASES = {"outgoing": lambda i: TriangleMultiplication(a.width, outgoing=True, implementation=i, p_drop=0.0),
         "incoming": lambda i: TriangleMultiplication(a.width, outgoing=False, implementation=i, p_drop=0.0),
         "bidirectional": lambda i: BidirectionalTriangleMultiplication(a.width, implementation=i, p_drop=0.0)}
res = {}
for case, make in CASES.items():
    ref_mod = init(make("pytorch"))
    with torch.no_grad():
        ref = copy.deepcopy(ref_mod).float()(x.float(), mask)
    m = make("anthropic").to("cuda").eval()
    m.load_state_dict(ref_mod.state_dict())
    for _n, prm in m.named_parameters():
        prm.data = prm.data.to(torch.bfloat16 if prm.ndim >= 2 else torch.float32)
    with torch.no_grad():
        e = err(m(x, mask), ref)
    res[case] = dict(us=graph_us(lambda: m(x, mask)), rel_rms=e)
    print("RESULT", case, json.dumps(res[case]), flush=True)
json.dump(dict(length=a.length, res=res), open(a.output, "w"), indent=1)
