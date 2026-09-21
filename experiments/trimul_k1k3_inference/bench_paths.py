"""Module-level time of every path on one session's clock: the payload, and the engine's own Triton and CuTe backends.

The CuTe backend has no tuned autotune cache for these ops on this checkout (it warns and falls back to the full grid), so its row is
the engine's DEFAULT today but not a tuned-kernel comparison; Triton is the engine's best measured path here.
"""
import argparse, json, math, statistics
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


x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
CASES = {"outgoing": lambda i: TriangleMultiplication(a.width, outgoing=True, implementation=i, p_drop=0.0),
         "incoming": lambda i: TriangleMultiplication(a.width, outgoing=False, implementation=i, p_drop=0.0),
         "bidirectional": lambda i: BidirectionalTriangleMultiplication(a.width, implementation=i, p_drop=0.0)}
res = {}
for case, make in CASES.items():
    row = {}
    for impl in ("anthropic", "triton", "cute"):
        try:
            m = init(make(impl))                      # built ONCE: constructing inside the timed lambda put module
            row[impl] = graph_us(lambda: m(x, mask))  # construction and its RNG inside the graph capture
        except Exception as exc:
            row[impl] = f"failed: {type(exc).__name__}: {exc}"[:160]
    res[case] = row
    print("RESULT", case, json.dumps(row), flush=True)
json.dump(dict(length=a.length, width=a.width, res=res), open(a.output, "w"), indent=1)
