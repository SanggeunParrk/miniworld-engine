"""Timing / error helpers shared by the backward benches (same protocol as ../a100_transition_fwd/bench.py)."""
import statistics

import torch


def rel_rms(y, ref):
    d = y.float() - ref.float()
    return float(d.square().mean().sqrt() / ref.float().square().mean().sqrt())


def graph_ms(fn):
    """CUDA-graph replay: warm ~300 ms, then the median of 7 x 50 replays (ms per call)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        g.replay()
    en.record()
    en.synchronize()
    warm = min(10000, max(30, int(300 / max(st.elapsed_time(en) / 10, 1e-3))))
    for _ in range(warm):
        g.replay()
    rounds = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50):
            g.replay()
        en.record()
        en.synchronize()
        rounds.append(st.elapsed_time(en) / 50)
    return statistics.median(rounds), rounds
