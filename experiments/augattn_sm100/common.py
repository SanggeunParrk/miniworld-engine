"""Shared pieces of the sm_100a augmented pair-bias attention experiment: inputs, the fp64 reference, graph timing.

The op (token DiT attention, AF3 Alg. 24 with A augmented samples sharing one pair bias):
    o[a, :, i, h] = softmax_j( q[a, i, h] . k[a, j, h] / sqrt(48) + bias[h, i, j] ) v[a, j, h]      H = 16, D = 48
"""
import math, statistics, torch

H, D = 16, 48


def make(A, L, seed=0, dtype=torch.bfloat16, dev="cuda"):
    """q, k, v [A, 1, L, H, D] (the engine's layout) and bias [H, L, L] (head-major, as the token DiT hoists it)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    q, k, v = (torch.randn(A, 1, L, H, D, device=dev, generator=g).to(dtype) for _ in range(3))
    bias = torch.randn(H, L, L, device=dev, generator=g).to(dtype)
    return q, k, v, bias


def reference(q, k, v, bias, dtype=torch.float64):
    """O [A, 1, L, H, D] in fp64."""
    A, _, L, _, _ = q.shape
    qh, kh, vh = (t.to(dtype)[:, 0].transpose(1, 2) for t in (q, k, v))      # [A, H, L, D]
    s = qh @ kh.transpose(-1, -2) / math.sqrt(D) + bias.to(dtype)[None]
    return (torch.softmax(s, -1) @ vh).transpose(1, 2)[:, None]


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def graph_time(fn, reps=20, rounds=5, warm=3):
    """CUDA-graph replay median (us per call). fn must be capture-safe."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    out = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        g.replay(); torch.cuda.synchronize()
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / reps)
    return statistics.median(out)
