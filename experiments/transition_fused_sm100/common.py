"""Shared pieces of the sm_100a Transition experiment: inputs, the H100 contract emulated in torch, graph timing."""
import statistics
import torch

D, H = 128, 512


def make_inputs(L, seed=2319, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = L * L
    x = torch.randn(M, D, generator=g).to(dev, torch.bfloat16)
    wa = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    wb = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    ws = (torch.randn(D, H, generator=g) * H ** -0.5).to(dev, torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    beta = (0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    return x, wa, wb, ws, gamma, beta


def contract_fwd(x, wa, wb, ws, gamma, beta, eps=1e-5):
    """The fused kernels' rounding points: xn bf16, a/b fp32, h bf16, acc fp32, out = bf16(x + acc)."""
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(-1, keepdim=True)
    rs = torch.rsqrt(var + eps)
    xn = ((xf - mean) * rs * gamma + beta).to(torch.bfloat16)
    a = xn.float() @ wa.float().t()
    b = xn.float() @ wb.float().t()
    h = (a * torch.sigmoid(a) * b).to(torch.bfloat16)
    acc = h.float() @ ws.float().t()
    out = (xf + acc).to(torch.bfloat16)
    return out, xn, rs.squeeze(-1), (mean * rs).squeeze(-1)


def fp32_fwd(x, wa, wb, ws, gamma, beta, eps=1e-5):
    xf = x.float()
    xn = torch.nn.functional.layer_norm(xf, (D,), gamma, beta, eps)
    return xf + (torch.nn.functional.silu(xn @ wa.float().t()) * (xn @ wb.float().t())) @ ws.float().t()


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def graph_time(fn, reps=20, rounds=5, warm=3):
    """CUDA-graph replay median (us per call). fn must be capture-safe."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    out = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        g.replay()
        torch.cuda.synchronize()
        st.record(); g.replay(); en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / reps)
    return statistics.median(out)
