"""Shared pieces of the sm_100a SWA atom DiT experiment: shapes, inputs, the fp64 / torch references, timing.

The block (team-gm SWAAtomBlock, block_style "esmfold2"; the H100 fused path is team-gm 14f2c73 swa_fused_triton.py = h100_fused.py):
    mod = silu(c_base) Wmod^T -> shift_a | scale_a | gate_a | shift_f | scale_f | gate_f            [B S, 6C] fp32, hoisted per atom
    x   = RMS(q) (1 + scale_a) + shift_a;   Q, K = rope(headRMS(x Wq^T)), rope(headRMS(x Wk^T));  V = x Wv^T;  G = x Wg^T
    o   = sliding-window attention (|i - j| <= 64, keys / queries < seqused), 4 heads x 32
    q1  = q + gate_a ((sigmoid(G) o) Wo^T);   y = RMS(q1) (1 + scale_f) + shift_f;   out = q1 + gate_f ((silu(y Wa^T) (y Wb^T)) Wd^T)
C = 128, SwiGLU hidden 256, eps = fp32 eps (RMS and qk-RMS). Rows are the flattened N = A * B samples x S atoms.
"""
import math, statistics, torch

C, H, D, NHID, HW = 128, 4, 32, 256, 64
EPS = float(torch.finfo(torch.float32).eps)


def make(A, S, B=1, seed=0, dev="cuda"):
    """q [N, S, C] bf16, c_base [B, S, C] bf16, cos / sin [B S, D/2] fp32, seqused [N] int32 (= S), weights bf16."""
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g)
    N = A * B
    q = r(N, S, C).to(torch.bfloat16)
    c_base = r(B, S, C).to(torch.bfloat16)
    ang = r(B * S, D // 2) * 3.0
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    seqused = torch.full((N,), S, device=dev, dtype=torch.int32)
    w = dict(wmod=(r(6 * C, C) * 0.05).to(torch.bfloat16), wqkv=(r(3 * C, C) / math.sqrt(C)).to(torch.bfloat16),
             wg=(r(C, C) / math.sqrt(C)).to(torch.bfloat16), wo=(r(C, C) / math.sqrt(C)).to(torch.bfloat16),
             wu=(r(2 * NHID, C) / math.sqrt(C)).to(torch.bfloat16), wd=(r(C, NHID) / math.sqrt(NHID)).to(torch.bfloat16))
    return q, c_base, cos, sin, seqused, w


def hoist_mod(c_base, wmod):
    a = torch.nn.functional.silu(c_base).float()
    return (a.reshape(-1, a.shape[-1]) @ wmod.float().t()).contiguous()


def _rope(x, cos, sin):
    """x [N, S, H, D]; cos / sin [N, S, D/2]: halves (x1 | x2) -> (x1 c - x2 s | x2 c + x1 s)."""
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    c, s = cos[:, :, None, :], sin[:, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def block_ref(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B=1, dtype=torch.float64, band=None):
    """The block in `dtype` (no intermediate rounding): torch ops, dense banded attention. q [N, S, C] -> out [N, S, C]."""
    N, S, _ = q.shape
    f = lambda t: t.to(dtype)
    q, mod, cos, sin = f(q), f(mod).view(B, S, 6 * C), f(cos).view(B, S, -1), f(sin).view(B, S, -1)
    wqkv, wg, wo, wu, wd = (f(t) for t in (wqkv, wg, wo, wu, wd))
    bi = torch.arange(N, device=q.device) % B
    sh_a, sc_a, g_a, sh_f, sc_f, g_f = (mod[bi][..., k * C:(k + 1) * C] for k in range(6))
    rms = lambda t, e: t * torch.rsqrt((t * t).mean(-1, keepdim=True) + e)
    x = rms(q, EPS) * (1 + sc_a) + sh_a
    p = x @ wqkv.t()
    Q, K, V = (p[..., k * C:(k + 1) * C].view(N, S, H, D) for k in range(3))
    Q, K = _rope(rms(Q, EPS), cos[bi], sin[bi]), _rope(rms(K, EPS), cos[bi], sin[bi])
    G = x @ wg.t()
    if band is None:
        i = torch.arange(S, device=q.device)
        band = (i[:, None] - i[None, :]).abs() <= HW
    Qh, Kh, Vh = (t.transpose(1, 2) for t in (Q, K, V))                          # [N, H, S, D]
    o = torch.nn.functional.scaled_dot_product_attention(Qh, Kh, Vh, attn_mask=band, scale=D ** -0.5)
    o = o.transpose(1, 2).reshape(N, S, C)
    q1 = q + g_a * ((torch.sigmoid(G) * o) @ wo.t())
    y = rms(q1, EPS) * (1 + sc_f) + sh_f
    a, b = (y @ wu.t()).split(NHID, -1)
    return q1 + g_f * ((torch.nn.functional.silu(a) * b) @ wd.t())


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def graph_time(fn, reps=10, rounds=5, warm=3):
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


def event_time(fn, reps=5, rounds=5, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        st.record()
        for _ in range(reps):
            fn()
        en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / reps)
    return statistics.median(out)
