"""Shared pieces of the sm_100a atom DiT experiment: shapes, block construction, timing.

The ordinary atom DiT block (AF3 Alg. 23 at atom widths, modules/dit.DiTBlock):
    a = a + AugmentedAttentionPairBias(a, s, z, mask)     d_single = d_cond = 128, d_pair = 16, 4 heads x 32
    a = a + ConditionedTransition(a, s)                   SwiGLU 128 -> 256 -> 128
at N = 8 * seq_len atoms; A = 5 samples (inference) / 48 (training) share one pair tensor z [1, N, N, 16].
"""
import statistics, torch

D_SINGLE, D_COND, D_PAIR, N_HEAD = 128, 128, 16, 4


def make_block(impl, dtype=torch.bfloat16, seed=0):
    from miniworld_engine.modules.dit import DiTBlock
    from miniworld_engine.modules.exceptions import ImplementationType
    torch.manual_seed(seed)
    blk = DiTBlock(d_single=D_SINGLE, d_cond=D_COND, d_pair=D_PAIR, n_head=N_HEAD, implementation=ImplementationType(impl))
    # the module's zero inits (to_out, squeeze, to_bias) would make the block an identity: randomize every parameter
    with torch.no_grad():
        for name, p in blk.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:                                        # norm weights around 1, biases around 0
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    return blk.to(device="cuda", dtype=dtype)


def make_inputs(A, L, dtype=torch.bfloat16, seed=0, grad=False):
    N = 8 * L
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(A, 1, N, D_SINGLE, device="cuda", generator=g).to(dtype).requires_grad_(grad)
    cond = torch.randn(A, 1, N, D_COND, device="cuda", generator=g).to(dtype).requires_grad_(grad)
    pair = torch.randn(1, N, N, D_PAIR, device="cuda", generator=g).to(dtype).requires_grad_(grad)
    return single, cond, pair


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def graph_time(fn, reps=10, rounds=5, warm=3):
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


def event_time(fn, reps=5, rounds=5, warm=3):
    """Plain stream timing (no graph), median us per call -- for training steps."""
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
