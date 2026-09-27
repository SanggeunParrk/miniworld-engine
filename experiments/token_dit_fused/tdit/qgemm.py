"""quack (CuTe DSL, sm_100) GEMMs with fused epilogues for the training block: the SwiGLU forward (gemm_act) and its
backward (gemm_dact, "dgated"). Weight rows are gate/up interleaved (a0, b0, a1, b1, ...), so the pre-activation and its
gradient are [M, 2n] interleaved too. The tile config is raced once per shape (eager timing, before any capture)."""
import torch

CFGS = ((128, 192, 2, 1, True), (128, 192, 1, 1, True), (128, 128, 2, 1, True), (128, 256, 1, 1, False),
        (128, 256, 2, 1, False), (256, 192, 2, 1, False), (256, 128, 2, 1, False), (128, 192, 1, 2, True))
_cfg = {}


def _mods():
    from quack.gemm_act import gemm_act
    from quack.gemm_dact import gemm_dact
    return gemm_act, gemm_dact


def _race(key, run):
    cfg = _cfg.get(key)
    if cfg is not None:
        return run(cfg)
    best = None
    for c in CFGS:
        try:
            run(c); torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(5):
                run(c)
            en.record(); torch.cuda.synchronize()
            t = st.elapsed_time(en)
            if best is None or t < best[0]:
                best = (t, c)
        except Exception:  # noqa: BLE001 -- a config this card / shape cannot take
            continue
    if best is None:
        raise RuntimeError(f"no quack config runs for {key}")
    _cfg[key] = best[1]
    return run(best[1])


def swiglu_fwd(x, w_i):
    """x [M, K] bf16, w_i [2n, K] interleaved -> (ab_i [M, 2n] pre-activation, h = silu(a) b [M, n]), one GEMM."""
    gemm_act, _ = _mods()
    M, n2 = x.shape[0], w_i.shape[0]
    ab = torch.empty(M, n2, device=x.device, dtype=x.dtype)
    h = torch.empty(M, n2 // 2, device=x.device, dtype=x.dtype)
    _race(("act", M, n2, x.shape[1]),
          lambda c: gemm_act(x[None], w_i[None], ab[None], None, h[None], None, "swiglu", c[0], c[1], c[2], c[3], pingpong=c[4]))
    return ab, h


def swiglu_bwd(dz, w_sq_t, ab):
    """dh = dz @ w_sq_t^T with the SwiGLU backward in the epilogue: dz [M, K] bf16, w_sq_t [n, K] (= Wsq^T contiguous),
    ab [M, 2n] interleaved pre-activation -> (dab [M, 2n] interleaved, h recomputed [M, n] for the squeeze wgrad)."""
    _, gemm_dact = _mods()
    M, n = dz.shape[0], w_sq_t.shape[0]
    dab = torch.empty_like(ab)
    h = torch.empty(M, n, device=dz.device, dtype=dz.dtype)
    _race(("dact", M, n, dz.shape[1]),
          lambda c: gemm_dact(dz[None], w_sq_t[None], dab[None], ab[None], h[None], None, "swiglu", c[0], c[1], c[2], c[3], pingpong=c[4]))
    return dab, h
