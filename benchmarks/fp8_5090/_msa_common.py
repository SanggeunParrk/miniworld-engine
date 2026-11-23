"""Shared case builders of the RTX 5090 OPM / PWA e4m3 benchmarks (the H100 scripts' constructions, docs/fp8.md "OuterProductMean" / "PWA").

OPM case  cm:cz:L:S[:a]   d_msa cm, d_pair cz, L tokens, depth S; ``a`` = the AF3 order (normalize_before_proj False); ragged MSA mask (the last
                          S / 8 rows cover half the tokens).
PWA case  D:C:cz:S:N      d_msa D, 8 heads x C, d_pair cz, depth S, N tokens; key mask with 3 masked tokens.
Random weights: W ~ N(0, 1 / fan_in), LayerNorm weight 1 +- 0.1, biases +- 0.1."""
import torch

flush = None


def init_params(m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m


def opm_case(spec, dev="cuda"):
    from miniworld_engine.modules.msa import OuterProductMean

    p = spec.split(":")
    cm, cz, L, S = map(int, p[:4])
    after = len(p) > 4 and p[4] == "a"
    m = OuterProductMean(cm, d_hidden=32, d_pair=cz, normalize_before_proj=not after, master_weights=True, backend="cuda").to(dev).eval()
    init_params(m, cm + cz)
    torch.manual_seed(0)
    x = torch.randn(1, S, L, cm, device=dev).bfloat16()
    mask = torch.ones(1, S, L, dtype=torch.bool, device=dev)
    mask[:, S - S // 8 :, : L // 2] = False
    return m, (x, mask)


def pwa_case(spec, dev="cuda"):
    from miniworld_engine.modules.msa import MSAPairWeightedAveraging

    D, C, cz, S, N = map(int, spec.split(":"))
    m = MSAPairWeightedAveraging(D, cz, n_head=8, d_hidden=C, master_weights=True, backend="cuda").to(dev).eval()
    init_params(m, D + cz + C)
    torch.manual_seed(0)
    x = torch.randn(1, S, N, D, device=dev).bfloat16()
    z = torch.randn(1, N, N, cz, device=dev).bfloat16()
    mask = torch.ones(1, N, dtype=torch.bool, device=dev)
    mask[0, torch.tensor([1, N // 2, N - 1])] = False
    return m, (x, z, mask)


def graph_time(f, n=3, reps=7):
    """us per call: CUDA-graph replay of n calls, 128 MB L2 flush before each (a flush-only graph subtracted), median of ``reps`` replays."""
    global flush
    if flush is None:
        flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            flush.zero_()
            f()
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n):
            flush.zero_()

    def t(gr):
        gr.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return sorted(ts)[reps // 2]

    out = (t(g) - t(gf)) / n * 1e3
    del g, gf
    return out


def err(y, y64, x=None):
    """rms = rms(y - y64) / rms(y64), max (judge) = max|y - y64| / max|y64|; with x: the update error rms / max of (y - y64) relative to y64 - x."""
    d = y.double() - y64
    r = {"rms": float(d.pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()), "judge": float(d.abs().max() / y64.abs().max()), "finite": bool(torch.isfinite(y).all())}
    if x is not None:
        u = y64 - x.double()
        r["upd_rms"] = float(d.pow(2).mean().sqrt() / u.pow(2).mean().sqrt())
        r["upd_max"] = float(d.abs().max() / u.abs().max())
    return r
