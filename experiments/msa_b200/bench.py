"""B200 OPM / PWA latency: PyTorch vs Anthropic opt_core vs this engine, inference and training.

Every number is the median of 7 x 50 CUDA-graph replays of one captured call (so launch overhead is
excluded for every column alike).  Training = forward + backward to every input and parameter
(autograd.grad), dropout 0.  Masks are ~10 % false.  Anthropic's opt_core MSA cells are forward-only:
their training cell is reported as n/a.

    python bench.py --op opm pwa --L 384 --S 1024 --impl pytorch anthropic ours --out res.json
    python bench.py --calibrate          # measured HBM copy bandwidth and cuBLAS bf16 peak (the SoL ceilings)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import types

import torch

D_MSA, D_PAIR, D_HID, HEADS = 64, 128, 32, 8


MIN_MODE = os.environ.get("BENCH_MIN") == "1"      # shared GPU: the minimum of many short rounds approximates an idle card


def timeit(fn, reps=50, rounds=7, warm=3):
    if MIN_MODE:
        reps, rounds = 5, 60
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / reps)
    return min(out) if MIN_MODE else statistics.median(out)


def make_inputs(op, L, S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = torch.bfloat16
    msa = torch.randn(1, S, L, D_MSA, device="cuda", dtype=bf, generator=g)
    pair = torch.randn(1, L, L, D_PAIR, device="cuda", dtype=bf, generator=g)
    if op == "opm":
        mask = torch.rand(1, S, L, device="cuda", generator=g) > 0.1
    else:
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    return msa, pair, mask


def make_module(op, impl, seed=0):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.msa_pair_weighted_averaging import MSAPairWeightedAveraging
    from miniworld_engine.modules.outer_product import OuterProductMean
    it = {"pytorch": IT.PYTORCH, "anthropic": IT.PYTORCH, "ours": IT.MINIWORLD}[impl]
    torch.manual_seed(seed)
    if op == "opm":
        m = OuterProductMean(D_MSA, D_PAIR, D_HID, implementation=it)
        torch.nn.init.normal_(m.to_out.weight, std=0.02)          # the zero init would hide the projection's gradients
        torch.nn.init.normal_(m.to_out.bias, std=0.02)
    else:
        m = MSAPairWeightedAveraging(D_MSA, D_PAIR, HEADS, D_HID, implementation=it)
        torch.nn.init.normal_(m.to_out.weight, std=0.02)
    for p in m.parameters():                                       # non-trivial LayerNorm affine
        if p.dim() == 1 and p.numel() in (D_MSA, D_PAIR):
            with torch.no_grad():
                p.add_(0.1 * torch.randn_like(p))
    return m.cuda().to(torch.bfloat16)


# ---- Anthropic opt_core cells, called as they ship ----
_OPT = {}


def opt_core():
    if not _OPT:
        root = os.environ.get("OPT_CORE_DIR")
        if not root:
            raise RuntimeError("OPT_CORE_DIR must name uplifting-biomolecular-modeling/common/opt_core")
        sys.path.insert(0, root)
        import importlib
        _OPT["opm"] = importlib.import_module("opt_core.ops.msa_opm")
        _OPT["pwa"] = importlib.import_module("opt_core.ops.msa_pwa")
        os.environ.setdefault("FPF_PWA_CFG", "g_fo4p")
    return _OPT


def anthropic_fn(op, mod, msa, pair, mask):
    oc = opt_core()
    if op == "opm":
        view = types.SimpleNamespace(norm=mod.ln_msa, proj_a=mod.to_left, proj_b=mod.to_right, proj_o=mod.to_out)
        return lambda: pair + oc["opm"].forward_mask_norm(view, msa, mask.to(torch.bfloat16))
    n = msa.shape[2]
    pm = mask[:, None, :].expand(-1, n, -1).to(torch.bfloat16)
    view = types.SimpleNamespace(norm_m=mod.ln_msa, proj_m=mod.to_value, proj_g=mod.to_gate, norm_z=mod.ln_pair,
                                 proj_z=mod.to_bias, proj_o=mod.to_out, inf=1e9, num_heads=mod.n_head,
                                 c_h=mod.to_value.weight.shape[0] // mod.n_head)
    return lambda: msa + oc["pwa"].forward_masked(view, msa, pair, pm, chunk_heads=False)


def module_fn(op, mod, msa, pair, mask):
    if op == "opm":
        return lambda: mod(msa, mask, residual=pair)
    return lambda: mod(msa, pair, mask)


def run_case(op, impl, mode, L, S):
    mod = make_module(op, impl)
    msa, pair, mask = make_inputs(op, L, S)
    if mode == "infer":
        mod.eval()
        with torch.no_grad():
            fn = anthropic_fn(op, mod, msa, pair, mask) if impl == "anthropic" else module_fn(op, mod, msa, pair, mask)
            return timeit(fn)
    if impl == "anthropic":
        return None                                                  # opt_core MSA cells have no backward
    mod.train()
    msa.requires_grad_(True)
    pair.requires_grad_(True)
    f = module_fn(op, mod, msa, pair, mask)
    params = [msa, pair, *mod.parameters()]
    gout = torch.randn(msa.shape if op == "pwa" else pair.shape, device="cuda", dtype=torch.bfloat16)

    def step():
        out = f()
        return torch.autograd.grad(out, params, gout)
    return timeit(step)


# ---- speed-of-light floors of THIS fusion algorithm (the H100 kernel boundaries, B200 kernels) ----
# per kernel: max(minimum DRAM bytes / HBM ceiling, FLOPs / tensor ceiling), summed over the module's kernels.
# Ceilings = the best rates measured on this B200: 6.9 TB/s (in-place read+write stream) and 1.73 PFLOP/s
# (cuBLAS on the module's own 12288^2 x 1024 GEMM).  Launch gaps are not in the floor.
HBM, TC = 6.9e12, 1.73e15


def sol_floor(op, mode, L, S, hbm=None, tc=None, energy=None):
    """energy = (J per HBM byte, J per FLOP, dynamic power budget W): on a power-capped card a kernel also needs
    (bytes e_b + FLOPs e_f) / (P_cap - P_idle) -- bytes and FLOPs share one power budget, so they add."""
    hbm, tc = hbm or HBM, tc or TC
    def k(byts, flops=0.0):
        t = max(byts / hbm, flops / tc)
        if energy is not None:
            eb, ef, pdyn = energy
            t = max(t, (byts * eb + flops * ef) / pdyn)
        return t
    n2, bf = L * L, 2
    m = S * L * 64 * bf                                  # an [S, N, 64] bf16 tensor
    if op == "opm":
        a = L * 32 * S * bf                              # A2 / BT
        o = (L * 32) ** 2 * bf                           # the grouped outer product O
        z = n2 * 128 * bf
        gemm = 2 * (L * 32) ** 2 * S
        fwd = [k(m + S * L + 2 * a + (S * L * 8 if mode == "train" else 0)),   # prologue (+ LN stats)
               k(2 * a + o, gemm),                                         # grouped GEMM
               k(o + 2 * z, 2 * n2 * 1024 * 128)]                          # epilogue (+ residual)
        if mode == "infer":
            return sum(fwd)
        bwd = [k(z + o + z, 2 * n2 * 128 * 1024),       # dgrad: dz -> dO (grouped) + dz/n
               k(o + 2 * a, gemm), k(o + 2 * a, gemm),   # dA, dB
               k(o + z, 2 * n2 * 1024 * 128),            # dWo
               k(2 * a + m + S * L * 8 + S * L + m)]     # prologue backward
        return sum(fwd) + sum(bwd)
    H, C = 8, 32
    v = S * L * H * C * bf
    zp = n2 * 128 * bf
    fwd = [k(zp + H * n2 * bf),                          # pair3: LN_z -> proj_z -> softmax
           k(m + m + v),                                 # ln_vg: y and head-major v
           k(v + 3 * m + (v if mode == "train" else 0), 2 * H * n2 * S * C + 2 * 2 * S * L * 64 * H * C)]   # fwd (+ o)
    if mode == "infer":
        return sum(fwd)
    bwd = [k(v + 2 * m + 2 * v),                         # glue: o, y, dres -> do, dgp
           k(2 * v, 2 * H * n2 * S * C),                 # plain: dv
           k(2 * v, 2 * H * n2 * S * C),                 # dw bmm
           k(2 * v + 4 * m),                             # dgv backward
           k(2 * zp + H * n2 * (2 + 4))]                 # pair backward
    return sum(fwd) + sum(bwd)


def calibrate():
    bf = torch.bfloat16
    x = torch.empty(2 * 1024 ** 3 // 2, dtype=bf, device="cuda")
    y = torch.empty_like(x)
    t = timeit(lambda: y.copy_(x), reps=20)
    bw = 2 * x.numel() * 2 / (t * 1e-3) / 1e12
    a = torch.randn(8192, 8192, dtype=bf, device="cuda")
    b = torch.randn(8192, 8192, dtype=bf, device="cuda")
    t = timeit(lambda: a @ b, reps=20)
    tf = 2 * 8192 ** 3 / (t * 1e-3) / 1e12
    return {"copy_TBps": bw, "bf16_gemm_TFLOPs": tf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", nargs="+", default=["opm", "pwa"])
    ap.add_argument("--impl", nargs="+", default=["pytorch", "anthropic", "ours"])
    ap.add_argument("--mode", nargs="+", default=["infer", "train"])
    ap.add_argument("--L", nargs="+", type=int, default=[384])
    ap.add_argument("--S", nargs="+", type=int, default=[1024])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
    if a.calibrate:
        res["calibration"] = calibrate()
        print(res["calibration"], flush=True)
    rows = []
    for op in a.op:
        for L in a.L:
            for S in a.S:
                for mode in a.mode:
                    for impl in a.impl:
                        try:
                            ms = run_case(op, impl, mode, L, S)
                            err = None
                        except Exception as exc:                     # a column that cannot run is reported, not fatal
                            ms, err = None, f"{type(exc).__name__}: {str(exc)[:300]}"
                        floor = sol_floor(op, mode, L, S) * 1e3
                        rows.append(dict(op=op, L=L, S=S, mode=mode, impl=impl, ms=ms, err=err, sol_floor_ms=floor))
                        sol = f"  SoL {100 * floor / ms:5.1f}% (floor {floor:.3f} ms)" if ms is not None and impl == "ours" else ""
                        print(f"{op} L{L} S{S} {mode:5s} {impl:9s} " + (f"{ms:.4f} ms" if ms is not None else f"n/a {err or ''}") + sol, flush=True)
                        torch.cuda.empty_cache()
    res["rows"] = rows
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
