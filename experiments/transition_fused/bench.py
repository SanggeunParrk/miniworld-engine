"""Correctness and timing for the fused Transition backward kernel (D = 128, H = 512, bf16).

Builds one Transition training case at M = length^2, computes the forward's saved tensors the way the engine's forward does
(fp32 LayerNorm statistics, bf16 normalised activations), runs the fused kernel, and checks its six gradients against an fp32
autograd reference. `--engine` additionally runs the engine's own backward on the same saved tensors as a speed and agreement
baseline; it is optional because that entry point moves between branches. Timing is CUDA-graph replay.

  python bench.py --length 384 [--dw-repl 8] [--cubin build/transition_bwd_r8.cubin] [--engine] [--save records/bench-L384.json]

Run on a compute node; needs torch with CUDA and `cuda.bindings` (the cu128 environment).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402


p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--width", type=int, default=128)
p.add_argument("--dw-repl", type=int, default=8, help="hidden-slice replicas: NDW = 8 R weight CTAs, NDX = ctas - NDW input CTAs")
p.add_argument("--ctas", type=int, default=132)
p.add_argument("--cubin", default="")
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--seed", type=int, default=2319)
p.add_argument("--eps", type=float, default=1e-5)
p.add_argument("--engine", action="store_true", help="also run and time the engine's backward on the same saved tensors")
p.add_argument("--save", default="")
a = p.parse_args()
D, H, ROWS = a.width, 4 * a.width, 128
SLICES = H // 64
NPART = 3 if D == 128 else 4                              # the D = 64 kernel keeps dWs^T as two row-half partial sums

NDW, NCTA = SLICES * a.dw_repl, a.ctas
NDX = NCTA - NDW
if NDX <= 0:
    raise SystemExit("no input CTAs left: lower --dw-repl")
M = a.length * a.length
if M % ROWS:
    raise SystemExit("M must be a multiple of %d" % ROWS)
tiles = M // ROWS
dev = "cuda"
torch.manual_seed(a.seed)

x = torch.randn(M, D, device=dev, dtype=torch.bfloat16)
gamma = torch.rand(D, device=dev) + 0.5
beta = torch.randn(D, device=dev) * 0.1
gb, bb = gamma.to(torch.bfloat16), beta.to(torch.bfloat16)      # the module pins the norm params fp32 and casts them to x.dtype
wa = (torch.randn(H, D, device=dev) * D ** -0.5).to(torch.bfloat16).contiguous()
wb = (torch.randn(H, D, device=dev) * D ** -0.5).to(torch.bfloat16).contiguous()
ws = (torch.randn(D, H, device=dev) * H ** -0.5).to(torch.bfloat16).contiguous()
dy = torch.randn(M, D, device=dev, dtype=torch.bfloat16)

with torch.no_grad():                                            # the forward's saves: fp32 statistics, bf16 normalised activations
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + a.eps)
    c1 = (mean * rstd).contiguous()
    rstd = rstd.contiguous()
    xn = ((xf * rstd[:, None] - c1[:, None]) * gb.float() + bb.float()).to(torch.bfloat16).contiguous()

NAMES = ("dx", "dgamma", "dbeta", "dWa", "dWb", "dWs")


def reference():
    """fp32 autograd through the same function: y = x + Ws(silu(Wa LN(x)) * (Wb LN(x)))."""
    xr = x.float().detach().requires_grad_(True)
    g = gb.float().detach().requires_grad_(True)
    b = bb.float().detach().requires_grad_(True)
    A = wa.float().detach().requires_grad_(True)
    B = wb.float().detach().requires_grad_(True)
    S = ws.float().detach().requires_grad_(True)
    y = F.layer_norm(xr, (D,), g, b, a.eps)
    (F.silu(y @ A.t()) * (y @ B.t()) @ S.t() + xr).backward(dy.float())
    return xr.grad, g.grad, b.grad, A.grad, B.grad, S.grad


cubin = Path(a.cubin) if a.cubin else HERE / "build" / f"transition_bwd_r{a.dw_repl}.cubin"
if not cubin.is_file():
    raise SystemExit(f"{cubin} missing: OUT=transition_bwd_r{a.dw_repl} ./build.sh -DDW_REPL={a.dw_repl}")
SMEM = 231424
k = drv.Kernel(str(cubin), "transition_bwd_fused", SMEM)
kr = drv.Kernel(str(cubin), "reduce_partials", 0)
print(f"{cubin.name}: regs {k.regs} lmem {k.lmem} smem {SMEM} | grid {NCTA}x256 (DW {NDW} / DX {NDX}) | "
      f"M {M} = {tiles} tiles of {ROWS} rows", flush=True)

tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
maps = (tm(dy, [D, M], D * 2, [64, 64]), tm(xn, [D, M], D * 2, [64, 64]), tm(x, [D, M], D * 2, [64, 64]),
        tm(ws, [H, D], H * 2, [64, D]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]))
gamma_f = gb.float().contiguous()
dx_k = torch.empty_like(x)
dgamma_k, dbeta_k = torch.zeros(D, device=dev), torch.zeros(D, device=dev)
partw = torch.empty(NDW * NPART * 64 * D, device=dev, dtype=torch.float32)
dgbw = torch.empty(NDX * 8 * 2 * D, device=dev, dtype=torch.float32)
dWa_k, dWb_k, dWs_k = torch.empty_like(wa), torch.empty_like(wb), torch.empty_like(ws)


def fused():
    k((NCTA, 1, 1), (256, 1, 1), *maps, rstd, c1, gamma_f, dx_k, dgamma_k, dbeta_k, partw, dgbw, int(M), int(tiles))
    kr(((3 * SLICES * 64 * D + 2 * D + 255) // 256, 1, 1), (256, 1, 1), partw, dWa_k, dWb_k, dWs_k, dgbw, dgamma_k, dbeta_k)
    return dx_k, dgamma_k, dbeta_k, dWa_k, dWb_k, dWs_k


engine = None
if a.engine:
    try:
        from miniworld_engine.autotune.shape_key import both_key
        from miniworld_engine.kernels.transition.triton.fused import _fused_bwd
        key = both_key(M)

        def engine():
            """The engine's own backward on the same saved tensors; the trailing fuse-residual flag exists only on some
            revisions. This is a convenience baseline, not part of the deliverable - check its own row against the fp32
            reference before quoting an agreement number from it."""
            try:
                return _fused_bwd(dy, x, rstd, c1, gb, bb, wa, wb, ws, xn, a.eps, True, [M, D], key, True)
            except (RuntimeError, TypeError):
                return _fused_bwd(dy, x, rstd, c1, gb, bb, wa, wb, ws, xn, a.eps, True, [M, D], key)
    except Exception as exc:  # noqa: BLE001
        print(f"engine backward unavailable on this branch ({exc!r}); skipping the baseline", flush=True)


def compare(tag, got, want):
    out = {}
    for n, g, w in zip(NAMES, got, want):
        gf, wf = g.float(), w.float()
        d = (gf - wf).abs()
        out[n] = dict(rel_rms=float(d.norm() / wf.norm().clamp_min(1e-20)), max_abs=float(d.max()),
                      frac_diff=float((d > 0).float().mean()), finite=bool(torch.isfinite(gf).all()))
        print(f"  {tag:>18s} {n:>6s}: rel_rms {out[n]['rel_rms']:.3e}  max_abs {out[n]['max_abs']:.3e}  "
              f"differing {out[n]['frac_diff'] * 100:6.3f}%  finite {out[n]['finite']}", flush=True)
    return out


rec = dict(length=a.length, M=M, D=D, H=H, dw_repl=a.dw_repl, ctas=NCTA, ndw=NDW, ndx=NDX,
           regs=k.regs, lmem=k.lmem, smem=SMEM, cubin=cubin.name)
got = [t.clone() for t in fused()]
torch.cuda.synchronize()
ref = reference()
rec["fused_vs_fp32"] = compare("fused vs fp32", got, ref)
again = [t.clone() for t in fused()]
torch.cuda.synchronize()
rec["reproducible"] = {n: bool(torch.equal(u, v)) for n, u, v in zip(NAMES, got, again)}
print("  bit-reproducible on replay:", rec["reproducible"], flush=True)
if engine is not None:
    eng = [t.clone() for t in engine()]
    torch.cuda.synchronize()
    rec["engine_vs_fp32"] = compare("engine vs fp32", eng, ref)
    rec["fused_vs_engine"] = compare("fused vs engine", got, eng)


def graph_of(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    return g


def time_us(g, reps):
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    st.record()
    for _ in range(reps):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000.0 / reps


graphs = {"fused": graph_of(fused)}
if engine is not None:
    try:
        graphs["engine"] = graph_of(engine)
    except Exception as exc:  # noqa: BLE001
        print(f"engine graph capture failed ({exc!r})", flush=True)
runs = {n: [] for n in graphs}
for r in range(a.rounds):
    for n in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
        runs[n].append(time_us(graphs[n], a.reps))
floor_us = 16.0 * M * D * H / (132 * 4096 * 1.755e9) * 1e6        # 132 SMs x 4096 dense bf16 FLOP per cycle at 1.755 GHz
rec["tensor_floor_us"] = floor_us
rec["time_us"] = {}
for n, v in runs.items():
    med = statistics.median(v)
    rec["time_us"][n] = dict(median=med, min=min(v), spread=max(v) - min(v), samples=v)
    print(f"RESULT {n:>8s}: {med:8.1f} us (min {min(v):.1f}, spread {max(v) - min(v):.1f}) | "
          f"tensor floor {floor_us:.1f} us -> {floor_us / med * 100:.1f}%", flush=True)
if a.save:
    out = Path(a.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("saved", out, flush=True)
