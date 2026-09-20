"""Shared setup for the K1/K3 inference measurements; nothing here needs a MiniWorld workspace.

The fp32 reference is the engine's PyTorch ``TriangleMultiplication``; its weights are handed to ``trimul_native.face.serve``; the face is
imported from ``$TRIMUL_NATIVE_BUILD_DIR/../python`` (a payload made by ``build_payload.py``).  The setup mirrors the adoption benchmark that
produced the engine-path numbers: seed 4103, ``mask[:, ::7] = False``, bf16 activations and weights, fp32 LayerNorm affine, TF32 off."""
import copy
import importlib
import math
import os
import statistics
import sys
import types

import torch

from miniworld_engine.modules import TriangleMultiplication

MASK_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "bool": torch.bool}


def init(m, dt=torch.bfloat16):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(dt)
            t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float()
            if name.endswith("weight"):
                t.data.normal_(1, .05)
            else:
                t.data.normal_(0, .05)
    return m


def weights_of(m):
    return {k: t.detach().contiguous() for k, t in dict(
        ln_in_w=m.ln_pair.weight, ln_in_b=m.ln_pair.bias, w_ag=m.to_left_gate.weight, w_ap=m.to_left.weight,
        w_bg=m.to_right_gate.weight, w_bp=m.to_right.weight, ln_out_w=m.ln_out.weight, ln_out_b=m.ln_out.bias,
        w_o=m.to_out.weight, w_og=m.to_gate.weight).items()}


def make(length, width=128, outgoing=True, mask_dtype="bf16", seed=4103):
    """One op's inputs: module, pair tensor, token mask, pair mask (in the dtype the caller hands the face), weights, fp32 reference."""
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    m = init(TriangleMultiplication(width, outgoing=outgoing, implementation="pytorch"))
    x = torch.randn(1, length, length, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
    mask[:, ::7] = False
    with torch.no_grad():
        ref = copy.deepcopy(m).float()(x.float(), mask)
    pairmask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(MASK_DTYPES[mask_dtype]).contiguous()
    return types.SimpleNamespace(m=m, x=x, mask=mask, pairmask=pairmask, weights=weights_of(m), ref=ref, eps=m.ln_pair.eps,
                                 direction="outgoing" if outgoing else "incoming", length=length, width=width)


def second(fx, outgoing=False):
    """A second module (the pairformer's next TriMul, incoming by default) applied to the first op's fp32 output."""
    m2 = init(TriangleMultiplication(fx.width, outgoing=outgoing, implementation="pytorch"))
    with torch.no_grad():
        ref2 = copy.deepcopy(m2).float()(fx.ref, fx.mask)
    return types.SimpleNamespace(m=m2, weights=weights_of(m2), ref=ref2, direction="outgoing" if outgoing else "incoming")


def face():
    """``trimul_native.face`` of the payload named by TRIMUL_NATIVE_BUILD_DIR (its python/ sits next to build/)."""
    bd = os.environ.get("TRIMUL_NATIVE_BUILD_DIR")
    if not bd:
        sys.exit("set TRIMUL_NATIVE_BUILD_DIR=<payload>/build (see build_payload.py)")
    py = os.path.join(os.path.dirname(os.path.abspath(bd)), "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    F = importlib.import_module("trimul_native.face")
    got = os.path.dirname(os.path.abspath(F.__file__))
    if got != os.path.join(py, "trimul_native"):
        sys.exit("trimul_native imported from %s, not the payload %s" % (got, py))
    return F


def error(y, r):
    d = y.float() - r.float()
    return dict(rel_rms=float(d.square().mean().sqrt() / (r.float().square().mean().sqrt() + 1e-12)), max_abs=float(d.abs().max()),
                finite=bool(torch.isfinite(y).all()))


def cupti_per_kernel(fn, iters):
    """Device durations per kernel class over ``iters`` eager calls (torch.profiler / CUPTI): k1 / k3 / cublas / other, us per call."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per, names = {}, set()
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        n = e.name
        key = "k1" if "tmn_k1" in n else "k3" if "tmn_k3" in n else "cublas" if ("nvjet" in n or "gemm" in n.lower() or "cutlass" in n.lower()) else "other"
        per.setdefault(key, []).append(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
        if "tmn_" in n:
            names.add(n[:70])
    return {k: dict(count=len(v) / iters, sum_per_call_us=sum(v) / iters, median_launch_us=statistics.median(v)) for k, v in per.items()}, sorted(names)


def capture(fn):
    """One-call CUDA graph of ``fn`` (after two side-stream warm calls); returns (graph, captured output)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    return g, out


def replay_us(g, warm=20, reps=100):
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / reps
