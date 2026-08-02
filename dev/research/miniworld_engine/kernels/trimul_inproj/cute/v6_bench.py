"""Single-direction v6 trimul TRAINING (fwd+bwd): verify grads + speed vs baselines.

Correctness: V6 (bf16) forward + grad_x vs fp32 torch reference (cos > 0.99).
Speed (fwd+bwd ms/layer): V6 vs pytorch(torch.compile) vs nvidia(dtv1) vs cuequiv.
do_bench full-mode (the team-gm training-regime measurement). pytorch is COMPILED
(HARD RULE). B=1, bf16, outgoing, d_pair=d_hidden=128. COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src_root = _here
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import os

import torch
import torch.nn as nn
import triton

from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_engine.kernels.trimul_inproj.cute.v6_training import V6TriMul
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_engine.modules.triangle_multiplication.module import TriangleMultiplication

D = int(os.environ.get("V6_D", "128"))           # d_pair = d_hidden
EPS = 1e-5
LS = [int(x) for x in os.environ.get("V6_LS", "256,384,512,768,1024").split(",")]
COLS = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v6"]


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")


def _bench_fwdbwd_eager(fn, pair, g):
    def step():
        pair.grad = None
        y = fn(pair)
        y.backward(g)
    try:
        for _ in range(3):
            step()
        return triton.testing.do_bench(step, warmup=10, rep=50, return_mode="median",
                                       grad_to_none=[pair])
    except Exception as e:  # noqa: BLE001
        print(f"   eager fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def _bench_fwdbwd_cudagraph(fn, pair, g):
    """Capture the fwd+bwd step in a CUDA graph (launch-overhead-free = the fair
    regime for our many-launch cute/triton path; memory: compile-vs-cudagraph-for-cute).
    Grads accumulate into static buffers across replays — fine for timing."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                pair.grad = None
                y = fn(pair)
                y.backward(g)
        torch.cuda.current_stream().wait_stream(s)
        pair.grad.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = fn(pair)
            y.backward(g)
        return triton.testing.do_bench(graph.replay, warmup=10, rep=50, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"   cudagraph fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def main():
    import os
    assert torch.cuda.is_available()
    DIR = os.environ.get("V6_DIR", "out")        # "out" (outgoing) or "in" (incoming)
    OUTGOING = DIR == "out"
    DIRSTR = "outgoing" if OUTGOING else "incoming"
    print(f"v6 single-dir [{DIRSTR}] D={D} fwd+bwd on {torch.cuda.get_device_name(0)}", flush=True)
    print("regime: ALL methods torch.compile (default) fwd+bwd, params require grad, event-timed",
          flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16

    base = TriangleMultiplication(d_pair=D, outgoing=OUTGOING,
                                  implementation=ImplementationType.PYTORCH).cuda()
    torch.manual_seed(0)
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)

    # --- correctness: V6 (bf16) vs fp32 torch reference (outgoing), at EVERY L ---
    import copy
    base_fp = copy.deepcopy(base).float()          # INDEPENDENT fp32 ref (base.to(dt) is in-place)
    v6 = V6TriMul(base.to(dt), direction=DIR)
    all_ok = True
    for Lc in LS:
        pair = torch.randn(1, Lc, Lc, D, device="cuda")
        dy = torch.randn_like(pair)
        xr = pair.float().clone().requires_grad_(True)
        gxr = torch.autograd.grad(base_fp(xr), xr, dy, retain_graph=False)[0]
        xo = pair.to(dt).clone().requires_grad_(True)
        yo = v6(xo)
        yo.backward(dy.to(dt))
        fwd_cos, dx_cos = cos(base_fp(xr), yo), cos(gxr, xo.grad)
        ok = min(fwd_cos, dx_cos) > 0.99
        all_ok &= ok
        print(f"  correctness L={Lc}: fwd cos={fwd_cos:.5f}  grad_x cos={dx_cos:.5f} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        del pair, dy, xr, xo, yo
    print(f"  -> correctness ALL L: {'PASS' if all_ok else 'FAIL'}", flush=True)
    torch.cuda.empty_cache()

    # --- speed: fwd+bwd ms/layer, ALL methods under torch.compile (reduce-overhead) ---
    # compile handles the fwd+bwd CUDA-graph AND param-grad accumulation automatically
    # (exact training, params require grad) — no manual capture / AccumulateGrad fighting.
    base_bf = base.to(dt)
    cq = TriangleMultiplication(d_pair=D, outgoing=OUTGOING,
                                implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt)
    cq.load_state_dict(base_bf.state_dict())

    class DtV1Mod(nn.Module):
        """Wrap dtv1 as a module so torch.compile sees its (base) params for fwd+bwd."""

        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, p):
            b = self.b
            return fused_triangle_multiplicative_update_dtv1(
                p, DIRSTR, None, eps=EPS,
                norm_in_weight=b.ln_pair.weight, norm_in_bias=b.ln_pair.bias,
                p_in_weight=torch.cat([b.to_left.weight, b.to_right.weight], dim=0),
                g_in_weight=torch.cat([b.to_left_gate.weight, b.to_right_gate.weight], dim=0),
                norm_out_weight=b.ln_out.weight, norm_out_bias=b.ln_out.bias,
                p_out_weight=b.to_out.weight, g_out_weight=b.to_gate.weight)

    mods = {
        "pytorch": base_bf,
        "nvidia(dtv1)": DtV1Mod(base.to(dt)),
        "cuequivariance": cq,
        "ours_v6": v6,
    }

    def bench_compiled_train(mod, p, gy):
        """fwd+bwd under torch.compile (default mode: Inductor fusion, params require grad
        = exact training). Manual CUDA-event timing (reduce-overhead's cudagraph-trees errors
        'output overwritten' under do_bench; default mode avoids that)."""
        comp = torch.compile(mod)
        params = [pr for pr in mod.parameters() if pr.requires_grad]

        def step():
            p.grad = None
            for pr in params:
                pr.grad = None
            comp(p).backward(gy)
        try:
            for _ in range(12):   # warm: compile + autotune
                step()
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(50):
                step()
            ev1.record()
            torch.cuda.synchronize()
            return ev0.elapsed_time(ev1) / 50
        except Exception as e:  # noqa: BLE001
            print(f"   compiled-train fail: {type(e).__name__}: {str(e)[:90]}", flush=True)
            return float("nan")

    # one method per process (argv) — compile/graph issues isolated.
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    cols = [only] if only else ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v6"]

    rows = {}
    for Lb in LS:
        p = torch.randn(1, Lb, Lb, D, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(p)
        t = {name: bench_compiled_train(mods[name], p, g) for name in cols}
        rows[Lb] = t
        print(f"[L={Lb}] " + " ".join(f"{c}={t[c]:.3f}" for c in cols), flush=True)
        del p, g
        torch.cuda.empty_cache()

    print("\n=== single-dir trimul fwd+bwd, ms/layer (ALL torch.compile default, event-timed) ===")
    print(f"{'L':>5} | " + " | ".join(f"{c:>14}" for c in cols))
    print("-" * 92)
    for Lb in LS:
        r = rows[Lb]
        print(f"{Lb:>5} | " + " | ".join(f"{r[c]:>14.3f}" for c in cols))
    for c in cols:
        print(f"DATA {c} " + ",".join(f"{Lb}:{rows[Lb][c]:.4f}" for Lb in LS), flush=True)


if __name__ == "__main__":
    main()
