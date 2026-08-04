"""Brute-force autotune capture for the CuTe/CUTLASS sm90 GEMM kernels.

For each wired op it times EVERY candidate GemmConfig (do_bench) across representative training
shapes and writes the ranked in-repo cache that ``cute_config.resolve_config`` reads. Run on the
target GPU (H100). Values are random (timing only) — the sweep never checks numerics.

    pixi run --frozen python submits/cute_autotune_capture.py [op ...]

With no args, sweeps all ops. Bucket strings MUST match what each kernel computes at runtime.
"""

from __future__ import annotations

import sys

import torch

from miniworld_engine.autotune.buckets import bucket_mixed
from miniworld_engine.autotune.cute_config import (
    gated_sm90_candidates,
    plain_sm90_candidates,
    sweep_and_cache,
)

BF16 = torch.bfloat16
DEV = "cuda"


def _rand(*shape, dtype=BF16):
    return torch.randn(*shape, device=DEV, dtype=dtype)


# --------------------------------------------------------------------------- #
# per-op case builders: (bucket, make_run(config) -> thunk)
# --------------------------------------------------------------------------- #
def cases_swiglu_fwd(shapes):
    from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import gemm_ln_swiglu

    cases = []
    for (M, K, N) in shapes:
        A = _rand(M, K); B = _rand(2 * N, K); PA = _rand(M, N)
        rstd = _rand(1, M, dtype=torch.float32); c1 = _rand(1, M, dtype=torch.float32)
        S = _rand(1, 2 * N, dtype=torch.float32); B2 = _rand(1, 2 * N, dtype=torch.float32)
        bucket = f"{bucket_mixed(M)}|k{K}"
        cases.append((bucket, lambda c, A=A, B=B, PA=PA, rstd=rstd, c1=c1, S=S, B2=B2:
                      (lambda: gemm_ln_swiglu(A, B, PA, rstd, c1, S, B2, config=c))))
    return "transition_swiglu_fwd", gated_sm90_candidates(), cases


def cases_gate_bwd(shapes):
    from miniworld_engine.kernels.transition.cute.backward_gatebwd import (
        transition_expand_gatebwd_cute,
    )

    cases = []
    for (M, K, N) in shapes:
        xn = _rand(M, K); ge = _rand(M, N); Wa = _rand(N, K) / (K ** 0.5); Wb = _rand(N, K) / (K ** 0.5)
        bucket = f"{bucket_mixed(M)}|k{K}"
        cases.append((bucket, lambda c, xn=xn, ge=ge, Wa=Wa, Wb=Wb:
                      (lambda: transition_expand_gatebwd_cute(xn, ge, Wa, Wb, config=c))))
    return "transition_gate_bwd", gated_sm90_candidates(), cases


def cases_dgrad(shapes):
    from miniworld_engine.autotune.cute_config import lnbwd_pp_candidates  # noqa: F401
    from miniworld_engine.kernels.layernorm_linear.cute.dgrad_lnbwd import dgrad_lnbwd_cute

    cases = []
    for (M, K, N) in shapes:
        dY = _rand(M, N); W = _rand(N, K) / (N ** 0.5); xhat = _rand(M, K)
        gamma = _rand(K); rstd = _rand(M, dtype=torch.float32).abs() + 0.5
        bucket = f"{bucket_mixed(M)}|k{K}"
        cases.append((bucket, lambda c, dY=dY, W=W, xhat=xhat, gamma=gamma, rstd=rstd:
                      (lambda: dgrad_lnbwd_cute(dY, W, xhat, gamma, rstd,
                                                tile_m=c.tile_m, cluster_m=c.cluster_m))))
    return "dgrad_lnbwd", lnbwd_pp_candidates(), cases


def cases_dab(shapes):
    from miniworld_engine.autotune.cute_config import lnbwd_pp_candidates
    from miniworld_engine.kernels.transition.cute.dab_lnbwd import transition_dab_lnbwd_cute

    cases = []
    for (M, K, N) in shapes:  # N = per-pair width; dAB is (M, 2N), w_ab (2N, K)
        dAB = _rand(M, 2 * N); w_ab = _rand(2 * N, K) / ((2 * N) ** 0.5); x = _rand(M, K)
        gamma = _rand(K); rstd = _rand(M, dtype=torch.float32).abs() + 0.5
        c1 = _rand(M, dtype=torch.float32) * 0.1
        bucket = f"{bucket_mixed(M)}|k{K}"
        cases.append((bucket, lambda c, dAB=dAB, w_ab=w_ab, x=x, gamma=gamma, rstd=rstd, c1=c1:
                      (lambda: transition_dab_lnbwd_cute(dAB, w_ab, x, gamma, rstd, c1,
                                                         tile_m=c.tile_m, cluster_m=c.cluster_m))))
    return "dab_lnbwd", lnbwd_pp_candidates(), cases


def cases_lnl_m1(shapes):
    from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
        layernorm_linear_cute,
    )

    cases = []
    for (M, K, N) in shapes:
        x = _rand(M, K); w = _rand(N, K); lw = _rand(K); lb = _rand(K)
        bucket = f"{bucket_mixed(M)}|n{N}"
        cases.append((bucket, lambda c, x=x, lw=lw, lb=lb, w=w:
                      (lambda: layernorm_linear_cute(x, lw, lb, w, None, config=c))))
    return "layernorm_linear_m1", plain_sm90_candidates(), cases


# representative training shapes (M = L*L pair rows; K = d; N = expand/output width)
_TRANSITION_SHAPES = [(65536, 128, 256), (262144, 128, 256), (65536, 256, 512), (262144, 256, 512)]
_LNL_SHAPES = [(65536, 128, 128), (262144, 128, 128), (65536, 128, 256), (262144, 128, 256),
               (65536, 256, 512), (262144, 256, 512)]

# LN-backward: reduction is over K; the tunable is tile_m over {64,128,192} where tile_n=K fits.
_LNBWD_SHAPES = [(65536, 128, 512), (262144, 128, 512), (65536, 256, 512), (262144, 256, 512)]

OPS = {
    "transition_swiglu_fwd": lambda: cases_swiglu_fwd(_TRANSITION_SHAPES),
    "transition_gate_bwd": lambda: cases_gate_bwd(_TRANSITION_SHAPES),
    "layernorm_linear_m1": lambda: cases_lnl_m1(_LNL_SHAPES),
    "dgrad_lnbwd": lambda: cases_dgrad(_LNBWD_SHAPES),
    "dab_lnbwd": lambda: cases_dab(_LNBWD_SHAPES),
}


def main(argv):
    which = argv or list(OPS)
    for name in which:
        if name not in OPS:
            print(f"!! unknown op {name}; known: {list(OPS)}", flush=True)
            continue
        op, candidates, cases = OPS[name]()
        print(f"\n=== sweeping {op}: {len(candidates)} configs x {len(cases)} shapes ===", flush=True)
        # per-bucket progress print
        def on_result(bucket, dtype, c, ms, _op=op):
            print(f"  [{_op}|{dtype}|{bucket}] tm{c.tile_m} tn{c.tile_n} "
                  f"cl({c.cluster_m},{c.cluster_n}) pp{int(c.pingpong)} -> {ms:.4f} ms", flush=True)
        sweep_and_cache(op, "torch.bfloat16", cases, candidates,
                        top_k=5, warmup=10, rep=30, on_result=on_result)
        print(f"=== done {op} ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
