"""Builder for the per-GPU autotune config cache (backend-agnostic: Triton / CuTe / CUDA).

Benches every candidate config across representative shape-buckets on the RUNNING GPU and
stores the top-K (by median ms) to the runtime cache
(``<cache-root>/autotune/<op>/<gpu_key>.json``); commit that JSON as the shipped default.

The core is :func:`tune_bucket` — it takes a list of candidate configs (``triton.Config`` OR
plain ``dict`` tile/cluster params) and a ``run_ms(cfg) -> float`` closure, so the SAME ranking
+ storage path serves every backend: Triton builders bench ``kernel.fn[grid]``; CuTe/CUDA
builders build+run the kernel with each config. Config choice is performance-only, so the
builder never affects correctness — it only records which configs are fastest here.

Run on the target GPU (via sbatch). CuTe/CUDA sweeps require sm90+ (skip on Ampere).

    PYTHONPATH=src python -m miniworld_engine.autotune.build --op all   # or --op <name>
"""

from __future__ import annotations

import argparse

import torch
import triton

from miniworld_engine.autotune.cache import (
    as_cfg_dict,
    config_space_hash,
    gpu_key,
    shape_bucket,
    store_ranked_configs,
)


def _bench(fn, grid, pos, meta) -> float:
    try:
        return triton.testing.do_bench(
            lambda: fn[grid](*pos, **meta), warmup=25, rep=100, return_mode="median"
        )
    except Exception as e:  # noqa: BLE001 -- a config that won't launch here is just skipped
        print(f"    skip {meta}: {type(e).__name__}")
        return float("inf")


def tune_bucket(op, gk, dtype_str, bucket, candidates, run_ms, csh, *, top_k=5):
    """Backend-agnostic core: bench each candidate via ``run_ms(cfg) -> ms``, store the top-K
    (fastest first) for ``(op, gpu, dtype, bucket)``. ``candidates`` may be ``triton.Config``
    objects (Triton) or plain dicts (CuTe/CUDA tile params). Returns the written path or None."""
    ranked = []
    for cfg in candidates:
        ms = run_ms(cfg)
        if ms != float("inf"):
            ranked.append((cfg, ms))
    ranked.sort(key=lambda t: t[1])
    if not ranked:
        print(f"  {bucket}: NO runnable config (!)")
        return None
    fp = store_ranked_configs(op, gk, dtype_str, bucket, ranked, csh, top_k=top_k)
    print(f"  {bucket}: top1={as_cfg_dict(ranked[0][0])['kwargs']} {ranked[0][1]:.4f}ms "
          f"-> stored {min(top_k, len(ranked))} to {fp}")
    return fp


def build_transition_split_fwd(d_hiddens=(256, 512), seq_lens=(384, 512, 768, 1024), n=4,
                               top_k=5, dtype=torch.bfloat16) -> None:
    """Split transition forward GEMM (main.py transition_fwd_kernel) — A100's default large-d
    (>=256) route. Bucket = (GROUP_M, n, N) to match key_bucket_of in main.py."""
    from miniworld_engine.kernels.transition.triton.main import (
        get_seq_group, transition_fwd_kernel,
    )

    kernel = transition_fwd_kernel
    full_grid = list(kernel.configs)
    csh = config_space_hash(full_grid)
    gk, op, dev = gpu_key(), "transition_split_fwd", "cuda"
    dtype_str = str(dtype).replace("torch.", "")
    print(f"[build] op={op} gpu={gk} configs={len(full_grid)} hash={csh}")
    for N in d_hiddens:
        for L in seq_lens:
            M = L * L
            x = torch.randn(M, N, device=dev, dtype=dtype)
            wa = torch.randn(n * N, N, device=dev, dtype=dtype) * 0.05
            wb = torch.randn(n * N, N, device=dev, dtype=dtype) * 0.05
            out = torch.empty(M, n * N, device=dev, dtype=dtype)
            gm = get_seq_group(M)
            pos = (x, wa, wb, out, M, n, N)
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(n * N, meta["BLOCK_N"]),
            )
            bucket = shape_bucket(GROUP_M=gm, n=n, N=N)
            run_ms = lambda cfg: _bench(  # noqa: E731
                kernel.fn, grid, pos,
                dict(cfg.kwargs, GROUP_M=gm, num_warps=cfg.num_warps, num_stages=cfg.num_stages))
            tune_bucket(op, gk, dtype_str, bucket, full_grid, run_ms, csh, top_k=top_k)


def build_trimul_bidir_front(d_pairs=(128, 256, 512), seq_lens=(384, 512, 768, 1024), top_k=5,
                             dtype=torch.bfloat16) -> None:
    from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B

    kernel = B._bidir_front_kernel
    full_grid = list(kernel.configs)
    csh = config_space_hash(full_grid)
    gk, op, dev = gpu_key(), "trimul_bidir_front", "cuda"
    dtype_str = str(dtype).replace("torch.", "")
    print(f"[build] op={op} gpu={gk} configs={len(full_grid)} hash={csh}")

    for d in d_pairs:
        h2 = 2 * d  # trimul: d_hidden == d_pair -> per-side H2 = 2*d
        for L in seq_lens:
            M = L * L
            x_flat = torch.randn(M, d, device=dev, dtype=dtype)
            WL = torch.randn(d, h2, device=dev, dtype=dtype) * 0.05
            WLg = torch.randn(d, h2, device=dev, dtype=dtype) * 0.05
            WR = torch.randn(d, h2, device=dev, dtype=dtype) * 0.05
            WRg = torch.randn(d, h2, device=dev, dtype=dtype) * 0.05
            left_w = torch.stack([WLg, WL], dim=2).reshape(d, 2 * h2)
            right_w = torch.stack([WRg, WR], dim=2).reshape(d, 2 * h2)
            Wlr = torch.cat([left_w, right_w], dim=1).contiguous()
            left = torch.empty(1, h2, L, L, device=dev, dtype=dtype)
            right = torch.empty(1, h2, L, L, device=dev, dtype=dtype)
            preact = torch.empty(4 * h2, M, device=dev, dtype=dtype)
            gm = B.get_seq_group(M)
            pos = (x_flat, Wlr, left, right, preact, M, M)
            grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
            bucket = shape_bucket(GM=gm, H2=h2, K=d)
            run_ms = lambda cfg: _bench(  # noqa: E731
                kernel.fn, grid, pos,
                dict(cfg.kwargs, K=d, H2=h2, GROUP_M=gm, SAVE_PREACT=True,
                     num_warps=cfg.num_warps, num_stages=cfg.num_stages))
            tune_bucket(op, gk, dtype_str, bucket, full_grid, run_ms, csh, top_k=top_k)


_BUILDERS = {
    "trimul_bidir_front": build_trimul_bidir_front,
    "transition_split_fwd": build_transition_split_fwd,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="all", choices=[*sorted(_BUILDERS), "all"])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "cache builder must run on a GPU"
    ops = sorted(_BUILDERS) if args.op == "all" else [args.op]
    for op in ops:
        _BUILDERS[op]()
    print("BUILD DONE")


if __name__ == "__main__":
    main()
