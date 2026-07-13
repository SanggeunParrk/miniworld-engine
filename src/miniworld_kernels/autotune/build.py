"""Builder for the per-GPU autotune config cache.

Benches every config in a kernel's grid across representative shape-buckets on the RUNNING
GPU and stores the top-K (by median ms) to the runtime cache
(``<cache-root>/autotune/<op>/<gpu_key>.json``); commit that JSON as the shipped default.

Run on the target GPU (via sbatch). Currently drives the pilot kernel ``trimul_bidir_front``;
extended per kernel as they adopt the cache prune. Config choice is performance-only, so the
builder never affects correctness — it only records which tiles are fastest here.

    PYTHONPATH=src python -m miniworld_kernels.autotune.build --op trimul_bidir_front
"""

from __future__ import annotations

import argparse

import torch
import triton


def _bench(fn, grid, pos, meta) -> float:
    try:
        return triton.testing.do_bench(
            lambda: fn[grid](*pos, **meta), warmup=25, rep=100, return_mode="median"
        )
    except Exception as e:  # noqa: BLE001 -- a config that won't launch here is just skipped
        print(f"    skip {meta}: {type(e).__name__}")
        return float("inf")


def build_transition_split_fwd(d_hiddens=(256, 512), seq_lens=(384, 512, 768, 1024), n=4,
                               top_k=5, dtype=torch.bfloat16) -> None:
    """Split transition forward GEMM (main.py transition_fwd_kernel) — A100's default large-d
    (>=256) route. Bucket = (GROUP_M, n, N) to match key_bucket_of in main.py."""
    from miniworld_kernels.autotune.cache import (
        config_space_hash, gpu_key, shape_bucket, store_ranked_configs,
    )
    from miniworld_kernels.kernels.transition.triton.main import (
        get_seq_group, transition_fwd_kernel,
    )

    kernel = transition_fwd_kernel
    full_grid = list(kernel.configs)
    csh = config_space_hash(full_grid)
    gk = gpu_key()
    op = "transition_split_fwd"
    dev = "cuda"
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
            ranked = []
            for cfg in full_grid:
                meta = dict(cfg.kwargs, GROUP_M=gm,
                            num_warps=cfg.num_warps, num_stages=cfg.num_stages)
                ms = _bench(kernel.fn, grid, pos, meta)
                if ms != float("inf"):
                    ranked.append((cfg, ms))
            ranked.sort(key=lambda t: t[1])
            if not ranked:
                print(f"  N={N} L={L} bucket={bucket}: NO launchable config (!)")
                continue
            fp = store_ranked_configs(op, gk, str(dtype).replace("torch.", ""), bucket,
                                      ranked, csh, top_k=top_k)
            print(f"  N={N} L={L} bucket={bucket}: top1={ranked[0][0].kwargs} "
                  f"{ranked[0][1]:.4f}ms -> stored {min(top_k, len(ranked))} to {fp}")


def build_trimul_bidir_front(d_pairs=(128, 256, 512), seq_lens=(384, 512, 768, 1024), top_k=5,
                             dtype=torch.bfloat16) -> None:
    from miniworld_kernels.autotune.cache import (
        config_space_hash, gpu_key, shape_bucket, store_ranked_configs,
    )
    from miniworld_kernels.kernels.trimul_inproj.triton import bidirectional as B

    kernel = B._bidir_front_kernel
    full_grid = list(kernel.configs)
    csh = config_space_hash(full_grid)
    gk = gpu_key()
    op = "trimul_bidir_front"
    dev = "cuda"
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
            ranked = []
            for cfg in full_grid:
                meta = dict(cfg.kwargs, K=d, H2=h2, GROUP_M=gm, SAVE_PREACT=True,
                            num_warps=cfg.num_warps, num_stages=cfg.num_stages)
                ms = _bench(kernel.fn, grid, pos, meta)
                if ms != float("inf"):
                    ranked.append((cfg, ms))
            ranked.sort(key=lambda t: t[1])
            if not ranked:
                print(f"  d={d} L={L} bucket={bucket}: NO launchable config (!)")
                continue
            fp = store_ranked_configs(op, gk, str(dtype).replace("torch.", ""), bucket,
                                      ranked, csh, top_k=top_k)
            best = ranked[0]
            print(f"  d={d} L={L} bucket={bucket}: top1={best[0].kwargs} {best[1]:.4f}ms "
                  f"-> stored {min(top_k, len(ranked))} to {fp}")


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
