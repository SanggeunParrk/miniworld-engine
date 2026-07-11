"""Tile-config sweep for the left+right gated GEMM (quack).

quack already autotunes (gemm_gated_tuned @autotune over get_all_configs("gated")),
so the high-level tuned call is the optimum. This sweeps the SM90 gated configs
explicitly to show the spread and confirm the autotuned best. Reports ms +
achieved GB/s (memory-bound). blld (contiguous) postact, no patch needed.
B=1, D=128, bf16. COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import triton
from quack.gemm_config import get_all_configs
from quack.gemm_interface import gemm_act, gemm_gated_tuned

from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand

PEAK_TBPS = 3.35


def _bench(fn, *, warmup=20, rep=60):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"gated-GEMM tile sweep on {torch.cuda.get_device_name(0)}", flush=True)
    dtype, D = torch.bfloat16, 128
    scale = D**-0.5

    def w():
        return (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()

    b_lr = prepack_lr_operand(w(), w(), w(), w())  # (D, 4D)

    sm90 = [c for c in get_all_configs("gated")
            if (c.device_capacity[0] if isinstance(c.device_capacity, (tuple, list))
                else c.device_capacity) == 9]
    print(f"sm90 gated configs to try: {len(sm90)}", flush=True)

    for L in (512, 1024):
        M = L * L
        x = torch.randn(M, D, device="cuda", dtype=dtype)
        post = torch.empty(M, 2 * D, device="cuda", dtype=dtype)  # blld contiguous
        bytes_ = M * D * 2 + M * (2 * D) * 2
        mem_floor = bytes_ / (PEAK_TBPS * 1e12) * 1e3

        # autotuned reference (high-level)
        t_auto = _bench(lambda: gemm_act(A=x, B=b_lr, activation="glu",
                                         store_preact=False, postact_out=post))

        results = []
        for cfg in sm90:
            try:
                t = _bench(lambda: gemm_gated_tuned.fn(
                    x, b_lr, None, post, activation="glu", config=cfg))
                results.append((t, cfg))
            except Exception:  # noqa: BLE001
                continue
        results.sort(key=lambda r: r[0])

        print(f"\n=== L={L} (M={M}) | autotuned={t_auto:.3f}ms "
              f"({bytes_/(t_auto*1e-3)/1e9:.0f} GB/s) | mem_floor={mem_floor:.3f}ms ===",
              flush=True)
        print(f"  {'rank':>4} | {'ms':>7} | {'GB/s':>6} | {'%floor':>6} | config")
        for i, (t, cfg) in enumerate(results[:6]):
            gbs = bytes_ / (t * 1e-3) / 1e9
            print(f"  {i:>4} | {t:>7.3f} | {gbs:>6.0f} | {mem_floor/t*100:>5.0f}% | "
                  f"tile_m={cfg.tile_m} tile_n={cfg.tile_n} pp={cfg.pingpong} "
                  f"cl=({cfg.cluster_m},{cfg.cluster_n}) swap_ab={cfg.swap_ab}", flush=True)
        if results:
            print(f"  ... worst of {len(results)} valid: {results[-1][0]:.3f}ms", flush=True)


if __name__ == "__main__":
    main()
