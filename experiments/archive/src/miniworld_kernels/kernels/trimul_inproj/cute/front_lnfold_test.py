"""Forward front: fold LN_in INTO the gated GEMM (stats kernel + gemm_ln_swiglu)
vs the current path (cuequiv LN_in + trimul_inproj). Correctness + speed.
The fold removes the separate LN kernel AND the x_n (256MB) materialization;
only stats (rstd,c1) are produced. COMPUTE NODE.
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
import torch.nn.functional as F
import triton
from quack.activation import gate_fn_map
from quack.gemm_config import GemmConfig


def _cfg(tm, tn, pp, cn):
    return GemmConfig(tile_m=tm, tile_n=tn, pingpong=pp, is_dynamic_persistent=False,
                      cluster_m=1, cluster_n=cn, swap_ab=False, max_swizzle_size=8, device_capacity=9)


_CFGS = [
    _cfg(256, 128, False, 2), _cfg(192, 128, True, 2), _cfg(128, 128, True, 1),
    _cfg(128, 128, False, 2), _cfg(128, 256, False, 1), _cfg(256, 256, False, 1),
    _cfg(192, 256, False, 1), _cfg(128, 512, False, 1), _cfg(64, 128, True, 2),
    _cfg(256, 128, True, 2), _cfg(192, 128, False, 2),
]

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear import fold_for_gemm
from miniworld_kernels.kernels.layernorm_linear.triton.stats import stats_triton
from miniworld_kernels.kernels.transition.cute.gemm_transition_swiglu import gemm_ln_swiglu
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns

D = 128
EPS = 1e-5
_GLU = gate_fn_map["glu"]


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def build_folded(WLg, WL, WRg, WR, g, b):
    """nn.Linear weights (D,D). Returns B_folded (4D,D), S (4D,), B2 (4D,) interleaved."""
    def il_rows(A, B):  # (D,D),(D,D) -> (2D,D) rows interleaved [A0,B0,A1,B1,...]
        return torch.stack([A, B], dim=1).reshape(2 * D, -1)

    def il_vec(a, b):
        return torch.stack([a, b], dim=1).reshape(2 * D)

    BwLg, SLg, B2Lg = fold_for_gemm(WLg, g, b, None)
    BwL, SL, B2L = fold_for_gemm(WL, g, b, None)
    BwRg, SRg, B2Rg = fold_for_gemm(WRg, g, b, None)
    BwR, SR, B2R = fold_for_gemm(WR, g, b, None)
    Bf = torch.cat([il_rows(BwLg, BwL), il_rows(BwRg, BwR)], 0).contiguous()      # (4D,D)
    S = torch.cat([il_vec(SLg, SL), il_vec(SRg, SR)]).float().contiguous()        # (4D,)
    B2 = torch.cat([il_vec(B2Lg, B2L), il_vec(B2Rg, B2R)]).float().contiguous()
    return Bf, S, B2


def main():
    assert torch.cuda.is_available()
    print(f"front LN-fold on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    _t1, _t2, _f, layer_norm_transpose = _load_cute_fns()
    torch.manual_seed(0)
    dt = torch.bfloat16

    for L in (256, 512, 1024):
        M = L * L
        x = torch.randn(1, L, L, D, device="cuda", dtype=dt) * 0.6
        g = torch.randn(D, device="cuda", dtype=dt)        # ln gamma
        b = torch.randn(D, device="cuda", dtype=dt) * 0.1  # ln beta
        WL, WLg, WR, WRg = (torch.randn(D, D, device="cuda", dtype=dt) * D**-0.5 for _ in range(4))
        WLt, WLgt, WRt, WRgt = WL.T.contiguous(), WLg.T.contiguous(), WR.T.contiguous(), WRg.T.contiguous()
        b_lr = prepack_lr_operand(WLt, WLgt, WRt, WRgt)
        Bf, S, B2 = build_folded(WLg, WL, WRg, WR, g, b)

        # --- fold path: stats + gemm_ln_swiglu (raw x, no LN kernel) ---
        xf = x.reshape(M, D)
        rstd, c1 = stats_triton(xf, EPS)
        post = torch.empty(M, 2 * D, device="cuda", dtype=dt)
        gemm_ln_swiglu(xf, Bf, post, rstd.view(1, M), c1.view(1, M),
                       S.view(1, 4 * D), B2.view(1, 4 * D), act_fn=_GLU)
        left_f, right_f = post[:, :D], post[:, D:]

        # --- reference (fp32) ---
        xn = F.layer_norm(x.float(), (D,), g.float(), b.float(), EPS).reshape(M, D)
        left_r = torch.sigmoid(xn @ WLg.float().T) * (xn @ WL.float().T)
        right_r = torch.sigmoid(xn @ WRg.float().T) * (xn @ WR.float().T)
        if L == 256:
            print(f"  cos left={cos(left_f, left_r):.5f} right={cos(right_f, right_r):.5f}", flush=True)

        # --- current path: cuequiv LN + trimul_inproj ---
        def cur():
            o = layer_norm_transpose(x.reshape(M, D), g, b, eps=EPS, layout="nd->nd")
            xn2 = (o[0] if isinstance(o, tuple) else o).view(1, L, L, D)
            return trimul_inproj_cute_forward(xn2, WLt, WLgt, WRt, WRgt, None,
                                              bdll_direct=True, compute_gate=False, b_lr=b_lr)

        def fold(cfg):
            r, c = stats_triton(xf, EPS)
            p = torch.empty(M, 2 * D, device="cuda", dtype=dt)
            gemm_ln_swiglu(xf, Bf, p, r.view(1, M), c.view(1, M),
                           S.view(1, 4 * D), B2.view(1, 4 * D), config=cfg, act_fn=_GLU)
            return p

        for _ in range(5):
            cur()
        tc = triton.testing.do_bench(cur, warmup=10, rep=50, return_mode="median")
        best, best_cfg = float("inf"), None
        for cfg in _CFGS:
            try:
                for _ in range(3):
                    fold(cfg)
                t = triton.testing.do_bench(lambda: fold(cfg), warmup=8, rep=30, return_mode="median")
            except Exception as e:  # noqa: BLE001
                continue
            if t < best:
                best, best_cfg = t, cfg
        bc = best_cfg
        print(f"  L={L:>4}: current {tc:.3f} ms | fold-best {best:.3f} ms "
              f"(tile_m={bc.tile_m} tile_n={bc.tile_n} pp={bc.pingpong} cn={bc.cluster_n})  "
              f"{'FOLD WINS' if best < tc else 'current wins'}", flush=True)

        if L == 1024:
            o = layer_norm_transpose(xf, g, b, eps=EPS, layout="nd->nd")
            xn_c = (o[0] if isinstance(o, tuple) else o).view(1, L, L, D)
            t_stats = triton.testing.do_bench(lambda: stats_triton(xf, EPS), warmup=10, rep=50, return_mode="median")
            t_ln = triton.testing.do_bench(lambda: layer_norm_transpose(xf, g, b, eps=EPS, layout="nd->nd"),
                                           warmup=10, rep=50, return_mode="median")
            t_inproj = triton.testing.do_bench(
                lambda: trimul_inproj_cute_forward(xn_c, WLt, WLgt, WRt, WRgt, None,
                                                   bdll_direct=True, compute_gate=False, b_lr=b_lr),
                warmup=10, rep=50, return_mode="median")

            def gln_only():
                p = torch.empty(M, 2 * D, device="cuda", dtype=dt)
                gemm_ln_swiglu(xf, Bf, p, rstd.view(1, M), c1.view(1, M),
                               S.view(1, 4 * D), B2.view(1, 4 * D), config=bc, act_fn=_GLU)
            t_gln = triton.testing.do_bench(gln_only, warmup=10, rep=50, return_mode="median")
            print(f"    breakdown @1024: cuequiv_LN={t_ln:.3f} + inproj_gemm_act={t_inproj:.3f}"
                  f"  vs  stats={t_stats:.3f} + gemm_ln_swiglu={t_gln:.3f}", flush=True)


if __name__ == "__main__":
    main()
