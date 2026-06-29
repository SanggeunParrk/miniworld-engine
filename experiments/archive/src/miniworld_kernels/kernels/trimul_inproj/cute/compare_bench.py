"""End-to-end trimul forward: OURS (best) vs cuequivariance vs nvidia dtv1
vs pytorch — ALL measured under the SAME harness, TWO regimes (eager + manual
CUDA-graph = the real launch-overhead-free training regime). B=1, D=128, bf16,
NO mask. Everything autotuned: front (quack tuned=True), LN_in (triton autotune),
v4 back (triton autotune), v5 back (per-L tile sweep), dtv1 (triton autotune).

This is the CANONICAL reconciled harness — supersedes the older mixed-methodology
numbers (compare_bench reduce-overhead-K8, teamgm_faithful with-mask,
whole_fwd_bench). LN_in is triton here (faster than cuequiv per ln_bench).

OURS = triton LN_in -> trimul_inproj(left+right, bdll) -> bmm -> fused back:
  ours_v4 : triton fused back (gate in-kernel)
  ours_v5 : cute layernorm_linear + folded gate-mul (gate precomputed), tuned
COMPUTE NODE only. Run with QUACK_CACHE_DIR fresh.
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
import torch.nn as nn
import triton

from miniworld_kernels.kernels.layernorm.triton.main import (
    triton_layernorm, triton_layernorm_masked,
)
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication, _load_cute_fns,
)

# v5 fused-back tile-config candidates (d=128, N=128). _FUSED_CONFIG is the first.
_V5_CONFIGS = [
    dict(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=64,  tile_n=128, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=192, tile_n=128, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=256, tile_n=128, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=128, tile_n=128, cluster_m=1, cluster_n=2, pingpong=True),
    dict(tile_m=64,  tile_n=128, cluster_m=1, cluster_n=2, pingpong=True),
    dict(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, pingpong=False),
    dict(tile_m=64,  tile_n=128, cluster_m=1, cluster_n=1, pingpong=False),
]


def _bench(fn, *, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"   bench fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def main():
    assert torch.cuda.is_available()
    print(f"trimul compare on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    _t1, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

    mods = {}
    mods["pytorch"] = TriangleMultiplication(
        d_pair=D, implementation=ImplementationType.PYTORCH).cuda().to(dtype)
    mods["cuequivariance"] = TriangleMultiplication(
        d_pair=D, implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dtype)

    # ours: build from a cute module's weights
    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)
    WL, WLg = mod.to_left.weight.T, mod.to_left_gate.weight.T
    WR, WRg, Wg = mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T
    gln_w, gln_b, eps = mod.ln_out.weight, mod.ln_out.bias, mod.ln_out.eps
    Wp_t, Wg_t = mod.to_out.weight.T, mod.to_gate.weight.T
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)

    # SAME weights everywhere so cos is meaningful (timing is data-independent anyway):
    # load the cute module's seeded weights into the pytorch / cuequiv reference modules.
    mods["pytorch"].load_state_dict(mod.state_dict())
    mods["cuequivariance"].load_state_dict(mod.state_dict())

    # dtv1 weights (cuequiv/dtv1 concatenated API), from the SAME mod weights.
    dtv1_kw = dict(
        norm_in_weight=mod.ln_pair.weight, norm_in_bias=mod.ln_pair.bias,
        p_in_weight=torch.cat([mod.to_left.weight, mod.to_right.weight], dim=0),
        g_in_weight=torch.cat([mod.to_left_gate.weight, mod.to_right_gate.weight], dim=0),
        norm_out_weight=mod.ln_out.weight, norm_out_bias=mod.ln_out.bias,
        p_out_weight=mod.to_out.weight, g_out_weight=mod.to_gate.weight,
    )

    def ours_nvidia_dtv1(pair, mask2d):
        return fused_triangle_multiplicative_update_dtv1(
            pair, "outgoing", mask2d, eps=eps, **dtv1_kw)

    def _ln_in(pair, rmask=None):
        # triton LN_in (faster than cuequiv per ln_bench), no transpose. When rmask
        # ([M] pair-mask) is given it is FOLDED into the LN epilogue (free) — proj(0)=0
        # so masked xn -> left/right exactly zero at masked positions (== AF's
        # mask*projection at every valid position; only the discarded output-gate at
        # padding differs). Avoids the 2 extra EW passes a post-front mask would cost.
        b, l1, l2, d = pair.shape
        xf = pair.reshape(b * l1 * l2, d)
        if rmask is None:
            xn = triton_layernorm(xf, mod.ln_pair.weight, mod.ln_pair.bias, mod.ln_pair.eps)
        else:
            xn = triton_layernorm_masked(xf, mod.ln_pair.weight, mod.ln_pair.bias,
                                         mod.ln_pair.eps, rmask)
        return xn.view(b, l1, l2, d)

    def ours_v4(pair, rmask):
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair, rmask)  # mask folded into LN (free) -> left/right zeroed
        left, right, _ = trimul_inproj_cute_forward(
            xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False, b_lr=b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return trimul_back_triton(tri, xn, Wp_t, Wg_t, gln_w, gln_b, eps)

    def make_ours_v5(cfg):
        def ours_v5(pair, rmask):
            b, l1, l2, d = pair.shape
            xn = _ln_in(pair, rmask)
            left, right, gate = trimul_inproj_cute_forward(
                xn, WL, WLg, WR, WRg, Wg, bdll_direct=True, compute_gate=True, b_lr=b_lr)
            tri = torch.einsum("bdik,bdjk->bdij", left, right)
            view = tri.reshape(b, d, l1 * l2)[0].t()
            y = layernorm_linear_cute_fused(view, gln_w, gln_b, mod.to_out.weight, None,
                                            eps=eps, gate=gate.reshape(b * l1 * l2, d),
                                            config=cfg)
            return y.view(b, l1, l2, d)
        return ours_v5

    def bench_cudagraph(fn, pair):
        """Manual CUDA-graph capture of ONE layer (launch-overhead-free = the real
        training regime; cleaner than reduce-overhead K-stack, no graph-break noise)."""
        try:
            with torch.no_grad():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(5):
                        fn(pair)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn(pair)
            return _bench(g.replay)
        except Exception as e:  # noqa: BLE001
            print(f"   cudagraph fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
            return float("nan")

    def sweep_v5(pair, rmask):
        """Pick the fastest v5 tile config for this L (eager single-launch ranking)."""
        best, best_cfg = float("inf"), None
        for cfg in _V5_CONFIGS:
            fn = make_ours_v5(cfg)
            try:
                with torch.no_grad():
                    for _ in range(3):
                        fn(pair, rmask)
                t = triton.testing.do_bench(lambda: fn(pair, rmask), warmup=10, rep=30,
                                            return_mode="median")
            except Exception as e:  # noqa: BLE001
                print(f"   v5 cfg {cfg['tile_m']}x{cfg['tile_n']}cn{cfg['cluster_n']}"
                      f"pp{int(cfg['pingpong'])} FAIL {type(e).__name__}", flush=True)
                continue
            if t < best:
                best, best_cfg = t, cfg
        print(f"   v5 best cfg = tile_m={best_cfg['tile_m']} tile_n={best_cfg['tile_n']} "
              f"cn={best_cfg['cluster_n']} pp={best_cfg['pingpong']} ({best:.3f} ms eager)",
              flush=True)
        return best_cfg

    def bench_eager(fn, pair):
        """Single-layer eager do_bench (no compile, no K-stack) — fair to dtv1,
        whose autograd Functions are @torch.compiler.disable()'d."""
        try:
            for _ in range(6):
                fn(pair)
            return _bench(lambda: fn(pair))
        except Exception as e:  # noqa: BLE001
            print(f"   eager fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
            return float("nan")

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return (a @ b / (a.norm() * b.norm() + 1e-20)).item()

    Ls = [256, 384, 512, 768, 1024]
    cols = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v4", "ours_v5"]
    rows_c, rows_e = {}, {}
    for L in Ls:
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        # mask EXACTLY like team-gm (torch.rand(1,L)>0.2). Canonical placement (pytorch
        # ref): mask the gated left/right (post-projection, pre-einsum), broadcast over
        # D; output gate stays on UNMASKED xn. v4/v5 get m2c=[1,1,L,L] for bdll.
        mask = torch.rand(1, L, device="cuda") > 0.2          # [1,L] bool
        mask2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)      # [1,L,L] bool (for dtv1)
        rmask = mask2d.reshape(-1).to(dtype)                  # [M] folded into ours' LN_in
        v5_cfg = sweep_v5(pair, rmask)
        v5fn = make_ours_v5(v5_cfg)
        variants = {
            "pytorch": lambda p: mods["pytorch"](p, mask),
            "nvidia(dtv1)": lambda p: ours_nvidia_dtv1(p, mask2d),
            "cuequivariance": lambda p: mods["cuequivariance"](p, mask),
            "ours_v4": lambda p: ours_v4(p, rmask),
            "ours_v5": lambda p: v5fn(p, rmask),
        }
        if L == 256:  # correctness vs pytorch ref, WITH mask
            ref = mods["pytorch"](pair, mask)
            print(f"   cos vs pytorch(mask): v4={cos(variants['ours_v4'](pair), ref):.5f} "
                  f"v5={cos(variants['ours_v5'](pair), ref):.5f} "
                  f"dtv1={cos(variants['nvidia(dtv1)'](pair), ref):.5f}", flush=True)
        tc, te = {}, {}
        for name in cols:
            with torch.no_grad():
                tc[name] = bench_cudagraph(variants[name], pair)
                te[name] = bench_eager(variants[name], pair)
        rows_c[L], rows_e[L] = tc, te
        print(f"[L={L}] cudagraph " + " ".join(f"{c}={tc[c]:.3f}" for c in cols), flush=True)
        print(f"[L={L}] eager     " + " ".join(f"{c}={te[c]:.3f}" for c in cols), flush=True)

    for tag, rows in (("CUDAGRAPH", rows_c), ("EAGER", rows_e)):
        print(f"\n=== {tag}, ms/layer ===")
        print(f"{'L':>5} | " + " | ".join(f"{c:>14}" for c in cols))
        print("-" * 92)
        for L in Ls:
            print(f"{L:>5} | " + " | ".join(f"{rows[L][c]:>14.3f}" for c in cols))
    print("\nDATA_CUDAGRAPH " + ";".join(
        f"{L}:" + ",".join(f"{rows_c[L][c]:.4f}" for c in cols) for L in Ls), flush=True)
    print("DATA_EAGER " + ";".join(
        f"{L}:" + ",".join(f"{rows_e[L][c]:.4f}" for c in cols) for L in Ls), flush=True)


if __name__ == "__main__":
    main()
