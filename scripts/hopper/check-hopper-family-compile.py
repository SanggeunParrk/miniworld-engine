"""Lower maintained Hopper CuTe families on a CPU compute node, with no launches."""
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import cutlass
import cutlass.cute as cute
from quack.compile_utils import make_fake_tensor

from miniworld_engine.autotune.cute_config import (
    gated_sm90_candidates,
    lnbwd_candidates,
    plain_sm90_candidates,
)
from miniworld_engine.kernels.layernorm_linear.cute.dgrad_lnbwd import _compile as dgrad
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
    _compile_gemm_lnl,
)
from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import TM2DualKernel
from miniworld_engine.kernels.transition.cute.backward_gatebwd import (
    _compile_gemm_dln_gatebwd,
)
from miniworld_engine.kernels.transition.cute.dab_lnbwd import _compile as dab
from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import (
    _compile_gemm_ln_swiglu,
)

bf, fp, device = cutlass.BFloat16, cutlass.Float32, (9, 0)


def check(name, fn, *args):
    print(f"compile {name}", flush=True)
    getattr(fn, "__wrapped__", fn)(*args)
    print(f"PASS {name}", flush=True)


for major in ("k", "m"):
    c = plain_sm90_candidates()[0]
    check(f"M1 {major}", _compile_gemm_lnl, bf, bf, bf, major, "k", "n", fp,
          (c.tile_m, c.tile_n), (c.cluster_m, c.cluster_n, 1), c.pingpong, True,
          c.is_dynamic_persistent, device)

for pingpong in (False, True):
    c = next(c for c in gated_sm90_candidates() if c.pingpong == pingpong)
    tile, cluster = (c.tile_m, c.tile_n), (c.cluster_m, c.cluster_n, 1)
    check(f"SwiGLU pingpong={pingpong}", _compile_gemm_ln_swiglu,
          bf, bf, bf, "k", "k", "n", fp, tile, cluster, pingpong, False, device)
    check(f"gate backward pingpong={pingpong}", _compile_gemm_dln_gatebwd,
          bf, bf, bf, bf, bf, "k", "k", "n", "n", "n", fp,
          tile, cluster, pingpong, False, device)

for width in (128, 256):
    c = lnbwd_candidates(width)[0]
    args = (bf, bf, bf, bf, "k", "k", "n", "n", fp,
            (c.tile_m, c.tile_n), (c.cluster_m, c.cluster_n, 1), True, True, False, device)
    check(f"dgrad LN width={width}", dgrad, *args)
    check(f"dAB LN width={width}", dab, 1, *args)

for rows, width, out_width, tile_m in ((128, 128, 128, 64), (256, 128, 128, 128),
                                      (64, 64, 32, 64), (64, 64, 48, 64), (64, 64, 16, 64)):
    x = make_fake_tensor(bf, (rows, width), leading_dim=1, divisibility=8)
    w = make_fake_tensor(bf, (out_width, width), leading_dim=1, divisibility=8)
    y = make_fake_tensor(bf, (rows, out_width), leading_dim=1, divisibility=8)
    check(f"TM2 M={rows} K={width} N={out_width} tile_m={tile_m}", cute.compile,
          TM2DualKernel(out_width, width, tile_m), x, x, w, w, y)
