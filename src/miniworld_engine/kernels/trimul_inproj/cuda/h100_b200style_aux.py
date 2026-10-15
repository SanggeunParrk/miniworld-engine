"""Driver bindings for B200's architecture-independent wide backward math.

The original CUDA device bodies are compiled without the ATen runtime launch
wrappers. This uses the same verified primary context as the Hopper WGMMA
kernels, including under Compute Sanitizer and CUDA Graph capture.
"""

import re
from pathlib import Path

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


def _device_source(name, namespace):
    source = (Path(__file__).with_name("b200_sources") / name).read_text()
    end = source.index("}  // namespace " + namespace) + len(
        "}  // namespace " + namespace
    )
    source = source[:end]
    source = re.sub(
        r"^#include <(?:torch|ATen|c10)/.*\n", "", source, flags=re.MULTILINE
    )
    source = source.replace(
        "__global__ void __launch_bounds__(256)", "__device__ __forceinline__ void"
    )
    return source


_STRUCTS = r"""
using bf = __nv_bfloat16;
struct Fold { const bf *wp; const float *go,*bo; const bf *wg;
  const float *gi,*bi; bf *wpf,*wgf; float *vec; int D,H; };
struct Gate { const bf *dy,*pv,*gv,*ds; const float *vec,*mu,*rs;
  bf *dpr,*dg; float *s1,*s2,*part; int ldg,M,L; };
struct Out { const bf *dor,*tri; const float *mu,*rs,*s1,*s2,*go;
  bf *dt; float *part; int M,H; };
struct In { const bf *dxn,*x,*dy; const float *mu,*rs,*gi;
  bf *dx; float *part; int M; };
struct Apply { const bf *x; const float *mu,*rs,*gi,*bi; bf *xn; int M; };
struct Sum { const float *part; float *out; int R,C; };
extern "C" __global__ void mw_b200style_fold(const Fold p) {
  wide::fold_prep_kernel(p.wp,p.go,p.bo,p.wg,p.gi,p.bi,p.wpf,p.wgf,p.vec,p.D,p.H);
}
extern "C" __global__ void mw_b200style_lnout(const Out p) {
  wbwd::lnout_bwd_kernel(p.dor,p.tri,p.mu,p.rs,p.s1,p.s2,p.go,p.dt,p.part,p.M,p.H);
}
extern "C" __global__ void mw_b200style_colsum(const Sum p) {
  wbwd::colsum_kernel(p.part,p.out,p.R,p.C);
}
"""


@T.device_cache
def _unit():
    body = (
        "#include <stdint.h>\n"
        + _device_source("wide_aux.cu", "wide")
        + "\n"
        + _device_source("wide_bwd.cu", "wbwd")
        + _STRUCTS
    )
    for d in (256, 384, 512):
        body += f"""
extern "C" __global__ void mw_b200style_gate_{d}(const Gate p) {{
  wbwd::gate_bwd_kernel<{d}>(p.dy,p.pv,p.gv,p.ds,p.vec,p.mu,p.rs,p.dpr,p.dg,p.ldg,p.s1,p.s2,p.part,p.M,p.L);
}}
extern "C" __global__ void mw_b200style_lnin_{d}(const In p) {{
  wbwd::lnin_bwd_kernel<{d}>(p.dxn,p.x,p.dy,p.mu,p.rs,p.gi,p.dx,p.part,p.M);
}}
extern "C" __global__ void mw_b200style_apply_{d}(const Apply p) {{
  wbwd::ln_apply_kernel<{d}>(p.x,p.mu,p.rs,p.gi,p.bi,p.xn,p.M);
}}
"""
    body = body.replace("__global__ void", "__global__ void __launch_bounds__(256)")
    flags = ["-std=c++17", "-O3", "-arch=sm_90a", "--cubin", "-lineinfo"]
    return T.load_unit(str(T.compile_text(body, flags)), "b200style_aux_driver")


# gate_bwd grid: one pass of 8 resident CTAs per SM is best from L384 up; below that the fixed 512-atomics-per-CTA epilogue
# dominates, so two CTAs per SM (measured on H100, D256-512, L128-768: L128 93 -> 34 us at D256).
GATE_BLOCKS = 132 * 8
GATE_BLOCKS_SMALL = 132 * 2


def _launch(name, grid, fields):
    _unit().kernel(name).launch(grid, (256, 1, 1), [T._launch_module().Struct(fields)])


class DriverAux:
    def fold(self, wp, go, bo, wg, gi, bi, wpf, wgf, vec):
        d, h = wp.shape
        _launch(
            "mw_b200style_fold",
            (2 * d, 1, 1),
            [wp, go, bo, wg, gi, bi, wpf, wgf, vec, d, h],
        )

    def gate_bwd(self, dy, pv, gv, ds, vec, mu, rs, dpr, dg, s1, s2, part, length):
        m, d = dy.shape
        s1.zero_()
        s2.zero_()
        part.zero_()
        _launch(
            f"mw_b200style_gate_{d}",
            (GATE_BLOCKS_SMALL if m < 384 * 384 else GATE_BLOCKS, 1, 1),
            [
                dy,
                pv,
                gv,
                ds,
                vec,
                mu,
                rs,
                dpr,
                dg,
                s1,
                s2,
                part,
                dg.stride(0),
                m,
                length,
            ],
        )

    def lnout_bwd(self, dor, tri, mu, rs, s1, s2, go, dt, dgb):
        h, m = tri.shape[0], tri.numel() // tri.shape[0]
        gx = m // 256
        part = torch.empty((gx * 2 * h,), device=tri.device, dtype=torch.float32)
        _launch(
            "mw_b200style_lnout",
            (gx, h // 64, 1),
            [dor, tri, mu, rs, s1, s2, go, dt, part, m, h],
        )
        _launch(
            "mw_b200style_colsum", ((2 * h + 31) // 32, 1, 1), [part, dgb, gx, 2 * h]
        )

    def lnin_bwd(self, dxn, x, dy, mu, rs, gi, dx, dgb):
        m, d = x.shape
        nb = min((m + 63) // 64, 148 * 4)
        part = torch.empty((nb * 2 * d,), device=x.device, dtype=torch.float32)
        _launch(
            f"mw_b200style_lnin_{d}", (nb, 1, 1), [dxn, x, dy, mu, rs, gi, dx, part, m]
        )
        _launch(
            "mw_b200style_colsum", ((2 * d + 31) // 32, 1, 1), [part, dgb, nb, 2 * d]
        )

    def ln_apply(self, x, mu, rs, gi, bi, xn):
        m, d = x.shape
        _launch(f"mw_b200style_apply_{d}", (148 * 8, 1, 1), [x, mu, rs, gi, bi, xn, m])
