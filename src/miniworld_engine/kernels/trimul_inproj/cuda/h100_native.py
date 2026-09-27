from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "wide"


@T.device_cache
def headers():
    src = T._upstream() / "csrc"
    dst = T.cache_dir() / ("native_headers_" + T._source_digest().hex()[:16])
    dst.mkdir(exist_ok=True)
    # Preserve upstream; relax only the minimum CTA occupancy for wide K1.
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        p = dst / name
        p.parent.mkdir(exist_ok=True)
        s = (src / name).read_text()
        if name == "tmn_kernels.cuh":
            needle = "static constexpr int MINB = NCWG == 1 ? 2 : 1;"
            assert needle in s
            s = s.replace(needle, "static constexpr int MINB = MW_MINB;")
        T.publish_header(p, s)
    return dst


def k1_smem(D, cfg):
    bi, bj, slots, sk, minb = cfg
    ng = bi * bj // 64
    spb = D // 64 // sk
    size = (
        bi * bj * D * 2
        + slots * sk * 8192
        + ng * 8192
        + 8 * D
        + ((2 + 2 * slots) * 8 + 127) // 128 * 128
    )
    if D // 64 % sk or slots < 2 * spb or size * minb > 232448:
        raise ValueError("K1 resources")
    return size


def configs(D):
    rows = []
    for bi, bj in ((1, 64), (2, 64), (1, 128)):
        for sk in (1, 2, 4, 6, 8):
            if D // 64 % sk:
                continue
            for slots in (2, 4, 6, 8):
                for minb in (2, 1) if bi * bj == 64 else (1,):
                    c = (bi, bj, slots, sk, minb)
                    try:
                        k1_smem(D, c)
                    except ValueError:
                        continue
                    rows.append(c)
    return rows


def k3_smem(D, cfg):
    bi, bj, slots, acc = cfg
    h = 2 * D
    bm = bi * bj
    if (bm == 64 and (acc != 1 or D // 64 % 2)) or slots < 4:
        raise ValueError("K3 scheduling")
    size = (
        bm * h * 2
        + bm * D * 2
        + slots * h * 64
        + 16384
        + 8 * (D + h)
        + ((2 * (D // 64) + 2 + 2 * slots) * 8 + 127) // 128 * 128
    )
    if size > 232448:
        raise ValueError("K3 resources")
    return size


@T.device_cache
def kernel(kind, D, cfg, emit_xn=True, normalize=True, hidden=None):
    hidden = 2 * D if hidden is None else hidden
    inc = headers()
    defines = ["-DMW_MINB=" + str(cfg[4] if kind == "k1" else 1)]
    if kind == "k1":
        bi, bj, slot, sk, mb = cfg
        body = f'using Cfg=tmn::K1Cfg<{D},{hidden},false,{bi},{bj},{slot},{sk}>;\nextern "C" __global__ __launch_bounds__(Cfg::NTHR,Cfg::MINB) void width_k1(__grid_constant__ const tmn::K1Params p){{tmn::sm90::k1_body<Cfg,true,{1 if normalize else 0},false,{str(emit_xn).lower()}>(p);}}\n'
        smem = k1_smem(D, cfg)
    else:
        bi, bj, slot, acc = cfg
        body = f'using Cfg=tmn::K3Cfg<{D},{2 * D},0,{bi},{bj},{slot},{acc}>;\nextern "C" __global__ __launch_bounds__(384,1) void width_k3(__grid_constant__ const tmn::K3Params p){{tmn::sm90::k3_body<Cfg,1>(p);}}\n'
        smem = k3_smem(D, cfg)
    source = (
        '// SPDX-License-Identifier: Apache-2.0\n// Anthropic v5 kernel body, width-specific experiment.\n#include "tmn_kernels.cuh"\n'
        + body
    )
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        "-I" + str(inc),
        *defines,
    ]
    out = T.compile_text(source, flags)
    launcher = T._launch_module()
    drv = launcher.BlockDriver(device=torch.cuda.current_device())
    mod = drv.load(out.read_bytes())
    u = launcher.Unit(
        kind, "sm_90a", torch.cuda.current_device(), drv.drv, mod, {}, str(out)
    )
    k = u.kernel("width_" + kind)
    k.set_max_dynamic_smem(smem)
    return k, smem, str(out)


class Front:
    def __init__(self, x, w, mask, g, b, cfg, emit_xn=True, normalize=True, saved=None, hidden=None):
        n = x.shape[1]
        D = x.shape[-1]
        hidden = 2 * D if hidden is None else hidden
        self.ab = (
            saved[0]
            if saved
            else torch.empty((2 * hidden, n, n), device=x.device, dtype=x.dtype)
        )
        self.xn = (
            saved[2] if saved else torch.empty_like(x) if emit_xn else x.new_empty((0,))
        )
        bi, bj, slots, sk, mb = cfg
        self.k, self.smem, self.path = kernel("k1", D, tuple(cfg), emit_xn, normalize, hidden)
        self.cfg = cfg
        U = T._launch_module()
        tm = lambda t, box, dims, strides: U.tensor_map(
            t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="128B"
        )
        mz = tm(x, [64, bj, bi], [D, n, n], [D * 2, n * D * 2])
        mw = tm(w, [64, 64], [D, 4 * hidden], [D * 2])
        ma = tm(self.ab, [64, 1, 32], [n, n, 2 * hidden], [n * 2, n * n * 2])
        tj = (n + bj - 1) // bj
        tiles = ((n + bi - 1) // bi) * tj
        self.params = U.Struct.fixed("h100_native:1",
            [
                mz,
                mw,
                ma,
                mask,
                g,
                b,
                self.ab,
                None,
                self.xn if emit_xn else None,
                n,
                n,
                tj,
                tiles,
                1,
                n,
                1,
                1e-5,
                n * D,
                D,
                0,
                0,
            ]
        )
        self.grid = min(
            tiles,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            * mb,
        )
        self.threads = 128 * (bi * bj // 64 + 1)

    def __call__(self):
        self.k.launch((self.grid, 1, 1), (self.threads, 1, 1), [self.params], self.smem)
        return self.ab, self.xn


class Output:
    def __init__(self, tri, x, wp, wg, gi, bi, go, bo, cfg):
        n = x.shape[1]
        D = x.shape[-1]
        self.y = torch.empty_like(x)
        bi0, bj, slots, acc = cfg
        self.k, self.smem, self.path = kernel("k3", D, tuple(cfg))
        U = T._launch_module()
        tm = lambda t, box, dims, strides: U.tensor_map(
            t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="128B"
        )
        maps = [
            tm(x, [64, bj, bi0], [D, n, n], [D * 2, n * D * 2]),
            tm(tri, [64, 1, 64], [n, n, 2 * D], [n * 2, n * n * 2]),
            tm(wg, [64, 32], [D, D], [D * 2]),
            tm(wp, [64, 32], [2 * D, D], [D * 4]),
            tm(self.y, [64, 16, 1], [D, n, n], [D * 2, n * D * 2]),
        ]
        tj = (n + bj - 1) // bj
        tiles = ((n + bi0 - 1) // bi0) * tj
        self.params = U.Struct.fixed("h100_native:2",
            [*maps, gi, bi, go, bo, x, self.y, None, n, n, tj, tiles, 1, 0, 1e-5, 0]
        )
        self.grid = min(
            tiles,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
        )

    def __call__(self):
        self.k.launch((self.grid, 1, 1), (384, 1, 1), [self.params], self.smem)
        return self.y
