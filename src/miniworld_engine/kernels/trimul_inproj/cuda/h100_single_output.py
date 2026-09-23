"""Single-direction output forward and on-chip B1 derivatives."""

import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

SMEM = 181248


@T.device_cache
def build(debug=False):
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        "-DSINGLE_DEBUG=" + str(int(debug)),
        "-I" + str(T._upstream() / "csrc"),
        "-I" + str(T.SOURCES / "common"),
        "-I" + str(T.SOURCES / "b1"),
    ]
    path = T.compile(T.SOURCES / "single_output/kernel.cu", flags)
    u = T.load_unit(str(path), "single_output")
    ks = {name: u.kernel("single_" + name) for name in ("output", "b1", "reduce")}
    for name in ("output", "b1"):
        ks[name].set_max_dynamic_smem(SMEM)
    return ks, path


class Plan:
    def __init__(self, x, xn, tri, wp, wg, gamma, beta, ds, dy, count=132, debug=False):
        self.ks, self.path = build(debug)
        self.count = count
        n = x.shape[1]
        m = n * n
        self.y = torch.empty_like(x)
        self.dg = x.new_empty((m, 128))
        self.dt = torch.empty_like(tri)
        self.dwg = torch.empty_like(wg)
        self.dwp = torch.empty_like(wp)
        self.dgamma = torch.empty_like(gamma)
        self.dbeta = torch.empty_like(beta)
        self.partial = torch.empty((count, 33024), device=x.device, dtype=torch.float32)
        U = T._launch_module()

        def tm(t, box, dims, stride):
            return U.tensor_map(
                t, box, dims=dims, strides_bytes=[stride], swizzle="128B", l2="128B"
            )

        row = lambda t: tm(t, [64, 64], [128, m], 256)
        maps = [
            row(xn),
            tm(tri, [64, 128], [m, 128], m * 2),
            row(dy),
            tm(wp, [64, 64], [128, 128], 256),
            tm(wg, [64, 64], [128, 128], 256),
            row(self.dg),
            tm(self.dt, [64, 16], [m, 128], m * 2),
        ]
        self.params = U.Struct(
            [
                *maps,
                x,
                ds,
                gamma,
                beta,
                self.y,
                self.dwg,
                self.dwp,
                self.dgamma,
                self.dbeta,
                self.partial,
                m,
                n,
            ]
        )
        self.inputs = (x, xn, tri, wp, wg, gamma, beta, ds, dy)

    def forward(self):
        self.ks["output"].launch((self.count, 1, 1), (256, 1, 1), [self.params], SMEM)
        return self.y

    def backward(self):
        self.ks["b1"].launch((self.count, 1, 1), (256, 1, 1), [self.params], SMEM)
        self.ks["reduce"].launch((129, 1, 1), (256, 1, 1), [self.params, self.count], 0)
        return self.dg, self.dwg, self.dt, self.dgamma, self.dbeta, self.dwp


@T.device_cache
def k3_build(cfg=(2, 64, 4, 1)):
    bi, bj, slots, acc = cfg
    defs = dict(
        MW_HIDDEN=128,
        MW_SAVE_PG=0,
        MW_PG_METHOD=0,
        XHAT_FP32=-1,
        MW_SAVE_IN=0,
        MW_SAVE_OUT=0,
        MW_STORE_METHOD=0,
        MW_SAVE_STATS_IN=0,
        MW_SAVE_STATS_OUT=0,
        MW_BI=bi,
        MW_BJ=bj,
        MW_SLOT=slots,
        MW_ACC=acc,
        TMN_K3_REGS_24_240=1,
        MW_SERIAL=1,
        MW_FUSED=1,
    )
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        "-I" + str(T._upstream() / "csrc"),
    ] + ["-D%s=%s" % v for v in defs.items()]
    path = T.compile(T.SOURCES / "output/save_k3.cu", flags)
    k = T.load_unit(str(path), "single_k3").kernel("save_k3")
    smem = (
        bi * bj * 128 * 4
        + slots * 128 * 64
        + 16384
        + 8 * 256
        + ((6 + 2 * slots) * 8 + 127) // 128 * 128
    )
    k.set_max_dynamic_smem(smem)
    return k, smem


class Output:
    def __init__(self, x, tri, wp, wg, gi, bi, go, bo, ds, cfg=(2, 64, 4, 1)):
        U = T._launch_module()
        n = x.shape[1]
        m = n * n
        self.k, self.smem = k3_build(cfg)
        self.y = torch.empty_like(x)
        tm = lambda t, box, dims, strides: U.tensor_map(
            t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="128B"
        )
        bi0, bj, _, _ = cfg
        maps = [
            tm(x, [64, bj, bi0], [128, n, n], [256, n * 256]),
            tm(tri, [64, 1, 64], [n, n, 128], [n * 2, m * 2]),
            tm(wg, [64, 32], [128, 128], [256]),
            tm(wp, [64, 32], [128, 128], [256]),
            tm(self.y, [64, 16, 1], [128, n, n], [256, n * 256]),
        ]
        tj = (n + bj - 1) // bj
        tiles = ((n + bi0 - 1) // bi0) * tj
        base = U.Struct(
            [*maps, gi, bi, go, bo, x, self.y, None, n, n, tj, tiles, 1, 0, 1e-5, 0]
        )
        self.params = U.Struct(
            [
                base,
                ds,
                maps[-1],
                maps[-1],
                None,
                None,
                None,
                None,
                None,
                None,
                maps[-1],
                maps[-1],
                None,
                None,
            ]
        )
        self.grid = min(
            tiles, torch.cuda.get_device_properties(x.device).multi_processor_count
        )
        self.inputs = (x, tri, wp, wg, gi, bi, go, bo, ds)

    def __call__(self):
        self.k.launch((self.grid, 1, 1), (384, 1, 1), [self.params], self.smem)
        return self.y
