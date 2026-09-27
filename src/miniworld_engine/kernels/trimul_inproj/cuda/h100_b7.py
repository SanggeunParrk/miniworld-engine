from pathlib import Path
import os, ctypes, torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


from types import SimpleNamespace


@T.device_cache
def kernel_plan(n, clusters, mode):
    R = T.SOURCES / ("b7_384" if n == 384 else "b7_768")
    cfg_values = SimpleNamespace()
    cfg_values.hwcluster = int(('2'))
    cfg_values.multicast = int(('0'))
    cfg_values.rings = int(('12'))
    cfg_values.clusters = clusters
    cfg_values.consumers = int(('10'))
    cfg_values.reuse_dp = 1
    cfg_values.count = clusters * (16 + cfg_values.consumers)
    inc = T._upstream() / "csrc"
    flags = [
        "-DB7_HW_CLUSTER=" + str(cfg_values.hwcluster),
        "-DB7_MULTICAST=" + str(cfg_values.multicast),
        "-DB7_RING_DEPTH=" + str(cfg_values.rings),
        "-DB7_TRANSPOSE_PART=" + str(bool(mode & 16) * 1),
        "-DB7_REDUCE_NATIVE=" + str(bool(mode & 32) * 1),
        "-DB7_PRODUCER_REGS=" + ('32'),
        "-DB7_RING_CHUNK="
        + str(8192 if mode & 8 else 16384 if mode & 4 else 32768),
        "-DB7_GATEFIRST=" + str(bool(mode & 2) * 1),
        "-DB7_RING_TENSOR=" + str(mode & 1),
        "-DB7_ROLLING=" + str(bool(mode & 4) * 1),
        "-DB7_LNPREFETCH=" + str(bool(mode & 8) * 1),
        "-DB7_PREFETCH=" + str(mode & 3),
        "-DB7_CONSUMERS=" + str(cfg_values.consumers),
        "-DB7_REUSE_DP=" + str(cfg_values.reuse_dp),
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        "-I" + str(inc),
        "-I" + str(T.SOURCES / "common"),
        "-I" + str(T.SOURCES / "common_b7"),
    ]
    out = T.compile(R / "joint.cu", flags)
    L = T._launch_module()
    unit = T.load_unit(str(out), "joint")
    drv = unit
    cfg_values.k = unit.kernel("b7_joint")
    cfg_values.k.set_max_dynamic_smem(114688)
    assert cfg_values.clusters > 0 and (16 + cfg_values.consumers) % cfg_values.hwcluster == 0
    dd = drv.drv.d
    cfg = dd.CUlaunchConfig()
    cfg.gridDimX = cfg_values.count
    cfg.gridDimY = 1
    cfg.gridDimZ = 1
    cfg.blockDimX = 256
    cfg.blockDimY = 1
    cfg.blockDimZ = 1
    cfg.sharedMemBytes = 114688
    cfg.hStream = dd.CUstream(int(torch.cuda.current_stream().cuda_stream))
    occ = dd.cuOccupancyMaxActiveClusters(dd.CUfunction(int(cfg_values.k.handle)), cfg)
    assert int(occ[0]) == 0 and cfg_values.count <= int(occ[1]) * cfg_values.hwcluster
    return cfg_values


class Plan:
    def __init__(self, d, dy, dl, dr, dg, xn, clusters=10, mode=52):
        R = T.SOURCES / ("b7_384" if d["n"] == 384 else "b7_768")
        self.d = d
        self.xn = xn
        static = kernel_plan(d["n"], clusters, mode)
        self.hwcluster, self.multicast, self.rings = static.hwcluster, static.multicast, static.rings
        self.clusters, self.consumers = static.clusters, static.consumers
        self.reuse_dp, self.count, self.k = static.reuse_dp, static.count, static.k
        x = d["x"]
        m = d["n"] ** 2
        self.dx = torch.empty((m, 128), device=x.device, dtype=x.dtype)
        self.dw = torch.empty((4, 128, 256), device=x.device, dtype=x.dtype)
        self.dgam = torch.empty(128, device=x.device)
        self.dbeta = torch.empty_like(self.dgam)
        self.partw = torch.zeros((self.clusters * 2, 16, 64, 128), device=x.device)
        self.partln = torch.empty(
            (self.clusters * self.consumers, 256), device=x.device
        )
        self.counts = torch.zeros(2, device=x.device, dtype=torch.int32)
        self.mask = d["mask"].bfloat16().reshape(-1)
        self.ring = torch.empty(
            (self.clusters, self.rings, 131072), device=x.device, dtype=torch.uint8
        )
        self.xring = torch.empty(
            (self.clusters, self.rings, 16384), device=x.device, dtype=torch.uint8
        )
        self.flags = torch.zeros(
            (self.clusters, self.rings, 18), device=x.device, dtype=torch.int32
        )
        self.outputs = (self.dx, *self.dw.unbind(), self.dgam, self.dbeta)
        self.bind(dl, dr, dg, dy, xn)

    def bind(self, dl, dr, dg, dy, xn=None):
        if xn is not None:
            self.xn = xn
        d = self.d
        n = d["n"]
        m = n * n
        L = T._launch_module()
        tm = lambda t, box, dims, strides: L.tensor_map(
            t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="256B"
        )
        row = lambda t: tm(t, [64, 64], [128, m], [256])
        wg = d["wt"][4]
        self.params = L.Struct.fixed("h100_b7:1",
            [
                row(self.xn),
                tm(d["w1"], [64, 64], [128, 1024], [256]),
                tm(dl, [64, 32], [m, 256], [m * 2]),
                tm(dr, [64, 32], [m, 256], [m * 2]),
                row(dg),
                tm(wg, [64, 128], [128, 128], [256]),
                row(d["x"]),
                row(dy),
                row(self.dx),
                *[
                    tm(w, [64, 128], [256, 128], [512])
                    for w in (d["wt"][1], d["wt"][0], d["wt"][3], d["wt"][2])
                ],
                L.tensor_map(
                    self.ring.view(torch.bfloat16),
                    [256, 8, 2],
                    dims=[256, 64, 4 * self.clusters * self.rings],
                    strides_bytes=[512, 32768],
                    swizzle="none",
                ),
                self.mask,
                d["gi"],
                d["bi"],
                self.dx,
                self.dw,
                self.dgam,
                self.dbeta,
                self.partw,
                self.partln,
                self.counts,
                self.dx.numel() // 128,
                m // 64,
                self.ring,
                self.xring,
                self.flags,
            ]
        )
        self.inputs = (dl, dr, dg, dy, d["w1"], self.xn, wg)

    def __call__(self):
        L = T._launch_module()
        drv = self.k.unit.drv
        args = L._Packed([self.params])
        drv._unwrap(
            "cuLaunchCooperativeKernel",
            drv.d.cuLaunchCooperativeKernel(
                drv.d.CUfunction(int(self.k.handle)),
                self.count,
                1,
                1,
                256,
                1,
                1,
                114688,
                drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),
                ctypes.addressof(args.array),
            ),
        )
        return self.outputs
