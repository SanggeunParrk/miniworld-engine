from pathlib import Path
import torch, json, ctypes, os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width_base as BASE
from miniworld_engine.kernels.trimul_inproj.cuda.h100_native import (
    Front,
    headers,
    k1_smem,
)
from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
    pack_into,
    normalize_into,
)

R = T.SOURCES / "wide"


def tuning(D):
    g = int((2 if D != 64 else 1))
    spl = int((128 if D == 64 else 32))
    return (
        g,
        spl,
        2 * 8192 * (1 + g * (int(('64')) // 64)) + 128,
    )


def gp_config(D):
    return (
        1,
        64,
        int((8 if D == 64 else 1)),
        int((1 if D == 64 else 6 if D == 384 else 4)),
        2,
    )


def gp_smem(D, cfg):
    bi, bj, slots, sk, mb = cfg
    return (
        bi * bj * D * 2
        + slots * sk * 8192
        + (bi * bj // 64) * 8192
        + 8 * D
        + ((2 + 2 * slots) * 8 + 127) // 128 * 128
    )


def gp_headers():
    import shutil

    dst = T.cache_dir() / ("gp_headers_" + T._source_digest().hex()[:16])
    dst.mkdir(exist_ok=True)
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        txt = (headers() / name).read_text()
        txt = txt.replace(
            "W_RESIDENT || NSLOT >= 2 * SPB",
            "W_RESIDENT || SCHED == 1 || NSLOT >= 2 * SPB",
        )
        p = dst / name
        p.parent.mkdir(exist_ok=True)
        T.publish_header(p, txt)
    return dst


def _compile(D, g, fused):
    inc = gp_headers()
    src = R / ('widths.cu')
    flags = [
        "-DGP_SHARED_A=" + ('1' if D >= 384 else '0'),
        "-DB7_MINB=" + ('2'),
        "-DMW_GP_STREAM=1",
        "-DFUSED_GP=" + str(int(fused)),
        "-DGP_BI=" + str(gp_config(D)[0]),
        "-DGP_SLOTS=" + str(gp_config(D)[2]),
        "-DGP_SK=" + str(gp_config(D)[3]),
        "-DMW_MINB=" + str(2),
        "-I" + str(R),
        "-DWIDTH_N=" + ('64'),
        "-DWIDTH_GROUPS=" + str(g),
        "-DEXTERNAL_GP=1",
        "-DPROFILE_STAGE=" + ('0'),
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        f"-DWIDTH={D}",
        f"-DWEIGHT_SPLITS={tuning(D)[1]}",
        "-I" + str(inc),
    ]
    out = T.compile(src, flags)
    L = T._launch_module()
    drv = L.BlockDriver(device=torch.cuda.current_device())
    mod = drv.load(out.read_bytes())
    u = L.Unit(
        "width_training",
        "sm_90a",
        torch.cuda.current_device(),
        drv.drv,
        mod,
        {},
        str(out),
    )
    ks = {n: u.kernel("width_" + n) for n in ("forward", "b1", "b7", "probe")}
    smem = 2 * 8192 * (1 + g * (int(('64')) // 64)) + 128
    for name, k in ks.items():
        k.set_max_dynamic_smem(
            max(smem, gp_smem(D, gp_config(D))) if name == "b7" and fused else smem
        )
    return ks, str(out)


@T.device_cache
def build(D):
    # Forward and B1 were not changed by this optimization. Reuse their proven
    # cubins, so register pressure from the new B7 cannot lower their occupancy.
    base, path = BASE.build(D)
    ks = dict(base)
    ks["b7"] = _compile(D, gp_config(D)[0] + 1, True)[0]["b7"]
    return ks, path


def launch(k, params, grid=132, cooperative=True, D=64, gp=None):
    threads = 128 * tuning(D)[0]
    smem = tuning(D)[2]
    if gp is not None:
        threads = gp.threads
        smem = max(2 * 8192 * (1 + threads // 128) + 128, gp.smem)
    L = T._launch_module()
    if not cooperative:
        k.launch((grid, 1, 1), (threads, 1, 1), [params], smem)
        return
    drv = k.unit.drv
    args = L._Packed([params] + ([gp.params] if gp is not None else []))
    drv._unwrap(
        "cuLaunchCooperativeKernel",
        drv.d.cuLaunchCooperativeKernel(
            drv.d.CUfunction(int(k.handle)),
            grid,
            1,
            1,
            threads,
            1,
            1,
            smem,
            drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),
            ctypes.addressof(args.array),
        ),
    )


def tm(t):
    assert t.ndim == 2 and t.is_contiguous() and t.dtype == torch.bfloat16
    return T._launch_module().tensor_map(
        t,
        [64, 64],
        dims=[t.shape[1], t.shape[0]],
        strides_bytes=[t.shape[1] * 2],
        swizzle="128B",
        l2="128B",
    )


class Training:
    def __init__(
        self, x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, mask, ds, dy, saved=None, packed=None, prepared_mask=None
    ):
        assert (
            x.dtype == torch.bfloat16 and x.shape[0] == 1 and x.shape[1] == x.shape[2]
        )
        n = x.shape[1]
        D = x.shape[-1]
        H = 2 * D
        M = n * n
        assert D in (64, 128, 256, 384, 512) and n in (384, 768)
        self.D, self.n, self.M = D, n, M
        self.weights = (wl, wlg, wr, wrg)
        self.x, self.dy, self.ds = x, dy, ds
        # Own the forward mask even when a direct caller already supplied FP32;
        # opaque forward outputs must not alias their inputs. Backward borrows
        # this exact saved conversion without another kernel launch.
        self.mask = (prepared_mask if prepared_mask is not None
                     else mask.to(dtype=torch.float32, copy=True).contiguous())
        self.ks, self.path = build(D)
        self.grid = torch.cuda.get_device_properties(
            x.device
        ).multi_processor_count * min(
            4,
            min(
                int(
                    k.unit.drv._unwrap(
                        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                        k.unit.drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                            k.unit.drv.d.CUfunction(int(k.handle)),
                            (128 * tuning(D)[0]),
                            (tuning(D)[2]),
                        ),
                    )
                )
                for k in (self.ks["forward"], self.ks["b1"])
            ),
        )
        self.w1 = packed if packed is not None else x.new_empty((8 * D, D))
        # Module backward reuses its own forward pack. Standalone callers with
        # only the old activation tuple still get a valid, freshly packed plan.
        if saved is not None and packed is None:
            pack_into(self.w1, *self.weights)
        if D == 128:
            cfg = dict(k1=[2, 64, 8, 2, 1], input_ln="fused")
        else:
            cfg = json.loads((R / "selection.json").read_text())[f"{D}-{n}"]
        self.separate = cfg["input_ln"] == "separate"
        self.gi, self.bi = gi, bi
        self.xn = saved[2] if saved else torch.empty_like(x) if self.separate else None
        self.front = Front(
            (self.xn if self.separate else x)[0],
            self.w1,
            self.mask,
            gi,
            bi,
            cfg["k1"],
            emit_xn=not self.separate,
            normalize=not self.separate,
            saved=saved,
        )
        if not self.separate:
            self.xn = self.front.xn
        self.tri = saved[1] if saved else x.new_empty((H, n, n))
        self.y = torch.empty_like(x)
        self.dx = torch.empty_like(x)
        self.dg = x.new_empty((M, D))
        self.dt = torch.empty_like(self.tri)
        self.gp_all = x.new_empty((4, H, M))
        self.gp = list(self.gp_all.unbind())
        self.dw = [torch.empty_like(w) for w in self.weights]
        self.dwp = torch.empty_like(wp)
        self.dwg = torch.empty_like(wg)
        norm = x.new_empty((M, H))
        dp = x.new_empty((M, D))
        dn = x.new_empty((M, H))
        dxn = x.new_empty((M, D))
        self.dl = torch.empty_like(self.tri)
        self.dr = torch.empty_like(self.tri)
        self.tensors = [
            x,
            self.tri,
            dy,
            ds,
            self.y,
            self.xn,
            norm,
            dp,
            self.dg,
            dn,
            dxn,
            self.dx,
            self.dt,
            *self.gp,
            *self.dw,
            self.dwp,
            self.dwg,
            None,
        ]
        self.floats = [
            gi,
            bi,
            go,
            bo,
            self.mask,
            *[torch.empty(M, device=x.device) for _ in range(2)],
            torch.empty((tuning(D)[1], 11 * D * D), device=x.device),
            *[torch.empty(c, device=x.device) for c in (D, D, H, H)],
            torch.empty(32, device=x.device, dtype=torch.int64),
        ]
        maps = [
            tm(self.xn.reshape(M, D)),
            tm(wp),
            tm(wg),
            tm(norm),
            tm(dp),
            tm(self.dg),
            *[tm(g) for g in self.gp],
            *[tm(w) for w in self.weights],
            tm(dxn),
            tm(dn),
        ]
        L = T._launch_module()
        self.params = L.Struct([*maps, *self.tensors, *self.floats, M, n])
        t7 = self.tensors.copy()
        t7[22:24] = [self.dl, self.dr]
        maps7 = maps.copy()
        maps7[14:16] = [tm(self.dl.reshape(H, M)), tm(self.dr.reshape(H, M))]
        self.params7 = L.Struct([*maps7, *t7, *self.floats, M, n])
        self.maps = maps
        self.maps7 = maps7
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_gp import GP

        self.fused = True
        self.gp_native = GP(self, gp_config(D), parameters_only=True)
        self.grid7 = self.grid
        if self.fused:
            k = self.ks["b7"]
            smem = max(
                2 * 8192 * (1 + self.gp_native.threads // 128) + 128,
                self.gp_native.smem,
            )
            occ = int(
                k.unit.drv._unwrap(
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                    k.unit.drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        k.unit.drv.d.CUfunction(int(k.handle)),
                        self.gp_native.threads,
                        smem,
                    ),
                )
            )
            self.grid7 = (
                torch.cuda.get_device_properties(x.device).multi_processor_count * occ
            )
        self.outputs = (self.dx, *self.dw, self.dwg, self.dwp, *self.floats[8:12])

    def forward(self):
        pack_into(self.w1, *self.weights)
        if self.separate:
            normalize_into(self.xn, self.x, self.gi, self.bi)
        ab, _ = self.front()
        d = self.D
        h = 2 * d
        torch.bmm(ab[:d], ab[h : h + d].transpose(-1, -2), out=self.tri[:d])
        torch.bmm(ab[d:h].transpose(-1, -2), ab[h + d :], out=self.tri[d:])
        launch(self.ks["forward"], self.params, self.grid, D=self.D)
        return self.y

    def bind_dy(self, dy):
        self.tensors[2] = dy.contiguous()
        U = T._launch_module()
        self.params = U.Struct(
            [*self.maps, *self.tensors, *self.floats, self.M, self.n]
        )
        t7 = self.tensors.copy()
        t7[22:24] = [self.dl, self.dr]
        self.params7 = U.Struct([*self.maps7, *t7, *self.floats, self.M, self.n])

    def backward(self, dy=None):
        if dy is not None:
            self.bind_dy(dy)
        launch(self.ks["b1"], self.params, self.grid, D=self.D)
        ab = self.front.ab
        d = self.D
        h = 2 * d
        torch.bmm(self.dt[:d], ab[h : h + d], out=self.dl[:d])
        torch.bmm(self.dt[:d].transpose(-1, -2), ab[:d], out=self.dr[:d])
        torch.bmm(ab[h + d :], self.dt[d:].transpose(-1, -2), out=self.dl[d:])
        torch.bmm(ab[d:h], self.dt[d:], out=self.dr[d:])
        if self.gp_native is not None and not self.fused:
            self.gp_native()
        launch(
            self.ks["b7"],
            self.params7,
            self.grid7,
            D=self.D,
            gp=self.gp_native if self.fused else None,
        )
        return self.outputs

    def __call__(self):
        return self.forward(), self.backward()
