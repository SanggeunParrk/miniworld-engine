from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "output"
K1 = (2, 64, 8, 2, -1, 232, 2)
K3 = (2, 64, 4, 1, 1, 1)


@T.device_cache
def kernel(kind, ln=0, stats=0, pg=False, method=0):
    inc = T._upstream() / "csrc"
    src = R / ("original_k1.cu" if kind == "k1" else "save_k3.cu")
    defs = dict(MW_SAVE_PG=int(pg), MW_PG_METHOD=method, XHAT_FP32=method)
    if kind == "k1":
        smem = T.k1_smem(K1) + (32768 if pg and method == 0 else 0)
        threads = 384
    else:
        bi, bj, slots, acc, regs, serial = K3
        defs.update(
            MW_SAVE_IN=ln & 1,
            MW_SAVE_OUT=0,
            MW_STORE_METHOD=0,
            MW_SAVE_STATS_IN=stats & 1,
            MW_SAVE_STATS_OUT=(stats >> 1) & 1,
            MW_BI=bi,
            MW_BJ=bj,
            MW_SLOT=slots,
            MW_ACC=acc,
            TMN_K3_REGS_24_240=regs,
            MW_SERIAL=serial,
            MW_FUSED=1,
        )
        smem = T.k3_smem(K3) + (32768 if pg and method == 0 else 0)
        threads = 384
    assert smem <= 232448
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        "-I" + str(inc),
    ] + ["-D%s=%s" % v for v in defs.items()]
    out = T.compile(src, flags)
    L = T._launch_module()
    drv = L.BlockDriver(device=torch.cuda.current_device())
    mod = drv.load(out.read_bytes())
    unit = L.Unit(
        "save_" + kind,
        "sm_90a",
        torch.cuda.current_device(),
        drv.drv,
        mod,
        {},
        str(out),
    )
    k = unit.kernel("infer_k1" if kind == "k1" else "save_k3")
    k.set_max_dynamic_smem(smem)
    return k, smem


def tm(t, box, dims, strides):
    return T._launch_module().tensor_map(
        t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="128B"
    )


def front(d, pg=False, method=0, bufs=None):
    n = d["n"]
    x = d["x"]
    m = n * n
    bi, bj = K1[:2]
    L = T._launch_module()
    k, smem = kernel("k1", pg=pg, method=method)
    ab, pre = (
        bufs
        if bufs
        else (
            torch.empty((512, n, n), device=x.device, dtype=x.dtype),
            torch.empty((1024, m), device=x.device, dtype=x.dtype) if pg else None,
        )
    )
    mz = tm(x, [64, bj, bi], [128, n, n], [256, n * 256])
    mw = tm(d["w1"], [64, 64], [128, 1024], [256])
    ma = tm(ab, [64, 1, 32], [n, n, 512], [n * 2, m * 2])
    mg = tm(pre, [64, 1, 32], [n, n, 512], [n * 2, m * 4]) if pg else ma
    mp = tm(pre[1], [64, 1, 32], [n, n, 512], [n * 2, m * 4]) if pg else ma
    tj = (n + bj - 1) // bj
    tiles = ((n + bi - 1) // bi) * tj
    base = L.Struct(
        [
            mz,
            mw,
            ma,
            d["mask"],
            d["gi"],
            d["bi"],
            ab,
            None,
            None,
            n,
            n,
            tj,
            tiles,
            1,
            n,
            1,
            1e-5,
            n * 128,
            128,
            0,
            0,
        ]
    )
    p = base
    k.launch(
        (
            min(
                tiles,
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count,
            ),
            1,
            1,
        ),
        (384, 1, 1),
        [p],
        smem,
    )
    return ab, pre


def output(d, tri, ln=1, stats=0, pg=False, method=0, bufs=None):
    n = d["n"]
    x = d["x"]
    m = n * n
    bi, bj = K3[:2]
    L = T._launch_module()
    k, smem = kernel("k3", ln, stats, pg, method)
    if bufs is None:
        e = lambda c: torch.empty((m, c), device=x.device, dtype=x.dtype)
        f = lambda: torch.empty(m, device=x.device, dtype=torch.float32)
        saves = dict(
            xn=e(128) if ln & 1 else None,
            xnout=tri
            if method == -1
            else torch.empty(
                (256, m),
                device=x.device,
                dtype=torch.float32
                if method == 1
                else torch.float16
                if method >= 2
                else x.dtype,
            )
            if ln & 2
            else None,
            mi=f() if stats & 1 else None,
            ri=f() if stats & 1 else None,
            mo=None,
            ro=torch.empty(
                m * (2 if method in (-1, 3) else 1),
                device=x.device,
                dtype=torch.float32,
            )
            if stats & 2
            else None,
            proj=e(128) if pg else None,
            gate=e(128) if pg else None,
        )
        y = torch.empty_like(x)
    else:
        y, saves = bufs
    if method in (-1, 3):
        saves["mo"] = saves["ro"][m:]
    if method == -1:
        saves["xnout"] = tri
    maps = [
        tm(x, [64, bj, bi], [128, n, n], [256, n * 256]),
        tm(tri, [64, 1, 64], [n, n, 256], [n * 2, m * 2]),
        tm(d["leaves"][5], [64, 32], [128, 128], [256]),
        tm(d["wp"], [64, 32], [256, 128], [512]),
        tm(y, [64, 16, 1], [128, n, n], [256, n * 256]),
    ]
    tj = (n + bj - 1) // bj
    tiles = ((n + bi - 1) // bi) * tj
    base = L.Struct(
        [
            *maps,
            d["gi"],
            d["bi"],
            d["go"],
            d["bo"],
            x,
            y,
            None,
            n,
            n,
            tj,
            tiles,
            1,
            0,
            1e-5,
            0,
        ]
    )
    om = lambda key, c: (
        tm(saves[key], [64, 16, 1], [c, n, n], [c * 2, n * c * 2])
        if saves[key] is not None
        else maps[-1]
    )
    outmap = (
        L.tensor_map(
            saves["xnout"],
            [16, 32],
            dims=[m, 256],
            strides_bytes=[m * 4],
            swizzle="64B",
            l2="128B",
        )
        if method == 1
        else maps[-1]
    )
    p = L.Struct(
        [
            base,
            d["ds"],
            om("xn", 128),
            outmap,
            saves["xn"],
            saves["xnout"],
            saves["mi"],
            saves["ri"],
            saves["mo"],
            saves["ro"],
            om("proj", 128),
            om("gate", 128),
            saves["proj"],
            saves["gate"],
        ]
    )
    k.launch(
        (
            min(
                tiles,
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count,
            ),
            1,
            1,
        ),
        (384, 1, 1),
        [p],
        smem,
    )
    return y, saves
