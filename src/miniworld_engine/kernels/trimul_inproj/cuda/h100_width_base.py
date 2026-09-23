import torch, os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "wide_base"


@T.device_cache
def build(D, hidden=None):
    inc = T._upstream() / "csrc"
    src = R / "widths.cu"
    flags = [
        "-DPROFILE_STAGE=" + ('0'),
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        "-Xptxas=-v",
        f"-DWIDTH={D}",
        f"-DHIDDEN={2 * D if hidden is None else hidden}",
        f"-DWEIGHT_SPLITS={128 if D == 64 else 32}",
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
    for k in ks.values():
        k.set_max_dynamic_smem(33024 if D == 64 else 49280)
    return ks, str(out)
