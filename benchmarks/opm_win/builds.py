"""Variant builds of the fused OPM extension (compile-time macros); shared by kbench.py and the CPU pre-compile job.
   python builds.py NAME...   compiles them (no GPU needed)."""
import sys
from pathlib import Path
from miniworld_engine.kernels.integrations import opm_train as ot

BUILDS = {
    "g2": ["-DOPM_FUSED_GROUP=2"],
    # (BK 64 needs 3 x 48 KiB stages + 2 weight stages: 240 KiB > 227; one weight stage deadlocks the projection loop)
}


def build(name):
    from miniworld_engine.kernels._nvcc import load_extension as load
    csrc = Path(ot.__file__).with_name("csrc")
    return load(name=f"miniworld_opm_fused_v_{name}", sources=[str(csrc / "opm_epilogue.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DMINIWORLD_OPM_FUSED=1", *BUILDS[name]], extra_cflags=["-O3"])


if __name__ == "__main__":
    for n in sys.argv[1:]:
        build(n); print("built", n, flush=True)
