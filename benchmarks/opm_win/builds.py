"""Variant builds of the fused OPM extension (compile-time macros); shared by kbench.py and the CPU pre-compile job.
   python builds.py NAME...   compiles them (no GPU needed)."""
import sys
from pathlib import Path
from miniworld_engine.kernels.integrations import opm_train as ot

BUILDS = {
    "g2": ["-DOPM_FUSED_GROUP=2"],
    "g1": ["-DOPM_PRO_G=1"],   # the wide prologue in the token-major tile order (before 2026-11-16)
    "noreg": ["-DOPM_PRO_REG=0"],   # the wide prologue without the setmaxnreg rebalance (128 regs everywhere: 112 B stack)
    "e64s3": ["-DOPM_EPI_BJ=64"],   # epilogue: 256 pairs per CTA (4 warpgroups), 3 stages
    "e64s4": ["-DOPM_EPI_BJ=64", "-DOPM_EPI_NSTAGE=4"],
    "e32s4": ["-DOPM_EPI_NSTAGE=4"],
    "g4": ["-DOPM_PRO_G=4"],
    "g64": ["-DOPM_PRO_G=64"],
    "g128": ["-DOPM_PRO_G=128"],
    "nst4": ["-DOPM_PRO_NST=4"],
    "nst6": ["-DOPM_PRO_NST=6"],
    "g256": ["-DOPM_PRO_G=256"],
    "g768": ["-DOPM_PRO_G=768"],
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


def build_old():
    """The fused-kernel build of the opm-bigs snapshot (benchmarks/opm_win/jobb.sh: git archive of 0ac874471)."""
    from miniworld_engine.kernels._nvcc import load_extension as load
    src = Path("/home/psk6950/miniworld-engine-scratch/opm-win-before/src/miniworld_engine/kernels/integrations/csrc/opm_epilogue.cu")
    return load(name="miniworld_opm_fused_old", sources=[str(src)],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DMINIWORLD_OPM_FUSED=1"], extra_cflags=["-O3"])
