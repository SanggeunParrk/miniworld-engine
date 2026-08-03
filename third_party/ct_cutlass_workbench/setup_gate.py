import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS = "/home/psk6950/miniworld-engine/_ct_cutlass/cutlass"

setup(
    name="ct_gate_ext",
    ext_modules=[
        CUDAExtension(
            name="ct_gate_ext",
            sources=["gate_gemm.cu"],
            include_dirs=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools", "util", "include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode", "arch=compute_90a,code=sm_90a",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
