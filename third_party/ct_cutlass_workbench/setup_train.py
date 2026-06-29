import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS = "/home/psk6950/miniworld-kernels/_ct_cutlass/cutlass"
KDIR = "/home/psk6950/miniworld-kernels/src/miniworld_kernels/kernels/conditioned_transition/cutlass"
setup(
    name="ct_train_ext",
    ext_modules=[
        CUDAExtension(
            name="ct_train_ext",
            sources=[os.path.join(KDIR, "ct_train.cu")],
            include_dirs=[os.path.join(CUTLASS, "include"),
                          os.path.join(CUTLASS, "tools", "util", "include"),
                          KDIR],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + (["-DCT_NO_FUSED_Y"] if os.environ.get("CT_NO_FUSED_Y") else []),
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                         "--expt-extended-lambda", "-gencode", "arch=compute_90a,code=sm_90a"]
                        + (["-DCT_NO_FUSED_Y"] if os.environ.get("CT_NO_FUSED_Y") else []),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
