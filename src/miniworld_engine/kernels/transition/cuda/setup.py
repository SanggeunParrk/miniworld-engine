# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/transition/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="transition_cuda_ext_v2",
    ext_modules=[
        CUDAExtension(
            name="transition_cuda_ext_v2",
            sources=["transition_cuda.cpp", "transition_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
            libraries=["cublas"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    options={"build": {"build_base": "build_v2"}},
)
