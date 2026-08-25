# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/setup.py
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# `setup()` under a __main__ guard. This file lives INSIDE the importable package, so
# without the guard `import miniworld_engine.kernels.<family>.cuda.setup` runs
# setuptools and dies with `SystemExit: usage: cli.py ...` -- which is what
# `dev audit`'s import sweep hit, reporting 2 not OK on every run. Running the script
# directly (`python setup.py build_ext --inplace`) is unchanged.
if __name__ == "__main__":
    setup(
        name="layer_norm_cuda",
        ext_modules=[
            CUDAExtension(
                name="layer_norm_cuda",
                sources=["layer_norm_cuda_kernel.cu"],
                extra_compile_args={
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-gencode=arch=compute_80,code=sm_80",   # A100
                        "-gencode=arch=compute_86,code=sm_86",   # RTX 3090 / A6000
                        "-gencode=arch=compute_89,code=sm_89",   # RTX 4090
                        "-gencode=arch=compute_90,code=sm_90",   # H100
                    ]
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
