# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/__init__.py
"""CUDA implementation of LayerNorm."""

from pathlib import Path

from torch.utils.cpp_extension import load

_dir = Path(__file__).parent

layer_norm_cuda = load(
    name="layer_norm_cuda",
    sources=[str(_dir / "layer_norm_cuda_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
