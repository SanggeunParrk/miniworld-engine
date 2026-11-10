"""Own CUDA packing with exact FP32 folds for no-QK-norm Token training."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

FIELDS = (
    "w1",
    "w2",
    "Wraw",
    "Wn",
    "Wg",
    "bs1",
    "bs2",
    "bg1",
    "bg2",
    "Wqkvg",
    "bqkvg",
    "nq",
    "nk",
    "wp",
    "Wb",
    "Wf",
    "Wf_at",
    "Wo",
    "Wab",
    "Wsq",
)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent
    return load_extension(
        name="token_dit_train_pack_sm80",
        sources=[str(directory / "token_train_pack_sm80.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(parameters):
    return dict(zip(FIELDS, _ext().result(list(parameters)), strict=True))
