"""Loader for the raw-CUDA/CuTe sm_100 (B200) fused transition-forward b2b kernel.

Ported from the Hopper hand-CUDA b2b (``transition_b2b_kernel.cu``). Blackwell tcgen05
has no register-source MMA, so the SwiGLU intermediate ``h`` round-trips
TMEM(acc) -> regs (tcgen05.ld) -> silu -> smem (sH) and the squeeze MMA reads sH (SS).

Build is via ``torch.utils.cpp_extension.load`` with CUTLASS 4.4.0 sm100 headers and
``arch=sm_100a``. The CUTLASS include root defaults to ``~/psk/_ct_cutlass_b200/cutlass``
and can be overridden with the ``CUTLASS_SM100_DIR`` env var.

Correctness (M=147456): out cos 0.999997 for both (K,ND,D)=(128,512,128) and (256,1024,256).
"""

from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load

_dir = Path(__file__).parent
_ct = Path(os.environ.get("CUTLASS_SM100_DIR", os.path.expanduser("~/psk/_ct_cutlass_b200/cutlass")))

_transition_b2b_sm100 = load(
    name="transition_b2b_sm100_cuda",
    sources=[str(_dir / "transition_b2b_sm100_kernel.cu")],
    extra_cuda_cflags=[
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DTRANSITION_B2B_STAGE=2",
        f"-I{_ct}/include",
        f"-I{_ct}/tools/util/include",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    ],
    extra_cflags=["-std=c++17"],
    verbose=False,
)


def transition_b2b_sm100_fwd(xn, wa, wb, ws):
    """out = (silu(xn@wa^T) * (xn@wb^T)) @ ws^T.

    xn:(M,K) pre-normalized bf16; wa/wb:(ND,K) bf16; ws:(D,ND) bf16 -> out:(M,D) bf16.
    Supports (K,ND,D) = (128,512,128) or (256,1024,256), M % 128 == 0.
    """
    return _transition_b2b_sm100.transition_b2b_fwd(
        xn.contiguous(), wa.contiguous(), wb.contiguous(), ws.contiguous()
    )
