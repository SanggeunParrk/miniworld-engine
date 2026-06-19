# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/test_layer_norm_cuda.py
"""
test_layer_norm_cuda.py

Build first:
    pip install -e . --no-build-isolation

or inline (JIT):
    from torch.utils.cpp_extension import load
    ext = load(name="layer_norm_cuda", sources=["layer_norm_cuda_kernel.cu"], verbose=True)
"""

from pathlib import Path

import torch
import torch.nn.functional as F

# ── JIT build (no pip install needed) ─────────────────────────────────────────
from torch.utils.cpp_extension import load

ext = load(
    name="layer_norm_cuda",
    sources=[str(Path(__file__).parent / "layer_norm_cuda_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


# ── helper ────────────────────────────────────────────────────────────────────
def ref_layernorm(x, w, b, eps=1e-5):
    """PyTorch reference (uses its own fused kernel, but correct)."""
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def test(dtype=torch.float32, shape=(512, 4096), eps=1e-5):
    M, N = shape
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    w = torch.randn(N, dtype=dtype, device="cuda")
    b = torch.randn(N, dtype=dtype, device="cuda")

    y_ref = ref_layernorm(x, w, b, eps)
    y_our, mean, rstd = ext.layer_norm_fwd(x.contiguous(), w, b, eps)

    atol = 1e-3 if dtype != torch.float32 else 1e-5
    max_err = (y_our - y_ref).abs().max().item()
    print(
        f"dtype={dtype}  shape={shape}  max_err={max_err:.2e}  "
        f"{'✓ PASS' if max_err < atol else '✗ FAIL'}"
    )

    # also verify mean / rstd
    x_f = x.float()
    ref_mean = x_f.mean(-1)
    ref_var = x_f.var(-1, unbiased=False)
    ref_rstd = (ref_var + eps).rsqrt()
    print(
        f"  mean max_err={(mean - ref_mean).abs().max():.2e}  "
        f"  rstd max_err={(rstd - ref_rstd).abs().max():.2e}"
    )


if __name__ == "__main__":
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in [(1, 256), (128, 1024), (512, 4096), (2048, 8192)]:
            test(dtype=dtype, shape=shape)
