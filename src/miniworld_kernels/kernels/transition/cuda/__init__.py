"""CUDA implementations for Transition forward kernels."""

from pathlib import Path

from torch.utils.cpp_extension import load

_dir = Path(__file__).parent

transition_b2b_cuda = load(
    name="transition_b2b_cuda",
    sources=[str(_dir / "transition_b2b_kernel.cu")],
    extra_cuda_cflags=[
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/include",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/external/cutlass/include",
        "-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    ],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

transition_expand_gate_cuda = load(
    name="transition_expand_gate_cuda",
    sources=[str(_dir / "transition_expand_gate_kernel.cu")],
    extra_cuda_cflags=[
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/include",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/external/cutlass/include",
        "-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    ],
    extra_cflags=["-std=c++17"],
    verbose=False,
)


transition_gatebwd_cuda = load(
    name="transition_gatebwd_cuda",
    sources=[str(_dir / "transition_gatebwd_kernel.cu")],
    extra_cuda_cflags=[
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/include",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/external/cutlass/include",
        "-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    ],
    extra_cflags=["-std=c++17"],
    verbose=False,
)


def transition_expand_gatebwd_wgmma(x, rstd, c1, g, beta, wa, wb, grad_expand):
    """Hopper WGMMA fused expand + SwiGLU gate backward. Returns (h, dAB, xn):
    h=(M,ND) silu(a)*b, dAB=(M,2ND) [dA|dB], xn=(M,K). Matches the Triton
    ``_transition_expand_gatebwd_stacked`` (Version A) for sm90 K in {128,256,512}."""
    return transition_gatebwd_cuda.transition_expand_gatebwd_wgmma(
        x, rstd, c1, g.contiguous(), beta.contiguous(),
        wa.contiguous(), wb.contiguous(), grad_expand.contiguous(),
    )


def transition_b2b_fwd(x, rstd, c1, g, beta, wa, wb, ws, add_residual=False):
    """Fused LN + SwiGLU expand + squeeze forward for fixed AF3 transition shapes.

    ``add_residual``: fold ``y = transition(x) + x`` into the squeeze epilogue (residual == x).
    """
    return transition_b2b_cuda.transition_b2b_fwd(
        x, rstd, c1, g, beta, wa, wb, ws, add_residual
    )


def transition_expand_gate_fwd(x, rstd, c1, g, beta, wa, wb):
    """Fused LN + SwiGLU expand/gate forward returning h[M, ND]."""
    return transition_expand_gate_cuda.transition_expand_gate_fwd(x, rstd, c1, g, beta, wa, wb)


def cuda_transition_b2b(x, ln_weight, ln_bias, wa, wb, ws, eps, add_residual=False):
    """Module-facing wrapper: LN stats (same ``stats_triton`` as the triton b2b path) +
    the hand-CUDA fused b2b forward. Fixed shapes only (K=128, ND=512, D=128 or
    K=256, ND=1024, D=256; n=4); the caller must gate on shape/dtype/M%128 before
    dispatching here.

    Weight layouts match ``nn.Linear.weight`` directly: ``wa``/``wb`` are ``[ND, K]`` and
    ``ws`` is ``[D, ND]`` — no transpose needed.

    ``add_residual``: fuse the post-transition residual add ``y = transition(x) + x`` into
    the squeeze output epilogue (the residual is the module input ``x`` itself; D == K).
    Saves the separate elementwise-add kernel + its M×D round-trip.
    """
    from miniworld_kernels.kernels.layernorm_linear.triton.stats import stats_triton

    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    rstd, c1 = stats_triton(x2, eps)
    out = transition_b2b_cuda.transition_b2b_fwd(
        x2,
        rstd,
        c1,
        ln_weight.contiguous(),
        ln_bias.contiguous(),
        wa.contiguous(),
        wb.contiguous(),
        ws.contiguous(),
        add_residual,
    )
    return out.reshape(*x.shape[:-1], out.shape[-1])


def cuda_transition_expand_gate(x, ln_weight, ln_bias, wa, wb, eps):
    """Module-facing wrapper: LN stats + hand-CUDA expand/gate forward.

    Returns the row-major hidden activation ``h`` with final shape ``(*x.shape[:-1], ND)``.
    The caller owns the squeeze GEMM/dispatch decision.
    """
    from miniworld_kernels.kernels.layernorm_linear.triton.stats import stats_triton

    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    rstd, c1 = stats_triton(x2, eps)
    h = transition_expand_gate_cuda.transition_expand_gate_fwd(
        x2,
        rstd,
        c1,
        ln_weight.contiguous(),
        ln_bias.contiguous(),
        wa.contiguous(),
        wb.contiguous(),
    )
    return h.reshape(*x.shape[:-1], h.shape[-1])
