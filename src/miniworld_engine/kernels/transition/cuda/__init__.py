"""CUDA implementations for Transition forward kernels."""

from pathlib import Path

import torch
from ..._compile import opaque


from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension, mathdx_includes

_dir = Path(__file__).parent
_EXTENSIONS = {}


def _ext(kind, width, config):
    from miniworld_engine.autotune.hopper_cuda_config import defines
    flags = defines(kind, width, config)
    key = (kind, width, tuple(sorted(config.items())))
    if key not in _EXTENSIONS:
        ensure_cuda_home()
        suffix = "_".join(f"{k}{v}" for k, v in sorted(config.items()))
        _EXTENSIONS[key] = load_extension(
            name=f"transition_{kind}_cuda_k{width}_{suffix}",
            sources=[str(_dir / f"transition_{kind}_kernel.cu")],
            extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "--use_fast_math",
                               "--expt-relaxed-constexpr", "--expt-extended-lambda",
                               *gencodes("90a"), *mathdx_includes(), *flags,
                               "-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT",
                               "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                               "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
            extra_cflags=["-std=c++17"], verbose=False,
        )
    return _EXTENSIONS[key]


def __b2b_launch_fake(x, rstd, c1, g, beta, wa, wb, ws, bn, stages, warpgroups, kt, min_blocks):
    """Allocate outputs with the same shape, dtype and strides as _b2b_launch."""
    return torch.empty_like(x)


@opaque(fake=__b2b_launch_fake, name="transition_b2b_fwd_cuda")
def _b2b_launch(
    x: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor,
    wb: torch.Tensor, ws: torch.Tensor, bn: int, stages: int,
    warpgroups: int, kt: int, min_blocks: int,
) -> torch.Tensor:
    """Native inference launch boundary; keep surrounding LN/reshapes traceable."""
    config = dict(bn=bn, stages=stages, warpgroups=warpgroups, kt=kt, min_blocks=min_blocks)
    return _ext("b2b", x.shape[-1], config).transition_b2b_fwd(x, rstd, c1, g, beta, wa, wb, ws, True)


def _run_configured(kind, symbol, tensors, *, config=None, residual=False):
    from miniworld_engine.autotune.hopper_cuda_config import candidates
    from miniworld_engine.autotune.native import choose_config, tensor_key
    x = tensors[0]
    width = x.shape[-1]
    grid = candidates(kind, width)
    # Current CUDA kernels require a whole warpgroup tile of rows. Configs
    # must also divide the actual workload, not just a coarse cache bucket.
    grid = [c for c in grid if x.shape[0] % (64 * c["warpgroups"]) == 0]
    if config is None:
        config = choose_config(
            {"b2b": "transition_fwd_b2b_sm90_cuda", "gatebwd": "transition_bwd_gate_sm90_cuda",
             "expand_gate": "transition_expand_gate_sm90_cuda"}[kind], grid, dtype=str(x.dtype),
            bucket=tensor_key(*tensors, extra=(residual,)), device_index=x.device.index,
            run=lambda c: _run_configured(kind, symbol, tensors, config=c, residual=residual),
        )
    if config not in grid:
        raise ValueError("CUDA transition config does not support the input row count")
    if kind == "b2b" and residual:
        return _b2b_launch(*tensors, **config)
    fn = getattr(_ext(kind, width, config), symbol)
    return fn(*tensors, True) if residual else fn(*tensors)


def transition_expand_gatebwd_wgmma(x, rstd, c1, g, beta, wa, wb, grad_expand, *, config=None):
    """Hopper WGMMA fused expand + SwiGLU gate backward. Returns (h, dAB, xn):
    h=(M,ND) silu(a)*b, dAB=(M,2ND) [dA|dB], xn=(M,K). Matches the Triton
    ``_transition_expand_gatebwd_stacked`` (Version A) for sm90 K in {128,256,512}."""
    return _run_configured("gatebwd", "transition_expand_gatebwd_wgmma",
                           (x, rstd, c1, g.contiguous(), beta.contiguous(),
                            wa.contiguous(), wb.contiguous(), grad_expand.contiguous()),
                           config=config)


def transition_b2b_fwd(x, rstd, c1, g, beta, wa, wb, ws, *, config=None):
    """Fused LN + SwiGLU expand + squeeze forward for fixed AF3 transition shapes.
    Returns ``y = transition(x) + x``.

    The residual is folded into the squeeze epilogue (residual == x) and is not optional: this
    is the Transition op, and every launcher of it defines the op that way. The C++ entry still
    takes the flag -- it is one `bool` in `transition_b2b_kernel.cu` -- but nothing can pass
    False through here, which is what "no variable" means for a Python caller.
    """
    return _run_configured("b2b", "transition_b2b_fwd",
                           (x, rstd, c1, g, beta, wa, wb, ws), config=config, residual=True)


def transition_expand_gate_fwd(x, rstd, c1, g, beta, wa, wb, *, config=None):
    """Fused LN + SwiGLU expand/gate forward returning h[M, ND]."""
    return _run_configured("expand_gate", "transition_expand_gate_fwd",
                           (x, rstd, c1, g, beta, wa, wb), config=config)


def cuda_transition_b2b(x, ln_weight, ln_bias, wa, wb, ws, eps):
    """Module-facing wrapper: LN stats (same ``stats_triton`` as the triton b2b path) +
    the hand-CUDA fused b2b forward. Fixed shapes only (K=128, ND=512, D=128 or
    K=256, ND=1024, D=256; n=4); the caller must gate on shape/dtype/M%128 before
    dispatching here.

    Weight layouts match ``nn.Linear.weight`` directly: ``wa``/``wb`` are ``[ND, K]`` and
    ``ws`` is ``[D, ND]`` — no transpose needed.

    Returns ``y = transition(x) + x``: the residual add is fused into the squeeze output
    epilogue (the residual is the module input ``x`` itself; D == K), saving the separate
    elementwise-add kernel and its M×D round-trip. It is part of the op, not a flag on it.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    rstd, c1 = stats_triton(x2, eps)
    out = transition_b2b_fwd(
        x2,
        rstd,
        c1,
        ln_weight.contiguous(),
        ln_bias.contiguous(),
        wa.contiguous(),
        wb.contiguous(),
        ws.contiguous(),
    )
    return out.reshape(*x.shape[:-1], out.shape[-1])


def cuda_transition_expand_gate(x, ln_weight, ln_bias, wa, wb, eps):
    """Module-facing wrapper: LN stats + hand-CUDA expand/gate forward.

    Returns the row-major hidden activation ``h`` with final shape ``(*x.shape[:-1], ND)``.
    The caller owns the squeeze GEMM/dispatch decision.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    rstd, c1 = stats_triton(x2, eps)
    h = transition_expand_gate_fwd(
        x2,
        rstd,
        c1,
        ln_weight.contiguous(),
        ln_bias.contiguous(),
        wa.contiguous(),
        wb.contiguous(),
    )
    return h.reshape(*x.shape[:-1], h.shape[-1])
