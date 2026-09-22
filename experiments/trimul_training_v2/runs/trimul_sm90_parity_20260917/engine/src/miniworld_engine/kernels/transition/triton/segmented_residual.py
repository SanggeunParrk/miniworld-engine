"""CSV/cache tuned segmented b2b, with shared inference/training forward."""
import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key, pack, rows_of
from miniworld_engine.kernels._compile import opaque
from .segmented_b2b import _segmented
from .b2b_residual import _B2B
from .residual import ResidualTransitionFunction


def _prune(configs, named_args, **kwargs):
    d = named_args['D']
    return [c for c in configs
            if c.kwargs['BLOCK_K'] == d
            and d % c.kwargs['BLOCK_O'] == 0
            and 2 <= d // c.kwargs['BLOCK_O'] <= 4
            and c.kwargs['BLOCK_M1'] * d < 255 * 32 * c.num_warps]


def _bench(fn, quantiles):
    # Rank configurations under the same CUDA Graph execution used by the cache
    # builder and module benchmarks, including on a bounded runtime cache miss.
    return triton.testing.do_bench_cudagraph(fn, rep=50, quantiles=quantiles)


@triton.autotune(configs=configs_for('transition_segmented_b2b_triton'),
                 key=['shape_key', 'NORMALIZE', 'SAVE_XN'], do_bench=_bench,
                 prune_configs_by={'early_config_prune': _prune})
@triton.jit
def _kernel(X, R, G, B, RS, C1, WA, WB, WS, Y, XN, M,
            D: tl.constexpr, NORMALIZE: tl.constexpr, SAVE_XN: tl.constexpr, shape_key,
            BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr, BLOCK_O: tl.constexpr):
    _segmented(X, R, G, B, RS, C1, WA, WB, WS, Y, XN, M, D, 4 * D,
               BLOCK_M1, BLOCK_N, BLOCK_K, BLOCK_O, NORMALIZE, SAVE_XN)


def _fake(x, residual, gamma, beta, rs, c1, wa, wb, ws, normalize, save_xn, shape_key):
    return torch.empty_like(x), torch.empty_like(x) if save_xn else x.new_empty(0)


@opaque(fake=_fake, name='transition_segmented_b2b')
def forward(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
            rs: torch.Tensor, c1: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
            ws: torch.Tensor, normalize: bool, save_xn: bool,
            shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    m, d = x.shape
    y = torch.empty_like(x)
    xn = torch.empty_like(x) if save_xn else x.new_empty(0)
    _kernel[lambda c: (triton.cdiv(m, c['BLOCK_M1']),)](
        x, residual, gamma, beta, rs, c1, wa, wb, ws, y, xn, m, d, normalize, save_xn,
        shape_key=pack(shape_key, D=d))
    return y, xn


class _Fused(_B2B):
    """Use the exact saved-input backward of the existing production b2b."""
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps, save_xn):
        from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
        ctx.shape = x.shape
        ctx.key = both_key(rows_of(x.shape))
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        rs, c1 = stats_triton(flat, eps, shape_key=ctx.key)
        y, xn = forward(flat, flat, gamma, beta, rs, c1, wa, wb, ws, True, save_xn, ctx.key)
        if save_xn:
            ctx.save_for_backward(flat, gamma, rs, c1, xn, wa, wb, ws)
        return y.reshape(ctx.shape)


class _Separate(ResidualTransitionFunction):
    """Use the split path's backward verbatim; only forward scheduling differs."""
    @staticmethod
    def forward(ctx, xn, wa, wb, ws, residual):
        ctx.shape = xn.shape
        flat = xn.reshape(-1, xn.shape[-1]).contiguous()
        ctx.save_for_backward(flat, wa, wb, ws)
        empty = flat.new_empty(0)
        y, _ = forward(flat, residual.reshape_as(flat), empty, empty, empty, empty,
                       wa, wb, ws, False, False, both_key(rows_of(ctx.shape)))
        return y.reshape(ctx.shape)


def transition_segmented(x, gamma, beta, wa, wb, ws, eps, *, fused_ln):
    if fused_ln:
        save = torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
        return _Fused.apply(x, gamma, beta, wa, wb, ws, eps, save)
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual
    xn, residual = input_ln_residual(x, gamma, beta, eps)
    return _Separate.apply(xn, wa, wb, ws, residual)
