"""Full-output b2b candidate and shared production Triton Transition dispatch."""
import torch
import triton
import triton.language as tl

from miniworld_engine import settings
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key, pack, rows_of
from miniworld_engine.kernels._compile import opaque
from .wide_b2b import _wide_b2b_kernel


def _prune(configs, named_args, **kwargs):
    d = named_args['D']
    return [c for c in configs
            if c.kwargs['BLOCK_K'] == triton.next_power_of_2(d)
            and c.kwargs['BLOCK_M1'] * triton.next_power_of_2(d) < 255 * 32 * c.num_warps]


@triton.autotune(configs=configs_for('transition_b2b_residual_triton'), key=['shape_key', 'SAVE_XN'],
                 prune_configs_by={'early_config_prune': _prune})
@triton.jit
def _kernel(X, G, B, RS, C1, WA, WB, WS, Y, XN, M,
            D: tl.constexpr, SAVE_XN: tl.constexpr, shape_key,
            BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    _wide_b2b_kernel(X, X, G, B, RS, C1, WA, WB, WS, Y, XN, M, D, 4 * D,
                     BLOCK_M1, BLOCK_N, BLOCK_K, tl.constexpr(triton.next_power_of_2(D)),
                     True, SAVE_XN)


def _fake(x, gamma, beta, rs, c1, wa, wb, ws, save_xn, shape_key):
    return torch.empty_like(x), torch.empty_like(x) if save_xn else x.new_empty(0)


@opaque(fake=_fake, name='transition_b2b_residual')
def forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
            rs: torch.Tensor, c1: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
            ws: torch.Tensor, save_xn: bool, shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    m, d = x.shape
    y = torch.empty_like(x)
    xn = torch.empty_like(x) if save_xn else x.new_empty(0)
    _kernel[lambda c: (triton.cdiv(m, c['BLOCK_M1']),)](
        x, gamma, beta, rs, c1, wa, wb, ws, y, xn, m, d, save_xn,
        shape_key=pack(shape_key, D=d))
    return y, xn


class _B2B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps, save_xn):
        from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
        ctx.shape = x.shape
        ctx.key = both_key(rows_of(x.shape))
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        rs, c1 = stats_triton(flat, eps, shape_key=ctx.key)
        y, xn = forward(flat, gamma, beta, rs, c1, wa, wb, ws, save_xn, ctx.key)
        if save_xn:
            ctx.save_for_backward(flat, gamma, rs, c1, xn, wa, wb, ws)
        return y.reshape(ctx.shape)

    @staticmethod
    def backward(ctx, dy):
        from .main import swiglu_squeeze_backward
        from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual_bwd
        x, gamma, rs, c1, xn, wa, wb, ws = ctx.saved_tensors
        flat_dy = dy.reshape_as(x).contiguous()
        dxn, dwa, dwb, dws = swiglu_squeeze_backward(
            xn, wa, wb, ws, flat_dy, ctx.shape, wa.shape[0])
        dx, dg, db = input_ln_residual_bwd(
            dxn.reshape_as(x), x, gamma, c1 / rs, rs, flat_dy, ctx.key)
        return dx.reshape(ctx.shape), dg, db, dwa, dwb, dws, None, None


def enabled(x, wa, wb, ws):
    """Measured SM90 D128/256 pair domain for the segmented production forward."""
    from miniworld_engine.modules.dispatch import is_sm90
    d = x.shape[-1]
    return (settings.current().transition_triton_b2b
            and not settings.current().transition_force_split
            and x.is_cuda and is_sm90(x.device) and d in (128, 256)
            and x.numel() // d >= 16384
            and wa.shape == wb.shape == (4 * d, d) and ws.shape == (d, 4 * d)
            and all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in (x, wa, wb, ws)))


def transition_b2b_residual(x, gamma, beta, wa, wb, ws, eps):
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
    return _B2B.apply(x, gamma, beta, wa, wb, ws, eps, save)


def transition_residual_dispatch(x, gamma, beta, wa, wb, ws, eps):
    from .residual import transition_residual
    if enabled(x, wa, wb, ws):
        from .segmented_residual import transition_segmented
        # Both inference and training use LN -> segmented expand/gate/squeeze/residual.
        # FP32 affine is preserved; the LN output is reused by the existing backward.
        return transition_segmented(x, gamma, beta, wa, wb, ws, eps, fused_ln=False)
    return transition_residual(x, gamma, beta, wa, wb, ws, eps)
