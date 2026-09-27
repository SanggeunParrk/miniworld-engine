"""B2B core and explicit experimental wide-D layouts.

One program owns every output channel for a row tile. Expand/SwiGLU is computed
once per hidden tile, and h never reaches HBM. Row, hidden and contraction tiles
are independent. Configs are supplied by the caller and benchmarked explicitly.
This full-output core remains a comparison candidate. Production D128/256
uses segmented_residual; D384/512 retains split.
"""
import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import opaque


@triton.jit
def _operand(X, G, B, RS, C1, rows, kk, M, D: tl.constexpr, NORMALIZE: tl.constexpr):
    x = tl.load(X + rows[:, None] * D + kk[None, :],
                (rows[:, None] < M) & (kk[None, :] < D), 0)
    if NORMALIZE:
        rs = tl.load(RS + rows, rows < M, 0)
        c1 = tl.load(C1 + rows, rows < M, 0)
        g = tl.load(G + kk, kk < D, 0).to(tl.float32)
        b = tl.load(B + kk, kk < D, 0).to(tl.float32)
        x = ((x.to(tl.float32) * rs[:, None] - c1[:, None]) * g[None, :] + b[None, :]).to(X.dtype.element_ty)
    return x


@triton.jit
def _packed_weights(WA, WB, kk, h0, D: tl.constexpr, H: tl.constexpr, BN: tl.constexpr):
    col = tl.arange(0, 2 * BN)
    nn = h0 + col % BN
    offset = nn[None, :] * D + kk[:, None]
    ptr = tl.where(col[None, :] < BN, WA + offset, WB + offset)
    return tl.load(ptr, (nn[None, :] < H) & (kk[:, None] < D), 0)


@triton.jit
def _wide_b2b_kernel(X, R, G, B, RS, C1, WA, WB, WS, Y, XN,
                     M, D: tl.constexpr, H: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     BD: tl.constexpr, NORMALIZE: tl.constexpr, SAVE_XN: tl.constexpr,
                     PACKED: tl.constexpr = False):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dc = tl.arange(0, BD)
    acc = tl.zeros((BM, BD), tl.float32)
    if BK >= D:
        kk = tl.arange(0, BK)
        xn = _operand(X, G, B, RS, C1, rows, kk, M, D, NORMALIZE)
        if SAVE_XN:
            tl.store(XN + rows[:, None] * D + kk[None, :], xn,
                     (rows[:, None] < M) & (kk[None, :] < D))
        for h0 in range(0, H, BN):
            nn = h0 + tl.arange(0, BN)
            wa = tl.load(WA + nn[None, :] * D + kk[:, None],
                         (nn[None, :] < H) & (kk[:, None] < D), 0)
            wb = tl.load(WB + nn[None, :] * D + kk[:, None],
                         (nn[None, :] < H) & (kk[:, None] < D), 0)
            if PACKED:
                if PACKED == 2:
                    ab = tl.dot(xn, _packed_weights(WA, WB, kk, h0, D, H, BN))
                    a, b = tl.split(tl.trans(tl.reshape(ab, (BM, 2, BN)), (0, 2, 1)))
                else:
                    ab = tl.dot(xn, tl.interleave(wa, wb))
                    a, b = tl.split(tl.reshape(ab, (BM, BN, 2)))
            else:
                a = tl.dot(xn, wa)
                b = tl.dot(xn, wb)
            h = (a * tl.sigmoid(a) * b).to(X.dtype.element_ty)
            ws = tl.load(WS + dc[None, :] * H + nn[:, None],
                         (dc[None, :] < D) & (nn[:, None] < H), 0)
            acc = tl.dot(h, ws, acc)
    else:
        # Keep the one-time saved-operand write outside the pipelined GEMM loops.
        # A conditional store inside every hidden iteration prevents the compiler
        # from scheduling the otherwise read-only operand pipeline effectively.
        if SAVE_XN:
            for k0 in range(0, D, BK):
                kk = k0 + tl.arange(0, BK)
                xn_save = _operand(X, G, B, RS, C1, rows, kk, M, D, NORMALIZE)
                tl.store(XN + rows[:, None] * D + kk[None, :], xn_save,
                         (rows[:, None] < M) & (kk[None, :] < D))
        for h0 in range(0, H, BN):
            nn = h0 + tl.arange(0, BN)
            if PACKED:
                ab = tl.zeros((BM, 2 * BN), tl.float32)
            else:
                a = tl.zeros((BM, BN), tl.float32)
                b = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, D, BK):
                kk = k0 + tl.arange(0, BK)
                xn = _operand(X, G, B, RS, C1, rows, kk, M, D, NORMALIZE)
                wa = tl.load(WA + nn[None, :] * D + kk[:, None],
                             (nn[None, :] < H) & (kk[:, None] < D), 0)
                wb = tl.load(WB + nn[None, :] * D + kk[:, None],
                             (nn[None, :] < H) & (kk[:, None] < D), 0)
                if PACKED:
                    if PACKED == 2:
                        ab = tl.dot(xn, _packed_weights(WA, WB, kk, h0, D, H, BN), ab)
                    else:
                        ab = tl.dot(xn, tl.interleave(wa, wb), ab)
                else:
                    a = tl.dot(xn, wa, a)
                    b = tl.dot(xn, wb, b)
            if PACKED:
                if PACKED == 2:
                    a, b = tl.split(tl.trans(tl.reshape(ab, (BM, 2, BN)), (0, 2, 1)))
                else:
                    a, b = tl.split(tl.reshape(ab, (BM, BN, 2)))
            h = (a * tl.sigmoid(a) * b).to(X.dtype.element_ty)
            ws = tl.load(WS + dc[None, :] * H + nn[:, None],
                         (dc[None, :] < D) & (nn[:, None] < H), 0)
            acc = tl.dot(h, ws, acc)
    mask = (rows[:, None] < M) & (dc[None, :] < D)
    residual = tl.load(R + rows[:, None] * D + dc[None, :], mask, 0).to(tl.float32)
    # Preserve the production squeeze BF16 rounding boundary BEFORE identity add.
    y = acc.to(Y.dtype.element_ty).to(tl.float32) + residual
    tl.store(Y + rows[:, None] * D + dc[None, :], y, mask)


def launch(x, residual, gamma, beta, rstd, c1, wa, wb, ws, *, config,
           normalize=False, save_xn=False, out=None, xn_out=None):
    """Raw explicit-config launch, also returning compiled resource metadata."""
    m, d = x.shape
    if d not in (128, 256, 384, 512) or wa.shape != (4*d, d) or wb.shape != wa.shape or ws.shape != (d, 4*d):
        raise ValueError('b2b experiment requires D128/256/384/512, expansion 4')
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x, residual, wa, wb, ws)):
        raise ValueError('wide b2b requires contiguous BF16 activation and weights')
    if residual.shape != x.shape or (save_xn and not normalize):
        raise ValueError('invalid residual shape or saved operand mode')
    # Those configurations need >=256 FP32 output accumulators per thread BEFORE
    # any expand state. Some Triton SM90 spill variants faulted during exploration;
    # do not expose them as supported launches. This is a resource safety filter,
    # not a width-specific winning configuration.
    if config['BM'] * triton.next_power_of_2(d) > 255 * 32 * config['num_warps']:
        raise ValueError('output accumulator alone exceeds the per-thread register budget')
    out = torch.empty_like(x) if out is None else out
    xn_out = (torch.empty_like(x) if save_xn else x.new_empty(0)) if xn_out is None else xn_out
    compiled = _wide_b2b_kernel[(triton.cdiv(m, config['BM']),)](
        x, residual, gamma, beta, rstd, c1, wa, wb, ws, out, xn_out,
        m, d, 4*d, BD=triton.next_power_of_2(d), NORMALIZE=normalize, SAVE_XN=save_xn,
        **config)
    return out, xn_out, compiled


def _fake(x, residual, gamma, beta, rstd, c1, wa, wb, ws,
          bm, bn, bk, warps, stages, normalize, save_xn, packed=0, output_tile=0):
    return torch.empty_like(x), torch.empty_like(x) if save_xn else x.new_empty(0)


@opaque(fake=_fake, name='transition_wide_b2b_experimental')
def _forward(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
             rstd: torch.Tensor, c1: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
             bm: int, bn: int, bk: int, warps: int, stages: int,
             normalize: bool, save_xn: bool, packed: int = 0, output_tile: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    selected_launch = launch
    config = dict(BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    if output_tile:
        from .segmented_b2b import launch as selected_launch
        config['BO'] = output_tile
    else:
        config['PACKED'] = packed
    y, xn, _ = selected_launch(x, residual, gamma, beta, rstd, c1, wa, wb, ws,
                      config=config,
                      normalize=normalize, save_xn=save_xn)
    return y, xn


class _NormalizedB2B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xn, wa, wb, ws, residual, config):
        ctx.shape = xn.shape
        ctx.save_for_backward(xn, wa, wb, ws)
        flat = xn.reshape(-1, xn.shape[-1]).contiguous()
        empty = flat.new_empty(0)
        y, _ = _forward(flat, residual.reshape_as(flat), empty, empty, empty, empty, wa, wb, ws,
                         config['BM'], config['BN'], config['BK'], config['num_warps'], config['num_stages'], False, False, int(config.get('PACKED', 0)), config.get('BO', 0))
        return y.reshape(ctx.shape)

    @staticmethod
    def backward(ctx, dy):
        from miniworld_engine.kernels.transition.triton.main import swiglu_squeeze_backward
        xn, wa, wb, ws = ctx.saved_tensors
        grads = swiglu_squeeze_backward(xn.reshape(-1, xn.shape[-1]), wa, wb, ws,
                                       dy.reshape(-1, dy.shape[-1]).contiguous(), ctx.shape, wa.shape[0])
        return (*grads, dy, None)


def transition_wide_b2b(x, gamma, beta, wa, wb, ws, eps, *, config):
    """LN + b2b; existing split backward unchanged, residual fused both ways."""
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual
    xn, residual = input_ln_residual(x, gamma, beta, eps)
    return _NormalizedB2B.apply(xn, wa, wb, ws, residual, config)


class _FusedNormB2B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps, config, save_xn):
        from miniworld_engine.autotune.shape_key import both_key, rows_of
        from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
        ctx.shape, ctx.eps = x.shape, eps
        ctx.key = both_key(rows_of(x.shape))
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        rs, c1 = stats_triton(flat, eps, shape_key=ctx.key)
        y, xn = _forward(flat, flat, gamma, beta, rs, c1, wa, wb, ws,
                         config['BM'], config['BN'], config['BK'], config['num_warps'], config['num_stages'], True, save_xn, int(config.get('PACKED', 0)), config.get('BO', 0))
        ctx.save_for_backward(flat, rs, c1, gamma, beta, wa, wb, ws, xn)
        ctx.has_xn = save_xn
        return y.reshape(ctx.shape)

    @staticmethod
    def backward(ctx, dy):
        from miniworld_engine.kernels.transition.triton.fused import _fused_bwd
        x, rs, c1, gamma, beta, wa, wb, ws, xn = ctx.saved_tensors
        if ctx.has_xn:
            # Match the production Triton backward when comparing forward layouts.
            from .main import swiglu_squeeze_backward
            from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual_bwd
            flat_dy = dy.reshape_as(x).contiguous()
            dxn, dwa, dwb, dws = swiglu_squeeze_backward(
                xn, wa, wb, ws, flat_dy, ctx.shape, wa.shape[0])
            dx, dg, db = input_ln_residual_bwd(
                dxn.reshape_as(x), x, gamma, c1 / rs, rs, flat_dy, ctx.key)
            return dx.reshape(ctx.shape), dg, db, dwa, dwb, dws, None, None, None
        grads = _fused_bwd(dy.contiguous(), x, rs, c1, gamma, beta, wa, wb, ws,
                           xn if ctx.has_xn else None, ctx.eps, ctx.has_xn,
                           list(ctx.shape), ctx.key, True)
        return (*grads, None, None, None)


def transition_wide_b2b_fused_ln(x, gamma, beta, wa, wb, ws, eps, *, config):
    """Stats + LN/expand/gate/squeeze/residual b2b, saving xn only for gradients."""
    save_xn = torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
    return _FusedNormB2B.apply(x, gamma, beta, wa, wb, ws, eps, config, save_xn)
