"""Experimental b2b with separately tiled output accumulators (no repeated expand)."""
import torch
import triton
import triton.language as tl
from .wide_b2b import _operand

@triton.jit
def _segmented(X, R, G, B, RS, C1, WA, WB, WS, Y, XN, M,
               D: tl.constexpr, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, BO: tl.constexpr, NORMALIZE: tl.constexpr, SAVE_XN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dc = tl.arange(0, BO)
    if D > 0 * BO:
        acc0 = tl.zeros((BM, BO), tl.float32)
    if D > 1 * BO:
        acc1 = tl.zeros((BM, BO), tl.float32)
    if D > 2 * BO:
        acc2 = tl.zeros((BM, BO), tl.float32)
    if D > 3 * BO:
        acc3 = tl.zeros((BM, BO), tl.float32)
    if BK >= D:
        kk_full = tl.arange(0, BK)
        xn_full = _operand(X, G, B, RS, C1, rows, kk_full, M, D, NORMALIZE)
        if SAVE_XN:
            tl.store(XN + rows[:, None] * D + kk_full[None, :], xn_full,
                     (rows[:, None] < M) & (kk_full[None, :] < D))
    elif SAVE_XN:
        for k0 in range(0, D, BK):
            kk = k0 + tl.arange(0, BK)
            xn_save = _operand(X, G, B, RS, C1, rows, kk, M, D, NORMALIZE)
            tl.store(XN + rows[:, None] * D + kk[None, :], xn_save,
                     (rows[:, None] < M) & (kk[None, :] < D))
    for h0 in range(0, H, BN):
        nn = h0 + tl.arange(0, BN)
        if BK >= D:
            wa = tl.load(WA + nn[None, :] * D + kk_full[:, None],
                         (nn[None, :] < H) & (kk_full[:, None] < D), 0)
            wb = tl.load(WB + nn[None, :] * D + kk_full[:, None],
                         (nn[None, :] < H) & (kk_full[:, None] < D), 0)
            a = tl.dot(xn_full, wa)
            b = tl.dot(xn_full, wb)
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
                a = tl.dot(xn, wa, a)
                b = tl.dot(xn, wb, b)
        h = (a * tl.sigmoid(a) * b).to(X.dtype.element_ty)
        if D > 0 * BO:
            col0 = dc + 0 * BO
            ws0 = tl.load(WS + col0[None, :] * H + nn[:, None],
                           (col0[None, :] < D) & (nn[:, None] < H), 0)
            acc0 = tl.dot(h, ws0, acc0)
        if D > 1 * BO:
            col1 = dc + 1 * BO
            ws1 = tl.load(WS + col1[None, :] * H + nn[:, None],
                           (col1[None, :] < D) & (nn[:, None] < H), 0)
            acc1 = tl.dot(h, ws1, acc1)
        if D > 2 * BO:
            col2 = dc + 2 * BO
            ws2 = tl.load(WS + col2[None, :] * H + nn[:, None],
                           (col2[None, :] < D) & (nn[:, None] < H), 0)
            acc2 = tl.dot(h, ws2, acc2)
        if D > 3 * BO:
            col3 = dc + 3 * BO
            ws3 = tl.load(WS + col3[None, :] * H + nn[:, None],
                           (col3[None, :] < D) & (nn[:, None] < H), 0)
            acc3 = tl.dot(h, ws3, acc3)
    if D > 0 * BO:
        col0 = dc + 0 * BO
        mask0 = (rows[:, None] < M) & (col0[None, :] < D)
        res0 = tl.load(R + rows[:, None] * D + col0[None, :], mask0, 0).to(tl.float32)
        y0 = acc0.to(Y.dtype.element_ty).to(tl.float32) + res0
        tl.store(Y + rows[:, None] * D + col0[None, :], y0, mask0)
    if D > 1 * BO:
        col1 = dc + 1 * BO
        mask1 = (rows[:, None] < M) & (col1[None, :] < D)
        res1 = tl.load(R + rows[:, None] * D + col1[None, :], mask1, 0).to(tl.float32)
        y1 = acc1.to(Y.dtype.element_ty).to(tl.float32) + res1
        tl.store(Y + rows[:, None] * D + col1[None, :], y1, mask1)
    if D > 2 * BO:
        col2 = dc + 2 * BO
        mask2 = (rows[:, None] < M) & (col2[None, :] < D)
        res2 = tl.load(R + rows[:, None] * D + col2[None, :], mask2, 0).to(tl.float32)
        y2 = acc2.to(Y.dtype.element_ty).to(tl.float32) + res2
        tl.store(Y + rows[:, None] * D + col2[None, :], y2, mask2)
    if D > 3 * BO:
        col3 = dc + 3 * BO
        mask3 = (rows[:, None] < M) & (col3[None, :] < D)
        res3 = tl.load(R + rows[:, None] * D + col3[None, :], mask3, 0).to(tl.float32)
        y3 = acc3.to(Y.dtype.element_ty).to(tl.float32) + res3
        tl.store(Y + rows[:, None] * D + col3[None, :], y3, mask3)

def launch(x, residual, gamma, beta, rstd, c1, wa, wb, ws, *, config,
           normalize=False, save_xn=False, out=None, xn_out=None):
    m,d=x.shape
    if d not in (128,256,384,512) or config['BO'] not in (32,64,128,256) or d>4*config['BO']:
        raise ValueError('segmented b2b requires D128/256/384/512 and at most four output tiles of width 32/64/128/256')
    if wa.shape != (4*d,d) or wb.shape != wa.shape or ws.shape != (d,4*d):
        raise ValueError('segmented b2b requires expansion 4')
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x,residual,wa,wb,ws)):
        raise ValueError('segmented b2b requires contiguous BF16 operands')
    if residual.shape != x.shape or (save_xn and not normalize):
        raise ValueError('invalid residual shape or saved operand mode')
    if config['BM']*triton.cdiv(d,config['BO'])*config['BO'] >= 255*32*config['num_warps']:
        raise ValueError('output accumulator register budget')
    out=torch.empty_like(x) if out is None else out
    xn_out=(torch.empty_like(x) if save_xn else x.new_empty(0)) if xn_out is None else xn_out
    k=_segmented[(triton.cdiv(m,config['BM']),)](x,residual,gamma,beta,rstd,c1,wa,wb,ws,out,xn_out,m,d,4*d,NORMALIZE=normalize,SAVE_XN=save_xn,**config)
    return out,xn_out,k
