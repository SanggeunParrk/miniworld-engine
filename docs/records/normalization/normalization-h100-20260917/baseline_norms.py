import triton
import triton.language as tl

@triton.jit
def _transition_ln_bwd_kernel(
    dxn_ptr, x_ptr, rstd_ptr, c1_ptr, g_ptr, dx_ptr, dg_ptr, db_ptr,
    # K is tl.constexpr (model d, fixed per module, already in this kernel's autotune key) so the
    # `BLOCK_K >= K` guard below resolves at COMPILE time and only one branch is emitted.
    M, K: tl.constexpr, shape_key,
    stride_m, stride_k,
    dg_stride_replica, dg_stride_k,
    db_stride_replica, db_stride_k,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_REPLICAS: tl.constexpr, PRIVATIZE_DGDB: tl.constexpr,
):
    # LayerNorm backward consuming the SAVED stats (rstd, c1=mean*rstd), one pass over K:
    #   x_hat = x*rstd - c1 = (x-mean)*rstd ; wdy = gamma*dxn
    #   dx = rstd*(wdy - mean_k(wdy) - x_hat*mean_k(wdy*x_hat))
    #   dgamma += sum_m(dxn*x_hat) ; dbeta += sum_m(dxn)   (atomic over M-blocks)
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=rmask, other=0.0)
    inv_k = 1.0 / K

    if BLOCK_K >= K:
        # COVERING TILE -> the pre-tiling single-pass schedule: dxn / x / gamma are read ONCE and
        # x_hat + wdy stay in registers for both the row reductions AND the dx epilogue, instead
        # of the two sweeps the general branch needs. Numerics are identical to the else-branch at
        # BLOCK_K >= K (its sweeps are single-trip and ca/cb start from an exact fp32 zero).
        k = tl.arange(0, BLOCK_K)
        kmask = k < K
        mask = rmask[:, None] & kmask[None, :]
        off = rows[:, None] * stride_m + k[None, :] * stride_k
        dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
        x_hat = tl.where(mask, x * rstd[:, None] - c1[:, None], 0.0)
        wdy = tl.where(mask, g[None, :] * dxn, 0.0)
        ca = tl.sum(x_hat * wdy, axis=1) * inv_k
        cb = tl.sum(wdy, axis=1) * inv_k
        dx = (wdy - (x_hat * ca[:, None] + cb[:, None])) * rstd[:, None]
        tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        pdg = tl.sum(dxn * x_hat, axis=0)
        pdb = tl.sum(dxn, axis=0)
        if PRIVATIZE_DGDB:
            replica = pid % NUM_REPLICAS
            tl.atomic_add(dg_ptr + replica * dg_stride_replica + k * dg_stride_k, pdg, mask=kmask)
            tl.atomic_add(db_ptr + replica * db_stride_replica + k * db_stride_k, pdb, mask=kmask)
        else:
            tl.atomic_add(dg_ptr + k, pdg, mask=kmask)
            tl.atomic_add(db_ptr + k, pdb, mask=kmask)
    else:
        # pass A: the two row reductions over ALL of K.
        ca = tl.zeros([BLOCK_M1], dtype=tl.float32)
        cb = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < K
            mask = rmask[:, None] & kmask[None, :]
            off = rows[:, None] * stride_m + k[None, :] * stride_k
            dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
            x_hat = tl.where(mask, x * rstd[:, None] - c1[:, None], 0.0)
            wdy = tl.where(mask, g[None, :] * dxn, 0.0)
            ca += tl.sum(x_hat * wdy, axis=1)
            cb += tl.sum(wdy, axis=1)
        ca = ca * inv_k
        cb = cb * inv_k

        # pass B: dx, plus the dgamma/dbeta column partials.
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < K
            mask = rmask[:, None] & kmask[None, :]
            off = rows[:, None] * stride_m + k[None, :] * stride_k
            dxn = tl.load(dxn_ptr + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + k, mask=kmask, other=0.0).to(tl.float32)
            x_hat = tl.where(mask, x * rstd[:, None] - c1[:, None], 0.0)
            wdy = tl.where(mask, g[None, :] * dxn, 0.0)
            dx = (wdy - (x_hat * ca[:, None] + cb[:, None])) * rstd[:, None]
            tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=mask)
            pdg = tl.sum(dxn * x_hat, axis=0)
            pdb = tl.sum(dxn, axis=0)
            if PRIVATIZE_DGDB:
                replica = pid % NUM_REPLICAS
                tl.atomic_add(dg_ptr + replica * dg_stride_replica + k * dg_stride_k, pdg, mask=kmask)
                tl.atomic_add(db_ptr + replica * db_stride_replica + k * db_stride_k, pdb, mask=kmask)
            else:
                tl.atomic_add(dg_ptr + k, pdg, mask=kmask)
                tl.atomic_add(db_ptr + k, pdb, mask=kmask)

@triton.jit
def _dgrad_condln_kernel(
    D, Wcat, Cond, MeanC, RstdC, LNW, DCond, DLNW, M, NC: tl.constexpr, K2,
    sd0, sd1, sw0, sw1, sc0, sc1, sdc0, sdc1,
    BLOCK_M1: tl.constexpr, BLOCK_K_NC: tl.constexpr, BLOCK_K_K2: tl.constexpr, shape_key):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    mean = tl.load(MeanC + rm, mask=rmask, other=0.0)[:, None]
    rstd = tl.load(RstdC + rm, mask=rmask, other=0.0)[:, None]
    inv_n = 1.0 / NC

    # TWO-PASS over the NC tiles. The cond LN-backward needs c1/c2 — reductions over the WHOLE
    # cond row of dcond_aff — before any element of dcond can be written, so once the NC axis is
    # tiled, dcond_aff has to be visited twice. Here the "row" being re-read is the in-kernel GEMM
    # result, so pass 2 RECOMPUTES the Dᵀ@Wcat tile rather than spilling it: the only alternative
    # was staging dcond_aff through the bf16 DCond buffer, which would round the gradient to bf16
    # before the LN-backward algebra. Correctness beats the extra MMA pass.
    #
    # dlnw and both c1/c2 are accumulated in pass 1 (they need acc but not c1/c2), so pass 2 only
    # redoes the GEMM and writes dcond.
    #
    # COVERING TILE (BLOCK_K_NC >= NC): the two NC loops are single-trip, so pass 2 recomputed the
    # SAME Dᵀ@Wcat tile — a second full pass of MMA plus a second read of D and Wcat, and neither
    # was CSE'd (the DLNW tl.atomic_add sits between them). `NC` is `tl.constexpr` (already this
    # kernel's autotune key, so a new d_cond already forced a re-tune and a fresh compile) which
    # makes the guard a TRACE-time comparison: one branch is emitted, and the fast path keeps the
    # single `acc` in REGISTERS across the c1/c2 reduction and the LN-backward algebra. That is the
    # point for this kernel — the recompute existed only to avoid staging dcond_aff through the
    # bf16 DCond buffer, and holding acc in fp32 registers avoids both. `reset_to_zero=["DLNW"]` is
    # unchanged and the fast path issues the same single tl.atomic_add per program.
    if BLOCK_K_NC >= NC:
        nc = tl.arange(0, BLOCK_K_NC)
        ncmask = nc < NC
        nmask2 = rmask[:, None] & ncmask[None, :]
        # in-kernel GEMM: dcond_aff[m,n] = Σ_k2 D[k2,m]·Wcat[k2,n] (= Dᵀ@Wcat). D=(2NX,M): a-tile
        # reads D[k,m] at m*sd1 + k*sd0 (m-contiguous since sd1=1), Wcat[k,n] at k*sw0 + n*sw1.
        acc = tl.zeros((BLOCK_M1, BLOCK_K_NC), dtype=tl.float32)
        for k0 in range(0, K2, BLOCK_K_K2):
            kk = k0 + tl.arange(0, BLOCK_K_K2)
            kmask = kk < K2
            a = tl.load(D + rm[:, None] * sd1 + kk[None, :] * sd0,
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            b = tl.load(Wcat + kk[:, None] * sw0 + nc[None, :] * sw1,
                        mask=kmask[:, None] & ncmask[None, :], other=0.0)
            acc += tl.dot(a, b, input_precision="tf32")
        cond = tl.load(Cond + rm[:, None] * sc0 + nc[None, :] * sc1,
                       mask=nmask2, other=0.0).to(tl.float32)
        g_w = tl.load(LNW + nc, mask=ncmask, other=0.0).to(tl.float32)[None, :]
        cnorm = tl.where(ncmask[None, :], (cond - mean) * rstd, 0.0)
        dxhat = acc * g_w
        c2 = tl.sum(tl.where(ncmask[None, :], dxhat, 0.0), axis=1) * inv_n
        c1 = tl.sum(tl.where(ncmask[None, :], dxhat * cnorm, 0.0), axis=1) * inv_n
        pdg = tl.sum(tl.where(nmask2, acc * cnorm, 0.0), axis=0)   # dlnw = Σ_m dcond_aff·cond̂
        tl.atomic_add(DLNW + nc, pdg, mask=ncmask)
        # cond LayerNorm backward (affine γ=lnw, no β) on the in-register dcond_aff.
        dcond = rstd * (dxhat - c2[:, None] - cnorm * c1[:, None])
        tl.store(DCond + rm[:, None] * sdc0 + nc[None, :] * sdc1,
                 dcond.to(DCond.dtype.element_ty), mask=nmask2)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, NC, BLOCK_K_NC):
            nc = n0 + tl.arange(0, BLOCK_K_NC)
            ncmask = nc < NC
            nmask2 = rmask[:, None] & ncmask[None, :]
            # in-kernel GEMM: dcond_aff[m,n] = Σ_k2 D[k2,m]·Wcat[k2,n] (= Dᵀ@Wcat). D=(2NX,M):
            # a-tile reads D[k,m] at m*sd1 + k*sd0 (m-contiguous since sd1=1), Wcat[k,n] at
            # k*sw0 + n*sw1.
            acc = tl.zeros((BLOCK_M1, BLOCK_K_NC), dtype=tl.float32)
            for k0 in range(0, K2, BLOCK_K_K2):
                kk = k0 + tl.arange(0, BLOCK_K_K2)
                kmask = kk < K2
                a = tl.load(D + rm[:, None] * sd1 + kk[None, :] * sd0,
                            mask=rmask[:, None] & kmask[None, :], other=0.0)
                b = tl.load(Wcat + kk[:, None] * sw0 + nc[None, :] * sw1,
                            mask=kmask[:, None] & ncmask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision="tf32")
            cond = tl.load(Cond + rm[:, None] * sc0 + nc[None, :] * sc1,
                           mask=nmask2, other=0.0).to(tl.float32)
            g_w = tl.load(LNW + nc, mask=ncmask, other=0.0).to(tl.float32)[None, :]
            cnorm = tl.where(ncmask[None, :], (cond - mean) * rstd, 0.0)
            dxhat = acc * g_w
            c2 += tl.sum(tl.where(ncmask[None, :], dxhat, 0.0), axis=1)
            c1 += tl.sum(tl.where(ncmask[None, :], dxhat * cnorm, 0.0), axis=1)
            pdg = tl.sum(tl.where(nmask2, acc * cnorm, 0.0), axis=0)   # dlnw = Σ_m dcond_aff·cond̂
            tl.atomic_add(DLNW + nc, pdg, mask=ncmask)
        c1 *= inv_n   # scale once at the end, as the untiled kernel did
        c2 *= inv_n

        # cond LayerNorm backward (affine γ=lnw, no β) on the recomputed dcond_aff.
        for n0 in range(0, NC, BLOCK_K_NC):
            nc = n0 + tl.arange(0, BLOCK_K_NC)
            ncmask = nc < NC
            nmask2 = rmask[:, None] & ncmask[None, :]
            acc = tl.zeros((BLOCK_M1, BLOCK_K_NC), dtype=tl.float32)
            for k0 in range(0, K2, BLOCK_K_K2):
                kk = k0 + tl.arange(0, BLOCK_K_K2)
                kmask = kk < K2
                a = tl.load(D + rm[:, None] * sd1 + kk[None, :] * sd0,
                            mask=rmask[:, None] & kmask[None, :], other=0.0)
                b = tl.load(Wcat + kk[:, None] * sw0 + nc[None, :] * sw1,
                            mask=kmask[:, None] & ncmask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision="tf32")
            cond = tl.load(Cond + rm[:, None] * sc0 + nc[None, :] * sc1,
                           mask=nmask2, other=0.0).to(tl.float32)
            g_w = tl.load(LNW + nc, mask=ncmask, other=0.0).to(tl.float32)[None, :]
            cnorm = tl.where(ncmask[None, :], (cond - mean) * rstd, 0.0)
            dxhat = acc * g_w
            dcond = rstd * (dxhat - c2[:, None] - cnorm * c1[:, None])
            tl.store(DCond + rm[:, None] * sdc0 + nc[None, :] * sdc1,
                     dcond.to(DCond.dtype.element_ty), mask=nmask2)

@triton.jit
def _layer_norm_linear_bwd(
    dout_ptr,
    x_ptr,
    lnw_ptr,
    pw_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dlnw_ptr,
    dpw_ptr,
    stride_m,
    M,
    N: tl.constexpr,
    NH: tl.constexpr,
    USE_DOT: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K_D: tl.constexpr,
    BLOCK_K_NH: tl.constexpr,
    shape_key,
):
    # BLOCK_K_D tiles d, BLOCK_K_NH tiles n_head. dx needs the row sums c1/c2 over ALL of d, so the
    # channel axis is walked TWICE: pass A accumulates c1/c2 (and emits the dpw/dlnw atomics),
    # pass B forms dx. At BLOCK_K_D >= N each loop is one iteration = the original single-tile
    # schedule, with the second pass reading an L2-hot row.
    rows = (tl.program_id(0) * BLOCK_M1 + tl.arange(0, BLOCK_M1)).to(tl.int64)
    row_mask = rows < M
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)

    c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
    c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_K_D):
        cols = n0 + tl.arange(0, BLOCK_K_D)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * stride_m + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = xhat * lnw[None, :]
        dy = tl.zeros((BLOCK_M1, BLOCK_K_D), dtype=tl.float32)
        if USE_DOT:
            for h0 in range(0, NH, BLOCK_K_NH):
                hcols = h0 + tl.arange(0, BLOCK_K_NH)
                h_mask = hcols < NH
                dout = tl.load(
                    dout_ptr + rows[:, None] * NH + hcols[None, :],
                    mask=row_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                pw = tl.load(
                    pw_ptr + hcols[:, None] * N + cols[None, :],
                    mask=h_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                dy = tl.dot(dout, pw, dy, allow_tf32=False)
                tl.atomic_add(
                    dpw_ptr + hcols[:, None] * N + cols[None, :],
                    tl.dot(tl.trans(dout), y, allow_tf32=False),
                    mask=h_mask[:, None] & col_mask[None, :],
                )
        else:
            for j in tl.static_range(NH):
                dout_j = tl.load(dout_ptr + rows * NH + j, mask=row_mask, other=0.0)
                pw = tl.load(pw_ptr + j * N + cols, mask=col_mask, other=0.0).to(tl.float32)
                dy += dout_j[:, None] * pw[None, :]
                tl.atomic_add(
                    dpw_ptr + j * N + cols,
                    tl.sum(dout_j[:, None] * y, axis=0),
                    mask=col_mask,
                )
        dy = tl.where(mask, dy, 0.0)
        dxhat = dy * lnw[None, :]
        c1 += tl.sum(dxhat * xhat, axis=1)
        c2 += tl.sum(dxhat, axis=1)
        tl.atomic_add(dlnw_ptr + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
    c1 = c1 / N
    c2 = c2 / N

    for n0 in range(0, N, BLOCK_K_D):
        cols = n0 + tl.arange(0, BLOCK_K_D)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * stride_m + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        dy = tl.zeros((BLOCK_M1, BLOCK_K_D), dtype=tl.float32)
        if USE_DOT:
            for h0 in range(0, NH, BLOCK_K_NH):
                hcols = h0 + tl.arange(0, BLOCK_K_NH)
                h_mask = hcols < NH
                dout = tl.load(
                    dout_ptr + rows[:, None] * NH + hcols[None, :],
                    mask=row_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                pw = tl.load(
                    pw_ptr + hcols[:, None] * N + cols[None, :],
                    mask=h_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                dy = tl.dot(dout, pw, dy, allow_tf32=False)
        else:
            for j in tl.static_range(NH):
                dout_j = tl.load(dout_ptr + rows * NH + j, mask=row_mask, other=0.0)
                pw = tl.load(pw_ptr + j * N + cols, mask=col_mask, other=0.0).to(tl.float32)
                dy += dout_j[:, None] * pw[None, :]
        dy = tl.where(mask, dy, 0.0)
        dxhat = dy * lnw[None, :]
        dx = (dxhat - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(dx_ptr + offs, dx, mask=mask)

@triton.jit
def rmsnorm_bwd_kernel(
    DX, DY, DW, X, W, Rstd,
    stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr,
):
    """``dx`` and, with a weight, ``dweight``.

    ``dweight`` is fp32 and accumulated across row tiles with ``atomic_add``, so its buffer is
    zeroed by the caller (not merely by the autotuner's ``reset_to_zero``, which fires only while
    a config is being benchmarked).
    """
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M
    rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)

    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                     mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, x * rstd[:, None], 0.0)
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
            wdy = tl.where(mask, dy * w[None, :], 0.0)
        else:
            wdy = tl.where(mask, dy, 0.0)
        # No c2: without the mean there is only the rstd term in the derivative.
        c1 = tl.sum(xhat * wdy, axis=1) / N
        dx = (wdy - xhat * c1[:, None]) * rstd[:, None]
        tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, x * rstd[:, None], 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
                wdy = tl.where(mask, dy * w[None, :], 0.0)
            else:
                wdy = tl.where(mask, dy, 0.0)
            c1 += tl.sum(xhat * wdy, axis=1)
        c1 = c1 / N

        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, x * rstd[:, None], 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                wdy = tl.where(mask, dy * w[None, :], 0.0)
            else:
                wdy = tl.where(mask, dy, 0.0)
            dx = (wdy - xhat * c1[:, None]) * rstd[:, None]
            tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)

@triton.jit
def rmsnorm_adamod_bwd_kernel(
    DQ, DSD, DW, DY, Q, C, WSC, W, Rstd,
    stride_qr, stride_qc, stride_cr, stride_cc, stride_wn, stride_wk,
    stride_sr, stride_sc,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_WEIGHT: tl.constexpr,
):
    """``dq``, ``dscale`` and ``dweight`` for the fused modulate. ``dshift`` IS ``dy``.

    ``scale`` is RECOMPUTED here (``c @ Wsc^T``) rather than saved: saving it is the 192 MB the
    forward fusion exists to not spend, so the backward pays one GEMM instead. The three large
    GEMMs the chain still needs -- ``dWsc``, ``dWsh``, ``dc`` -- stay in the caller, on cuBLAS,
    which is better at them than a hand-tiled `tl.dot` and needs no scratch here.

    The covering-tile branch (BLOCK_N >= N, which is the atom-DiT d_model of 128) keeps
    everything in registers and recomputes nothing twice. The tiled branch cannot: ``c1`` reduces
    over the whole row, so the GEMM is recomputed in the second pass rather than spilling
    ``dnormed`` to a [M, N] scratch buffer.
    """
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M
    rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)

    if BLOCK_N >= N:
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        acc_sc = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                          mask=k_mask[:, None] & col_mask[None, :], other=0.0)
            acc_sc += tl.dot(c, wsc, input_precision="ieee")
        q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                    mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                     mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, q * rstd[:, None], 0.0)
        normed = xhat
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
            normed = xhat * w[None, :]
        # dscale and dy go into ONE [M, 2N] buffer, side by side. Both are already in
        # registers here, so the second store is the whole cost of a concatenation the
        # caller would otherwise pay for -- and it turns the caller's four GEMMs into two:
        #   dWpair = [dscale|dy]^T @ cs      (was dWsc and dWsh separately)
        #   dc     = [dscale|dy] @ [Wsc;Wsh] (was two [M,N]@[N,K] and an add)
        tl.store(DSD + rows[:, None] * stride_sr + cols[None, :] * stride_sc,
                 dy * normed, mask=mask)
        tl.store(DSD + rows[:, None] * stride_sr + (N + cols[None, :]) * stride_sc,
                 dy, mask=mask)
        dnormed = tl.where(mask, dy * (1.0 + acc_sc), 0.0)
        if HAS_WEIGHT:
            tl.atomic_add(DW + cols, tl.sum(dnormed * xhat, axis=0), mask=col_mask)
            wdy = tl.where(mask, dnormed * w[None, :], 0.0)
        else:
            wdy = dnormed
        c1 = tl.sum(xhat * wdy, axis=1) / N
        dq = (wdy - xhat * c1[:, None]) * rstd[:, None]
        tl.store(DQ + rows[:, None] * stride_qr + cols[None, :] * stride_qc, dq, mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            acc_sc = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                              mask=k_mask[:, None] & col_mask[None, :], other=0.0)
                acc_sc += tl.dot(c, wsc, input_precision="ieee")
            q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, q * rstd[:, None], 0.0)
            normed = xhat
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                normed = xhat * w[None, :]
            # dscale and dy go into ONE [M, 2N] buffer, side by side. Both are already in
            # registers here, so the second store is the whole cost of a concatenation the
            # caller would otherwise pay for -- and it turns the caller's four GEMMs into two:
            #   dWpair = [dscale|dy]^T @ cs      (was dWsc and dWsh separately)
            #   dc     = [dscale|dy] @ [Wsc;Wsh] (was two [M,N]@[N,K] and an add)
            tl.store(DSD + rows[:, None] * stride_sr + cols[None, :] * stride_sc,
                     dy * normed, mask=mask)
            tl.store(DSD + rows[:, None] * stride_sr + (N + cols[None, :]) * stride_sc,
                     dy, mask=mask)
            dnormed = tl.where(mask, dy * (1.0 + acc_sc), 0.0)
            if HAS_WEIGHT:
                tl.atomic_add(DW + cols, tl.sum(dnormed * xhat, axis=0), mask=col_mask)
                wdy = tl.where(mask, dnormed * w[None, :], 0.0)
            else:
                wdy = dnormed
            c1 += tl.sum(xhat * wdy, axis=1)
        c1 = c1 / N

        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            acc_sc = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                c = tl.load(C + rows[:, None] * stride_cr + ks[None, :] * stride_cc,
                            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                wsc = tl.load(WSC + cols[None, :] * stride_wn + ks[:, None] * stride_wk,
                              mask=k_mask[:, None] & col_mask[None, :], other=0.0)
                acc_sc += tl.dot(c, wsc, input_precision="ieee")
            q = tl.load(Q + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                        mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_qr + cols[None, :] * stride_qc,
                         mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, q * rstd[:, None], 0.0)
            dnormed = tl.where(mask, dy * (1.0 + acc_sc), 0.0)
            if HAS_WEIGHT:
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                wdy = tl.where(mask, dnormed * w[None, :], 0.0)
            else:
                wdy = dnormed
            dq = (wdy - xhat * c1[:, None]) * rstd[:, None]
            tl.store(DQ + rows[:, None] * stride_qr + cols[None, :] * stride_qc, dq, mask=mask)
