"""OuterProductMean in Triton, forward and backward -- the portable path (any GPU Triton supports).

The decomposition keeps the module's statements and fuses only where the PyTorch path pays extra memory passes:

  forward   layernorm_gemm   LN(msa) -> left / right projections -> mask                       a, b [S, L, CH] bf16
            outer product    O = a^T b in the grouped layout [(i, d), (j, e)]  (cuBLAS: the one layout that is a matmul)
            counts           n = mask^T mask (fp32, clamp 1)
            epilogue         out[i, j] = W_out . vec(O_ij) / n_ij + bias (+ residual)   the [i,j,d,e] permute folded into the loads
  backward  bwd_epilogue_dx  dzn = dz / n, dO = dzn . W_out, written in the grouped layout
            dA, dB           dA = b . dO^T, dB = a . dO  (cuBLAS)
            bwd_epilogue_dw  dW_out = sum_ij dzn_ij (x) O_ij off the kept grouped O  (M = c, N = (d, e), K split over i)
            bwd_layernorm_gemm   mask, dy = [dA | dB] W, dW_left / dW_right, LayerNorm backward -> dmsa, dgamma, dbeta

Every tile, warp count, stage count and persistent program count is an autotune axis (``autotune/configs/grid/<op>.csv``);
the constants in the kernels are the model dimensions (constexpr). Offsets that scale with L^2 CH^2 or S L are 64-bit.
``normalize_before_proj=True`` (AF3 order) divides before the projection; False (ESMFold2) divides after it, bias included.

d_msa and d_hidden must be powers of two >= 16 (``tl.arange`` over the whole row); d_pair only has to be a multiple of 32,
because it is tiled, never spanned (the smallest output tile of the ladder is 32). ``refusal`` says which.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import length_of, token_key
from miniworld_engine.kernels._compile import opaque


def _divides(**axes):
    """``early_config_prune``: keep the configs whose tile axis ``ax`` divides the kernel argument ``axes[ax]`` names (a tile over
    an unmasked extent must divide it). A correctness exclusion only; performance is the autotuner's to decide."""
    def prune(configs, named_args, **kw):
        args = {**named_args, **kw}
        keep = [c for c in configs if all(args[dim] % c.kwargs[ax] == 0 for ax, dim in axes.items())]
        if not keep:
            raise ValueError(f"no config tiles {', '.join(f'{d}={args[d]}' for d in axes.values())}")
        return keep
    return prune


def _nsm(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------------------------------- kernels
@triton.autotune(configs=configs_for("outer_product_mean_layernorm_gemm_triton"), key=["shape_key", "HAS_MASK"])
@triton.jit
def _opm_layernorm_gemm_kernel(M, MASK, LNW, LNB, WL, WR, A, B, T_, eps, shape_key,
                               CM: tl.constexpr, CH: tl.constexpr, HAS_MASK: tl.constexpr, BT: tl.constexpr):
    # a[t] = mask[t] * LN(msa[t]) . W_left^T, b likewise; one program = BT rows of the flattened [S L] tokens
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    rm = rows < T_
    cols = tl.arange(0, CM)
    x = tl.load(M + rows[:, None] * CM + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    mu = tl.sum(x, axis=1) / CM
    xc = x - mu[:, None]
    rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / CM + eps)
    y = (xc * rs[:, None] * tl.load(LNW + cols)[None, :] + tl.load(LNB + cols)[None, :]).to(tl.bfloat16)
    ch = tl.arange(0, CH)
    wl = tl.load(WL + ch[None, :] * CM + cols[:, None])                                   # [CM, CH] = W_left^T
    wr = tl.load(WR + ch[None, :] * CM + cols[:, None])
    a = tl.dot(y, wl).to(tl.bfloat16)
    b = tl.dot(y, wr).to(tl.bfloat16)
    if HAS_MASK:
        mk = tl.load(MASK + rows, mask=rm, other=0) != 0
        a = tl.where(mk[:, None], a, 0.0)
        b = tl.where(mk[:, None], b, 0.0)
    tl.store(A + rows[:, None] * CH + ch[None, :], a, mask=rm[:, None])
    tl.store(B + rows[:, None] * CH + ch[None, :], b, mask=rm[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_epilogue_triton"), key=["shape_key", "HAS_RES", "NORM_FIRST"],
                 prune_configs_by={"early_config_prune": _divides(BN="CZ", BKD="CH")})
@triton.jit
def _opm_epilogue_kernel(O, WOT, BIAS, NORM, RES, OUT, L, shape_key,
                         CH: tl.constexpr, CZ: tl.constexpr, HAS_RES: tl.constexpr, NORM_FIRST: tl.constexpr,
                         TI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr, BKD: tl.constexpr):
    # tile = TI rows i x BJ columns j (M = TI BJ pairs, so W_out is re-read once per TI BJ pairs) x BN outputs; K = CH^2 in steps of
    # BKD values of d (a K chunk of BKD CH): A[(i, j), (d, e)] = O[i CH + d, j CH + e]
    r = tl.arange(0, TI * BJ)
    i = tl.program_id(0) * TI + r // BJ
    j = tl.program_id(1) * BJ + r % BJ
    rm = (i < L) & (j < L)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BKD * CH)
    dd = kk // CH
    e = kk % CH
    ldo = L.to(tl.int64) * CH
    arow = i.to(tl.int64) * CH * ldo + j.to(tl.int64) * CH
    acc = tl.zeros((TI * BJ, BN), dtype=tl.float32)
    for d0 in range(0, CH, BKD):
        a = tl.load(O + arow[:, None] + ((d0 + dd).to(tl.int64) * ldo + e)[None, :], mask=rm[:, None], other=0.0)
        w = tl.load(WOT + ((d0 + dd) * CH + e)[:, None] * CZ + n[None, :])
        acc += tl.dot(a, w)
    nrm = tl.load(NORM + i.to(tl.int64) * L + j, mask=rm, other=1.0)
    bias = tl.load(BIAS + n)
    if NORM_FIRST:
        z = acc / nrm[:, None] + bias[None, :]
    else:
        z = (acc + bias[None, :]) / nrm[:, None]
    ptr = (i.to(tl.int64) * L + j)[:, None] * CZ + n[None, :]
    if HAS_RES:
        z += tl.load(RES + ptr, mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(OUT + ptr, z.to(tl.bfloat16), mask=rm[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_bwd_epilogue_dx_triton"), key=["shape_key"],
                 prune_configs_by={"early_config_prune": _divides(BKC="CZ", BN="CH2")})
@triton.jit
def _opm_bwd_epilogue_dx_kernel(DZ, NORM, WO, DZN, DO, L, CH2, shape_key,
                                CH: tl.constexpr, CZ: tl.constexpr, TI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
                                BKC: tl.constexpr):
    # dzn = dz / n (either order: the projection is linear), dO[(i, d), (j, e)] = sum_c dzn[i, j, c] W_out[c, d CH + e].
    # tile = TI i x BJ j pairs x BN of the CH^2 (d, e) outputs, K = CZ in BKC steps; the n-block-0 programs also store dzn.
    r = tl.arange(0, TI * BJ)
    i = tl.program_id(0) * TI + r // BJ
    j = tl.program_id(1) * BJ + r % BJ
    rm = (i < L) & (j < L)
    pn = tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    d = n // CH
    e = n % CH
    nrm = tl.load(NORM + i.to(tl.int64) * L + j, mask=rm, other=1.0)
    prow = (i.to(tl.int64) * L + j) * CZ
    acc = tl.zeros((TI * BJ, BN), dtype=tl.float32)
    for c0 in range(0, CZ, BKC):
        c = c0 + tl.arange(0, BKC)
        dz = tl.load(DZ + prow[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        dzn = (dz / nrm[:, None]).to(tl.bfloat16)
        if pn == 0:
            tl.store(DZN + prow[:, None] + c[None, :], dzn, mask=rm[:, None])
        w = tl.load(WO + c[:, None] * (CH * CH) + n[None, :])                                # [BKC, BN]
        acc += tl.dot(dzn, w)
    ldo = L.to(tl.int64) * CH
    orow = i.to(tl.int64) * CH * ldo + j.to(tl.int64) * CH
    tl.store(DO + orow[:, None] + (d.to(tl.int64) * ldo + e)[None, :], acc.to(tl.bfloat16), mask=rm[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_bwd_epilogue_dw_triton"), key=["shape_key"], reset_to_zero=["DWO"],
                 prune_configs_by={"early_config_prune": _divides(BM="CZ", BN="CH2")})
@triton.jit
def _opm_bwd_epilogue_dw_kernel(DZN, O, DWO, L, CH2, shape_key,
                                CH: tl.constexpr, CZ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                SPLIT: tl.constexpr):
    # dWo[c, (d, e)] = sum_(i, j) dzn[i, j, c] O[i CH + d, j CH + e]: an M = CZ, N = CH^2, K = L^2 GEMM, tile BM x BN, K split over
    # SPLIT ranges of i (fp32 atomics once per program)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    d = n // CH
    e = n % CH
    sp = tl.program_id(2)
    i_per = tl.cdiv(L, SPLIT)
    ldo = L.to(tl.int64) * CH
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ii in range(i_per):
        i = sp * i_per + ii
        if i < L:
            for j0 in range(0, L, BK):
                j = j0 + kk
                jm = j < L
                a = tl.load(DZN + (i.to(tl.int64) * L + j)[:, None] * CZ + m[None, :], mask=jm[:, None], other=0.0)     # [BK, BM]
                b = tl.load(O + i.to(tl.int64) * CH * ldo + j[:, None].to(tl.int64) * CH + (d.to(tl.int64) * ldo + e)[None, :],
                            mask=jm[:, None], other=0.0)                                                             # [BK, BN]
                acc += tl.dot(tl.trans(a), b)
    tl.atomic_add(DWO + m[:, None] * (CH * CH) + n[None, :], acc)


@triton.autotune(configs=configs_for("outer_product_mean_bwd_layernorm_gemm_triton"), key=["shape_key", "HAS_MASK"],
                 reset_to_zero=["DWL", "DWR", "DG", "DBETA"])
@triton.jit
def _opm_bwd_layernorm_gemm_kernel(M, MASK, LNW, LNB, WL, WR, DA, DB, DM, DWL, DWR, DG, DBETA, T_, eps, shape_key,
                                   CM: tl.constexpr, CH: tl.constexpr, HAS_MASK: tl.constexpr, BT: tl.constexpr, PPS: tl.constexpr):
    # persistent (PPS programs per SM): the program walks token blocks pid, pid + nprog, ..., keeping dW_left / dW_right / dgamma /
    # dbeta in registers, and adds them once at the end
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, CM)
    ch = tl.arange(0, CH)
    g = tl.load(LNW + cols)
    bta = tl.load(LNB + cols)
    wl = tl.load(WL + ch[:, None] * CM + cols[None, :])                                   # [CH, CM]
    wr = tl.load(WR + ch[:, None] * CM + cols[None, :])
    dwl = tl.zeros((CH, CM), dtype=tl.float32)
    dwr = tl.zeros((CH, CM), dtype=tl.float32)
    dgs = tl.zeros((CM,), dtype=tl.float32)
    dbs = tl.zeros((CM,), dtype=tl.float32)
    nblk = tl.cdiv(T_, BT)
    for blk in range(pid, nblk, nprog):
        rows = blk.to(tl.int64) * BT + tl.arange(0, BT)
        rm = rows < T_
        x = tl.load(M + rows[:, None] * CM + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(x, axis=1) / CM
        xc = x - mu[:, None]
        rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / CM + eps)
        xh = xc * rs[:, None]
        y = (xh * g[None, :] + bta[None, :]).to(tl.bfloat16)
        da = tl.load(DA + rows[:, None] * CH + ch[None, :], mask=rm[:, None], other=0.0)
        db = tl.load(DB + rows[:, None] * CH + ch[None, :], mask=rm[:, None], other=0.0)
        if HAS_MASK:
            mk = tl.load(MASK + rows, mask=rm, other=0) != 0
            da = tl.where(mk[:, None], da, 0.0).to(tl.bfloat16)
            db = tl.where(mk[:, None], db, 0.0).to(tl.bfloat16)
        dwl += tl.dot(tl.trans(da), y)
        dwr += tl.dot(tl.trans(db), y)
        dy = tl.dot(da, wl) + tl.dot(db, wr)                                                 # [BT, CM] fp32
        dgs += tl.sum(dy * xh, axis=0)
        dbs += tl.sum(dy, axis=0)
        dxh = dy * g[None, :]
        m1 = tl.sum(dxh, axis=1) / CM
        m2 = tl.sum(dxh * xh, axis=1) / CM
        dx = rs[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
        tl.store(DM + rows[:, None] * CM + cols[None, :], dx.to(tl.bfloat16), mask=rm[:, None])
    tl.atomic_add(DWL + ch[:, None] * CM + cols[None, :], dwl)
    tl.atomic_add(DWR + ch[:, None] * CM + cols[None, :], dwr)
    tl.atomic_add(DG + cols, dgs)
    tl.atomic_add(DBETA + cols, dbs)


# ---------------------------------------------------------------------------------------------------- launches (opaque to Dynamo)
def _pow2_16(n: int) -> bool:
    return n >= 16 and n & (n - 1) == 0


def refusal(msa: torch.Tensor, d_hidden: int, d_pair: int, *, interchain: bool, residual: torch.Tensor | None = None) -> str | None:
    """None if the Triton path can run this OuterProductMean call, else why it cannot. Never raises."""
    try:
        if interchain:
            return "mask_interchain is applied after the projection and is not fused here"
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        if msa.dtype != torch.bfloat16:
            return f"the Triton path is bf16, got {msa.dtype}"
        if residual is not None and (residual.dtype != torch.bfloat16 or residual.shape != (msa.shape[0], msa.shape[2], msa.shape[2], d_pair)):
            return "the pair residual must be bf16 and [B, L, L, d_pair] to be fused"
        d_msa = msa.shape[-1]
        if not (_pow2_16(d_msa) and _pow2_16(d_hidden)):
            return f"d_msa and d_hidden must be powers of two >= 16 (spanned by one tile), got {d_msa}, {d_hidden}"
        if d_pair % 32:
            return f"d_pair must be a multiple of 32 (the smallest output tile), got {d_pair}"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _fwd_fake(msa, mask, lnw, lnb, wl, wr, wo, bo, residual, eps, norm_first):
    """Shapes of `_fwd`'s outputs: out [L, L, CZ], a / b [S L, CH], O [L CH, L CH] bf16, n [L, L] fp32."""
    S, L, _ = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    new = lambda *shape, dt=torch.bfloat16: msa.new_empty(shape, dtype=dt)  # noqa: E731
    return [new(L, L, CZ), new(S * L, CH), new(S * L, CH), new(L * CH, L * CH), new(L, L, dt=torch.float32)]


@opaque(fake=_fwd_fake, name="outer_product_mean_triton_fwd")
def _fwd(msa: torch.Tensor, mask: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor,
         wo: torch.Tensor, bo: torch.Tensor, residual: torch.Tensor, eps: float, norm_first: bool) -> list[torch.Tensor]:
    """One stack: msa [S, L, CM], mask [S, L] bool (or empty: no mask), residual [L, L, CZ] (or empty).
    Returns [out [L, L, CZ], a, b [S L, CH], O [L CH, L CH], n [L, L]]; the last four are what the backward reads."""
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    T_ = S * L
    bf = torch.bfloat16
    has_mask, has_res = mask.numel() > 0, residual.numel() > 0
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if has_mask else x
    w_l, w_r = wl.to(bf).contiguous(), wr.to(bf).contiguous()
    a = msa.new_empty((T_, CH), dtype=bf)
    b = msa.new_empty((T_, CH), dtype=bf)
    _opm_layernorm_gemm_kernel[lambda meta: (triton.cdiv(T_, meta["BT"]),)](
        x, m8, lnw.float().contiguous(), lnb.float().contiguous(), w_l, w_r, a, b, T_, eps,
        shape_key=token_key(length_of(msa.shape), CM=CM, CH=CH), CM=CM, CH=CH, HAS_MASK=has_mask)
    o = torch.mm(a.view(S, L * CH).t(), b.view(S, L * CH))                                  # [(i, d), (j, e)]
    if has_mask:
        mf = mask.reshape(S, L).float()
        norm = (mf.t() @ mf).clamp_(min=1)
    else:
        norm = msa.new_full((L, L), float(S), dtype=torch.float32)
    out = msa.new_empty((L, L, CZ), dtype=bf)
    res = residual.reshape(L * L, CZ).contiguous() if has_res else out
    _opm_epilogue_kernel[lambda meta: (triton.cdiv(L, meta["TI"]), triton.cdiv(L, meta["BJ"]), triton.cdiv(CZ, meta["BN"]))](
        o, wo.to(bf).t().contiguous(), bo.float().contiguous(), norm, res, out, L,
        shape_key=token_key(length_of(msa.shape), CH=CH, CZ=CZ), CH=CH, CZ=CZ, HAS_RES=has_res, NORM_FIRST=norm_first)
    return [out, a, b, o, norm]


def _bwd_fake(dz, msa, mask, a, b, o, norm, lnw, lnb, wl, wr, wo, eps, norm_first):
    """Shapes of `_bwd`'s outputs: dmsa like msa, every parameter gradient fp32 in its parameter's shape."""
    f32 = torch.float32
    return [torch.empty_like(msa), lnw.new_empty(lnw.shape, dtype=f32), lnb.new_empty(lnb.shape, dtype=f32),
            wl.new_empty(wl.shape, dtype=f32), wr.new_empty(wr.shape, dtype=f32), wo.new_empty(wo.shape, dtype=f32),
            wo.new_empty((wo.shape[0],), dtype=f32)]


@opaque(fake=_bwd_fake, name="outer_product_mean_triton_bwd")
def _bwd(dz: torch.Tensor, msa: torch.Tensor, mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor, o: torch.Tensor,
         norm: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor, wo: torch.Tensor, eps: float,
         norm_first: bool) -> list[torch.Tensor]:
    """Gradients of one stack: [dmsa, dgamma, dbeta, dW_left, dW_right, dW_out, db_out] (fp32 but dmsa)."""
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    T_ = S * L
    bf = torch.bfloat16
    has_mask = mask.numel() > 0
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if has_mask else x
    dzc = dz.reshape(L * L, CZ).contiguous()
    dzn = dz.new_empty((L * L, CZ), dtype=bf)
    dO = dz.new_empty((L * CH, L * CH), dtype=bf)
    _opm_bwd_epilogue_dx_kernel[lambda meta: (triton.cdiv(L, meta["TI"]), triton.cdiv(L, meta["BJ"]), triton.cdiv(CH * CH, meta["BN"]))](
        dzc, norm, wo.to(bf).contiguous(), dzn, dO, L, CH * CH,
        shape_key=token_key(length_of(msa.shape), CH=CH, CZ=CZ), CH=CH, CZ=CZ)
    dbo = dzc.float().sum(0) if norm_first else (dzc.float() / norm.reshape(L * L, 1)).sum(0)   # ESMFold2 order: bias divided too
    dA = torch.mm(b.view(S, L * CH), dO.t())                                                # [S, (i, d)]
    dB = torch.mm(a.view(S, L * CH), dO)                                                    # [S, (j, e)]
    del dO
    dwo = dz.new_zeros((CZ, CH * CH), dtype=torch.float32)
    _opm_bwd_epilogue_dw_kernel[lambda meta: (triton.cdiv(CZ, meta["BM"]), triton.cdiv(CH * CH, meta["BN"]), meta["SPLIT"])](
        dzn, o, dwo, L, CH * CH, shape_key=token_key(length_of(msa.shape), CH=CH, CZ=CZ), CH=CH, CZ=CZ)
    dm = torch.empty_like(x)
    red = dz.new_zeros((2 * CH * CM + 2 * CM,), dtype=torch.float32)
    dwl, dwr = red[:CH * CM].view(CH, CM), red[CH * CM:2 * CH * CM].view(CH, CM)
    dg, dbeta = red[2 * CH * CM:2 * CH * CM + CM], red[2 * CH * CM + CM:]
    nsm = _nsm(msa.device)
    _opm_bwd_layernorm_gemm_kernel[lambda meta: (min(triton.cdiv(T_, meta["BT"]), meta["PPS"] * nsm),)](
        x, m8, lnw.float().contiguous(), lnb.float().contiguous(), wl.to(bf).contiguous(), wr.to(bf).contiguous(),
        dA.view(T_, CH), dB.view(T_, CH), dm, dwl, dwr, dg, dbeta, T_, eps,
        shape_key=token_key(length_of(msa.shape), CM=CM, CH=CH), CM=CM, CH=CH, HAS_MASK=has_mask)
    return [dm.view(S, L, CM), dg.clone(), dbeta.clone(), dwl.clone(), dwr.clone(), dwo, dbo]


class _OuterProductMeanTriton(torch.autograd.Function):
    """One MSA stack (no batch axis). ``mask`` / ``residual`` are empty tensors when absent."""

    @staticmethod
    def forward(ctx, msa, mask, lnw, lnb, wl, wr, wo, bo, residual, eps, norm_first):
        out, a, b, o, norm = _fwd(msa, mask, lnw, lnb, wl, wr, wo, bo, residual, eps, norm_first)
        ctx.save_for_backward(msa, mask, a, b, o, norm, lnw, lnb, wl, wr, wo, bo)
        ctx.eps, ctx.norm_first, ctx.has_res = eps, norm_first, residual.numel() > 0
        return out

    @staticmethod
    def backward(ctx, dz):
        msa, mask, a, b, o, norm, lnw, lnb, wl, wr, wo, bo = ctx.saved_tensors
        dm, dg, dbeta, dwl, dwr, dwo, dbo = _bwd(dz.contiguous(), msa, mask, a, b, o, norm, lnw, lnb, wl, wr, wo, ctx.eps,
                                                 ctx.norm_first)
        return (dm, None, dg.to(lnw.dtype), dbeta.to(lnb.dtype), dwl.to(wl.dtype), dwr.to(wr.dtype), dwo.to(wo.dtype),
                dbo.to(bo.dtype), dz if ctx.has_res else None, None, None)


def triton_outer_product_mean(msa: torch.Tensor, mask: torch.Tensor | None, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                              w_left: torch.Tensor, w_right: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor,
                              residual: torch.Tensor | None = None, *, eps: float = 1e-5, normalize_before_proj: bool = True) -> torch.Tensor:
    """``residual + OPM(msa)`` (``OPM(msa)`` without a residual), differentiable. msa [B, S, L, CM] bf16, mask [B, S, L] bool.

    Stacks run one at a time: the mask counts and the grouped outer product are per stack. Check :func:`refusal` first."""
    empty = msa.new_empty((0,))
    outs = [_OuterProductMeanTriton.apply(msa[bi], empty if mask is None else mask[bi], ln_weight, ln_bias, w_left, w_right, w_out,
                                          b_out, empty if residual is None else residual[bi], float(eps), bool(normalize_before_proj))
            for bi in range(msa.shape[0])]
    return outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs, 0)
