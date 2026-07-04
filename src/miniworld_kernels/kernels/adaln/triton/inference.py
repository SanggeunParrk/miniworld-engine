"""Inference-only AdaptiveLayerNorm: materialize-then-GEMM, fused LN+sigmoid-gate epilogue.

The reference op (team-gm DiT adaLN, see modules/adaptive_layernorm/module.py PYTORCH path):

    x_norm    = LayerNorm(x)                      # d_hidden, NO affine (γ=1, β=0)
    cond_norm = LayerNorm(cond) * lnw             # d_cond, weight=lnw, NO bias
    scale     = cond_norm @ Wscaleᵀ + scale_b     # Linear(d_cond → d_hidden) w/ bias
    bias      = cond_norm @ Wbiasᵀ                # Linear(d_cond → d_hidden) no bias
    y         = sigmoid(scale) * x_norm + bias

This is structurally a 2-output LayerNormLinear + sigmoid-gate. For INFERENCE we save nothing
(no x_hat/cond_norm/gate/rstd for backward), so we maximize fusion and minimize traffic.

Structure (mirrors layernorm_linear/te_style.py: cuBLAS GEMM = fast, Triton LN = portable):
  1. cond_aff = LN(cond)·lnw                          (Triton, strided cond → contiguous (M,NC))
  2. [scale|bias] = cond_aff @ [Wscale|Wbias]ᵀ + [scale_b|0]   (ONE cuBLAS GEMM, M×NC×2NX)
  3. y = sigmoid(scale)·LN(x) + bias                  (Triton fused: LN(x) + gate epilogue)

The single fused GEMM (concat Wscale,Wbias along out-dim) replaces the two separate GEMMs the
eager/compiled path runs — one launch, cond_aff read once, TF32 policy for fp32.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# fp32 matmul precision policy (shared idea with layernorm_linear/te_style).
_FP32_MATMUL_PRECISION = "high"  # "high" → TF32 (fast); "highest" → true fp32


def set_fp32_matmul_precision(mode: str) -> None:
    """'high' = TF32 cuBLAS for fp32 GEMM (fast); 'highest' = true fp32 (accurate, slower)."""
    global _FP32_MATMUL_PRECISION  # noqa: PLW0603
    assert mode in ("high", "highest")
    _FP32_MATMUL_PRECISION = mode


# bf16 operands (fp32 accumulate) for the fp32 GEMM — ~1.6× faster than TF32, cos≈0.9999.
_GEMM_BF16 = False

# Cache for adaln_inference_lnfold's prefolded GEMM operands, keyed on (fixed) weight identities +
# dtype. Inference weights are static, so folding once and reusing avoids a per-forward fold pass.
_LNFOLD_CACHE: dict = {}


def set_gemm_bf16(flag: bool) -> None:
    """Enable bf16 operands (fp32 accumulate) for the GEMM on fp32 inputs (faster, cos≈0.9999)."""
    global _GEMM_BF16  # noqa: PLW0603
    _GEMM_BF16 = flag


@contextlib.contextmanager
def _fp32_matmul_ctx(dtype):
    if dtype is not torch.float32:
        yield
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = _FP32_MATMUL_PRECISION == "high"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


_LN_CONFIGS = [
    triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
    for bm in (1, 2, 4, 8, 16, 32)
    for nw in (4, 8, 16)
    for ns in (2, 3, 4)
]


# ───────────────────── step 1: cond_aff = LN(cond) · lnw  (no bias) ─────────────────────
@triton.autotune(configs=_LN_CONFIGS, key=["N", "DT"])
@triton.jit
def _cond_affine_kernel(
    Cond, CondAff, LnW, M, N, eps,
    sc0, sc1, sa0, sa1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DT: tl.constexpr,
):
    row = tl.program_id(0)
    rm = row * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    c = tl.load(Cond + rm[:, None] * sc0 + cols[None, :] * sc1, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(c, axis=1) / N
    cc = tl.where(cmask[None, :], c - mean[:, None], 0.0)
    var = tl.sum(cc * cc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    lnw = tl.load(LnW + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    aff = cc * rstd[:, None] * lnw
    tl.store(CondAff + rm[:, None] * sa0 + cols[None, :] * sa1,
             aff.to(CondAff.dtype.element_ty), mask=mask)


def _cond_affine(cond: torch.Tensor, lnw: torch.Tensor, eps: float,
                 out_dtype: torch.dtype | None = None) -> torch.Tensor:
    M, N = cond.shape
    aff = torch.empty(M, N, device=cond.device, dtype=out_dtype or cond.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731
    _cond_affine_kernel[grid](
        cond, aff, lnw, M, N, eps,
        cond.stride(0), cond.stride(1), aff.stride(0), aff.stride(1),
        BLOCK_N=triton.next_power_of_2(N), DT=cond.element_size(),
    )
    return aff


# ───── step 3: y = sigmoid(scale)·LN(x) + bias  (fused LN(x) + gate epilogue) ─────
@triton.autotune(configs=_LN_CONFIGS, key=["N", "DT"])
@triton.jit
def _adaln_epilogue_kernel(
    X, SB, Y, M, N, eps,
    sx0, sx1, ss0, ss1, sy0, sy1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DT: tl.constexpr,
):
    # SB is (M, 2N): cols [0:N] = scale (incl. scale_b), [N:2N] = bias.
    row = tl.program_id(0)
    rm = row * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = xc * rstd[:, None]
    scale = tl.load(SB + rm[:, None] * ss0 + cols[None, :] * ss1, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(SB + rm[:, None] * ss0 + (cols[None, :] + N) * ss1, mask=mask, other=0.0).to(tl.float32)
    y = tl.sigmoid(scale) * x_hat + bias
    tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, y.to(Y.dtype.element_ty), mask=mask)


def _adaln_epilogue(x: torch.Tensor, sb: torch.Tensor, eps: float) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731
    _adaln_epilogue_kernel[grid](
        x, sb, y, M, N, eps,
        x.stride(0), x.stride(1), sb.stride(0), sb.stride(1), y.stride(0), y.stride(1),
        BLOCK_N=triton.next_power_of_2(N), DT=x.element_size(),
    )
    return y


def adaln_inference_materialize(
    x: torch.Tensor,
    cond: torch.Tensor,
    cond_ln_weight: torch.Tensor,   # lnw, (NC,)
    scale_weight: torch.Tensor,     # to_scale.weight, (NX, NC)
    scale_bias: torch.Tensor,       # to_scale.bias, (NX,)
    bias_weight: torch.Tensor,      # to_bias.weight, (NX, NC)
    eps_x: float,
    eps_cond: float,
    *,
    weight_cat: torch.Tensor | None = None,   # optional cached cat([Wscale,Wbias],0) (2NX,NC)
    bias_cat: torch.Tensor | None = None,      # optional cached cat([scale_b,0])     (2NX,)
) -> torch.Tensor:
    """Inference adaLN via materialize+cuBLAS (best at large d, e.g. token d=768)."""
    orig_x_shape = x.shape
    nx = orig_x_shape[-1]
    x2d = x.reshape(-1, nx)
    cond2d = cond.reshape(-1, cond.shape[-1])
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    if cond2d.stride(-1) != 1:
        cond2d = cond2d.contiguous()

    cond_aff = _cond_affine(cond2d, cond_ln_weight, eps_cond)

    if weight_cat is None:
        weight_cat = torch.cat([scale_weight, bias_weight], dim=0)  # (2NX, NC)
    if bias_cat is None:
        bias_cat = torch.cat([scale_bias, scale_bias.new_zeros(nx)], dim=0)  # (2NX,)

    if _GEMM_BF16 and x.dtype == torch.float32:
        sb = torch.matmul(cond_aff.to(torch.bfloat16), weight_cat.t().to(torch.bfloat16))
        sb = sb.float() + bias_cat
    else:
        with _fp32_matmul_ctx(x.dtype):
            sb = F.linear(cond_aff, weight_cat, bias_cat)  # (M, 2NX) = [scale | bias]

    y = _adaln_epilogue(x2d, sb, eps_x)
    return y.reshape(orig_x_shape)


def adaln_inference_lnfold(
    x: torch.Tensor,
    cond: torch.Tensor,
    cond_ln_weight: torch.Tensor,   # lnw, (NC,)
    scale_weight: torch.Tensor,     # (NX, NC)
    scale_bias: torch.Tensor,       # (NX,)
    bias_weight: torch.Tensor,      # (NX, NC)
    eps_x: float,
    eps_cond: float,
    *,
    weight_cat: torch.Tensor | None = None,   # cached cat([Wscale,Wbias],0) (2NX,NC)
    bias_cat: torch.Tensor | None = None,       # cached cat([scale_b,0])    (2NX,)
    prefolded=None,                             # cached fold_for_gemm(weight_cat, lnw, 0, bias_cat)
) -> torch.Tensor:
    """Inference adaLN via FUSED LN(cond)+GEMM (kernel A, cute layernorm_linear) + fused epilogue
    (kernel B, _adaln_epilogue). The cond LayerNorm is folded into the GEMM prologue, so cond_aff
    never hits HBM — the materialize pass of ``adaln_inference_materialize`` is gone. Best at token
    d (>=256); LOSES to materialize at atom d=128 (cute stats+launch overhead > the small
    materialize saving there). Weights fixed → prefold once and pass ``weight_cat``/``prefolded``."""
    from miniworld_kernels.kernels.layernorm_linear.cute import fold_for_gemm, layernorm_linear

    orig_x_shape = x.shape
    nx = orig_x_shape[-1]
    x2d = x.reshape(-1, nx)
    cond2d = cond.reshape(-1, cond.shape[-1])
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    if cond2d.stride(-1) != 1:
        cond2d = cond2d.contiguous()

    # Prefold the (fixed-weight) GEMM operands ONCE. Doing it per call adds a ~(2NX,NC) fold pass
    # every forward (captured & re-run on each cudagraph replay) which erases the win — so cache it
    # keyed on the weight identities + dtype (inference weights are static). Callers may also pass
    # weight_cat/bias_cat/prefolded explicitly to bypass the cache.
    if prefolded is None and weight_cat is None:
        key = (scale_weight.data_ptr(), bias_weight.data_ptr(), scale_bias.data_ptr(),
               cond_ln_weight.data_ptr(), x.dtype)
        hit = _LNFOLD_CACHE.get(key)
        if hit is None:
            weight_cat = torch.cat([scale_weight, bias_weight], dim=0)          # (2NX, NC)
            bias_cat = torch.cat([scale_bias, scale_bias.new_zeros(nx)], dim=0)  # (2NX,)
            ln_bias = cond_ln_weight.new_zeros(cond_ln_weight.shape)             # cond LN: no bias
            prefolded = fold_for_gemm(weight_cat, cond_ln_weight, ln_bias, bias_cat, w2_dtype=x.dtype)
            _LNFOLD_CACHE[key] = (weight_cat, bias_cat, prefolded)
        else:
            weight_cat, bias_cat, prefolded = hit
    else:
        if weight_cat is None:
            weight_cat = torch.cat([scale_weight, bias_weight], dim=0)
        if bias_cat is None:
            bias_cat = torch.cat([scale_bias, scale_bias.new_zeros(nx)], dim=0)
        if prefolded is None:
            ln_bias = cond_ln_weight.new_zeros(cond_ln_weight.shape)
            prefolded = fold_for_gemm(weight_cat, cond_ln_weight, ln_bias, bias_cat, w2_dtype=x.dtype)
    ln_bias = cond_ln_weight.new_zeros(cond_ln_weight.shape)

    # kernel A: [scale|bias] = LN(cond) @ [Wscale|Wbias]ᵀ + [scale_b|0], LN folded into prologue.
    sb = layernorm_linear(cond2d, cond_ln_weight, ln_bias, weight_cat, bias_cat, eps_cond,
                          prefolded=prefolded)                              # (M, 2NX)
    y = _adaln_epilogue(x2d, sb, eps_x)                                     # kernel B
    return y.reshape(orig_x_shape)


# ───────────────── single-fused inference kernel (best at small d, e.g. atom d=128) ─────────────
# One kernel: LN(cond)·lnw, in-kernel GEMM → scale,bias, LN(x), sigmoid-gate → Y. Writes ONLY Y
# (no x_hat/cond_norm/gate/rstd saves), so it strips the backward-materialization traffic the
# training fwd kernel pays — a pure win in the memory-bound small-d regime.
_FUSED_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_NX": bnx, "BLOCK_NC": bnc}, num_warps=nw, num_stages=ns)
    for bm in (16, 32, 64)
    for bnx in (64, 128)
    for bnc in (64, 128)
    for nw in (4, 8)
    for ns in (2, 3)
]


@triton.autotune(configs=_FUSED_CONFIGS, key=["NX", "NC", "DT"])
@triton.jit
def _adaln_fused_kernel(  # noqa: PLR0915
    X, Cond, LnW, ScaleW, ScaleB, BiasW, Y,
    sxr, sxc, scr, scc, swr, swc, sbwr, sbwc, syr, syc,
    M, NX: tl.constexpr, NC: tl.constexpr, eps_x, eps_cond,
    USE_LOW: tl.constexpr, DT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_NX: tl.constexpr, BLOCK_NC: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # --- LN(cond) stats ---
    mean_c = tl.zeros([BLOCK_M], dtype=tl.float32)
    for cs in range(0, NC, BLOCK_NC):
        cc = cs + tl.arange(0, BLOCK_NC)
        cm = row_mask[:, None] & (cc[None, :] < NC)
        co = rows[:, None] * scr + cc[None, :] * scc
        v = tl.load(Cond + co, mask=cm, other=0.0).to(tl.float32)
        mean_c += tl.sum(tl.where(cm, v, 0.0), axis=1)
    mean_c /= NC
    var_c = tl.zeros([BLOCK_M], dtype=tl.float32)
    for cs in range(0, NC, BLOCK_NC):
        cc = cs + tl.arange(0, BLOCK_NC)
        cm = row_mask[:, None] & (cc[None, :] < NC)
        co = rows[:, None] * scr + cc[None, :] * scc
        v = tl.load(Cond + co, mask=cm, other=0.0).to(tl.float32)
        vc = tl.where(cm, v - mean_c[:, None], 0.0)
        var_c += tl.sum(vc * vc, axis=1)
    rstd_c = tl.rsqrt(var_c / NC + eps_cond)

    # --- LN(x) stats ---
    mean_x = tl.zeros([BLOCK_M], dtype=tl.float32)
    for xs in range(0, NX, BLOCK_NX):
        xc = xs + tl.arange(0, BLOCK_NX)
        xm = row_mask[:, None] & (xc[None, :] < NX)
        xo = rows[:, None] * sxr + xc[None, :] * sxc
        v = tl.load(X + xo, mask=xm, other=0.0).to(tl.float32)
        mean_x += tl.sum(tl.where(xm, v, 0.0), axis=1)
    mean_x /= NX
    var_x = tl.zeros([BLOCK_M], dtype=tl.float32)
    for xs in range(0, NX, BLOCK_NX):
        xc = xs + tl.arange(0, BLOCK_NX)
        xm = row_mask[:, None] & (xc[None, :] < NX)
        xo = rows[:, None] * sxr + xc[None, :] * sxc
        v = tl.load(X + xo, mask=xm, other=0.0).to(tl.float32)
        vc = tl.where(xm, v - mean_x[:, None], 0.0)
        var_x += tl.sum(vc * vc, axis=1)
    rstd_x = tl.rsqrt(var_x / NX + eps_x)

    # --- per x-block: GEMM scale,bias over cond, then gate ---
    for xs in range(0, NX, BLOCK_NX):
        xc = xs + tl.arange(0, BLOCK_NX)
        xm = row_mask[:, None] & (xc[None, :] < NX)
        xo = rows[:, None] * sxr + xc[None, :] * sxc
        xv = tl.load(X + xo, mask=xm, other=0.0).to(tl.float32)
        x_hat = (xv - mean_x[:, None]) * rstd_x[:, None]

        scale_b = tl.load(ScaleB + xc, mask=xc < NX, other=0.0).to(tl.float32)
        scale = tl.zeros([BLOCK_M, BLOCK_NX], dtype=tl.float32)
        bias = tl.zeros([BLOCK_M, BLOCK_NX], dtype=tl.float32)
        for cs in range(0, NC, BLOCK_NC):
            cc = cs + tl.arange(0, BLOCK_NC)
            cm = row_mask[:, None] & (cc[None, :] < NC)
            co = rows[:, None] * scr + cc[None, :] * scc
            v = tl.load(Cond + co, mask=cm, other=0.0).to(tl.float32)
            cond_norm = (v - mean_c[:, None]) * rstd_c[:, None]
            lnw = tl.load(LnW + cc, mask=cc < NC, other=0.0).to(tl.float32)
            cond_aff = cond_norm * lnw[None, :]
            sw = tl.load(ScaleW + (cc[:, None] * swc + xc[None, :] * swr),
                         mask=(cc[:, None] < NC) & (xc[None, :] < NX), other=0.0).to(tl.float32)
            bw = tl.load(BiasW + (cc[:, None] * sbwc + xc[None, :] * sbwr),
                         mask=(cc[:, None] < NC) & (xc[None, :] < NX), other=0.0).to(tl.float32)
            if USE_LOW:
                cond_aff = cond_aff.to(X.dtype.element_ty)
                sw = sw.to(X.dtype.element_ty)
                bw = bw.to(X.dtype.element_ty)
            scale += tl.dot(cond_aff, sw, out_dtype=tl.float32)
            bias += tl.dot(cond_aff, bw, out_dtype=tl.float32)
        scale += scale_b[None, :]
        y = tl.sigmoid(scale) * x_hat + bias
        yo = rows[:, None] * syr + xc[None, :] * syc
        tl.store(Y + yo, y.to(Y.dtype.element_ty), mask=xm)


def adaln_inference_fused(
    x: torch.Tensor,
    cond: torch.Tensor,
    cond_ln_weight: torch.Tensor,
    scale_weight: torch.Tensor,
    scale_bias: torch.Tensor,
    bias_weight: torch.Tensor,
    eps_x: float,
    eps_cond: float,
) -> torch.Tensor:
    """Inference adaLN via ONE fused kernel (best at small d, e.g. atom d=128)."""
    orig_x_shape = x.shape
    nx = orig_x_shape[-1]
    x2d = x.reshape(-1, nx)
    cond2d = cond.reshape(-1, cond.shape[-1])
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    if cond2d.stride(-1) != 1:
        cond2d = cond2d.contiguous()
    m, _ = x2d.shape
    nc = cond2d.shape[-1]
    y = torch.empty_like(x2d)
    use_low = x.dtype in (torch.bfloat16, torch.float16)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]),)  # noqa: E731
    _adaln_fused_kernel[grid](
        x2d, cond2d, cond_ln_weight, scale_weight, scale_bias, bias_weight, y,
        x2d.stride(0), x2d.stride(1), cond2d.stride(0), cond2d.stride(1),
        scale_weight.stride(0), scale_weight.stride(1),
        bias_weight.stride(0), bias_weight.stride(1), y.stride(0), y.stride(1),
        m, NX=nx, NC=nc, eps_x=eps_x, eps_cond=eps_cond, USE_LOW=use_low, DT=x.element_size(),
    )
    return y.reshape(orig_x_shape)


def adaln_inference(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight,
                    eps_x, eps_cond, **kw):
    """Dispatch: small d (≤256) → single fused kernel; token d (>256) → LN-folded cute GEMM
    (kernel A) + fused epilogue (kernel B), which beats the materialize path 1.12-1.21x by
    dropping the cond_aff HBM round-trip. ``kw`` (weight_cat/bias_cat/prefolded) is forwarded so
    a caller with fixed weights can prefold once.

    lnfold's cute GEMM (quack SM90) is 16/8-bit ONLY, so fp32 falls back to materialize+cuBLAS —
    there is no fast fused fp32/TF32 GEMM here (triton's TF32 GEMM is ~0.5× cuBLAS)."""
    if x.shape[-1] <= 256:  # noqa: PLR2004
        return adaln_inference_fused(x, cond, cond_ln_weight, scale_weight, scale_bias,
                                     bias_weight, eps_x, eps_cond)
    if x.dtype in (torch.float16, torch.bfloat16):
        return adaln_inference_lnfold(x, cond, cond_ln_weight, scale_weight, scale_bias,
                                      bias_weight, eps_x, eps_cond, **kw)
    return adaln_inference_materialize(x, cond, cond_ln_weight, scale_weight, scale_bias,
                                       bias_weight, eps_x, eps_cond, **kw)
