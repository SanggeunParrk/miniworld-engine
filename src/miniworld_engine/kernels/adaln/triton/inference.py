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
from miniworld_engine.autotune.configs import configs_for

# The weighted-LayerNorm kernel that used to live here was fused3.py's `_ln_kernel` with
# HAS_W=True -- bitwise equal on the output (.bench/direct.out). Imported now.
from .fused3 import _ln_kernel

import contextlib

import torch

from miniworld_engine.kernels._compile import opaque
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


# Both axes are tuned tiles. BLOCK_N used to arrive as next_pow2(N) from the launcher (the whole
# row), which is also why BLOCK_M1 was pinned to 1..32. N here is the REDUCE axis (mean/var over
# d_cond / d_hidden = 128..1024), so it is a CSV tile and
# would force a two-pass over X on every shape instead of leaving the whole-row tile reachable.


# ───────────────────── step 1: cond_aff = LN(cond) · lnw  (no bias) ─────────────────────


# shape_key is keyed, and its value is L -- the atom count (this family is level=atom in
# kernels/registry.csv) -- not the flattened row count M = B*A the kernels iterate.
# `_cond_affine` and `_adaln_epilogue` are INNER launchers that only see the (M, D) matrix, so each
# takes the key from the caller that still holds the pre-flatten shape; the default covers a caller
# that hands in a genuinely 2-D activation (nothing folded into the rows, so shape[-2] IS L), which
# is what the drivers and checkers do.
from miniworld_engine.autotune.shape_key import atom_key, length_of


@opaque(fake=lambda cond, lnw, eps, out_dtype=None, shape_key=None: cond.new_empty(
            cond.shape, dtype=out_dtype or cond.dtype),
        name="adaln_cond_affine")
def _cond_affine(cond: torch.Tensor, lnw: torch.Tensor, eps: float,
                 out_dtype: torch.dtype | None = None,
                 shape_key: int | None = None) -> torch.Tensor:
    M, N = cond.shape
    if shape_key is None:
        shape_key = atom_key(length_of(cond.shape))
    aff = torch.empty(M, N, device=cond.device, dtype=out_dtype or cond.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # N is tl.constexpr now (it drives the BLOCK_N >= N fold) -> pass a plain python int.
    _ln_kernel[grid](
        cond, aff, lnw, M, int(N), eps,
        cond.stride(0), cond.stride(1), aff.stride(0), aff.stride(1),
        HAS_W=True, shape_key=shape_key,
    )
    return aff


# ───── step 3: y = sigmoid(scale)·LN(x) + bias  (fused LN(x) + gate epilogue) ─────


@triton.autotune(configs=configs_for("adaln_epilogue_triton"), key=['N', 'shape_key'])
@triton.jit
def _adaln_epilogue_kernel(
    X, SB, Y, M, N: tl.constexpr, eps,
    sx0, sx1, ss0, ss1, sy0, sy1,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, shape_key):
    # SB is (M, 2N): cols [0:N] = scale (incl. scale_b), [N:2N] = bias.
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    # TWO-PASS: pass 1 accumulates Σx / Σx² over the N tiles (fp32, plain sums → exact across
    # tiles); pass 2 re-reads x and streams the scale/bias halves of SB for the gate epilogue.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip but their tl.loads of X sit in
    # separate scf.for regions and are NOT CSE'd, so the covering config read x twice. `N` is
    # `tl.constexpr` (already this kernel's autotune key, so a new d_hidden already forced a
    # re-tune and a fresh compile) which makes the guard a TRACE-time comparison — exactly one
    # branch is emitted and the covering tile is back to the untiled single-read schedule. The
    # fast path uses the CENTRED variance Σ(x-mean)²/N (numerically stabler, and x is already in
    # registers); the uncentered Σx²/N - mean² stays in the tiled branch, where it is what keeps
    # that branch to one read per tile.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        mask = rmask[:, None] & (cols[None, :] < N)
        x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        x_hat = xc * rstd[:, None]
        scale = tl.load(SB + rm[:, None] * ss0 + cols[None, :] * ss1, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(SB + rm[:, None] * ss0 + (cols[None, :] + N) * ss1,
                       mask=mask, other=0.0).to(tl.float32)
        y = tl.sigmoid(scale) * x_hat + bias
        tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, y.to(Y.dtype.element_ty), mask=mask)
    else:
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            mask = rmask[:, None] & (cols[None, :] < N)
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=1)
            ss += tl.sum(x * x, axis=1)
        mean = s / N
        var = ss / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            mask = rmask[:, None] & (cols[None, :] < N)
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean[:, None]) * rstd[:, None]
            scale = tl.load(SB + rm[:, None] * ss0 + cols[None, :] * ss1, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(SB + rm[:, None] * ss0 + (cols[None, :] + N) * ss1,
                           mask=mask, other=0.0).to(tl.float32)
            y = tl.sigmoid(scale) * x_hat + bias
            tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, y.to(Y.dtype.element_ty), mask=mask)


@opaque(fake=lambda x, sb, eps, shape_key=None: torch.empty_like(x),
        name="adaln_inference_epilogue")
def _adaln_epilogue(x: torch.Tensor, sb: torch.Tensor, eps: float,
                    shape_key: int | None = None) -> torch.Tensor:
    M, N = x.shape
    if shape_key is None:
        shape_key = atom_key(length_of(x.shape))
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # N is tl.constexpr now (it drives the BLOCK_N >= N fold) -> pass a plain python int.
    _adaln_epilogue_kernel[grid](
        x, sb, y, M, int(N), eps,
        x.stride(0), x.stride(1), sb.stride(0), sb.stride(1), y.stride(0), y.stride(1),
        shape_key=shape_key,
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

    # L from the pre-flatten shapes (shape[-2]), read before the reshapes above discarded them.
    key_x = atom_key(length_of(orig_x_shape))
    cond_aff = _cond_affine(cond2d, cond_ln_weight, eps_cond,
                            shape_key=atom_key(length_of(cond.shape)))

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

    y = _adaln_epilogue(x2d, sb, eps_x, shape_key=key_x)
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
    from miniworld_engine.kernels.layernorm_linear.cute import fold_for_gemm, layernorm_linear

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
    y = _adaln_epilogue(x2d, sb, eps_x,
                        shape_key=atom_key(length_of(orig_x_shape)))         # kernel B
    return y.reshape(orig_x_shape)


# ───────────────── single-fused inference kernel (best at small d, e.g. atom d=128) ─────────────
# One kernel: LN(cond)·lnw, in-kernel GEMM → scale,bias, LN(x), sigmoid-gate → Y. Writes ONLY Y
# (no x_hat/cond_norm/gate/rstd saves), so it strips the backward-materialization traffic the
# training fwd kernel pays — a pure win in the memory-bound small-d regime.




# USE_LOW is deliberately NOT in the key. It picks the tl.dot operand precision (16-bit
# tensor-core MMA vs the fp32 dot), so it IS a code path -- but it is set at the one launch site
# below as `x.dtype in (bfloat16, float16)`, a pure function of X's dtype, and X is the first
# tensor operand, i.e. exactly what the cache's own `dtype` component reads (`dtype_of_args` in
# autotune/cache.py). The dtype component is strictly finer (it separates bf16 from fp16, which
# USE_LOW does not), so keying on USE_LOW as well would add no partition. Unlike the training
# kernels in main.py, nothing here consults autocast, so the two cannot diverge.
@triton.autotune(configs=configs_for("adaln_fwd_triton"), key=['NX', 'NC', 'shape_key'])
@triton.jit
def _adaln_fused_kernel(  # noqa: PLR0915
    X, Cond, LnW, ScaleW, ScaleB, BiasW, Y,
    sxr, sxc, scr, scc, swr, swc, sbwr, sbwc, syr, syc,
    M, NX: tl.constexpr, NC: tl.constexpr, eps_x, eps_cond,
    USE_LOW: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_K_NX: tl.constexpr, BLOCK_K_NC: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M

    # --- LN(cond) stats ---
    mean_c = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for cs in range(0, NC, BLOCK_K_NC):
        cc = cs + tl.arange(0, BLOCK_K_NC)
        cm = row_mask[:, None] & (cc[None, :] < NC)
        co = rows[:, None] * scr + cc[None, :] * scc
        v = tl.load(Cond + co, mask=cm, other=0.0).to(tl.float32)
        mean_c += tl.sum(tl.where(cm, v, 0.0), axis=1)
    mean_c /= NC
    var_c = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for cs in range(0, NC, BLOCK_K_NC):
        cc = cs + tl.arange(0, BLOCK_K_NC)
        cm = row_mask[:, None] & (cc[None, :] < NC)
        co = rows[:, None] * scr + cc[None, :] * scc
        v = tl.load(Cond + co, mask=cm, other=0.0).to(tl.float32)
        vc = tl.where(cm, v - mean_c[:, None], 0.0)
        var_c += tl.sum(vc * vc, axis=1)
    rstd_c = tl.rsqrt(var_c / NC + eps_cond)

    # --- LN(x) stats ---
    mean_x = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for xs in range(0, NX, BLOCK_K_NX):
        xc = xs + tl.arange(0, BLOCK_K_NX)
        xm = row_mask[:, None] & (xc[None, :] < NX)
        xo = rows[:, None] * sxr + xc[None, :] * sxc
        v = tl.load(X + xo, mask=xm, other=0.0).to(tl.float32)
        mean_x += tl.sum(tl.where(xm, v, 0.0), axis=1)
    mean_x /= NX
    var_x = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for xs in range(0, NX, BLOCK_K_NX):
        xc = xs + tl.arange(0, BLOCK_K_NX)
        xm = row_mask[:, None] & (xc[None, :] < NX)
        xo = rows[:, None] * sxr + xc[None, :] * sxc
        v = tl.load(X + xo, mask=xm, other=0.0).to(tl.float32)
        vc = tl.where(xm, v - mean_x[:, None], 0.0)
        var_x += tl.sum(vc * vc, axis=1)
    rstd_x = tl.rsqrt(var_x / NX + eps_x)

    # --- per x-block: GEMM scale,bias over cond, then gate ---
    for xs in range(0, NX, BLOCK_K_NX):
        xc = xs + tl.arange(0, BLOCK_K_NX)
        xm = row_mask[:, None] & (xc[None, :] < NX)
        xo = rows[:, None] * sxr + xc[None, :] * sxc
        xv = tl.load(X + xo, mask=xm, other=0.0).to(tl.float32)
        x_hat = (xv - mean_x[:, None]) * rstd_x[:, None]

        scale_b = tl.load(ScaleB + xc, mask=xc < NX, other=0.0).to(tl.float32)
        scale = tl.zeros([BLOCK_M1, BLOCK_K_NX], dtype=tl.float32)
        bias = tl.zeros([BLOCK_M1, BLOCK_K_NX], dtype=tl.float32)
        for cs in range(0, NC, BLOCK_K_NC):
            cc = cs + tl.arange(0, BLOCK_K_NC)
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


@opaque(fake=lambda x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight,
               eps_x, eps_cond, length=None: torch.empty_like(x),
        name="adaln_inference_fused")
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
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M1"]),)  # noqa: E731
    _adaln_fused_kernel[grid](
        x2d, cond2d, cond_ln_weight, scale_weight, scale_bias, bias_weight, y,
        x2d.stride(0), x2d.stride(1), cond2d.stride(0), cond2d.stride(1),
        scale_weight.stride(0), scale_weight.stride(1),
        bias_weight.stride(0), bias_weight.stride(1), y.stride(0), y.stride(1),
        m, NX=nx, NC=nc, eps_x=eps_x, eps_cond=eps_cond, USE_LOW=use_low,
        # L is x's pre-flatten shape[-2]; m = B*A is the row count, not the shape.
        shape_key=atom_key(length_of(orig_x_shape)),
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
    # lnfold's fused GEMM is the cute quack path imported directly from
    # layernorm_linear.cute (SM90 WGMMA/TMA, sm_90a-only) with NO internal fallback — unlike
    # the top-level layernorm_linear(), it does not self-dispatch by arch. So gate it on
    # Hopper *exactly* (major == 9); on pre-Hopper (sm_80 / A100) and Blackwell (sm_100) use
    # the portable materialize + cuBLAS path. Without this, bf16/fp16 d>256 crashes on A100.
    if x.dtype in (torch.float16, torch.bfloat16) and (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(x.device)[0] == 9  # noqa: PLR2004
    ):
        return adaln_inference_lnfold(x, cond, cond_ln_weight, scale_weight, scale_bias,
                                      bias_weight, eps_x, eps_cond, **kw)
    return adaln_inference_materialize(x, cond, cond_ln_weight, scale_weight, scale_bias,
                                       bias_weight, eps_x, eps_cond, **kw)
