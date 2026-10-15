"""OuterProductMean with measured H100 dispatch for BF16 inputs and BF16/FP32 master weights.

H100 uses a WGMMA prologue, deterministic CUDA mask/parameter passes, cuBLAS contractions and CUDA epilogues.
The seven inference cells and every training cell at S1024/2048/4096 use the persistent TMA prologue (the non-TMA WGMMA prologue is 5-6x
slower than the legacy Triton one in training; TMA matches or beats legacy at S1024-4096, L128-768, 2026-10-07).
S1024/L768 training owns cuBLAS workspaces for safe independent CUDA-graph lifetimes.
S1024/L384 and S2048/L384 inference also use the gathered-TMA, overlapping-WGMMA epilogue.
Other required inference cells retain the faster accurate historical composition;
training retains zero-gamma-safe CUDA backward. The all-shape CUDA-only goal remains open.
The H100 shape routing lives in `_h100_path` / `_epilogue_*_enabled` / `_owned_blas_for`; the former environment switches for
the TMA prologue, the cuBLAS mask count and the WIDE epilogue are the private `_CANDIDATE_*` constants below (all False).

The historical TRAINING path: one autograd.Function over the fused OPM kernels developed in MiniWorld's
`runs/msa_bench_20260921` (2026-09-22): the upstream-style fused Triton PROLOGUE (LayerNorm + both projections + mask
straight into the GEMM layouts, vendored below with the LayerNorm statistics kept for the backward), cuBLAS for the
grouped outer product, and this repo's `csrc/opm_epilogue.cu` for everything else:

Forward:   fused_prologue -> A2 [(i,c), s], BT [(j,e), s] bf16 (+ stats [S,N,2]);  O = A2 . BT^T (cuBLAS, grouped layout);
           norm = mask^T mask (fp32, clamp 1);  opm_epilogue: [i,j,c,e] -> [(i,j),(c,e)] permute, / norm, bf16, c_hidden^2 -> c_z + bias.
Backward:  opm_dgrad (dz/norm . Wo straight into the grouped layout, dbo on the side), two cuBLAS GEMMs for dA / dB,
           opm_dwo (dWo off the grouped O: kept from the forward when settings.opm_train_save_o != 0, else one GEMM recomputes it),
           opm_prologue_bwd (mask, both projections and the LayerNorm backward in one pass -> dm, dWa, dWb, dgamma, dbeta).
Measured H100, L=384, S=1024, bf16: fwd+bwd 2.23 ms / 796 MB (O kept) or 2.63 ms / 557 MB, vs this engine's own path
5.80 ms / 1624 MB; every gradient checked against fp32 autograd of the same statements.

Automatic for supported inputs (settings.opm_train=0 disables it; no external payload needed). Serves a module built with
implementation="miniworld" or "anthropic" when: d_msa=64, d_hidden=32, d_pair=128, batch 1, N % 64 == 0, S % 256 == 0,
bf16 on sm_90a, normalize_before_proj, no interchain mask. Fuses the optional pair residual in the output epilogue. Inference omits LN statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl

from miniworld_engine import settings

CM, CH, CZ = 64, 32, 128
# (d_msa, d_pair) served on sm_90a: MiniWorld, Protenix-v2 / ESMFold2, OpenDDE (d_hidden 32 throughout).  The widths other than
# MiniWorld's run the wide composition: persistent TMA prologue at d_msa 64 / 128, cuBLAS contraction, the c_z-sliced epilogue.
SHAPES = ((64, 128), (128, 256), (128, 384))
_EXT: dict[str, Any] = {}


def _enabled() -> bool:
    """``settings.opm_train``: False disables the path; None (the default) is automatic."""
    from miniworld_engine import settings

    value = settings.current().opm_train
    return True if value is None else value


# Candidate build variants that stay compiled/loadable but are not selected by any shape (the former diagnostic environment switches).
_CANDIDATE_TMA_PROLOGUE = False  # TMA prologue on every shape (``_h100_path`` already picks it for the qualified cells)
_CANDIDATE_CUBLAS_COUNT = False  # FP32 cuBLAS mask count instead of the exact bit-mask popcount kernel (needs the TMA prologue)
_CUDA_PROLOGUE_FOR_LEGACY = True  # inference cells that used the Triton prologue run the CUDA one (`opm_prologue_wide`): 1.5-5.5 % faster at all 11 such cells (2026-10-13)
_CANDIDATE_EPILOGUE_WIDE = (
    False  # wide gathered-TMA epilogue (mutually exclusive with HALF)
)


def _h100_path(s: int, n: int, *, training: bool, wide: bool = False) -> str:
    """Only stable paired wins below the eligible historical timing are promoted."""
    if wide:  # d_msa / d_pair other than MiniWorld's: the wide composition (persistent TMA prologue at that d_msa)
        return "wide"
    if _CANDIDATE_TMA_PROLOGUE:
        return "tma"
    if training:
        return "tma" if s in (1024, 2048, 4096) else "native"
    if (s, n) in (
        (1024, 128),
        (1024, 256),
        (1024, 384),
        (2048, 128),
        (2048, 256),
        (2048, 384),
        (4096, 128),
    ):
        return "tma"
    if s in (1024, 2048, 4096) and n in (128, 256, 384, 512, 640, 768):
        return "legacy"
    return "native"


def wanted(implementation) -> bool:
    from miniworld_engine import settings
    from miniworld_engine.modules.exceptions import ImplementationType

    return (
        _enabled()
        and settings.current().engine_backend != "triton"
        and implementation
        in (ImplementationType.MINIWORLD, ImplementationType.ANTHROPIC)
    )


def refusal(
    msa: torch.Tensor,
    d_msa: int,
    d_hidden: int,
    d_pair: int,
    *,
    interchain: bool,
    normalize_before_proj: bool,
) -> str | None:
    """None if this path can run this training call, else why it cannot. Never raises."""
    try:
        if not _enabled():
            return "settings.opm_train=False disables the native path"
        if d_hidden != CH or (d_msa, d_pair) not in SHAPES:
            return f"the kernels serve d_hidden={CH} and (d_msa, d_pair) in {SHAPES}, got ({d_msa}, {d_hidden}, {d_pair})"
        if interchain:
            return (
                "mask_interchain is applied after the projection and is not fused here"
            )
        if msa.dtype != torch.bfloat16:
            return f"the kernels are bf16, got {msa.dtype}"
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        cap = torch.cuda.get_device_capability(msa.device)
        if cap not in ((9, 0), (10, 0)):
            return "the kernels are built for sm_90a and sm_100a"
        if cap == (10, 0) and torch.is_grad_enabled() and not _SM100_TRAIN:
            return "the sm_100a backward kernels are not built yet"
        if cap == (10, 0) and (
            (d_msa, d_pair) != (CM, CZ) or not normalize_before_proj
        ):
            return "the sm_100a kernels serve d_msa 64 / d_pair 128 with the AF3 normalisation order"
        if msa.shape[0] != 1:
            return f"one MSA stack per call, got batch {msa.shape[0]}"
        s, n = msa.shape[1], msa.shape[2]
        if n == 0 or s == 0:
            return f"empty MSA stack, got N={n}, S={s}"
        # a length not a multiple of 64 / a depth not a multiple of 256 (the prologue backward's 32 x 8 row blocks) is padded
        # with masked rows by `outer_product_mean`
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


STATS: dict[str, Any] = {
    "served": 0,
    "refused": {},
}  # how often the path ran / why it did not (for a real-model check)


def serves(*a, **kw) -> bool:
    why = refusal(*a, **kw)
    # Counters are eager diagnostics; mutating them while tracing adds recompilation guards.
    if torch.compiler.is_compiling():
        return why is None
    if why is None:
        STATS["served"] += 1
        return True
    if (
        settings.current().opm_train is not None and why not in STATS["refused"]
    ):  # opted in but not served: say why, once per reason
        import logging

        logging.getLogger("miniworld_engine").warning(
            "opm_train: the fused training path is not taken: %s", why
        )
    STATS["refused"][why] = STATS["refused"].get(why, 0) + 1
    return False


_SM100_TRAIN = True  # the sm_100a backward kernels (csrc/sm100/opm_sm100.cu) serve grad-enabled calls


def _epilogue_tma_enabled(s: int, n: int, *, training: bool) -> bool:
    """Measured default: the gathered-TMA epilogue serves S1024/2048 x N384 inference."""
    return not training and (s, n) in ((1024, 384), (2048, 384))


def _epilogue_half_enabled(s: int, n: int, *, training: bool) -> bool:
    return not training and (s, n) == (1024, 384)


def _owned_blas_for(s: int, n: int, *, training: bool) -> bool:
    """Graph-owned cuBLAS for the qualified exact shapes."""
    return (training and (s, n) == (1024, 768)) or (
        not training and (s, n) == (4096, 128)
    )


def _owned_mm(a, b, enabled: bool):
    if not enabled:
        return torch.mm(a, b)
    from miniworld_engine.kernels.conditioned_transition.cuda.train import (
        owned_gemm_ext,
    )

    return owned_gemm_ext().owned_mm(a, b, torch.backends.cuda.matmul.allow_tf32, False)


def _ext(
    device=None,
    *,
    native=None,
    tma=None,
    epi_tma=None,
    epi_half=None,
    owned=None,
    cm: int = CM,
    cz: int = CZ,
):
    """This repo's fused OPM kernels for the current card: `csrc/opm_epilogue.cu` (sm_90a) or
    `csrc/sm100/opm_sm100.cu` (sm_100a, the same kernels and signatures on tcgen05 / TMEM).
    (cm, cz) other than MiniWorld's build the same source with -DOPM_CM / -DOPM_CZ (one extension per width)."""
    sm100 = torch.cuda.get_device_capability(device) == (10, 0)
    wide = (cm, cz) != (CM, CZ)
    native = True if native is None else native
    tma = _CANDIDATE_TMA_PROLOGUE if tma is None else tma
    count = native and tma and _CANDIDATE_CUBLAS_COUNT
    owned = not sm100 and native and bool(owned)
    # Gathered O/W TMA stages with overlapping WGMMA groups (chosen per shape by the callers).
    epi_tma = not sm100 and native and tma and bool(epi_tma)
    epi_half = epi_tma and bool(epi_half)
    epi_wide = epi_tma and _CANDIDATE_EPILOGUE_WIDE
    if epi_half and epi_wide:
        raise ValueError("OPM epilogue HALF and WIDE candidates are mutually exclusive")
    key = (
        "sm100"
        if sm100
        else (
            "mod_native_tma_cublas_count"
            if count
            else "mod_native_tma_static"
            if native and tma
            else "mod_native"
            if native
            else "mod"
        )
    )
    if epi_tma:
        key += "_epi_tma_overlap"
    if epi_half:
        key += "_half"
    if epi_wide:
        key += "_wide"
    if owned:
        key += "_owned_blas"
    if wide and not sm100:
        key += f"_{cm}_{cz}"
    if key in _EXT:
        return _EXT[key]
    from miniworld_engine.kernels._nvcc import load_extension as load

    root = settings.engine_jit_root()
    csrc = Path(__file__).with_name("csrc")
    if sm100:
        build = root / "opm_sm100"
        build.mkdir(parents=True, exist_ok=True)
        _EXT[key] = load(
            name="miniworld_opm_sm100",
            sources=[str(csrc / "sm100" / "opm_sm100.cu")],
            build_directory=str(build),
            extra_include_paths=[str(csrc / "sm100")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
            extra_cflags=["-O3"],
        )
    else:
        suffix = (
            "_native_tma_cublas_count"
            if count
            else "_native_tma_static"
            if native and tma
            else "_native"
            if native
            else ""
        )
        if epi_tma:
            suffix += "_epi_tma_overlap"
        if epi_half:
            suffix += "_half"
        if epi_wide:
            suffix += "_wide"
        if owned:
            suffix += "_owned_blas"
        if wide:
            suffix += f"_{cm}_{cz}"
        build = root / ("opm_epilogue" + suffix)
        build.mkdir(parents=True, exist_ok=True)
        _EXT[key] = load(
            name="miniworld_opm_epilogue" + suffix,
            sources=[str(csrc / "opm_epilogue.cu")],
            build_directory=str(build),
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
            + ([f"-DOPM_CM={cm}", f"-DOPM_CZ={cz}"] if wide else [])
            + (["-DMINIWORLD_OPM_CUBLAS_COUNT=1"] if count else [])
            + (["-DMINIWORLD_OPM_OWNED_BLAS=1"] if owned else [])
            + (["-DMINIWORLD_OPM_EPILOGUE_TMA_OVERLAP=1"] if epi_tma else [])
            + (["-DMINIWORLD_OPM_EPILOGUE_TMA_HALF=1"] if epi_half else [])
            + (["-DMINIWORLD_OPM_EPILOGUE_TMA_WIDE=1"] if epi_wide else []),
            extra_cflags=["-O3"],
            extra_ldflags=["-lcublas"] if owned else [],
        )
    return _EXT[key]


# ---- the fused prologue (vendored from the upstream msa_opm cell, with the LayerNorm statistics kept) ----
@triton.jit(
    do_not_specialize=["S"]
)  # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _opm_prologue_kernel(
    M,
    MASK,
    LNW,
    LNB,
    WA,
    WB,
    BA,
    BB,
    A2,
    BT,
    STATS,
    S,
    N,
    SK,
    NA,
    NBj,
    stride_ms,
    stride_mi,
    eps,
    CM: tl.constexpr,
    CH: tl.constexpr,
    BI: tl.constexpr,
    BS: tl.constexpr,
    BIP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    LN_AFFINE: tl.constexpr,
    MASK_I64: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    # Fused stock prologue: y = LN(m[s, i, :]) in fp32 (torch.nn.LayerNorm math: biased var, rsqrt(var + eps), affine), y -> bf16 (autocast),
    # a = y16 @ Wa^T, b = y16 @ Wb^T with fp32 accumulation (+ bf16-rounded bias in fp32, one rounding: the cuBLASLt epilogue), -> bf16,
    # * mask (0/1: exact).  Written directly in the GEMM layouts: A2[(iblk, c, bi), s] (iblk = i // BI) and BT[(i, e), s], zero-padded to
    # (NA | NBj) x SK.  Tile = BIP tokens x BS MSA rows; rows ordered (i, s) so that the stores are contiguous along s.
    pid_s = tl.program_id(
        0
    ).to(
        tl.int64
    )  # int64 program indices: every m / mask / A2 / BT element offset below is formed in int64
    pid_i = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BIP
    R: tl.constexpr = BIP * BS
    rr = tl.arange(0, R)
    r_i = i0 + rr // BS
    r_s = s0 + rr % BS
    valid = (r_s < S) & (r_i < N)
    cols = tl.arange(0, CM)
    x = tl.load(
        M + r_s[:, None] * stride_ms + r_i[:, None] * stride_mi + cols[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)  # [R, CM]
    mean = tl.sum(x, axis=1) / CM
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / CM
    rstd = 1.0 / tl.sqrt(var + eps)
    if SAVE_STATS:  # the LayerNorm statistics, for a backward that would otherwise re-derive them from m: STATS[s, i] = (mean, rstd)
        tl.store(STATS + (r_s * N + r_i) * 2, mean, mask=valid)
        tl.store(STATS + (r_s * N + r_i) * 2 + 1, rstd, mask=valid)
    y = xc * rstd[:, None]
    if LN_AFFINE:
        y = (
            y * tl.load(LNW + cols).to(tl.float32)[None, :]
            + tl.load(LNB + cols).to(tl.float32)[None, :]
        )
    y16 = y.to(tl.bfloat16)
    ch = tl.arange(0, CH)
    wa = tl.load(WA + cols[:, None] * CH + ch[None, :])  # [CM, CH] bf16 = Wa^T
    wb = tl.load(WB + cols[:, None] * CH + ch[None, :])
    a = tl.dot(y16, wa)  # [R, CH] fp32
    b = tl.dot(y16, wb)
    if HAS_BIAS:
        a += tl.load(BA + ch).to(tl.float32)[None, :]
        b += tl.load(BB + ch).to(tl.float32)[None, :]
    a16 = a.to(tl.bfloat16)
    b16 = b.to(tl.bfloat16)
    if HAS_MASK:
        if MASK_I64:
            mk = tl.load(MASK + r_s * N + r_i, mask=valid, other=0).to(tl.float32)
        else:
            mk = tl.load(MASK + r_s * N + r_i, mask=valid, other=0.0).to(tl.float32)
        a16 = (a16.to(tl.float32) * mk[:, None]).to(tl.bfloat16)
        b16 = (b16.to(tl.float32) * mk[:, None]).to(tl.bfloat16)
    a16 = tl.where(
        valid[:, None], a16, a16 * 0.0
    )  # zero padding rows (s >= S or i >= N)
    b16 = tl.where(valid[:, None], b16, b16 * 0.0)
    a_rows = ((r_i // BI) * CH)[:, None] + ch[None, :]
    a_ptr = A2 + (a_rows * BI + (r_i % BI)[:, None]) * SK + r_s[:, None]
    tl.store(a_ptr, a16, mask=((r_i < NA) & (r_s < SK))[:, None])
    b_ptr = BT + (r_i[:, None] * CH + ch[None, :]) * SK + r_s[:, None]
    tl.store(b_ptr, b16, mask=((r_i < NBj) & (r_s < SK))[:, None])


def fused_prologue(
    m,
    mask,
    lnw,
    lnb,
    eps,
    wa_t,
    wb_t,
    BI=1,
    BJ=1,
    BK=64,
    num_warps=8,
    BS=32,
    BIP=8,
    save_stats=True,
):
    """m: [S, N, CM] bf16, mask: [S, N] bf16 0/1 -> (A2 [N*CH, SK], BT [N*CH, SK]) bf16 with A2 row = i*CH + c, and stats [S, N, 2] fp32 (mean, rstd)."""
    S, N, cm = m.shape
    NA = triton.cdiv(N, BI) * BI
    NBj = triton.cdiv(N, BJ) * BJ
    SK = triton.cdiv(S, BK) * BK
    A2 = torch.empty(NA * CH, SK, device=m.device, dtype=torch.bfloat16)
    BT = torch.empty(NBj * CH, SK, device=m.device, dtype=torch.bfloat16)
    assert SK % BS == 0 and m.stride(2) == 1, (SK, BS, m.stride())
    grid = (SK // BS, triton.cdiv(max(NA, NBj), BIP))
    mask_c = mask.contiguous()
    stats = (
        torch.empty(S, N, 2, device=m.device, dtype=torch.float32)
        if save_stats
        else None
    )
    cast(Any, _opm_prologue_kernel)[grid](
        m,
        mask_c,
        lnw,
        lnb,
        wa_t,
        wb_t,
        wa_t,
        wa_t,
        A2,
        BT,
        stats if stats is not None else A2,
        S,
        N,
        SK,
        NA,
        NBj,
        m.stride(0),
        m.stride(1),
        float(eps),
        CM=cm,
        CH=CH,
        BI=BI,
        BS=BS,
        BIP=BIP,
        HAS_MASK=True,
        HAS_BIAS=False,
        LN_AFFINE=True,
        MASK_I64=False,
        SAVE_STATS=save_stats,
        num_warps=num_warps,
        num_stages=1,
    )
    return A2, BT, stats


def _forward_sm100(ext, msa, mask, lnw, lnb, wa, wb, wo, bo, eps, residual, save_stats):
    """B200: the same three stages on sm_100a kernels -- prologue (LN + both projections + mask -> A2 / BT, the LN
    statistics as [N][S] and the mask as bits), cuBLAS for the grouped outer product, and the epilogue, which counts
    the mask per pair from the bits instead of taking a separate [N, N] count."""
    bf = torch.bfloat16
    m = msa[0].contiguous()
    mk = mask[0].contiguous()
    if mk.dtype != torch.bool:
        mk = mk != 0
    a2, bt, _, stats, bits = ext.opm_prologue(
        m,
        mk,
        lnw.detach().contiguous(),
        lnb.detach().contiguous(),
        float(eps),
        wa.detach().to(bf).contiguous(),
        wb.detach().to(bf).contiguous(),
        save_stats,
        False,
    )
    o = torch.matmul(a2, bt.t())  # the grouped outer product [(i,c), (j,e)], cuBLAS NT
    n = m.shape[1]
    z = ext.opm_epilogue(
        o,
        bits,
        wo.detach().to(bf).contiguous(),
        bo.detach().to(bf).contiguous(),
        n,
        n,
        residual,
    )
    return z, m, mk, bits, a2, bt, stats, o


def _forward_wide(
    ctx, msa, mask, lnw, lnb, wa, wb, wo, bo, eps, save_o, residual, save_stats, after
):
    """The wide composition (d_msa 64 / 128, d_pair 128 / 256 / 384, either normalisation order): the persistent TMA prologue
    (LN + both projections + mask -> A2 / BT, the statistics and the exact mask count), cuBLAS for the grouped outer product,
    and the epilogue, which cuts c_z into 128-wide slices.  `after` divides by the count after the projection and its bias."""
    bf = torch.bfloat16
    m = msa[0].contiguous()
    cm, cz = m.shape[2], wo.shape[0]
    n = m.shape[1]
    ext = _ext(msa.device, native=True, tma=False, cm=cm, cz=cz)
    mk = mask[0].contiguous()
    if mk.dtype != torch.bool:
        mk = mk != 0
    a2, bt, stats, mask16, norm = ext.opm_prologue_wide(
        m,
        mk,
        lnw.detach().float().contiguous(),
        lnb.detach().float().contiguous(),
        wa.detach().to(bf).contiguous(),
        wb.detach().to(bf).contiguous(),
        float(eps),
        bool(save_stats),
    )
    o = torch.mm(a2, bt.t())  # the grouped outer product [(i,c), (j,e)], cuBLAS NT
    z = ext.opm_epilogue(
        o,
        norm,
        wo.detach().to(bf).contiguous(),
        bo.detach().to(bf).float().contiguous(),
        n,
        n,
        residual,
        int(after),
    )
    ctx.save_o = bool(save_o)
    ctx.after = bool(after)
    ctx.save_for_backward(
        m,
        mask16,
        norm,
        a2,
        bt,
        stats,
        lnw,
        lnb,
        wa,
        wb,
        wo,
        bo,
        *((o,) if save_o else ()),
    )
    ctx.eps = eps
    return z


def _backward_wide(ctx, dz):
    """Backward of `_forward_wide`: the composition of the MiniWorld training path (dgrad into the grouped layout, two cuBLAS
    GEMMs, dWo off the grouped O, the prologue backward), at this build's widths."""
    bf = torch.bfloat16
    m, mask16, norm, a2, bt, stats, lnw, lnb, wa, wb, wo, bo = ctx.saved_tensors[:12]
    o = ctx.saved_tensors[12] if ctx.save_o else None
    s, n, cm = m.shape[0], m.shape[1], m.shape[2]
    ext = _ext(dz.device, native=True, tma=False, cm=cm, cz=wo.shape[0])
    bw = (
        wo.detach().t().contiguous().to(bf)
    )  # Wo^T [CH*CH, CZ]: wgmma wants k = z contiguous
    dO, dzp, dbo = ext.opm_dgrad(dz[0].contiguous(), norm, bw, n, n, int(ctx.after))
    dA = torch.mm(bt.t(), dO.t())  # [s, (i,c)]
    dB = torch.mm(a2.t(), dO)  # [s, (j,e)]
    del dO
    if o is None:
        o = torch.mm(a2, bt.t())
    dWo = ext.opm_dwo(dzp, o, n, n)
    del o
    g = lnw.detach().float()[None, :]
    bwp = (
        torch.cat([(wa.detach().float() * g).t(), (wb.detach().float() * g).t()], dim=1)
        .to(bf)
        .contiguous()
    )  # [CM, 2*CH]: gamma folded in
    dm, dWa, dWb, dgam, dbet = ext.opm_prologue_bwd(
        dA.contiguous(),
        dB.contiguous(),
        m,
        stats,
        mask16,
        lnw.detach().float().contiguous(),
        lnb.detach().float().contiguous(),
        float(ctx.eps),
        bwp,
        n,
        s,
    )
    return (
        dm.view(1, s, n, cm),
        None,
        dgam.to(lnw.dtype),
        dbet.to(lnb.dtype),
        dWa.to(wa.dtype),
        dWb.to(wb.dtype),
        dWo.to(wo.dtype),
        dbo.to(bo.dtype),
        None,
        None,
    )


class _OpmMath(torch.autograd.Function):
    """OPM with optional residual; weights and returned gradients keep their own dtype."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(
        ctx,
        msa,
        mask,
        lnw,
        lnb,
        wa,
        wb,
        wo,
        bo,
        eps,
        save_o,
        residual=None,
        save_stats=True,
        after=False,
    ):
        cm, cz = msa.shape[3], wo.shape[0]
        path = _h100_path(msa.shape[1], msa.shape[2], training=save_stats)
        if (
            (cm, cz) != (CM, CZ)
            or after
            or (path == "legacy" and _CUDA_PROLOGUE_FOR_LEGACY)
        ):
            return _forward_wide(
                ctx,
                msa,
                mask,
                lnw,
                lnb,
                wa,
                wb,
                wo,
                bo,
                eps,
                save_o,
                residual,
                save_stats,
                after,
            )
        owned = _owned_blas_for(msa.shape[1], msa.shape[2], training=save_stats)
        ext = _ext(
            msa.device,
            native=path != "legacy",
            tma=path == "tma",
            epi_tma=_epilogue_tma_enabled(
                msa.shape[1], msa.shape[2], training=save_stats
            ),
            epi_half=_epilogue_half_enabled(
                msa.shape[1], msa.shape[2], training=save_stats
            ),
            owned=owned,
        )
        if torch.cuda.get_device_capability(msa.device) == (10, 0):
            z, m, mk, bits, a2, bt, stats, o = _forward_sm100(
                ext, msa, mask, lnw, lnb, wa, wb, wo, bo, eps, residual, save_stats
            )
            ctx.save_o = bool(save_o)
            ctx.save_for_backward(
                m,
                mk,
                bits,
                a2,
                bt,
                stats,
                lnw,
                lnb,
                wa,
                wb,
                wo,
                bo,
                *((o,) if save_o else ()),
            )
            ctx.eps = eps
            return z
        bf = torch.bfloat16
        if path != "legacy":
            m = msa[0]
            packed = ext.opm_native_pack([lnw, lnb, wa, wb, wo, bo])
            prologue = (
                ext.opm_native_prologue_tma
                if path == "tma"
                else ext.opm_native_prologue
            )
            a2, bt, stats, mask16, norm = prologue(
                m, mask[0], packed, float(eps), save_stats
            )
            o = _owned_mm(a2, bt.t(), owned)
            n = m.shape[1]
            z = ext.opm_epilogue(o, norm, packed[4], packed[5], n, n, residual)
            ctx.save_o = bool(save_o)
            ctx.save_for_backward(
                m,
                mask16,
                norm,
                a2,
                bt,
                stats,
                lnw,
                lnb,
                wa,
                wb,
                wo,
                bo,
                *((o,) if save_o else ()),
            )
            ctx.eps = eps
            return z
        m = msa[0].contiguous()
        mask16 = mask[0].to(bf).contiguous()
        _s, n = m.shape[0], m.shape[1]
        a2, bt, stats = fused_prologue(
            m,
            mask16,
            lnw.detach().float().contiguous(),
            lnb.detach().float().contiguous(),
            eps,
            wa.detach().to(bf).t().contiguous(),
            wb.detach().to(bf).t().contiguous(),
            save_stats=save_stats,
        )
        o = torch.matmul(
            a2, bt.t()
        )  # the grouped outer product [(i,c), (j,e)], cuBLAS NT
        mf = mask16.float()
        norm = (
            (mf.t() @ mf).clamp_(min=1).contiguous()
        )  # the module's own fp32 mask count
        z = ext.opm_epilogue(
            o,
            norm,
            wo.detach().contiguous().to(bf),
            bo.detach().to(bf).float().contiguous(),
            n,
            n,
            residual,
        )
        ctx.save_o = bool(save_o)
        ctx.save_for_backward(
            m,
            mask16,
            norm,
            a2,
            bt,
            stats,
            lnw,
            lnb,
            wa,
            wb,
            wo,
            bo,
            *((o,) if save_o else ()),
        )
        ctx.eps = eps
        return z  # the epilogue already returns [1, N, N, c_z]

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dz):
        bf = torch.bfloat16
        saved = ctx.saved_tensors
        m, mask16, norm, a2, bt, stats, lnw, lnb, wa, wb, wo, bo = saved[:12]
        o = saved[12] if ctx.save_o else None
        if m.shape[2] != CM or wo.shape[0] != CZ or getattr(ctx, "after", False):
            return _backward_wide(ctx, dz)
        s, n = m.shape[0], m.shape[1]
        path = _h100_path(s, n, training=True)
        owned = _owned_blas_for(s, n, training=True)
        ext = _ext(
            dz.device,
            native=path != "legacy",
            tma=path == "tma",
            epi_tma=_epilogue_tma_enabled(s, n, training=True),
            epi_half=_epilogue_half_enabled(s, n, training=True),
            owned=owned,
        )
        if torch.cuda.get_device_capability(dz.device) == (10, 0):
            # sm_100a: `mask16` is the bool mask, `norm` the bit mask, `stats` [N][S][2] -- see _forward_sm100
            dO, dzp, dbo = ext.opm_dgrad(
                dz[0].contiguous(),
                norm,
                wo.detach().to(bf).contiguous(),
                n,
                n,
                int(bo.dtype == bf),
            )
            dA = torch.mm(bt.t(), dO.t())  # [s, (i,c)]
            dB = torch.mm(a2.t(), dO)  # [s, (j,e)]
            del dO
            if o is None:
                o = torch.matmul(a2, bt.t())
            dWo = ext.opm_dwo(
                dzp, o, n, n, int(wo.dtype == bf)
            )  # gradients come back in the parameters' dtype
            del o
            dm, dWa, dWb, dgam, dbet = ext.opm_prologue_bwd(
                dA,
                dB,
                m,
                stats,
                mask16,
                lnw.detach(),
                lnb.detach(),
                wa.detach().to(bf).contiguous(),
                wb.detach().to(bf).contiguous(),
            )
            return (
                dm,
                None,
                dgam.to(lnw.dtype),
                dbet.to(lnb.dtype),
                dWa.to(wa.dtype),
                dWb.to(wb.dtype),
                dWo.to(wo.dtype),
                dbo.to(bo.dtype),
                None,
                None,
            )
        if path != "legacy":
            if dz.dtype != bf or not dz.is_contiguous():
                raise ValueError(
                    "native OPM requires contiguous BF16 upstream gradients"
                )
            packed = ext.opm_native_pack([lnw, lnb, wa, wb, wo, bo])
            dO, dzp, dbo = ext.opm_dgrad(dz[0], norm, packed[6], n, n)
            dA = _owned_mm(bt.t(), dO.t(), owned)
            dB = _owned_mm(a2.t(), dO, owned)
            if o is None:
                o = _owned_mm(a2, bt.t(), owned)
            dWo = ext.opm_dwo(dzp, o, n, n)
            dm, dWa, dWb, dgam, dbet = ext.opm_prologue_bwd(
                dA,
                dB,
                m,
                stats,
                mask16,
                packed[0],
                packed[1],
                float(ctx.eps),
                packed[7],
                n,
                s,
                True,
                packed[2:4],
            )
            grads = ext.opm_native_cast_grads(
                [dgam, dbet, dWa, dWb, dWo, dbo], [lnw, lnb, wa, wb, wo, bo]
            )
            return (dm, None, *grads, None, None)
        bw = (
            wo.detach().t().contiguous().to(bf)
        )  # Wo^T [CH*CH, CZ]: wgmma wants k = z contiguous
        dO, dzp, dbo = ext.opm_dgrad(
            dz[0].contiguous(), norm, bw, n, n
        )  # (dz / norm) . Wo in the grouped layout; dbo falls out
        dA = torch.mm(
            bt.t(), dO.t()
        )  # [s, (i,c)]: the prologue backward wants a row's channels contiguous
        dB = torch.mm(a2.t(), dO)  # [s, (j,e)]
        del dO
        if o is None:
            o = torch.matmul(
                a2, bt.t()
            )  # 309 GFLOP once, instead of 302 MB kept from the forward
        dWo = ext.opm_dwo(dzp, o, n, n)  # fp32 [CZ, CH*CH], read off the grouped O
        del o
        g = lnw.detach().float()[None, :]
        bwp = (
            torch.cat(
                [(wa.detach().float() * g).t(), (wb.detach().float() * g).t()], dim=1
            )
            .to(bf)
            .contiguous()
        )  # [CM, 2*CH]: gamma folded in
        dm, dWa, dWb, dgam, dbet = ext.opm_prologue_bwd(
            dA.contiguous(),
            dB.contiguous(),
            m,
            stats,
            mask16,
            lnw.detach().float().contiguous(),
            lnb.detach().float().contiguous(),
            float(ctx.eps),
            bwp,
            n,
            s,
        )
        return (
            dm.view(1, s, n, CM),
            None,
            dgam.to(lnw.dtype),
            dbet.to(lnb.dtype),
            dWa.to(wa.dtype),
            dWb.to(wb.dtype),
            dWo.to(wo.dtype),
            dbo.to(bo.dtype),
            None,
            None,
        )


def outer_product_mean(
    module, msa: torch.Tensor, mask: torch.Tensor, residual: torch.Tensor | None = None
) -> torch.Tensor:
    """OPM with optional pair residual fused into the CUDA epilogue; no OPM dropout in the model.

    A length that is not a multiple of 64 or an MSA depth that is not a multiple of 256 is padded with rows that carry mask 0
    (they add nothing to any count or product and the prologue zeroes their projections), and the pair output is cut back; the
    gradient to the padded rows is dropped by the pad's own backward.  The pair residual is fused only for an unpadded length."""
    eps = float(getattr(module.ln_msa, "eps", 1e-5))
    after = not getattr(module, "normalize_before_proj", True)
    cz = module.to_out.weight.shape[0]
    n0 = msa.shape[2]
    msa_in = msa
    pn, ps = -n0 % 64, -msa.shape[1] % 256
    if pn or ps:
        msa = torch.nn.functional.pad(msa, (0, 0, 0, pn, 0, ps))
        mask = torch.nn.functional.pad(mask, (0, pn, 0, ps))
    fuse = (
        residual is not None
        and not pn
        and residual.dtype == msa.dtype
        and residual.shape == (1, msa.shape[2], msa.shape[2], cz)
    )
    r = residual.contiguous() if fuse else msa.new_empty((0,))
    tensors = [
        msa.contiguous(),
        mask,
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.to_out.weight,
        module.to_out.bias,
        r,
    ]
    if torch.is_grad_enabled():
        save_o = settings.current().opm_train_save_o
        out = OpmTrainFn.apply(*tensors, eps, save_o, after)
        if torch.cuda.get_device_capability(msa.device) == (9, 0):
            from miniworld_engine.integrations.token_dit_train import (
                _native_caller_output,
            )

            leaves = (msa_in, *module.parameters(), *((residual,) if fuse else ()))
            out = _native_caller_output(out, *leaves, force=True)
    else:
        out = _inference_op(tensors, eps, after)
    if pn:
        out = out[:, :n0, :n0]
    return out + residual if residual is not None and not fuse else out


class _SavedContext:
    eps: float
    save_o: bool
    after: bool = False

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def _forward_fake(args, eps, save_o, after):
    m = args[0]
    _, s, n, _ = m.shape
    CZ = args[6].shape[0]  # d_pair of this call (the module-level CZ is MiniWorld's)
    if torch.cuda.get_device_capability(m.device) == (
        10,
        0,
    ):  # sm_100a keeps (nothing for the mask, its bits, A2, BT, stats [N, S, 2])
        return [
            torch.empty((1, n, n, CZ), device=m.device, dtype=torch.bfloat16),
            torch.empty((0,), device=m.device, dtype=m.dtype),
            torch.empty((n, s // 32), device=m.device, dtype=torch.int32),
            torch.empty((n * CH, s), device=m.device, dtype=torch.bfloat16),
            torch.empty((n * CH, s), device=m.device, dtype=torch.bfloat16),
            torch.empty((n, s, 2), device=m.device, dtype=torch.float32),
            torch.empty(
                (n * CH, n * CH) if save_o else (0,),
                device=m.device,
                dtype=torch.bfloat16,
            ),
        ]
    return [
        torch.empty((1, n, n, CZ), device=m.device, dtype=torch.bfloat16),
        torch.empty((s, n), device=m.device, dtype=torch.bfloat16),
        torch.empty((n, n), device=m.device, dtype=torch.float32),
        torch.empty((n * CH, s), device=m.device, dtype=torch.bfloat16),
        torch.empty((n * CH, s), device=m.device, dtype=torch.bfloat16),
        torch.empty((s, n, 2), device=m.device, dtype=torch.float32),
        torch.empty(
            (n * CH, n * CH) if save_o else (0,), device=m.device, dtype=torch.bfloat16
        ),
    ]


from miniworld_engine.kernels._compile import opaque


@opaque(fake=_forward_fake, name="opm_h100_fwd")
def _forward_op(
    args: list[torch.Tensor], eps: float, save_o: bool, after: bool
) -> list[torch.Tensor]:
    ctx = _SavedContext()
    out = _OpmMath.forward(
        ctx,
        *args[:8],
        eps,
        save_o,
        residual=args[8] if args[8].numel() else None,
        after=after,
    )
    sv = ctx.saved_tensors
    kept = list(sv[1:6])
    if sv[1].dtype == torch.bool:
        kept[0] = args[0].new_empty(
            (0,)
        )  # sm_100a keeps the bool mask itself, a view of an input: re-derived in the backward
    return [out, *kept, sv[12] if save_o else args[0].new_empty((0,))]


def _backward_fake(args, kept, dz, eps, save_o, after):
    return [torch.empty_like(args[i]) for i in (0, 2, 3, 4, 5, 6, 7)]


@opaque(fake=_backward_fake, name="opm_h100_bwd")
def _backward_op(
    args: list[torch.Tensor],
    kept: list[torch.Tensor],
    dz: torch.Tensor,
    eps: float,
    save_o: bool,
    after: bool,
) -> list[torch.Tensor]:
    ctx = _SavedContext()
    ctx.eps = eps
    ctx.save_o = save_o
    ctx.after = after
    k5 = list(kept[:5])
    if k5[0].numel() == 0:
        k5[0] = args[1][0] if args[1].dtype == torch.bool else args[1][0] != 0
    ctx.saved_tensors = (args[0][0], *k5, *args[2:8], *((kept[5],) if save_o else ()))
    gradients = _OpmMath.backward(ctx, dz)
    return [gradients[i] for i in (0, 2, 3, 4, 5, 6, 7)]


class OpmTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = list(args[:9])
        ctx.eps, ctx.save_o, ctx.after = args[9:]
        ctx.has_residual = bool(tensors[8].numel())
        out, *kept = _forward_op(tensors, ctx.eps, ctx.save_o, ctx.after)
        ctx.save_for_backward(*tensors, *kept)
        return out

    @staticmethod
    def backward(ctx, dz):
        vals = ctx.saved_tensors
        grads = _backward_op(
            list(vals[:9]), list(vals[9:]), dz, ctx.eps, ctx.save_o, ctx.after
        )
        return (
            grads[0],
            None,
            *grads[1:],
            dz if ctx.has_residual else None,
            None,
            None,
            None,
        )


def _inference_fake(args, eps, after):
    m = args[0]
    return torch.empty(
        (1, m.shape[2], m.shape[2], args[6].shape[0]),
        device=m.device,
        dtype=torch.bfloat16,
    )


@opaque(fake=_inference_fake, name="opm_h100_infer")
def _inference_op(args: list[torch.Tensor], eps: float, after: bool) -> torch.Tensor:
    # No training activations or LN statistics escape the inference boundary.
    if torch.cuda.get_device_capability(args[0].device) == (10, 0):
        return _forward_sm100(
            _ext(args[0].device),
            *args[:8],
            eps=eps,
            residual=args[8] if args[8].numel() else None,
            save_stats=False,
        )[0]
    return _OpmMath.forward(
        _SavedContext(),
        *args[:8],
        eps,
        False,
        residual=args[8] if args[8].numel() else None,
        save_stats=False,
        after=after,
    )
