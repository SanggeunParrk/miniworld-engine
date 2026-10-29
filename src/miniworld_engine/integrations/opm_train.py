"""OuterProductMean with measured H100 dispatch for BF16 inputs and BF16/FP32 master weights.

H100 uses a WGMMA prologue, deterministic CUDA mask/parameter passes, cuBLAS contractions and CUDA epilogues.
The seven inference cells and every training cell at S1024/2048/4096 use the persistent TMA prologue (the non-TMA WGMMA prologue is 5-6x
slower than the legacy Triton one in training; TMA matches or beats legacy at S1024-4096, L128-768, 2026-10-07).
S1024/L768 training owns cuBLAS workspaces for safe independent CUDA-graph lifetimes.
S1024/L384 and S2048/L384 inference also use the gathered-TMA, overlapping-WGMMA epilogue.
Other required inference cells retain the faster accurate historical composition;
training retains zero-gamma-safe CUDA backward. The all-shape CUDA-only goal remains open.
The H100 shape routing lives in `_h100_path` / `_epilogue_*_enabled` / `_owned_blas_for`.

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

import os
import warnings
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl

from miniworld_engine import settings
from miniworld_engine.kernels import _cast_cuda

CM, CH, CZ = 64, 32, 128
# (d_msa, d_pair) served on sm_90a: MiniWorld, Protenix-v2 / ESMFold2, OpenDDE (d_hidden 32 throughout).  The widths other than
# MiniWorld's run the wide composition: persistent TMA prologue at d_msa 64 / 128, cuBLAS contraction, the c_z-sliced epilogue.
SHAPES = ((64, 128), (128, 256), (128, 384))
# sm_100a: MiniWorld's width on csrc/sm100/opm_sm100.cu; every other (d_msa, d_pair) pair of these and RoseTTAFold3's projection biases on
# csrc/sm100/opm_wide_sm100.cu (the same stages: d_msa 128 / 256 prologue / prologue backward, c_z cut into 128-wide slices); d_msa 256 /
# d_pair 512 is IntelliFold-v2's
SM100_D_MSA, SM100_D_PAIR = (64, 128, 256), (128, 256, 384, 512)
_EXT: dict[str, Any] = {}


def _enabled() -> bool:
    """``settings.opm_train``: False disables the path; None (the default) is automatic."""
    from miniworld_engine import settings

    value = settings.current().opm_train
    return True if value is None else value


_CUDA_PROLOGUE_FOR_LEGACY = True  # inference cells that used the Triton prologue run the CUDA one (`opm_prologue_wide`): 1.5-5.5 % faster at all 11 such cells (2026-10-13)


def _h100_path(s: int, n: int, *, training: bool, wide: bool = False) -> str:
    """Only stable paired wins below the eligible historical timing are promoted."""
    if wide:  # d_msa / d_pair other than MiniWorld's: the wide composition (persistent TMA prologue at that d_msa)
        return "wide"
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

    # ``ANTHROPIC`` is not asked here: an explicit ``implementation="anthropic"`` is the Anthropic release's path
    # (integrations.anthropic_msa), not this engine's own H100 path.
    return (
        _enabled()
        and (not settings.current().force_portable_kernels)
        and implementation == ImplementationType.MINIWORLD
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
        sm100 = msa.is_cuda and torch.cuda.get_device_capability(msa.device) == (10, 0)
        if d_hidden != CH or (not sm100 and (d_msa, d_pair) not in SHAPES):
            return f"the kernels serve d_hidden={CH} and (d_msa, d_pair) in {SHAPES}, got ({d_msa}, {d_hidden}, {d_pair})"
        if interchain and not sm100:
            return (
                "mask_interchain is applied after the projection and is not fused here (sm_90a)"
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
        if cap == (10, 0) and (d_msa not in SM100_D_MSA or d_pair not in SM100_D_PAIR):
            return f"the sm_100a kernels serve d_msa in {SM100_D_MSA} and d_pair in {SM100_D_PAIR} (either normalisation order)"
        if msa.shape[0] != 1 and not sm100:   # sm_100a: `outer_product_mean` runs the batch one MSA stack at a time
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


def _fused_enabled(s: int, n: int, *, training: bool, cm: int = CM, cz: int = CZ) -> bool:
    """The inference outer product + projection as ONE kernel (`opm_fused_infer`, csrc/opm_epilogue.cu): the grouped product never goes to HBM.
    MiniWorld / RF3 widths (d_msa 64, d_pair 128), N % 16, S % 64, either normalisation order, with or without the pair residual."""
    return (not training and (cm, cz) == (CM, CZ) and n % 16 == 0 and s % 64 == 0 and s >= 64
            and not settings.current().force_portable_kernels)


def _fused_weight(wo):
    """Wo [128, 1024] bf16 with k = (c, e) permuted to (c // 8, e // 8, c % 8, e % 8): the layout `opm_fused_infer` keeps its products in."""
    from miniworld_engine.kernels import _capture

    return _capture.param_forms(
        "opm_fused_w", [wo],
        lambda: wo.detach().to(torch.bfloat16).reshape(128, 4, 8, 4, 8).permute(0, 1, 3, 2, 4).reshape(128, 1024).contiguous())


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
    tma=False,
    epi_tma=None,
    epi_half=None,
    cm: int = CM,
    cz: int = CZ,
    fused: bool = False,
):
    """This repo's fused OPM kernels for the current card: `csrc/opm_epilogue.cu` (sm_90a) or
    `csrc/sm100/opm_sm100.cu` (sm_100a, the same kernels and signatures on tcgen05 / TMEM).
    (cm, cz) other than MiniWorld's build the same source with -DOPM_CM / -DOPM_CZ (one extension per width)."""
    sm100 = torch.cuda.get_device_capability(device) == (10, 0)
    wide = (cm, cz) != (CM, CZ)
    native = True if native is None else native
    # Gathered O/W TMA stages with overlapping WGMMA groups (chosen per shape by the callers).
    epi_tma = not sm100 and native and tma and bool(epi_tma)
    epi_half = epi_tma and bool(epi_half)
    key = (
        "sm100"
        if sm100
        else (
            "mod_native_tma_static"
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
    if fused and not sm100 and not wide:   # the fused outer product + projection kernel (csrc/opm_epilogue.cu, MINIWORLD_OPM_FUSED)
        key += "_fused" + os.environ.get("OPM_FUSED_DEFS", "").replace(" ", "").replace("=", "").replace("-D", "_")
    if wide and not sm100:
        key += f"_{cm}_{cz}"
    if key in _EXT:
        return _EXT[key]
    from miniworld_engine.kernels._nvcc import load_extension as load

    csrc = Path(__file__).with_name("csrc")
    if sm100:
        _EXT[key] = load(
            name="miniworld_opm_sm100",
            sources=[str(csrc / "sm100" / "opm_sm100.cu")],
            extra_include_paths=[str(csrc / "sm100")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
            extra_cflags=["-O3"],
        )
    else:
        suffix = (
            "_native_tma_static"
            if native and tma
            else "_native"
            if native
            else ""
        )
        if epi_tma:
            suffix += "_epi_tma_overlap"
        if epi_half:
            suffix += "_half"
        if fused and not wide:
            suffix += "_fused" + os.environ.get("OPM_FUSED_DEFS", "").replace(" ", "").replace("=", "").replace("-D", "_")
        if wide:
            suffix += f"_{cm}_{cz}"
        _EXT[key] = load(
            name="miniworld_opm_epilogue" + suffix,
            sources=[str(csrc / "opm_epilogue.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
            + ([f"-DOPM_CM={cm}", f"-DOPM_CZ={cz}"] if wide else [])
            + (["-DMINIWORLD_OPM_EPILOGUE_TMA_OVERLAP=1"] if epi_tma else [])
            + (["-DMINIWORLD_OPM_EPILOGUE_TMA_HALF=1"] if epi_half else [])
            + (["-DMINIWORLD_OPM_FUSED=1", *os.environ.get("OPM_FUSED_DEFS", "").split()] if fused and not wide else []),
            extra_cflags=["-O3"],
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


def _fused_sm100_enabled(s: int, n: int) -> bool:
    """B200 inference: the outer product + projection as ONE kernel (`opm_fused_infer`, csrc/sm100/opm_sm100.cu) at the cells where it
    measured faster than cuBLAS + the epilogue kernel (`_FUSED_SM100_CELLS`; docs/gpus/b200/opm/opm.md)."""
    return (n, s) in _FUSED_SM100_CELLS and not settings.current().force_portable_kernels


# (N, S) after the wrapper's padding (N % 64, S % 256) where the fused kernel is served: none, as measured 2026-10-29 (B200, CUDA graph).
# Module harness (bench.py, ms, two runs each; today's path -> fused): S1024 L128 0.057 / 0.055 -> 0.057 / 0.057, L384 0.309 / 0.295 ->
# 0.303 / 0.294, L512 0.496 / 0.494 -> 0.503 / 0.514, L640 0.729 / 0.731 -> 0.808 / 0.795, L768 1.043 / 1.058 -> 1.153 / 1.051;
# S4096 L768 3.528 / 3.687 -> 3.472 / 3.651.  Kernel pair in steady state under the 1000 W cap (us, cuBLAS + epilogue -> fused): S1024
# L384 293 -> 288, L512 513 -> 505, L640 800 -> 788, L768 1144 -> 1141; S2048 +4-8 %; S4096 +9-10 %.  A cell goes here once it wins
# on the harness (docs/gpus/b200/opm/opm.md: what is left is the fused mainloop's energy per product).
_FUSED_SM100_CELLS: frozenset[tuple[int, int]] = frozenset()


def _ext_wide(device=None):
    """`csrc/sm100/opm_wide_sm100.cu`: the B200 OPM kernels at the other widths (d_msa 64 / 128, d_pair 128 / 256 / 384) and with
    RoseTTAFold3's projection biases; MiniWorld's width keeps `opm_sm100.cu`."""
    if "sm100_wide" not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        csrc = Path(__file__).with_name("csrc")
        _EXT["sm100_wide"] = load(
            name="miniworld_opm_wide_sm100",
            sources=[str(csrc / "sm100" / "opm_wide_sm100.cu")],
            extra_include_paths=[str(csrc / "sm100")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
            extra_cflags=["-O3"],
        )
    return _EXT["sm100_wide"]


# B200 training backward: dWo (an HBM-bound hand kernel) runs on a side stream on a few SMs while cuBLAS computes dA and dB on the
# others (an SM-count target for those two GEMMs).  The GEMMs are bound by the 1000 W cap, not by their SM count, so they lose little
# of their rate to the carve-out while dWo's time leaves the step.  {c_z: CTA budget} where it measured faster (docs/gpus/b200/opm/opm.md).
_DWO_SIDE_CTAS: dict[int, int] = {128: 16, 256: 16}
_DWO_SIDE_MIN_N = 128      # (L128 S1024 MiniWorld width: 0.167 -> 0.154 ms; shorter lengths are not measured)
_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}
# torch warns once a process that its SM carve-out is experimental; the engine sets it only around those two GEMMs (and checks it exists)
warnings.filterwarnings("ignore", message="Setting the SM carveout for matmuls is a temporary experimental mitigation")


def _dwo_side_ctas(cz: int, s: int, n: int) -> int:
    """The dWo CTA budget beside the gradient GEMMs (0: dWo after them on every SM)."""
    if n < _DWO_SIDE_MIN_N or not hasattr(torch._C, "_set_sm_carveout_experimental"):
        return 0
    return _DWO_SIDE_CTAS.get(cz, 0)


def _side_stream(device) -> torch.cuda.Stream:
    idx = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    if idx not in _SIDE_STREAMS:
        _SIDE_STREAMS[idx] = torch.cuda.Stream(device=idx)
    return _SIDE_STREAMS[idx]


class _sm_carveout:
    """cuBLAS leaves `n` SMs free for the GEMMs issued inside (CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, recorded into a CUDA graph)."""

    def __init__(self, n: int):
        self.n = n

    def __enter__(self):
        self.prev = torch._C._get_sm_carveout_experimental()
        torch._C._set_sm_carveout_experimental(self.n)

    def __exit__(self, *exc):
        torch._C._set_sm_carveout_experimental(self.prev)
        return False


def _bf16_params(ext, ps):
    """bf16 forms of the parameters `ps` (Wa, Wb, Wo, the bias[, the projection biases]): with fp32 master parameters ONE launch writes
    them all into one flat buffer (`opm_cast_bf16`, segments at 64-element boundaries) -- a cast kernel each, in the forward and again in
    the backward, were seven launches of the training step.  Returns the views and the buffer (None when every parameter is bf16 already,
    as with bf16 parameters: no kernel)."""
    if all(p.dtype == torch.bfloat16 for p in ps):
        return [p.detach().contiguous() for p in ps], None
    flat = ext.opm_cast_bf16([p.detach().contiguous() for p in ps])
    return _bf16_views(flat, ps), flat


def _bf16_views(flat, ps):
    views, off = [], 0
    for p in ps:
        views.append(flat[off:off + p.numel()].view(p.shape))
        off += (p.numel() + 63) // 64 * 64
    return views


def _forward_sm100(ext, msa, mask, lnw, lnb, wa, wb, wo, bo, eps, residual, save_stats, after=False, pbias=None, asym=None):
    """B200: the same three stages on sm_100a kernels -- prologue (LN + both projections + mask -> A2 / BT, the LN
    statistics as [N][S] and the mask as bits), cuBLAS for the grouped outer product, the mask counts [N, N] off the bits (one small
    kernel, kept for the dgrad), and the epilogue.  `after`: the count divides the projection
    and its bias (ESMFold2's order, and AF3's as released).  `pbias` (RoseTTAFold3's left | right projection biases, [64]):
    the prologue adds them to the projections, masked like them; widths other than MiniWorld's and the biases run
    `opm_wide_sm100.cu` (the epilogue cuts c_z into 128-wide slices).  `asym` (int32 [N] chain ids, the interchain mask): the epilogue
    zeroes the update of a pair of two chains (the residual passes through).  The last value returned is the flat bf16 buffer of the
    parameters' bf16 forms (`_bf16_params`; None with bf16 parameters), which the training backward reuses."""
    bf = torch.bfloat16
    m = msa[0].contiguous()
    mk = mask[0].contiguous()
    if mk.dtype != torch.bool:
        mk = mk != 0
    cm, cz = m.shape[2], wo.shape[0]
    (wa16, wb16, wo16, bo16, *pb16), w16 = _bf16_params(ext, [wa, wb, wo, bo, *((pbias,) if pbias is not None else ())])
    if (cm, cz) != (CM, CZ) or pbias is not None:
        wext = _ext_wide(msa.device)
        # the biases in bf16, as the module's bf16 Linear adds them (its fp32 accumulation, one rounding of the sum)
        pb = pb16[0].float() if pbias is not None else m.new_empty((0,), dtype=torch.float32)
        a2, bt, stats, bits = wext.opm_wprologue(
            m, mk, lnw.detach().contiguous(), lnb.detach().contiguous(), float(eps), wa16, wb16, pb, bool(save_stats))
        n = m.shape[1]
        if not save_stats and cz == CZ and asym is None and _fused_sm100_enabled(m.shape[0], n):
            z = ext.opm_fused_infer(a2, bt, bits, _fused_weight(wo), bo16, residual, int(after))
            return z, m, mk, bits, a2, bt, stats, None, w16
        o = torch.matmul(a2, bt.t())
        norm = ext.opm_norm(bits)   # the mask counts [N, N] (one small kernel; the epilogue and the dgrad read them, one tile ahead)
        if cz == CZ:
            z = ext.opm_epilogue(o, norm, wo16, bo16, n, n, residual, int(after), asym)
        else:
            z = wext.opm_wepilogue(o, norm, wo16, bo16, n, n, residual, int(after), asym)
        return z, m, mk, norm, a2, bt, stats, o, w16
    a2, bt, _, stats, bits = ext.opm_prologue(m, mk, lnw.detach().contiguous(), lnb.detach().contiguous(), float(eps), wa16, wb16,
                                              save_stats, False)
    n = m.shape[1]
    if not save_stats and asym is None and _fused_sm100_enabled(m.shape[0], n):
        # inference: the products stay on chip (k-permuted weight as H100's fused kernel; the bias rounded to bf16 as below)
        z = ext.opm_fused_infer(a2, bt, bits, _fused_weight(wo), bo16, residual, int(after))
        return z, m, mk, bits, a2, bt, stats, None, w16
    o = torch.matmul(a2, bt.t())  # the grouped outer product [(i,c), (j,e)], cuBLAS NT
    norm = ext.opm_norm(bits)     # the mask counts [N, N], read by the epilogue and the dgrad
    z = ext.opm_epilogue(o, norm, wo16, bo16, n, n, residual, int(after), asym)
    return z, m, mk, norm, a2, bt, stats, o, w16
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
    n = m.shape[1]
    if not save_stats and asym is None and _fused_sm100_enabled(m.shape[0], n):
        # inference: the products stay on chip (k-permuted weight as H100's fused kernel; the bias rounded to bf16 as below)
        z = ext.opm_fused_infer(a2, bt, bits, _fused_weight(wo), bo.detach().to(bf).contiguous(), residual, int(after))
        return z, m, mk, bits, a2, bt, stats, None
    o = torch.matmul(a2, bt.t())  # the grouped outer product [(i,c), (j,e)], cuBLAS NT
    norm = ext.opm_norm(bits)     # the mask counts [N, N], read by the epilogue and the dgrad
    z = ext.opm_epilogue(
        o,
        norm,
        wo.detach().to(bf).contiguous(),
        bo.detach().to(bf).contiguous(),
        n,
        n,
        residual,
        int(after),
        asym,
    )
    return z, m, mk, norm, a2, bt, stats, o


def _forward_wide(
    ctx, msa, mask, lnw, lnb, wa, wb, wo, bo, eps, save_o, residual, save_stats, after, pbias=None
):
    """The wide composition (d_msa 64 / 128, d_pair 128 / 256 / 384, either normalisation order): the persistent TMA prologue
    (LN + both projections + mask -> A2 / BT, the statistics and the exact mask count), cuBLAS for the grouped outer product,
    and the epilogue, which cuts c_z into 128-wide slices.  `after` divides by the count after the projection and its bias."""
    bf = torch.bfloat16
    m = msa[0].contiguous()
    cm, cz = m.shape[2], wo.shape[0]
    n = m.shape[1]
    fused = _fused_enabled(m.shape[0], n, training=save_stats, cm=cm, cz=cz) and torch.cuda.get_device_capability(msa.device) == (9, 0)
    ext = _ext(msa.device, native=True, tma=False, cm=cm, cz=cz, fused=fused)
    mk = mask[0].contiguous()
    if mk.dtype != torch.bool:
        mk = mk != 0
    def forms():
        return (
            lnw.detach().float().contiguous(),
            lnb.detach().float().contiguous(),
            wa.detach().to(bf).contiguous(),
            wb.detach().to(bf).contiguous(),
            wo.detach().to(bf).contiguous(),
            bo.detach().to(bf).float().contiguous(),
            pbias.detach().float().contiguous() if pbias is not None else None,   # RoseTTAFold3's projection biases [left | right]
        )

    if save_stats:
        gam, bet, wa16, wb16, wo16, ob, pb = forms()
    else:   # inference: the kernel-dtype forms of the weights are made once while the weights are unchanged (no cast kernels per call)
        from miniworld_engine.kernels import _capture

        gam, bet, wa16, wb16, wo16, ob, pb = _capture.param_forms(
            "opm_wide_infer", [t for t in (lnw, lnb, wa, wb, wo, bo, pbias) if t is not None], forms)
    # inference: the epilogue takes the mask count from the mask bit rows (`norm` is that int32 [N, S/32] table), no separate count kernel
    a2, bt, stats, mask16, norm = ext.opm_prologue_wide(m, mk, gam, bet, wa16, wb16, float(eps), bool(save_stats), pb, bool(save_stats))
    ctx.has_bias = pbias is not None
    if fused:   # inference: the products stay in shared memory (norm is the mask bit table here)
        z = ext.opm_fused_infer(a2, bt, norm, _fused_weight(wo), ob, residual, int(after))
        o = None
    else:
        o = torch.mm(a2, bt.t())  # the grouped outer product [(i,c), (j,e)], cuBLAS NT
        z = ext.opm_epilogue(o, norm, wo16, ob, n, n, residual, int(after))
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
    # the projection biases (RoseTTAFold3): left[s, i, c] = mask[s, i] (proj + b_L[c]), so db_L[c] = sum_{s, i} mask[s, i] dA[s, i, c]
    # (dA is the gradient of the grouped A2 in [s, (i, c)] order), one fp32-accumulated GEMV per side
    db = None
    if getattr(ctx, "has_bias", False):
        ch = dA.shape[1] // n
        mk = mask16.reshape(1, -1)
        db = torch.cat([torch.mm(mk, dA.reshape(-1, ch), out_dtype=torch.float32).reshape(-1),
                        torch.mm(mk, dB.reshape(-1, ch), out_dtype=torch.float32).reshape(-1)])
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
        db,
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
        pbias=None,
        asym=None,
    ):
        cm, cz = msa.shape[3], wo.shape[0]
        if torch.cuda.get_device_capability(msa.device) == (10, 0):   # B200: every width of `refusal`, either order, the biases, interchain
            z, m, mk, bits, a2, bt, stats, o, w16 = _forward_sm100(
                _ext(msa.device), msa, mask, lnw, lnb, wa, wb, wo, bo, eps, residual, save_stats, after, pbias, asym
            )
            ctx.w16 = w16   # the parameters' bf16 forms (one flat buffer, or None), reused by the backward
            ctx.save_o = bool(save_o)
            ctx.has_bias = pbias is not None
            ctx.asym = asym
            ctx.save_for_backward(m, mk, bits, a2, bt, stats, lnw, lnb, wa, wb, wo, bo, *((o,) if save_o else ()))
            ctx.eps = eps
            return z
        path = _h100_path(msa.shape[1], msa.shape[2], training=save_stats)
        if (
            (cm, cz) != (CM, CZ)
            or after
            or pbias is not None
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
                pbias,
            )
        owned = _owned_blas_for(msa.shape[1], msa.shape[2], training=save_stats)
        fused = path != "legacy" and _fused_enabled(msa.shape[1], msa.shape[2], training=save_stats) \
            and torch.cuda.get_device_capability(msa.device) == (9, 0)
        ext = _ext(
            msa.device,
            native=path != "legacy",
            tma=path == "tma",
            epi_tma=not fused and _epilogue_tma_enabled(
                msa.shape[1], msa.shape[2], training=save_stats
            ),
            epi_half=not fused and _epilogue_half_enabled(
                msa.shape[1], msa.shape[2], training=save_stats
            ),
            fused=fused,
        )
        bf = torch.bfloat16
        if path != "legacy":
            m = msa[0]
            if save_stats:
                packed = ext.opm_native_pack([lnw, lnb, wa, wb, wo, bo], False)
            else:   # inference: no transposed Wo / backward weight; the packed forms are cached while the weights are unchanged
                from miniworld_engine.kernels import _capture

                packed = _capture.param_forms(
                    "opm_native_pack_infer",
                    [lnw, lnb, wa, wb, wo, bo],
                    lambda: ext.opm_native_pack([lnw, lnb, wa, wb, wo, bo], True),
                )
            prologue = (
                ext.opm_native_prologue_tma
                if path == "tma"
                else ext.opm_native_prologue
            )
            # inference: the mask count is taken by the epilogue from the mask bit rows (`norm` is then that int32 [N, S/32] table)
            a2, bt, stats, mask16, norm = prologue(
                m, mask[0], packed, float(eps), save_stats, save_stats
            )
            n = m.shape[1]
            if fused:   # inference: the products stay in shared memory (norm is the mask bit table here)
                z = ext.opm_fused_infer(a2, bt, norm, _fused_weight(wo), packed[5], residual, 0)
                o = None
            else:
                o = _owned_mm(a2, bt.t(), owned)
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
        sm100 = torch.cuda.get_device_capability(dz.device) == (10, 0)
        if not sm100 and (m.shape[2] != CM or wo.shape[0] != CZ or getattr(ctx, "after", False) or getattr(ctx, "has_bias", False)):
            return _backward_wide(ctx, dz)
        s, n = m.shape[0], m.shape[1]
        if sm100:
            ext = _ext(dz.device)
        else:
            path = _h100_path(s, n, training=True)
            owned = _owned_blas_for(s, n, training=True)
            ext = _ext(
                dz.device,
                native=path != "legacy",
                tma=path == "tma",
                epi_tma=_epilogue_tma_enabled(s, n, training=True),
                epi_half=_epilogue_half_enabled(s, n, training=True),
            )
        if sm100:
            # sm_100a: `mask16` is the bool mask, `norm` the fp32 [N, N] mask counts, `stats` [N][S][2] -- see _forward_sm100.  The dgrad is
            # the CTA-pair kernel of opm_wide_sm100.cu at every c_z (MiniWorld's 128 included: 327 -> 292 us at L768 S1024); MiniWorld's width
            # keeps opm_sm100.cu's dWo and prologue backward, c_z 256 / 384 / 512 the c_z-sliced dWo and d_msa 128 / 256 the wide prologue
            # backward (opm_wide_sm100.cu)
            cm, cz = m.shape[2], wo.shape[0]
            wext = _ext_wide(dz.device)
            w16 = getattr(ctx, "w16", None)
            if w16 is not None:   # the forward's bf16 forms of Wa, Wb, Wo (no cast kernel here)
                wa16, wb16, wo16 = _bf16_views(w16, [wa, wb, wo])
            else:
                wa16, wb16, wo16 = (w.detach().to(bf).contiguous() for w in (wa, wb, wo))
            dgrad = wext.opm_wdgrad
            # the interchain mask (chain ids): a pair of two chains passes no gradient (dO, dz/n and dbias without it)
            # z = (O Wo^T + b) / n (`after`): the dgrad's dbias is sum dz / n
            dO, dzp, dbo = dgrad(dz[0].contiguous(), norm, wo16, n, n, int(bo.dtype == bf), getattr(ctx, "asym", None),
                                 int(getattr(ctx, "after", False)))
            if o is None:
                o = torch.matmul(a2, bt.t())
            dwo = ext.opm_dwo if cz == CZ else wext.opm_wdwo
            side = _dwo_side_ctas(cz, s, n)
            if side:
                # dWo (HBM-bound) on `side` SMs of a side stream while the two power-bound gradient GEMMs run on the rest
                main = torch.cuda.current_stream(dz.device)
                aux = _side_stream(dz.device)
                aux.wait_stream(main)
                with torch.cuda.stream(aux):
                    dWo = dwo(dzp, o, n, n, int(wo.dtype == bf), side)
                dzp.record_stream(aux)
                o.record_stream(aux)
                with _sm_carveout(side):
                    dA = torch.mm(bt.t(), dO.t())  # [s, (i,c)]
                    dB = torch.mm(a2.t(), dO)  # [s, (j,e)]
                main.wait_stream(aux)
                dWo.record_stream(main)
            else:
                dA = torch.mm(bt.t(), dO.t())  # [s, (i,c)]
                dB = torch.mm(a2.t(), dO)  # [s, (j,e)]
                dWo = dwo(dzp, o, n, n, int(wo.dtype == bf))  # gradients come back in the parameters' dtype
            del dO, o
            if cm == CM:
                # ssa: the masked column sums of [dA dB] (the kernel's row 64 of R), = RoseTTAFold3's projection-bias gradients
                dm, dWa, dWb, dgam, dbet, red = ext.opm_prologue_bwd(dA, dB, m, stats, mask16, lnw.detach(), lnb.detach(), wa16, wb16)
                ssa = red.reshape(-1)[CM * CM:CM * CM + 2 * CH]
            else:
                # d_msa 128: the masked column sums leave the kernel (its dW GEMM's 128 rows are the channels): one GEMV a side
                mkf = mask16.reshape(1, -1).to(bf)
                ssa = torch.cat([torch.mm(mkf, dA.view(-1, CH), out_dtype=torch.float32).view(-1),
                                 torch.mm(mkf, dB.view(-1, CH), out_dtype=torch.float32).view(-1)])
                dm, dWa, dWb, dgam, dbet = wext.opm_wprologue_bwd(dA, dB, m, stats, mask16, lnw.detach(), lnb.detach(), wa16, wb16, ssa)
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
                ssa.clone() if getattr(ctx, "has_bias", False) else None,
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
            grads = _cast_cuda.cast(
                [dgam, dbet, dWa, dWb, dWo, dbo],
                [torch.float32 if p.dtype == torch.float32 else torch.bfloat16 for p in (lnw, lnb, wa, wb, wo, bo)],
                force=True,
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
    module, msa: torch.Tensor, mask: torch.Tensor, residual: torch.Tensor | None = None, token_asym_id: torch.Tensor | None = None
) -> torch.Tensor:
    """OPM with optional pair residual fused into the CUDA epilogue; no OPM dropout in the model.

    A length that is not a multiple of 64 or an MSA depth that is not a multiple of 256 is padded with rows that carry mask 0
    (they add nothing to any count or product and the prologue zeroes their projections), and the pair output is cut back; the
    gradient to the padded rows is dropped by the pad's own backward.  The pair residual is fused only for an unpadded length.

    B200 (sm_100a): `token_asym_id` [B, N] (given only for the module's interchain mask) zeroes the update of a pair of two chains in the
    epilogue and its gradient in the dgrad; a batch of MSA stacks runs one stack at a time (each its own autograd node, the parameter
    gradients summed by autograd)."""
    if msa.shape[0] > 1:
        outs = [
            outer_product_mean(module, msa[b:b + 1], mask[b:b + 1], None if residual is None else residual[b:b + 1],
                               None if token_asym_id is None else token_asym_id[b:b + 1])
            for b in range(msa.shape[0])
        ]
        return torch.cat(outs, 0)
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
    # RoseTTAFold3: the left | right projection biases as one fp32 vector (an empty tensor: no biases)
    pbias = (torch.cat([module.to_left.bias, module.to_right.bias]) if module.to_left.bias is not None
             else msa.new_empty((0,), dtype=torch.float32))
    # the interchain mask: chain ids as int32 [N] (padded tokens -1: their pairs are cut away), or an empty tensor
    asym = msa.new_empty((0,), dtype=torch.int32)
    if token_asym_id is not None:
        asym = token_asym_id[0].to(torch.int32)
        if pn:
            asym = torch.nn.functional.pad(asym, (0, pn), value=-1)
        asym = asym.contiguous()
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
        pbias,
        asym,
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
    ):  # sm_100a keeps (nothing for the mask, the mask counts [N, N], A2, BT, stats [N, S, 2])
        return [
            torch.empty((1, n, n, CZ), device=m.device, dtype=torch.bfloat16),
            torch.empty((0,), device=m.device, dtype=m.dtype),
            torch.empty((n, n), device=m.device, dtype=torch.float32),
            torch.empty((n * CH, s), device=m.device, dtype=torch.bfloat16),
            torch.empty((n * CH, s), device=m.device, dtype=torch.bfloat16),
            torch.empty((n, s, 2), device=m.device, dtype=torch.float32),
            torch.empty(
                (n * CH, n * CH) if save_o else (0,),
                device=m.device,
                dtype=torch.bfloat16,
            ),
            torch.empty((_bf16_flat_numel(args),), device=m.device, dtype=torch.bfloat16),
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
        torch.empty((0,), device=m.device, dtype=torch.bfloat16),
    ]


def _bf16_flat_numel(args) -> int:
    """The length of `_bf16_params`' flat buffer for these op arguments (0: every parameter bf16, no buffer)."""
    ps = [args[4], args[5], args[6], args[7], *((args[9],) if args[9].numel() else ())]
    if all(p.dtype == torch.bfloat16 for p in ps):
        return 0
    return sum((p.numel() + 63) // 64 * 64 for p in ps)


from miniworld_engine.kernels._compile import opaque


@opaque(fake=_forward_fake, name="opm_train_fwd")   # (was opm_h100_fwd: the sm_100a forward now keeps the mask counts, a stale cached graph must not match)
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
        pbias=args[9].detach() if args[9].numel() else None,
        asym=args[10] if args[10].numel() else None,
    )
    sv = ctx.saved_tensors
    kept = list(sv[1:6])
    if sv[1].dtype == torch.bool:
        kept[0] = args[0].new_empty(
            (0,)
        )  # sm_100a keeps the bool mask itself, a view of an input: re-derived in the backward
    w16 = getattr(ctx, "w16", None)   # sm_100a: the parameters' bf16 forms (flat), for the backward
    return [out, *kept, sv[12] if save_o else args[0].new_empty((0,)),
            w16 if w16 is not None else args[0].new_empty((0,), dtype=torch.bfloat16)]


def _backward_fake(args, kept, dz, eps, save_o, after):
    return [torch.empty_like(args[i]) for i in (0, 2, 3, 4, 5, 6, 7, 9)]


@opaque(fake=_backward_fake, name="opm_train_bwd")
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
    ctx.has_bias = bool(args[9].numel())
    ctx.asym = args[10] if args[10].numel() else None
    k5 = list(kept[:5])
    if k5[0].numel() == 0:
        k5[0] = args[1][0] if args[1].dtype == torch.bool else args[1][0] != 0
    ctx.saved_tensors = (args[0][0], *k5, *args[2:8], *((kept[5],) if save_o else ()))
    ctx.w16 = kept[6] if len(kept) > 6 and kept[6].numel() else None
    gradients = _OpmMath.backward(ctx, dz)
    db = gradients[10].to(args[9].dtype) if ctx.has_bias else args[9].new_empty((0,))
    return [*(gradients[i] for i in (0, 2, 3, 4, 5, 6, 7)), db]


class OpmTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = list(args[:11])
        ctx.eps, ctx.save_o, ctx.after = args[11:]
        ctx.has_residual = bool(tensors[8].numel())
        ctx.has_bias = bool(tensors[9].numel())
        out, *kept = _forward_op(tensors, ctx.eps, ctx.save_o, ctx.after)
        ctx.save_for_backward(*tensors, *kept)
        return out

    @staticmethod
    def backward(ctx, dz):
        vals = ctx.saved_tensors
        grads = _backward_op(
            list(vals[:11]), list(vals[11:]), dz, ctx.eps, ctx.save_o, ctx.after
        )
        return (
            grads[0],
            None,
            *grads[1:7],
            dz if ctx.has_residual else None,
            grads[7] if ctx.has_bias else None,
            None,
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
        # the projection biases (args[9]) go into the prologue: no caller has to add them afterwards
        return _forward_sm100(
            _ext(args[0].device),
            *args[:8],
            eps=eps,
            residual=args[8] if args[8].numel() else None,
            save_stats=False,
            after=after,
            pbias=args[9] if args[9].numel() else None,
            asym=args[10] if args[10].numel() else None,
        )[0]
    return _OpmMath.forward(
        _SavedContext(),
        *args[:8],
        eps,
        False,
        residual=args[8] if args[8].numel() else None,
        save_stats=False,
        after=after,
        pbias=args[9] if args[9].numel() else None,
    )
