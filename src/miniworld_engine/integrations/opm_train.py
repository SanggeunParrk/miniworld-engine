"""The OuterProductMean TRAINING path: one autograd.Function over the fused OPM kernels developed in MiniWorld's
`runs/msa_bench_20260921` (2026-09-22): the upstream-style fused Triton PROLOGUE (LayerNorm + both projections + mask
straight into the GEMM layouts, vendored below with the LayerNorm statistics kept for the backward), cuBLAS for the
grouped outer product, and this repo's `csrc/opm_epilogue.cu` for everything else:

Forward:   fused_prologue -> A2 [(i,c), s], BT [(j,e), s] bf16 (+ stats [S,N,2]);  O = A2 . BT^T (cuBLAS, grouped layout);
           norm = mask^T mask (fp32, clamp 1);  opm_epilogue: [i,j,c,e] -> [(i,j),(c,e)] permute, / norm, bf16, c_hidden^2 -> c_z + bias.
Backward:  opm_dgrad (dz/norm . Wo straight into the grouped layout, dbo on the side), two cuBLAS GEMMs for dA / dB,
           opm_dwo (dWo off the grouped O: kept from the forward when MINIWORLD_OPM_TRAIN_SAVE_O != 0, else one GEMM recomputes it),
           opm_prologue_bwd (mask, both projections and the LayerNorm backward in one pass -> dm, dWa, dWb, dgamma, dbeta).
Measured H100, L=384, S=1024, bf16: fwd+bwd 2.23 ms / 796 MB (O kept) or 2.63 ms / 557 MB, vs this engine's own path
5.80 ms / 1624 MB; every gradient checked against fp32 autograd of the same statements.

Opt-in: MINIWORLD_OPM_TRAIN=1 (no external payload needed). Serves a grad-enabled call of a module built with
implementation="miniworld" or "anthropic" when: d_msa=64, d_hidden=32, d_pair=128, batch 1, N % 64 == 0, S % 256 == 0,
bf16 on sm_90a, normalize_before_proj, no interchain mask. Returns OPM(msa) WITHOUT the residual (the module adds it).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

ENV = "MINIWORLD_OPM_TRAIN"
CM, CH, CZ = 64, 32, 128
_EXT: dict[str, Any] = {}


def wanted(implementation) -> bool:
    from miniworld_engine.modules.exceptions import ImplementationType
    return bool(os.environ.get(ENV)) and implementation in (ImplementationType.MINIWORLD, ImplementationType.ANTHROPIC)


def refusal(msa: torch.Tensor, d_msa: int, d_hidden: int, d_pair: int, *, interchain: bool, normalize_before_proj: bool) -> str | None:
    """None if this path can run this training call, else why it cannot. Never raises."""
    try:
        if not os.environ.get(ENV):
            return f"{ENV} is not set"
        if (d_msa, d_hidden, d_pair) != (CM, CH, CZ):
            return f"the kernels serve (d_msa={CM}, d_hidden={CH}, d_pair={CZ}), got ({d_msa}, {d_hidden}, {d_pair})"
        if interchain:
            return "mask_interchain is applied after the projection and is not fused here"
        if not normalize_before_proj:
            return "the epilogue divides by the mask count BEFORE the projection (AF3 order)"
        if msa.dtype != torch.bfloat16:
            return f"the kernels are bf16, got {msa.dtype}"
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        if torch.cuda.get_device_capability(msa.device) != (9, 0):
            return "the kernels are built for sm_90a"
        if msa.shape[0] != 1:
            return f"one MSA stack per call, got batch {msa.shape[0]}"
        s, n = msa.shape[1], msa.shape[2]
        if n % 64 or s % 256:
            return f"N must be a multiple of 64 and S of 256 (the prologue backward's 32 x 8 row blocks), got N={n}, S={s}"
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


STATS: dict[str, Any] = {"served": 0, "refused": {}}       # how often the path ran / why it did not (for a real-model check)


def serves(*a, **kw) -> bool:
    why = refusal(*a, **kw)
    if why is None:
        STATS["served"] += 1
        return True
    if os.environ.get(ENV) and why not in STATS["refused"]:     # opted in but not served: say why, once per reason
        import logging
        logging.getLogger("miniworld_engine").warning("%s: the fused training path is not taken: %s", ENV, why)
    STATS["refused"][why] = STATS["refused"].get(why, 0) + 1
    return False


def _ext():
    """This repo's fused OPM kernels (the same extension the inference path builds)."""
    if "mod" in _EXT:
        return _EXT["mod"]
    from torch.utils.cpp_extension import load
    src = Path(__file__).with_name("csrc") / "opm_epilogue.cu"
    build = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit")) / "opm_epilogue"
    build.mkdir(parents=True, exist_ok=True)
    _EXT["mod"] = load(name="miniworld_opm_epilogue", sources=[str(src)], build_directory=str(build),
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], extra_cflags=["-O3"])
    return _EXT["mod"]


# ---- the fused prologue (vendored from the upstream msa_opm cell, with the LayerNorm statistics kept) ----
@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _opm_prologue_kernel(M, MASK, LNW, LNB, WA, WB, BA, BB, A2, BT, STATS,
                         S, N, SK, NA, NBj, stride_ms, stride_mi, eps,
                         CM: tl.constexpr, CH: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, BIP: tl.constexpr,
                         HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, LN_AFFINE: tl.constexpr, MASK_I64: tl.constexpr,
                         SAVE_STATS: tl.constexpr):
    # Fused stock prologue: y = LN(m[s, i, :]) in fp32 (torch.nn.LayerNorm math: biased var, rsqrt(var + eps), affine), y -> bf16 (autocast),
    # a = y16 @ Wa^T, b = y16 @ Wb^T with fp32 accumulation (+ bf16-rounded bias in fp32, one rounding: the cuBLASLt epilogue), -> bf16,
    # * mask (0/1: exact).  Written directly in the GEMM layouts: A2[(iblk, c, bi), s] (iblk = i // BI) and BT[(i, e), s], zero-padded to
    # (NA | NBj) x SK.  Tile = BIP tokens x BS MSA rows; rows ordered (i, s) so that the stores are contiguous along s.
    pid_s = tl.program_id(0).to(tl.int64)     # int64 program indices: every m / mask / A2 / BT element offset below is formed in int64
    pid_i = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BIP
    R: tl.constexpr = BIP * BS
    rr = tl.arange(0, R)
    r_i = i0 + rr // BS
    r_s = s0 + rr % BS
    valid = (r_s < S) & (r_i < N)
    cols = tl.arange(0, CM)
    x = tl.load(M + r_s[:, None] * stride_ms + r_i[:, None] * stride_mi + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)   # [R, CM]
    mean = tl.sum(x, axis=1) / CM
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / CM
    rstd = 1.0 / tl.sqrt(var + eps)
    if SAVE_STATS:   # the LayerNorm statistics, for a backward that would otherwise re-derive them from m: STATS[s, i] = (mean, rstd)
        tl.store(STATS + (r_s * N + r_i) * 2, mean, mask=valid)
        tl.store(STATS + (r_s * N + r_i) * 2 + 1, rstd, mask=valid)
    y = xc * rstd[:, None]
    if LN_AFFINE:
        y = y * tl.load(LNW + cols).to(tl.float32)[None, :] + tl.load(LNB + cols).to(tl.float32)[None, :]
    y16 = y.to(tl.bfloat16)
    ch = tl.arange(0, CH)
    wa = tl.load(WA + cols[:, None] * CH + ch[None, :])          # [CM, CH] bf16 = Wa^T
    wb = tl.load(WB + cols[:, None] * CH + ch[None, :])
    a = tl.dot(y16, wa)                                           # [R, CH] fp32
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
    a16 = tl.where(valid[:, None], a16, a16 * 0.0)                # zero padding rows (s >= S or i >= N)
    b16 = tl.where(valid[:, None], b16, b16 * 0.0)
    a_rows = ((r_i // BI) * CH)[:, None] + ch[None, :]
    a_ptr = A2 + (a_rows * BI + (r_i % BI)[:, None]) * SK + r_s[:, None]
    tl.store(a_ptr, a16, mask=((r_i < NA) & (r_s < SK))[:, None])
    b_ptr = BT + (r_i[:, None] * CH + ch[None, :]) * SK + r_s[:, None]
    tl.store(b_ptr, b16, mask=((r_i < NBj) & (r_s < SK))[:, None])


def fused_prologue(m, mask, lnw, lnb, eps, wa_t, wb_t, BI=1, BJ=1, BK=64, num_warps=8, BS=32, BIP=8):
    """m: [S, N, CM] bf16, mask: [S, N] bf16 0/1 -> (A2 [N*CH, SK], BT [N*CH, SK]) bf16 with A2 row = i*CH + c, and stats [S, N, 2] fp32 (mean, rstd)."""
    S, N, cm = m.shape
    NA = triton.cdiv(N, BI) * BI; NBj = triton.cdiv(N, BJ) * BJ; SK = triton.cdiv(S, BK) * BK
    A2 = torch.empty(NA * CH, SK, device=m.device, dtype=torch.bfloat16)
    BT = torch.empty(NBj * CH, SK, device=m.device, dtype=torch.bfloat16)
    assert SK % BS == 0 and m.stride(2) == 1, (SK, BS, m.stride())
    grid = (SK // BS, triton.cdiv(max(NA, NBj), BIP))
    mask_c = mask.contiguous()
    stats = torch.empty(S, N, 2, device=m.device, dtype=torch.float32)
    _opm_prologue_kernel[grid](m, mask_c, lnw, lnb, wa_t, wb_t, wa_t, wa_t, A2, BT, stats,
                               S, N, SK, NA, NBj, m.stride(0), m.stride(1), float(eps),
                               CM=cm, CH=CH, BI=BI, BS=BS, BIP=BIP, HAS_MASK=True, HAS_BIAS=False, LN_AFFINE=True,
                               MASK_I64=False, SAVE_STATS=True, num_warps=num_warps, num_stages=1)
    return A2, BT, stats


class OpmTrainFn(torch.autograd.Function):
    """pair = OPM(msa, mask) without the residual. Weights in their own dtype (fp32 or bf16); gradients returned in it."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, msa, mask, lnw, lnb, wa, wb, wo, bo, eps, save_o):
        ext = _ext()
        bf = torch.bfloat16
        m = msa[0].contiguous(); mask16 = mask[0].to(bf).contiguous()
        s, n = m.shape[0], m.shape[1]
        a2, bt, stats = fused_prologue(m, mask16, lnw.detach().float().contiguous(), lnb.detach().float().contiguous(), eps,
                                       wa.detach().to(bf).t().contiguous(), wb.detach().to(bf).t().contiguous())
        o = torch.matmul(a2, bt.t())                                                # the grouped outer product [(i,c), (j,e)], cuBLAS NT
        mf = mask16.float()
        norm = (mf.t() @ mf).clamp_(min=1).contiguous()                             # the module's own fp32 mask count
        z = ext.opm_epilogue(o, norm, wo.detach().contiguous().to(bf), bo.detach().to(bf).float().contiguous(), n, n)
        ctx.save_o = bool(save_o)
        ctx.save_for_backward(m, mask16, norm, a2, bt, stats, lnw, lnb, wa, wb, wo, bo, *((o,) if save_o else ()))
        ctx.eps = eps
        return z                                                                    # the epilogue already returns [1, N, N, c_z]

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dz):
        ext = _ext()
        bf = torch.bfloat16
        saved = ctx.saved_tensors
        m, mask16, norm, a2, bt, stats, lnw, lnb, wa, wb, wo, bo = saved[:12]
        o = saved[12] if ctx.save_o else None
        s, n = m.shape[0], m.shape[1]
        bw = wo.detach().t().contiguous().to(bf)                                     # Wo^T [CH*CH, CZ]: wgmma wants k = z contiguous
        dO, dzp, dbo = ext.opm_dgrad(dz[0].contiguous(), norm, bw, n, n)             # (dz / norm) . Wo in the grouped layout; dbo falls out
        dA = torch.mm(bt.t(), dO.t())                                                # [s, (i,c)]: the prologue backward wants a row's channels contiguous
        dB = torch.mm(a2.t(), dO)                                                    # [s, (j,e)]
        del dO
        if o is None:
            o = torch.matmul(a2, bt.t())                                             # 309 GFLOP once, instead of 302 MB kept from the forward
        dWo = ext.opm_dwo(dzp, o, n, n)                                              # fp32 [CZ, CH*CH], read off the grouped O
        del o
        g = lnw.detach().float()[None, :]
        bwp = torch.cat([(wa.detach().float() * g).t(), (wb.detach().float() * g).t()], dim=1).to(bf).contiguous()   # [CM, 2*CH]: gamma folded in
        dm, dWa, dWb, dgam, dbet = ext.opm_prologue_bwd(dA.contiguous(), dB.contiguous(), m, stats, mask16,
                                                        lnw.detach().float().contiguous(), lnb.detach().float().contiguous(), float(ctx.eps), bwp, n, s)
        return (dm.view(1, s, n, CM), None, dgam.to(lnw.dtype), dbet.to(lnb.dtype), dWa.to(wa.dtype), dWb.to(wb.dtype),
                dWo.to(wo.dtype), dbo.to(bo.dtype), None, None)


def outer_product_mean(module, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """OuterProductMean(msa, mask) WITHOUT the residual -- the module adds its own. mask: [1, S, N] bool."""
    eps = float(getattr(module.ln_msa, "eps", 1e-5))
    save_o = os.environ.get("MINIWORLD_OPM_TRAIN_SAVE_O", "1") != "0"
    return OpmTrainFn.apply(msa, mask, module.ln_msa.weight, module.ln_msa.bias, module.to_left.weight, module.to_right.weight,
                            module.to_out.weight, module.to_out.bias, eps, save_o)
