"""The MSAPairWeightedAveraging TRAINING path: one autograd.Function whose forward and backward are the fused
CUDA / Triton kernels developed in MiniWorld's `runs/msa_bench_20260921` (2026-09-22), served from this
repo's `csrc/` and the Triton kernels below. No external payload is needed.

Forward (3 launches):  pair_fwd3 (LN_z -> proj_z -> key mask -> softmax on tensor cores, csrc/pair3.cu; Triton pair_fwd below as the fallback)
                       ln_vg     (LN_m + value projection, v head-major [H][N][S*C] and y = LN(m), all TMA)
                       pwa_fwd2  (contraction + gate + out-projection + residual, warp-specialized TMA; keeps o)
Backward (6 launches): pwa_glue3  (du, gate glue from the saved o -> do head-major, dgp into the dgp | dv buffer, and the
                                   dWo partials accumulated in-kernel: the g*o tensor never touches memory),
                       pwa_plain2 (dv = w^T . do into the shared [S,N,2*HC] buffer), cuBLAS dw,
                       dgv_bwd (dWgv, dy in registers, LayerNorm backward + residual -> dm, dgamma/dbeta in one pass),
                       pair_bwd (softmax-bwd -> proj_z-bwd -> LN_z-bwd -> dz, dWb, dgamma_z/dbeta_z).
Measured H100, L=384, S=1024, bf16: fwd+bwd 1.81 ms vs this engine's own path 4.22 ms, 1426 MB vs 1424,
every gradient within 6e-3 of the engine's (the LayerNorm backward is CLOSER to fp32 truth than the engine's:
dy never rounds to bf16).

Automatic in auto mode; MINIWORLD_PWA_TRAIN=0 disables this path. It serves a grad-enabled call of a module built with
implementation="miniworld" or "anthropic" when the shapes fit (d_msa=64, d_pair=128, 8 heads x 32, batch 1,
N a multiple of 128, S even, bf16 activations on sm_90a); anything else falls through to the module's own statements. The residual AND the
module's row-broadcast dropout on the update are applied inside the kernel: the module returns this path's output as is.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl

ENV = "MINIWORLD_PWA_TRAIN"
H, C, D, DZ = 8, 32, 64, 128
HC = H * C
_EXT: dict[str, Any] = {}


def payload_dir():  # kept for symmetry with anthropic_msa: this path needs no payload
    return None


def wanted(implementation) -> bool:
    """Grad-enabled calls: opted in by MINIWORLD_PWA_TRAIN. Grad-free calls (inference) take the same forward kernels
    whenever MINIWORLD_PWA_TRAIN or MINIWORLD_PWA_INFER is set (no payload needed; faster than the upstream cell)."""
    from miniworld_engine import settings
    from miniworld_engine.modules.exceptions import ImplementationType
    name = ENV if torch.is_grad_enabled() else "MINIWORLD_PWA_INFER"
    on = os.environ.get(name, os.environ.get(ENV, "1")) != "0" and settings.current().engine_backend != "triton"
    return on and implementation in (ImplementationType.MINIWORLD, ImplementationType.ANTHROPIC)


def refusal(msa: torch.Tensor, pair: torch.Tensor, d_msa: int, d_pair: int, n_head: int, d_hidden: int, *, dropout: bool = False) -> str | None:
    """None if this path can run this training call, else why it cannot. Never raises."""
    try:
        name = ENV if torch.is_grad_enabled() else "MINIWORLD_PWA_INFER"
        if os.environ.get(name, os.environ.get(ENV, "1")) == "0":
            return f"{ENV} disables the native path"
        if (d_msa, d_pair, n_head, d_hidden) != (D, DZ, H, C):
            return f"the kernels serve (d_msa={D}, d_pair={DZ}, n_head={H}, d_hidden={C}), got ({d_msa}, {d_pair}, {n_head}, {d_hidden})"
        if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
            return f"the kernels are bf16, got {msa.dtype} / {pair.dtype}"
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        if torch.cuda.get_device_capability(msa.device) != (9, 0):
            return "the kernels are built for sm_90a"
        if msa.shape[0] != 1:
            return f"one MSA stack per call, got batch {msa.shape[0]}"
        n, s = msa.shape[2], msa.shape[1]
        if n % 128 or s % 2:
            return f"N must be a multiple of 128 and S even, got N={n}, S={s}"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


STATS: dict[str, Any] = {"served": 0, "refused": {}}       # how often the path ran / why it did not (for a real-model check)


def serves(*a, **kw) -> bool:
    why = refusal(*a, **kw)
    # Counters are eager diagnostics; mutating them while tracing adds recompilation guards.
    if torch.compiler.is_compiling():
        return why is None
    if why is None:
        STATS["served"] += 1
        return True
    if os.environ.get(ENV) and why not in STATS["refused"]:     # opted in but not served: say why, once per reason
        import logging
        logging.getLogger("miniworld_engine").warning("%s: the fused training path is not taken: %s", ENV, why)
    STATS["refused"][why] = STATS["refused"].get(why, 0) + 1
    return False


def _build(name: str, src_name: str):
    if name in _EXT:
        return _EXT[name]
    from miniworld_engine.kernels._nvcc import load_extension as load
    src = Path(__file__).with_name("csrc") / src_name
    build = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit")) / name
    build.mkdir(parents=True, exist_ok=True)
    _EXT[name] = load(name=name, sources=[str(src)], build_directory=str(build),
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"], extra_cflags=["-O3"])
    return _EXT[name]


def _k():
    """The CUDA extensions, built once (JIT, cached under MINIWORLD_ENGINE_JIT_ROOT). The forward comes from
    `pwa_fwd3.cu` (register-resident y tiles, 48 KB stages) when that file is present, else from `pwa_fwd2.cu`."""
    if "all" not in _EXT:
        k = {"fwd": _build("miniworld_pwa_fwd2", "pwa_fwd2.cu"), "ctr": _build("miniworld_pwa_ctr", "pwa_ctr.cu"),
                 "dgv": _build("miniworld_pwa_dgv_bwd", "dgv_bwd.cu"), "lnvg": _build("miniworld_pwa_ln_vg", "ln_vg.cu")}
        k["glue3"] = _build("miniworld_pwa_glue3", "pwa_glue3.cu") if (Path(__file__).with_name("csrc") / "pwa_glue3.cu").is_file() else None
        k["pair3"] = _build("miniworld_pwa_pair3", "pair3.cu") if (Path(__file__).with_name("csrc") / "pair3.cu").is_file() else None   # tensor-core pair forward
        # the row-broadcast dropout keep-mask (training) is applied inside the residual epilogue of either forward
        fwd2 = lambda w16, v, y, wg16, wo16, m, dmask, dscale: k["fwd"].pwa_fwd2(w16, v, y, wg16, wo16, m, True, 1, 4, dmask, dscale)
        if (Path(__file__).with_name("csrc") / "pwa_fwd3.cu").is_file():
            f3 = _build("miniworld_pwa_fwd3", "pwa_fwd3.cu"); k["fwd3"] = f3
            k["forward"] = lambda w16, v, y, wg16, wo16, m, dmask, dscale: f3.pwa_fwd3(w16, v, y, wg16, wo16, m, True, 1, 3, 1, 0, dmask, dscale)
        else:
            k["forward"] = fwd2
        _EXT["all"] = k
    return _EXT["all"]


def colsum(x):
    """deterministic column sums of a 2-D fp32 tensor as a cuBLAS GEMV: torch's dim-0 reduce of a tall
    thin buffer ran at a tenth of bandwidth (20 us for 3 MB)"""
    R = x.shape[0]
    return torch.matmul(torch.ones(R, dtype=x.dtype, device=x.device), x.reshape(R, -1)).reshape(x.shape[1:])



# ---------------------------------------------------------------------------------------------
# The pair side, fused.  Forward: w[h,i,:] = softmax_j(mask ? LN(z[i,j,:]) Wb^T : -inf), one program per
# row i, two passes over j (max/sum, then normalise) recomputing the LayerNorm + 8-wide projection --
# cheaper than holding [N, 128] in registers.  Backward: db = w (dw - sum_j w dw), dzn = db Wb,
# LN backward -> dz, plus per-program partials of dWb and dgamma/dbeta.  It replaces six forward
# launches and six backward ones, all on tensors small enough that launch count was the cost.
@triton.jit
def _pair_fwd_kernel(Z, MASK, LNW, LNB, WBT, W, N, eps, DZ: tl.constexpr, HP: tl.constexpr, H: tl.constexpr, BJ: tl.constexpr, BJO: tl.constexpr):
    i = tl.program_id(0)
    jo = tl.program_id(1)                                                               # this program's output block of j
    d = tl.arange(0, DZ)
    hh = tl.arange(0, HP)
    lw = tl.load(LNW + d).to(tl.float32)
    lb = tl.load(LNB + d).to(tl.float32)
    wbt = tl.load(WBT + d[:, None] * HP + hh[None, :])                                  # [DZ, HP] bf16, columns >= H are zero
    # pass 1: online max / rescaled sum per head
    mx = tl.full((HP,), -1e30, dtype=tl.float32)
    den = tl.zeros((HP,), dtype=tl.float32)
    for j0 in range(0, N, BJ):
        j = j0 + tl.arange(0, BJ)
        x = tl.load(Z + (i * N + j)[:, None] * DZ + d[None, :]).to(tl.float32)          # [BJ, DZ]
        mean = tl.sum(x, 1) / DZ
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / DZ + eps)
        zn = (xc * rstd[:, None] * lw[None, :] + lb[None, :]).to(tl.bfloat16)
        b = tl.dot(zn, wbt).to(tl.bfloat16).to(tl.float32)                              # [BJ, HP]: the stock proj_z output is bf16
        m = tl.load(MASK + i * N + j).to(tl.float32)
        b = tl.where(m[:, None] > 0.5, b, -1e30)
        mn = tl.maximum(mx, tl.max(b, 0))
        den = den * tl.exp(mx - mn) + tl.sum(tl.exp(b - mn[None, :]), 0)
        mx = mn
    for j0 in range(jo * BJO, (jo + 1) * BJO, BJ):
        j = j0 + tl.arange(0, BJ)
        x = tl.load(Z + (i * N + j)[:, None] * DZ + d[None, :]).to(tl.float32)
        mean = tl.sum(x, 1) / DZ
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / DZ + eps)
        zn = (xc * rstd[:, None] * lw[None, :] + lb[None, :]).to(tl.bfloat16)
        b = tl.dot(zn, wbt).to(tl.bfloat16).to(tl.float32)
        m = tl.load(MASK + i * N + j).to(tl.float32)
        b = tl.where(m[:, None] > 0.5, b, -1e30)
        w = tl.exp(b - mx[None, :]) / den[None, :]                                        # [BJ, HP]
        tl.store(W + (hh * N + i)[None, :] * N + j[:, None], w.to(W.dtype.element_ty), mask=(hh < H)[None, :])


@triton.jit
def _pair_bwd_kernel(Z, W, DW, SDOT, LNW, LNB, WB, DZO, PWB, PLN, N, eps, DZ: tl.constexpr, HP: tl.constexpr, H: tl.constexpr, BJ: tl.constexpr, BJO: tl.constexpr, HAS_SDOT: tl.constexpr):
    i = tl.program_id(0)
    jo = tl.program_id(1)
    pid = i * tl.num_programs(1) + jo
    d = tl.arange(0, DZ)
    hh = tl.arange(0, HP)
    hmask = hh < H
    lw = tl.load(LNW + d).to(tl.float32)
    lb = tl.load(LNB + d).to(tl.float32)
    wb = tl.load(WB + hh[:, None] * DZ + d[None, :], mask=hmask[:, None], other=0.0)     # [HP, DZ] bf16 (Wb rows, zero-padded)
    # sum_j w dw per head: precomputed (a tiny GEMV-like pass) when the row is split over programs
    if HAS_SDOT:
        sdot = tl.load(SDOT + hh * N + i, mask=hmask, other=0.0)
    else:
        sdot = tl.zeros((HP,), dtype=tl.float32)
        for j0 in range(0, N, BJ):
            j = j0 + tl.arange(0, BJ)
            off = (hh * N + i)[None, :] * N + j[:, None]
            w = tl.load(W + off, mask=hmask[None, :], other=0.0).to(tl.float32)
            dw = tl.load(DW + off, mask=hmask[None, :], other=0.0)
            sdot += tl.sum(w * dw, 0)
    pwb = tl.zeros((HP, DZ), dtype=tl.float32)
    pg = tl.zeros((DZ,), dtype=tl.float32)
    pb = tl.zeros((DZ,), dtype=tl.float32)
    for j0 in range(jo * BJO, (jo + 1) * BJO, BJ):
        j = j0 + tl.arange(0, BJ)
        off = (hh * N + i)[None, :] * N + j[:, None]
        w = tl.load(W + off, mask=hmask[None, :], other=0.0).to(tl.float32)
        dw = tl.load(DW + off, mask=hmask[None, :], other=0.0)
        db = w * (dw - sdot[None, :])                                                     # [BJ, HP] softmax backward
        x = tl.load(Z + (i * N + j)[:, None] * DZ + d[None, :]).to(tl.float32)            # [BJ, DZ]
        mean = tl.sum(x, 1) / DZ
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / DZ + eps)
        xh = xc * rstd[:, None]
        zn = (xh * lw[None, :] + lb[None, :]).to(tl.bfloat16)
        db16 = db.to(tl.bfloat16)
        dzn = tl.dot(db16, wb).to(tl.float32)                                             # [BJ, DZ] = db Wb  (the stock proj_z is a bf16 GEMM)
        pwb += tl.dot(tl.trans(db16), zn)                                                 # [HP, DZ] += db^T zn
        pg += tl.sum(dzn * xh, 0)
        pb += tl.sum(dzn, 0)
        g = dzn * lw[None, :]
        gs = tl.sum(g, 1) / DZ
        gx = tl.sum(g * xh, 1) / DZ
        dz = rstd[:, None] * (g - gs[:, None] - xh * gx[:, None])
        tl.store(DZO + (i * N + j)[:, None] * DZ + d[None, :], dz.to(DZO.dtype.element_ty))
    tl.store(PWB + pid * HP * DZ + hh[:, None] * DZ + d[None, :], pwb)
    tl.store(PLN + pid * 2 * DZ + d, pg)
    tl.store(PLN + pid * 2 * DZ + DZ + d, pb)


def pair_fwd(z, mask, ln_w, ln_b, eps, wb, H, BJ=64, num_warps=4, BJO=None):
    """z [N,N,DZ] bf16, mask [N,N] (bf16 0/1), wb [H, DZ] (proj_z.weight) -> w [H,N,N] bf16 (softmax over the last dim).
    BJO: j columns written per program (the row statistics are recomputed by each); default: the whole row."""
    N, _, DZ = z.shape; HP = 16
    BJO = BJO or N
    assert N % BJO == 0 and BJO % BJ == 0
    wbt = torch.zeros((DZ, HP), dtype=torch.bfloat16, device=z.device); wbt[:, :H] = wb.to(torch.bfloat16).t()
    w = torch.empty((H, N, N), dtype=torch.bfloat16, device=z.device)
    cast(Any, _pair_fwd_kernel)[(N, N // BJO)](z, mask, ln_w, ln_b, wbt, w, N, float(eps), DZ=DZ, HP=HP, H=H, BJ=BJ, BJO=BJO, num_warps=num_warps)
    return w


def pair_bwd(z, w16, dw, ln_w, ln_b, eps, wb, BJ=32, num_warps=4, BJO=None):
    """-> dz [N,N,DZ] bf16, dWb [H, DZ] fp32, dgamma_z, dbeta_z [DZ] fp32.  dw [H,N,N] fp32."""
    N, _, DZ = z.shape; H = w16.shape[0]; HP = 16
    BJO = BJO or N
    assert N % BJO == 0 and BJO % BJ == 0
    nprog = N * (N // BJO)
    wbp = torch.zeros((HP, DZ), dtype=torch.bfloat16, device=z.device); wbp[:H] = wb.to(torch.bfloat16)
    dz = torch.empty_like(z)
    pwb = torch.empty((nprog, HP, DZ), dtype=torch.float32, device=z.device)
    pln = torch.empty((nprog, 2 * DZ), dtype=torch.float32, device=z.device)
    dwc = dw.contiguous()
    sdot = (w16.float() * dwc).sum(-1).contiguous() if BJO < N else dwc   # [H, N]; only needed when the row is split
    cast(Any, _pair_bwd_kernel)[(N, N // BJO)](z, w16, dwc, sdot, ln_w, ln_b, wbp, dz, pwb, pln, N, float(eps), DZ=DZ, HP=HP, H=H, BJ=BJ, BJO=BJO,
                                    HAS_SDOT=BJO < N, num_warps=num_warps)
    ps = colsum(pln)
    return dz, colsum(pwb)[:H], ps[:DZ], ps[DZ:]


class _PwaMath(torch.autograd.Function):
    """out = msa + PWA(msa, pair, key mask). Weights in their own dtype (fp32 or bf16); gradients returned in it."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, msa, pair, pm, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, eps_m, eps_z, p_drop):
        k = _k()
        m = msa[0].contiguous(); z = pair[0].contiguous()
        N = m.shape[1]
        bf = torch.bfloat16
        wv16 = wv.detach().to(bf).contiguous(); wg16 = wg.detach().to(bf).contiguous(); wo16 = wo.detach().to(bf).contiguous()
        if k["pair3"] is not None and N % 16 == 0 and N <= 1024:
            w16 = k["pair3"].pair_fwd3(z, pm, lnz_w.detach().float().contiguous(), lnz_b.detach().float().contiguous(), eps_z, wb.detach())
        else:
            w16 = pair_fwd(z, pm, lnz_w.detach().float().contiguous(), lnz_b.detach().float().contiguous(), eps_z, wb.detach(), H, BJ=64)
        v, y = k["lnvg"].ln_vg(m, lnm_w.detach().float().contiguous(), lnm_b.detach().float().contiguous(), wv16, eps_m, 2, 3, 1)
        # the module's drop_msa (Dropout(broadcast_dim=1)): one keep-mask per (token, channel) shared over the MSA rows,
        # x * mask / (1 - p); applied to the bf16 update inside the kernel's residual epilogue
        dmask, dscale = None, 1.0
        if p_drop > 0:
            dmask = (torch.rand(N, D, device=m.device, dtype=bf) > p_drop).to(bf)
            dscale = 1.0 / (1.0 - p_drop)
        out, o = k["forward"](w16, v, y, wg16, wo16, m, dmask, dscale)           # residual fused; o kept for the backward
        ctx.save_for_backward(m, z, w16, v, y, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, *((dmask,) if dmask is not None else ()))
        ctx.eps = (eps_m, eps_z); ctx.dscale = dscale
        return out[None]

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dres):
        k = _k()
        saved = ctx.saved_tensors
        m, z, w16, v, y, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = saved[:14]
        dmask = saved[14] if len(saved) > 14 else None
        eps_m, eps_z = ctx.eps
        S, N = m.shape[0], m.shape[1]
        bf = torch.bfloat16
        dres0 = dres[0].contiguous()
        # the residual's gradient is dres itself; the update's is dres * mask / (1 - p) (what autograd of x * mask / (1-p) gives)
        dout = dres0 if dmask is None else (dres0 * dmask[None] * ctx.dscale).contiguous()
        wv16 = wv.to(bf).contiguous(); wg16 = wg.to(bf).contiguous(); wo16 = wo.to(bf).contiguous()
        dgv = torch.empty((S, N, 2 * HC), dtype=bf, device=m.device)               # dgp | dv: one [S,N,512] buffer
        if k["glue3"] is not None:                                                  # dWo partials fused: go never touches memory
            d_o, _, dWo = k["glue3"].pwa_glue3(o, y, dout, wg16, wo16.t().contiguous(), dgv, 2)
        else:
            d_o, _, go = k["ctr"].pwa_glue_o(o, y, dout, wg16, wo16.t().contiguous(), 1, dgv)
            dWo = dout.view(S * N, D).t() @ go.view(S * N, HC)                      # [D][HC]
        k["fwd"].pwa_plain2(w16.transpose(1, 2).contiguous(), d_o, 4, dgv)         # dv -> dgv[..., HC:]
        dw = torch.bmm(d_o, v.transpose(1, 2)).float()                              # [H][N][N], K = S*C
        wgvT = torch.cat([wg16, wv16], 0).t().contiguous()
        dm, dWgv, dlw, dlb = k["dgv"].dgv_bwd(dgv.view(S * N, 2 * HC), y.view(S * N, D), m.view(S * N, D), dres0.view(S * N, D),
                                              wgvT, lnm_w.float().contiguous(), eps_m, 1, 8, True)
        dWg, dWv = dWgv[:HC], dWgv[HC:]
        dz, dWb, dzw, dzb = pair_bwd(z, w16, dw, lnz_w.float().contiguous(), lnz_b.float().contiguous(), eps_z, wb, BJ=32)
        # Separate small weight/LN gradient views at the custom-op boundary.
        return (dm.view(S, N, D)[None], dz[None], None, dlw.to(lnm_w.dtype, copy=True), dlb.to(lnm_b.dtype, copy=True), dWv.to(wv.dtype, copy=True), dWg.to(wg.dtype, copy=True),
                dzw.to(lnz_w.dtype, copy=True), dzb.to(lnz_b.dtype, copy=True), dWb.to(wb.dtype), dWo.to(wo.dtype), None, None, None)


@torch.no_grad()
@torch.autocast("cuda", enabled=False)
def _inference_math(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """msa + PWA(msa, pair, key mask) for a grad-free call: the same three forward kernels, o not kept, no dropout."""
    k = _k(); bf = torch.bfloat16
    m = msa[0].contiguous(); z = pair[0].contiguous(); n = m.shape[1]
    pm = torch.ones(n, n, dtype=bf, device=m.device) if mask is None else mask[0].to(bf)[None, :].expand(n, n).contiguous()
    lnz_w = module.ln_pair.weight.detach().float().contiguous(); lnz_b = module.ln_pair.bias.detach().float().contiguous()
    eps_m = float(getattr(module.ln_msa, "eps", 1e-5)); eps_z = float(getattr(module.ln_pair, "eps", 1e-5))
    if k["pair3"] is not None and n % 16 == 0 and n <= 1024:
        w16 = k["pair3"].pair_fwd3(z, pm, lnz_w, lnz_b, eps_z, module.to_bias.weight.detach())
    else:
        w16 = pair_fwd(z, pm, lnz_w, lnz_b, eps_z, module.to_bias.weight.detach(), H, BJ=64)
    v, y = k["lnvg"].ln_vg(m, module.ln_msa.weight.detach().float().contiguous(), module.ln_msa.bias.detach().float().contiguous(),
                           module.to_value.weight.detach().to(bf).contiguous(), eps_m, 2, 3, 1)
    wg16 = module.to_gate.weight.detach().to(bf).contiguous(); wo16 = module.to_out.weight.detach().to(bf).contiguous()
    if "fwd3" in k:
        out, _ = k["fwd3"].pwa_fwd3(w16, v, y, wg16, wo16, m, False, 1, 3, 1, 0, None, 1.0)
    else:
        out, _ = k["fwd"].pwa_fwd2(w16, v, y, wg16, wo16, m, False, 1, 4, None, 1.0)
    return out[None]


def pair_weighted_averaging(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """msa + MSAPairWeightedAveraging(msa, pair, mask): the residual is INCLUDED (the forward kernel adds it)."""
    n = msa.shape[2]
    if mask is None:
        pm = torch.ones(n, n, dtype=torch.bfloat16, device=msa.device)
    else:
        pm = mask[0].to(torch.bfloat16)[None, :].expand(n, n).contiguous()          # the module masks the key axis j
    eps_m = float(getattr(module.ln_msa, "eps", 1e-5)); eps_z = float(getattr(module.ln_pair, "eps", 1e-5))
    p_drop = float(module.drop_msa.p_drop) if module.training else 0.0          # the module's drop_msa, fused into the kernel
    return PwaTrainFn.apply(msa.contiguous(), pair.contiguous(), pm, module.ln_msa.weight, module.ln_msa.bias, module.to_value.weight, module.to_gate.weight,
                            module.ln_pair.weight, module.ln_pair.bias, module.to_bias.weight, module.to_out.weight, eps_m, eps_z, p_drop)


class _SavedContext:
    eps: tuple[float, float]
    dscale: float
    def save_for_backward(self,*tensors):self.saved_tensors=tensors


def _forward_fake(args,eps_m,eps_z,p_drop):
    m=args[0];_,s,n,_=m.shape
    return [torch.empty_like(m),torch.empty((H,n,n),device=m.device,dtype=torch.bfloat16),torch.empty((H,n,s*C),device=m.device,dtype=torch.bfloat16),
            torch.empty((s,n,D),device=m.device,dtype=torch.bfloat16),torch.empty((s,n,HC),device=m.device,dtype=torch.bfloat16),
            torch.empty((n,D) if p_drop else (0,),device=m.device,dtype=torch.bfloat16)]


from miniworld_engine.kernels._compile import opaque


@opaque(fake=_forward_fake,name="pwa_h100_fwd")
def _forward_op(args:list[torch.Tensor],eps_m:float,eps_z:float,p_drop:float)->list[torch.Tensor]:
    ctx=_SavedContext();out=_PwaMath.forward(ctx,*args,eps_m,eps_z,p_drop);sv=ctx.saved_tensors
    return [out,*sv[2:6],sv[14] if p_drop else args[0].new_empty((0,))]


def _backward_fake(args,kept,dy,eps_m,eps_z,p_drop):return [torch.empty_like(args[i]) for i in (0,1,3,4,5,6,7,8,9,10)]


@opaque(fake=_backward_fake,name="pwa_h100_bwd")
def _backward_op(args:list[torch.Tensor],kept:list[torch.Tensor],dy:torch.Tensor,eps_m:float,eps_z:float,p_drop:float)->list[torch.Tensor]:
    ctx=_SavedContext();ctx.eps=(eps_m,eps_z);ctx.dscale=1/(1-p_drop) if p_drop else 1.
    ctx.saved_tensors=(args[0][0],args[1][0],*kept[:4],*args[3:],*((kept[4],) if p_drop else ()))
    gradients=_PwaMath.backward(ctx,dy)
    return [gradients[i] for i in (0,1,3,4,5,6,7,8,9,10)]


class PwaTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,*args):
        tensors=list(args[:11]);ctx.eps_m,ctx.eps_z,ctx.p_drop=args[11:]
        out,*kept=_forward_op(tensors,ctx.eps_m,ctx.eps_z,ctx.p_drop)
        ctx.save_for_backward(*tensors,*kept)
        return out
    @staticmethod
    def backward(ctx,dy):
        vals=ctx.saved_tensors
        grads=_backward_op(list(vals[:11]),list(vals[11:]),dy,ctx.eps_m,ctx.eps_z,ctx.p_drop)
        return (*grads[:2],None,*grads[2:],None,None,None)



def _infer_fake(args,eps_m,eps_z):return torch.empty_like(args[0])


@opaque(fake=_infer_fake,name="pwa_h100_infer")
def _inference_op(args:list[torch.Tensor],eps_m:float,eps_z:float)->torch.Tensor:
    from types import SimpleNamespace as NS
    msa,pair,mask,lmw,lmb,wv,wg,lzw,lzb,wb,wo=args
    module=NS(ln_msa=NS(weight=lmw,bias=lmb,eps=eps_m),ln_pair=NS(weight=lzw,bias=lzb,eps=eps_z),
              to_value=NS(weight=wv),to_gate=NS(weight=wg),to_bias=NS(weight=wb),to_out=NS(weight=wo))
    return _inference_math(module,msa,pair,mask)


def pair_weighted_averaging_inference(module,msa,pair,mask):
    mask=mask if mask is not None else torch.ones((1,msa.shape[2]),device=msa.device,dtype=torch.bool)
    args=[msa.contiguous(),pair.contiguous(),mask.contiguous(),module.ln_msa.weight,module.ln_msa.bias,module.to_value.weight,module.to_gate.weight,
          module.ln_pair.weight,module.ln_pair.bias,module.to_bias.weight,module.to_out.weight]
    return _inference_op(args,module.ln_msa.eps,module.ln_pair.eps)
