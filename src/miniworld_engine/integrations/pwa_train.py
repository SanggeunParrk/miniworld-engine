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
                       pair_bwd (softmax-bwd -> proj_z-bwd -> LN_z-bwd -> dz, dWb, dgamma_z).
Measured H100, L=384, S=1024, bf16: fwd+bwd 1.81 ms vs this engine's own path 4.22 ms, 1426 MB vs 1424,
every gradient within 6e-3 of the engine's (the LayerNorm backward is CLOSER to fp32 truth than the engine's:
dy never rounds to bf16).

On sm_100a (B200) the same boundaries run `csrc/sm100/pwa_sm100.cu`: forward pair_fwd (LN_z folded into proj_z), ln_vg,
cuBLAS bmm (o = w . v), pwa_gate_out; backward pwa_glue2 (do, dyg = dgp . Wg, dWo, dWg -- dgp stays on chip), cuBLAS bmm
(dv = w^T . do, dw = do . v^T), dv_bwd, pair_bwd.

Automatic in auto mode; settings.pwa_train=0 disables this path. It serves a grad-enabled call of a module built with
implementation="miniworld" or "anthropic" when the shapes fit (d_msa=64, d_pair=128, 8 heads x 32, batch 1,
N a multiple of 128, S even, bf16 activations on sm_90a); anything else falls through to the module's own statements. The residual AND the
module's row-broadcast dropout on the update are applied inside the kernel: the module returns this path's output as is.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl

from miniworld_engine import settings
from miniworld_engine.integrations import pwa_wide as _wide

H, C, D, DZ = 8, 32, 64, 128
HC = H * C
_EXT: dict[str, Any] = {}
_NATIVE_TRAINING_SCOPE: ContextVar[bool] = ContextVar(
    "pwa_native_training", default=False
)


_SPLIT_MIN_N = 512


NMAX_INFER = 2048  # longest sequence of the inference path (pair_fwd3's shared-memory row of logits)


def _split_contraction(n: int) -> bool:
    """N >= 512: the contraction runs on cuBLAS (head-major o) and the gate / out pass in pwa_gateout; below, the fused kernel wins
    (the o round trip and the extra launch cost more than the faster contraction saves). o is rounded to bf16 BEFORE the gate here, as in
    the module's bf16 statements and the sm_100a path, so the forward output is not bit-identical to the fused kernel's; the gradients are."""
    return n >= _SPLIT_MIN_N


def _h100_training_default(msa: torch.Tensor) -> bool:
    """The six required training shapes with full native accuracy qualification."""
    return (
        msa.device.type == "cuda"
        and torch.cuda.get_device_capability(msa.device) == (9, 0)
        and msa.ndim == 4
        and msa.shape[0] == 1
        and msa.shape[3] == D
        and msa.shape[1] in (1024, 2048, 4096)
        and msa.shape[2] in (384, 768)
    )


def native_aux_enabled(msa: torch.Tensor | None = None) -> bool:
    """Qualified training defaults (the six required H100 shapes, or the active native training scope)."""
    return (
        _h100_training_default(msa) if msa is not None else _NATIVE_TRAINING_SCOPE.get()
    )


def native_inference_enabled(device: torch.device) -> bool:
    """Validated H100 inference default."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


def payload_dir():  # kept for symmetry with anthropic_msa: this path needs no payload
    return None


def _enabled() -> bool:
    """Grad-enabled calls follow ``settings.pwa_train``; inference follows ``pwa_infer`` and falls back to it. None = automatic."""
    current = settings.current()
    value = current.pwa_train
    if not torch.is_grad_enabled() and current.pwa_infer is not None:
        value = current.pwa_infer
    return True if value is None else value


def wanted(implementation) -> bool:
    """Grad-enabled calls: ``settings.pwa_train``. Grad-free calls (inference) take the same forward kernels, controlled by
    ``settings.pwa_infer`` and, when that is unset, ``settings.pwa_train`` (no payload needed; faster than the upstream cell)."""
    from miniworld_engine.modules.exceptions import ImplementationType

    on = _enabled() and (not settings.current().force_portable_kernels)
    # ``ANTHROPIC`` is not asked here: an explicit ``implementation="anthropic"`` is the Anthropic release's cell
    # (integrations.anthropic_msa), not this engine's own H100 path.
    return on and implementation == ImplementationType.MINIWORLD


def refusal(
    msa: torch.Tensor,
    pair: torch.Tensor,
    d_msa: int,
    d_pair: int,
    n_head: int,
    d_hidden: int,
    *,
    dropout: bool = False,
) -> str | None:
    """None if this path can run this training call, else why it cannot. Never raises."""
    try:
        if not _enabled():
            return "settings.pwa_train / pwa_infer disables the native path"
        wide = (d_msa, d_pair, n_head, d_hidden) != (D, DZ, H, C)
        if wide and not _wide.supported(d_msa, d_pair, n_head, d_hidden):
            return (
                f"the kernels serve (d_msa={D}, d_pair={DZ}, n_head={H}, d_hidden={C}) and, wide, d_msa in {_wide.DMSA}, "
                f"d_pair in {_wide.DPAIR}, {H} heads x d_hidden in {_wide.DHID}; got ({d_msa}, {d_pair}, {n_head}, {d_hidden})"
            )
        if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
            return f"the kernels are bf16, got {msa.dtype} / {pair.dtype}"
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        cap = torch.cuda.get_device_capability(msa.device)
        if cap not in ((9, 0), (10, 0)):
            return "the kernels are built for sm_90a and sm_100a"
        if cap == (10, 0) and torch.is_grad_enabled() and not _SM100_TRAIN:
            return "the sm_100a backward kernels are not built yet"
        if msa.shape[0] != 1:
            return f"one MSA stack per call, got batch {msa.shape[0]}"
        n, s = msa.shape[2], msa.shape[1]
        native = native_aux_enabled(msa) or (
            not torch.is_grad_enabled() and native_inference_enabled(msa.device)
        )
        # the forward pair kernel holds H rows of N logits in shared memory (N <= 2048); the backward one is built for N <= 1024
        n_max = 1024 if torch.is_grad_enabled() else NMAX_INFER
        if cap == (9, 0) and native and (n == 0 or s == 0 or n > n_max):
            return f"the CUDA-only H100 path needs positive S and N <= {n_max}, got N={n}, S={s}"
        if wide:  # the wide path pads the length to a multiple of 16 itself; any depth
            if cap != (9, 0):
                return "the wide kernels are built for sm_90a"
            if not 0 < n <= _wide.NMAX or s == 0:
                return f"the wide pair kernels hold one row of logits: 0 < N <= {_wide.NMAX}, got N={n}, S={s}"
            return None
        if n % 128 or s % 2:
            return f"N must be a multiple of 128 and S even, got N={n}, S={s}"
        if cap == (10, 0) and (
            s % 128 or n > 1024
        ):  # ln_vg / dv_bwd tiles are 128 MSA rows; pair_fwd holds a row of N
            return f"the sm_100a kernels need S a multiple of 128 and N <= 1024, got N={n}, S={s}"
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
        settings.current().pwa_train is not None and why not in STATS["refused"]
    ):  # opted in but not served: say why, once per reason
        import logging

        logging.getLogger("miniworld_engine").warning(
            "pwa_train: the fused training path is not taken: %s", why
        )
    STATS["refused"][why] = STATS["refused"].get(why, 0) + 1
    return False


def _build(
    name: str, src_name: str, native: bool | None = None, fp32_logits: bool = True
):
    native = (native_aux_enabled() if native is None else native) and src_name in (
        "pair3.cu",
        "dgv_bwd.cu",
        "pwa_glue3.cu",
    )
    if native:
        name += "_native_v1"
    if (
        src_name == "pair3.cu" and not fp32_logits
    ):  # the opt-in BF16 logit rounding (-DMW_PWA_ROUND_LOGITS), either build
        name += "_rounded_logits"
    if name in _EXT:
        return _EXT[name]
    from miniworld_engine.kernels._nvcc import load_extension as load

    src = Path(__file__).with_name("csrc") / src_name
    _EXT[name] = load(
        name=name,
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            "--use_fast_math",
            *(["-DMW_PWA_CUDA_AUX=1"] if native else []),
            *(
                ["-DMW_PWA_ROUND_LOGITS=1"]
                if src_name == "pair3.cu" and not fp32_logits
                else []
            ),
        ],
        extra_cflags=["-O3"],
    )
    return _EXT[name]


_SM100_TRAIN = True  # the sm_100a backward kernels (csrc/sm100/pwa_sm100.cu) serve grad-enabled calls


def _k100():
    """B200: `csrc/sm100/pwa_sm100.cu` (the pair side, ln_vg, the gate / out pass and the backward on tcgen05 / TMEM), built for
    sm_100a; the contractions are cuBLAS bmm."""
    if "sm100" not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        csrc = Path(__file__).with_name("csrc")
        pwa = load(
            name="miniworld_pwa_sm100",
            sources=[str(csrc / "sm100" / "pwa_sm100.cu")],
            extra_include_paths=[str(csrc / "sm100")],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--use_fast_math",
            ],
            extra_cflags=["-O3"],
        )
        _EXT["sm100"] = {"pwa": pwa}
    return _EXT["sm100"]


def _k(native: bool | None = None, fp32_logits: bool = True):
    """The CUDA extensions, built once (JIT, cached under settings.engine_jit_root). The forward comes from
    `pwa_fwd3.cu` (register-resident y tiles, 48 KB stages) when that file is present, else from `pwa_fwd2.cu`."""
    native = native_aux_enabled() if native is None else native
    key = "all_native" if native else "all"
    if not fp32_logits:
        key += "_rounded_logits"
    if key not in _EXT:
        build = lambda name, src: _build(name, src, native, fp32_logits)
        k = {
            "fwd": build("miniworld_pwa_fwd2", "pwa_fwd2.cu"),
            "ctr": build("miniworld_pwa_ctr", "pwa_ctr.cu"),
            "dgv": build("miniworld_pwa_dgv_bwd", "dgv_bwd.cu"),
            "lnvg": build("miniworld_pwa_ln_vg", "ln_vg.cu"),
        }
        k["glue3"] = (
            build("miniworld_pwa_glue3", "pwa_glue3.cu")
            if (Path(__file__).with_name("csrc") / "pwa_glue3.cu").is_file()
            else None
        )
        k["pair3"] = (
            build("miniworld_pwa_pair3", "pair3.cu")
            if (Path(__file__).with_name("csrc") / "pair3.cu").is_file()
            else None
        )  # tensor-core pair forward
        k["gateout"] = (
            build("miniworld_pwa_gateout", "pwa_gateout.cu")
            if (Path(__file__).with_name("csrc") / "pwa_gateout.cu").is_file()
            else None
        )  # gate / out pass over a cuBLAS contraction
        # the row-broadcast dropout keep-mask (training) is applied inside the residual epilogue of either forward
        fwd2 = lambda w16, v, y, wg16, wo16, m, dmask, dscale: k["fwd"].pwa_fwd2(
            w16, v, y, wg16, wo16, m, True, 1, 4, dmask, dscale
        )
        if (Path(__file__).with_name("csrc") / "pwa_fwd3.cu").is_file():
            f3 = build("miniworld_pwa_fwd3", "pwa_fwd3.cu")
            k["fwd3"] = f3
            k["forward"] = lambda w16, v, y, wg16, wo16, m, dmask, dscale: f3.pwa_fwd3(
                w16, v, y, wg16, wo16, m, True, 1, 3, 1, 0, dmask, dscale
            )
        else:
            k["forward"] = fwd2
        _EXT[key] = k
    return _EXT[key]


def colsum(x):
    """deterministic column sums of a 2-D fp32 tensor as a cuBLAS GEMV: torch's dim-0 reduce of a tall
    thin buffer ran at a tenth of bandwidth (20 us for 3 MB)"""
    R = x.shape[0]
    return torch.matmul(
        torch.ones(R, dtype=x.dtype, device=x.device), x.reshape(R, -1)
    ).reshape(x.shape[1:])


# ---------------------------------------------------------------------------------------------
# The pair side, fused.  Forward: w[h,i,:] = softmax_j(mask ? LN(z[i,j,:]) Wb^T : -inf), one program per
# row i, two passes over j (max/sum, then normalise) recomputing the LayerNorm + 8-wide projection --
# cheaper than holding [N, 128] in registers.  Backward: db = w (dw - sum_j w dw), dzn = db Wb,
# LN backward -> dz, plus per-program partials of dWb and dgamma/dbeta.  It replaces six forward
# launches and six backward ones, all on tensors small enough that launch count was the cost.
@triton.jit
def _pair_fwd_kernel(
    Z,
    MASK,
    LNW,
    WBT,
    W,
    N,
    eps,
    DZ: tl.constexpr,
    HP: tl.constexpr,
    H: tl.constexpr,
    BJ: tl.constexpr,
    BJO: tl.constexpr,
):
    i = tl.program_id(0)
    jo = tl.program_id(1)  # this program's output block of j
    d = tl.arange(0, DZ)
    hh = tl.arange(0, HP)
    lw = tl.load(LNW + d).to(tl.float32)
    wbt = tl.load(
        WBT + d[:, None] * HP + hh[None, :]
    )  # [DZ, HP] bf16, columns >= H are zero
    # pass 1: online max / rescaled sum per head
    mx = tl.full((HP,), -1e30, dtype=tl.float32)
    den = tl.zeros((HP,), dtype=tl.float32)
    for j0 in range(0, N, BJ):
        j = j0 + tl.arange(0, BJ)
        x = tl.load(Z + (i * N + j)[:, None] * DZ + d[None, :]).to(
            tl.float32
        )  # [BJ, DZ]
        mean = tl.sum(x, 1) / DZ
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / DZ + eps)
        zn = (xc * rstd[:, None] * lw[None, :]).to(tl.bfloat16)
        b = (
            tl.dot(zn, wbt).to(tl.bfloat16).to(tl.float32)
        )  # [BJ, HP]: the stock proj_z output is bf16
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
        zn = (xc * rstd[:, None] * lw[None, :]).to(tl.bfloat16)
        b = tl.dot(zn, wbt).to(tl.bfloat16).to(tl.float32)
        m = tl.load(MASK + i * N + j).to(tl.float32)
        b = tl.where(m[:, None] > 0.5, b, -1e30)
        w = tl.exp(b - mx[None, :]) / den[None, :]  # [BJ, HP]
        tl.store(
            W + (hh * N + i)[None, :] * N + j[:, None],
            w.to(W.dtype.element_ty),
            mask=(hh < H)[None, :],
        )


@triton.jit
def _pair_bwd_kernel(
    Z,
    W,
    DW,
    SDOT,
    LNW,
    WB,
    DZO,
    PWB,
    PLN,
    N,
    eps,
    DZ: tl.constexpr,
    HP: tl.constexpr,
    H: tl.constexpr,
    BJ: tl.constexpr,
    BJO: tl.constexpr,
    HAS_SDOT: tl.constexpr,
):
    i = tl.program_id(0)
    jo = tl.program_id(1)
    pid = i * tl.num_programs(1) + jo
    d = tl.arange(0, DZ)
    hh = tl.arange(0, HP)
    hmask = hh < H
    lw = tl.load(LNW + d).to(tl.float32)
    wb = tl.load(
        WB + hh[:, None] * DZ + d[None, :], mask=hmask[:, None], other=0.0
    )  # [HP, DZ] bf16 (Wb rows, zero-padded)
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
    for j0 in range(jo * BJO, (jo + 1) * BJO, BJ):
        j = j0 + tl.arange(0, BJ)
        off = (hh * N + i)[None, :] * N + j[:, None]
        w = tl.load(W + off, mask=hmask[None, :], other=0.0).to(tl.float32)
        dw = tl.load(DW + off, mask=hmask[None, :], other=0.0)
        db = w * (dw - sdot[None, :])  # [BJ, HP] softmax backward
        x = tl.load(Z + (i * N + j)[:, None] * DZ + d[None, :]).to(
            tl.float32
        )  # [BJ, DZ]
        mean = tl.sum(x, 1) / DZ
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / DZ + eps)
        xh = xc * rstd[:, None]
        zn = (xh * lw[None, :]).to(tl.bfloat16)
        db16 = db.to(tl.bfloat16)
        dzn = tl.dot(db16, wb).to(
            tl.float32
        )  # [BJ, DZ] = db Wb  (the stock proj_z is a bf16 GEMM)
        pwb += tl.dot(tl.trans(db16), zn)  # [HP, DZ] += db^T zn
        pg += tl.sum(dzn * xh, 0)
        g = dzn * lw[None, :]
        gs = tl.sum(g, 1) / DZ
        gx = tl.sum(g * xh, 1) / DZ
        dz = rstd[:, None] * (g - gs[:, None] - xh * gx[:, None])
        tl.store(
            DZO + (i * N + j)[:, None] * DZ + d[None, :], dz.to(DZO.dtype.element_ty)
        )
    tl.store(PWB + pid * HP * DZ + hh[:, None] * DZ + d[None, :], pwb)
    tl.store(PLN + pid * DZ + d, pg)


def pair_fwd(z, mask, ln_w, eps, wb, H, BJ=64, num_warps=4, BJO=None):
    """z [N,N,DZ] bf16, mask [N,N] (bf16 0/1), wb [H, DZ] (proj_z.weight) -> w [H,N,N] bf16 (softmax over the last dim).
    BJO: j columns written per program (the row statistics are recomputed by each); default: the whole row."""
    N, _, DZ = z.shape
    HP = 16
    BJO = BJO or N
    assert N % BJO == 0 and BJO % BJ == 0
    wbt = torch.zeros((DZ, HP), dtype=torch.bfloat16, device=z.device)
    wbt[:, :H] = wb.to(torch.bfloat16).t()
    w = torch.empty((H, N, N), dtype=torch.bfloat16, device=z.device)
    cast(Any, _pair_fwd_kernel)[(N, N // BJO)](
        z,
        mask,
        ln_w,
        wbt,
        w,
        N,
        float(eps),
        DZ=DZ,
        HP=HP,
        H=H,
        BJ=BJ,
        BJO=BJO,
        num_warps=num_warps,
    )
    return w


def pair_bwd(z, w16, dw, ln_w, eps, wb, BJ=32, num_warps=4, BJO=None):
    """-> dz [N,N,DZ] bf16, dWb [H, DZ] fp32, dgamma_z [DZ] fp32.  dw [H,N,N] fp32."""
    N, _, DZ = z.shape
    H = w16.shape[0]
    HP = 16
    BJO = BJO or N
    assert N % BJO == 0 and BJO % BJ == 0
    nprog = N * (N // BJO)
    wbp = torch.zeros((HP, DZ), dtype=torch.bfloat16, device=z.device)
    wbp[:H] = wb.to(torch.bfloat16)
    dz = torch.empty_like(z)
    pwb = torch.empty((nprog, HP, DZ), dtype=torch.float32, device=z.device)
    pln = torch.empty((nprog, DZ), dtype=torch.float32, device=z.device)
    dwc = dw.contiguous()
    sdot = (
        (w16.float() * dwc).sum(-1).contiguous() if BJO < N else dwc
    )  # [H, N]; only needed when the row is split
    cast(Any, _pair_bwd_kernel)[(N, N // BJO)](
        z,
        w16,
        dwc,
        sdot,
        ln_w,
        wbp,
        dz,
        pwb,
        pln,
        N,
        float(eps),
        DZ=DZ,
        HP=HP,
        H=H,
        BJ=BJ,
        BJO=BJO,
        HAS_SDOT=BJO < N,
        num_warps=num_warps,
    )
    return dz, colsum(pwb)[:H], colsum(pln)


def _cast_h100_matrices(*weights):
    """Batch master-to-BF16 copies without changing the B200 path."""
    if native_aux_enabled() and torch.cuda.get_device_capability(weights[0].device) == (
        9,
        0,
    ):
        return tuple(_k()["pair3"].cast(list(weights), [0] * len(weights), False))
    if torch.cuda.get_device_capability(weights[0].device) == (9, 0) and any(
        w.dtype != torch.bfloat16 for w in weights
    ):
        from miniworld_engine.integrations.h100_master import pack

        return tuple(pack(weights))
    return tuple(w.to(torch.bfloat16).contiguous() for w in weights)


def _h100_float(*weights):
    if native_aux_enabled():
        return tuple(_k()["pair3"].cast(list(weights), [1] * len(weights), False))
    return tuple(w.float().contiguous() for w in weights)


class _PwaMath(torch.autograd.Function):
    """out = msa + PWA(msa, pair, key mask). Weights in their own dtype (fp32 or bf16); gradients returned in it."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(
        ctx, msa, pair, pm, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo, eps_m, eps_z, p_drop
    ):
        m = msa[0].contiguous()
        z = pair[0].contiguous()
        N = m.shape[1]
        bf = torch.bfloat16
        wv16, wg16, wo16 = _cast_h100_matrices(wv.detach(), wg.detach(), wo.detach())
        sm100 = torch.cuda.get_device_capability(m.device) == (10, 0)
        k = _k100() if sm100 else _k()
        if not sm100 and native_aux_enabled() and pm.numel() <= N:
            ctx.key_mask = pm
            pm = k["pair3"].mask(m, None if pm.numel() == 0 else pm, N)
        if sm100:  # pm: the [N] bool key mask (pair_weighted_averaging)
            w16 = k["pwa"].pair_fwd(
                z, pm, lnz_w.detach().contiguous(), eps_z, wb.detach().contiguous()
            )
        elif k["pair3"] is not None and N % 16 == 0 and N <= 1024:
            (zw,) = _h100_float(lnz_w.detach())
            wb16 = (
                _cast_h100_matrices(wb.detach())[0]
                if native_aux_enabled()
                else wb.detach()
            )
            w16 = k["pair3"].pair_fwd3(z, pm, zw, eps_z, wb16)
        else:
            w16 = pair_fwd(
                z, pm, lnz_w.detach().float().contiguous(), eps_z, wb.detach(), H, BJ=64
            )
        if sm100:
            v, y = k["pwa"].ln_vg(
                m, lnm_w.detach().contiguous(), lnm_b.detach().contiguous(), wv16, eps_m
            )
        else:
            mw, mb = _h100_float(lnm_w.detach(), lnm_b.detach())
            v, y = k["lnvg"].ln_vg(m, mw, mb, wv16, eps_m, 2, 3, 1)
        # the module's drop_msa (Dropout(broadcast_dim=1)): one keep-mask per (token, channel) shared over the MSA rows,
        # x * mask / (1 - p); applied to the bf16 update inside the kernel's residual epilogue
        dmask, dscale = None, 1.0
        if p_drop > 0:
            if not sm100 and native_aux_enabled():
                from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import (
                    extension,
                )

                dmask = extension("pack").drop_keep(m, N, D, float(p_drop))
            else:
                dmask = (torch.rand(N, D, device=m.device, dtype=bf) > p_drop).to(bf)
            dscale = 1.0 / (1.0 - p_drop)
        if sm100:
            o = torch.bmm(
                w16, v
            )  # the contraction on cuBLAS: 5-15 % faster than pwa_ctr at L 256-1024, bit-identical
            out = k[
                "pwa"
            ].pwa_gate_out(
                o,
                m,
                lnm_w.detach().contiguous(),
                lnm_b.detach().contiguous(),
                eps_m,
                wg16,
                wo16,
                dmask,
                dscale,
            )  # split: contraction, then gate / out (y recomputed; ln_vg's y kept for the backward)
        else:
            if (
                native_aux_enabled()
                and k.get("gateout") is not None
                and k["glue3"] is not None
                and _split_contraction(N)
            ):
                o = torch.bmm(
                    w16, v
                )  # head-major o, kept for the backward (glue3 reads it as such)
                out = k["gateout"].pwa_gateout(o, y, wg16, wo16, m, 1, 4, dmask, dscale)
            else:
                out, o = k["forward"](
                    w16, v, y, wg16, wo16, m, dmask, dscale
                )  # residual fused; o kept for the backward
        ctx.save_for_backward(
            m,
            z,
            w16,
            v,
            y,
            o,
            lnm_w,
            lnm_b,
            wv,
            wg,
            lnz_w,
            wb,
            wo,
            *((dmask,) if dmask is not None else ()),
        )
        ctx.eps = (eps_m, eps_z)
        ctx.dscale = dscale
        return out[None]

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dres):
        k = None if torch.cuda.get_device_capability(dres.device) == (10, 0) else _k()
        saved = ctx.saved_tensors
        m, z, w16, v, y, o, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo = saved[:13]
        dmask = saved[13] if len(saved) > 13 else None
        eps_m, eps_z = ctx.eps
        S, N = m.shape[0], m.shape[1]
        bf = torch.bfloat16
        dres0 = dres[0].contiguous()
        # the residual's gradient is dres itself; the update's is dres * mask / (1 - p) (what autograd of x * mask / (1-p) gives)
        wv16, wg16, wo16 = _cast_h100_matrices(wv, wg, wo)
        if torch.cuda.get_device_capability(m.device) == (10, 0):
            k = _k100()["pwa"]
            # the parameter gradients come out of the kernels in the parameters' dtype (.to below is then a no-op): no cast launches
            wot, wgvT = k.pwa_wprep(wg16, wv16, wo16)  # Wo^T, [Wg; Wv]^T: one launch
            # glue2: dgp never leaves the chip -- its consumers (dWg = dgp^T y, dyg = dgp Wg) run inside; dv_bwd adds dv Wv to dyg
            d_o, dyg, dWo, dWg = k.pwa_glue2(
                o, y, dres0, wg16, wot, dmask, ctx.dscale, int(wo.dtype == bf)
            )
            dv = torch.bmm(
                w16.transpose(1, 2), d_o
            )  # head-major [H, N, S*C], cuBLAS (as the forward)
            dw = torch.bmm(
                d_o, v.transpose(1, 2)
            )  # [H][N][N] bf16 (PyTorch's dw is bf16 too), K = S*C
            dm, dWv, dlw, dlb = k.dv_bwd(
                dv,
                dyg,
                y,
                m,
                dres0,
                wgvT,
                lnm_w.detach().contiguous(),
                eps_m,
                int(wv.dtype == bf),
                int(lnm_w.dtype == bf),
            )
            dz, dWb, dzw = k.pair_bwd(
                z,
                w16,
                dw,
                lnz_w.detach().contiguous(),
                eps_z,
                wb.detach().contiguous(),
                int(wb.dtype == bf),
                int(lnz_w.dtype == bf),
            )
            return (
                dm[None],
                dz[None],
                None,
                dlw.to(lnm_w.dtype),
                dlb.to(lnm_b.dtype),
                dWv.to(wv.dtype),
                dWg.to(wg.dtype),
                dzw.to(lnz_w.dtype),
                dWb.to(wb.dtype),
                dWo.to(wo.dtype),
                None,
                None,
                None,
            )
        assert (
            k is not None
        )  # the sm_100 branch above returned; dres and m share a device, so _k() ran
        native = native_aux_enabled()
        wot = k["pair3"].transpose(wo16) if native else wo16.t().contiguous()
        dgv = torch.empty(
            (S, N, 2 * HC), dtype=bf, device=m.device
        )  # dgp | dv: one [S,N,512] buffer
        if k["glue3"] is not None:  # dWo partials fused: go never touches memory
            d_o, _, dWo = k["glue3"].pwa_glue3(
                o,
                y,
                dres0,
                wg16,
                wot,
                dgv,
                2,
                1,
                dmask,
                ctx.dscale,
                bool(native and k.get("gateout") is not None and _split_contraction(N)),
            )
        else:
            dout = (
                dres0
                if dmask is None
                else (dres0 * dmask[None] * ctx.dscale).contiguous()
            )
            d_o, _, go = k["ctr"].pwa_glue_o(
                o, y, dout, wg16, wo16.t().contiguous(), 1, dgv
            )
            dWo = dout.view(S * N, D).t() @ go.view(S * N, HC)  # [D][HC]
        # dv = w^T . d_o. On the native path cuBLAS runs it (~2x faster than pwa_plain2 for the same FLOPs) and dgv_bwd reads its head-major
        # result tile by tile; otherwise pwa_plain2 writes dv into dgv[..., HC:].
        dv_hm = torch.bmm(w16.transpose(1, 2), d_o) if native and N % 64 == 0 else None
        if dv_hm is None:
            wt = (
                k["pair3"].transpose(w16)
                if native
                else w16.transpose(1, 2).contiguous()
            )
            k["fwd"].pwa_plain2(wt, d_o, 4, dgv)  # dv -> dgv[..., HC:]
        dw = torch.bmm(d_o, v.transpose(1, 2))  # [H][N][N], K = S*C
        dw, mw, zw = _h100_float(dw, lnm_w, lnz_w)
        wgvT = (
            k["pair3"].wgv(wg16, wv16)
            if native
            else torch.cat([wg16, wv16], 0).t().contiguous()
        )
        dm, dWgv, dlw, dlb = k["dgv"].dgv_bwd(
            dgv.view(S * N, 2 * HC),
            y.view(S * N, D),
            m.view(S * N, D),
            dres0.view(S * N, D),
            wgvT,
            mw,
            eps_m,
            1,
            8,
            True,
            dv_hm,
        )
        dWg, dWv = dWgv[:HC], dWgv[HC:]
        if native:
            (wb16,) = _cast_h100_matrices(wb)
            key = getattr(ctx, "key_mask", None)
            key = key if key is not None and key.numel() else None
            dz, dWb, dzw = k["pair3"].pair_bwd3(z, w16, dw, zw, eps_z, wb16, key)
            leaves = k["pair3"].cast(
                [dlw, dlb, dWv, dWg, dzw, dWb, dWo],
                [
                    int(w.dtype == torch.float32)
                    for w in (lnm_w, lnm_b, wv, wg, lnz_w, wb, wo)
                ],
                True,
            )
            return (dm.view(S, N, D)[None], dz[None], None, *leaves, None, None, None)
        dz, dWb, dzw = pair_bwd(z, w16, dw, zw, eps_z, wb, BJ=32)
        # Separate small weight/LN gradient views at the custom-op boundary.
        return (
            dm.view(S, N, D)[None],
            dz[None],
            None,
            dlw.to(lnm_w.dtype, copy=True),
            dlb.to(lnm_b.dtype, copy=True),
            dWv.to(wv.dtype, copy=True),
            dWg.to(wg.dtype, copy=True),
            dzw.to(lnz_w.dtype, copy=True),
            dWb.to(wb.dtype),
            dWo.to(wo.dtype),
            None,
            None,
            None,
        )


@torch.no_grad()
@torch.autocast("cuda", enabled=False)
def _inference_math(
    module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """msa + PWA(msa, pair, key mask) for a grad-free call: the same three forward kernels, o not kept, no dropout."""
    bf = torch.bfloat16
    m = msa[0].contiguous()
    z = pair[0].contiguous()
    n = m.shape[1]
    eps_m = float(getattr(module.ln_msa, "eps", 1e-5))
    eps_z = float(getattr(module.ln_pair, "eps", 1e-5))
    if torch.cuda.get_device_capability(m.device) == (10, 0):
        k = _k100()
        km = (
            torch.ones(n, dtype=torch.bool, device=m.device)
            if mask is None
            else mask[0].to(torch.bool).contiguous()
        )
        w16 = k["pwa"].pair_fwd(
            z,
            km,
            module.ln_pair.weight.detach().contiguous(),
            eps_z,
            module.to_bias.weight.detach().contiguous(),
        )
        lnw, lnb = (
            module.ln_msa.weight.detach().contiguous(),
            module.ln_msa.bias.detach().contiguous(),
        )
        v, _ = k["pwa"].ln_vg(
            m,
            lnw,
            lnb,
            module.to_value.weight.detach().to(bf).contiguous(),
            eps_m,
            False,
        )  # y is recomputed downstream
        out = k["pwa"].pwa_gate_out(
            torch.bmm(w16, v),
            m,
            lnw,
            lnb,
            eps_m,
            module.to_gate.weight.detach().to(bf).contiguous(),
            module.to_out.weight.detach().to(bf).contiguous(),
            None,
            1.0,
        )
        return out[None]
    native = native_inference_enabled(m.device)
    # FP32 logits, as in the training forward (both builds): bitwise parity with it, and the logits are not rounded to BF16.
    k = _k(native)
    pm = (
        k["pair3"].mask(m, mask, n)
        if native
        else (
            torch.ones(n, n, dtype=bf, device=m.device)
            if mask is None
            else mask[0].to(bf)[None, :].expand(n, n).contiguous()
        )
    )
    if native:
        lnz_w, lnm_w, lnm_b = k["pair3"].cast(
            [
                module.ln_pair.weight.detach(),
                module.ln_msa.weight.detach(),
                module.ln_msa.bias.detach(),
            ],
            [1] * 3,
            False,
        )
        wb16, wv16, wg16, wo16 = k["pair3"].cast(
            [
                module.to_bias.weight.detach(),
                module.to_value.weight.detach(),
                module.to_gate.weight.detach(),
                module.to_out.weight.detach(),
            ],
            [0] * 4,
            False,
        )
    else:
        lnz_w = module.ln_pair.weight.detach().float().contiguous()
        wb16 = module.to_bias.weight.detach()
    if k["pair3"] is not None and n % 16 == 0 and n <= NMAX_INFER:
        w16 = k["pair3"].pair_fwd3(z, pm, lnz_w, eps_z, wb16)
    else:
        w16 = pair_fwd(z, pm, lnz_w, eps_z, module.to_bias.weight.detach(), H, BJ=64)
    if native:
        v, y = k["lnvg"].ln_vg(m, lnm_w, lnm_b, wv16, eps_m, 2, 3, 1)
    else:
        v, y = k["lnvg"].ln_vg(
            m,
            module.ln_msa.weight.detach().float().contiguous(),
            module.ln_msa.bias.detach().float().contiguous(),
            module.to_value.weight.detach().to(bf).contiguous(),
            eps_m,
            2,
            3,
            1,
        )
        wg16 = module.to_gate.weight.detach().to(bf).contiguous()
        wo16 = module.to_out.weight.detach().to(bf).contiguous()
    if native and k.get("gateout") is not None and _split_contraction(n):
        # contraction on cuBLAS (it runs the same FLOPs ~2x faster than the fused kernel's n=64 wgmma tiles), gate / out pass in its own kernel;
        # at small N the fused kernel wins (the o round trip and the extra launch cost more than the faster contraction saves)
        out = k["gateout"].pwa_gateout(
            torch.bmm(w16, v), y, wg16, wo16, m, 1, 4, None, 1.0
        )
    elif "fwd3" in k:
        out, _ = k["fwd3"].pwa_fwd3(
            w16, v, y, wg16, wo16, m, False, 1, 3, 1, 0, None, 1.0
        )
    else:
        out, _ = k["fwd"].pwa_fwd2(w16, v, y, wg16, wo16, m, False, 1, 4, None, 1.0)
    return out[None]


def _is_wide(module) -> bool:
    """True for a module whose widths are not MiniWorld's (served by `pwa_wide`)."""
    return (
        module.to_value.weight.shape[1],
        module.to_bias.weight.shape[1],
        module.n_head,
        module.to_value.weight.shape[0] // module.n_head,
    ) != (D, DZ, H, C)


def pair_weighted_averaging(
    module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """msa + MSAPairWeightedAveraging(msa, pair, mask): the residual is INCLUDED (the forward kernel adds it)."""
    if _is_wide(module):
        return _wide.pair_weighted_averaging(module, msa, pair, mask)
    n = msa.shape[2]
    if torch.cuda.get_device_capability(msa.device) == (
        10,
        0,
    ):  # sm_100a: the [N] key mask itself
        pm = (
            torch.ones(n, dtype=torch.bool, device=msa.device)
            if mask is None
            else mask[0].to(torch.bool).contiguous()
        )
    elif native_aux_enabled(msa):
        # Keep the CUDA launch inside the opaque forward so torch.compile can
        # trace the module without following a pybind extension call.
        pm = (
            mask
            if mask is not None
            else torch.empty((0,), device=msa.device, dtype=torch.bool)
        )
    elif mask is None:
        pm = torch.ones(n, n, dtype=torch.bfloat16, device=msa.device)
    else:
        pm = (
            mask[0].to(torch.bfloat16)[None, :].expand(n, n).contiguous()
        )  # the module masks the key axis j
    eps_m = float(getattr(module.ln_msa, "eps", 1e-5))
    eps_z = float(getattr(module.ln_pair, "eps", 1e-5))
    p_drop = (
        float(module.drop_msa.p_drop) if module.training else 0.0
    )  # the module's drop_msa, fused into the kernel
    out = PwaTrainFn.apply(
        msa.contiguous(),
        pair.contiguous(),
        pm,
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_value.weight,
        module.to_gate.weight,
        module.ln_pair.weight,
        module.to_bias.weight,
        module.to_out.weight,
        eps_m,
        eps_z,
        p_drop,
    )
    if native_aux_enabled(msa) and torch.cuda.get_device_capability(msa.device) == (
        9,
        0,
    ):
        from miniworld_engine.integrations.token_dit_train import _native_caller_hooks

        # Keep the result's autograd edges alive before hooking leaf nodes.
        _native_caller_hooks(
            msa, pair, *module.parameters(), force=_h100_training_default(msa)
        )
    return out


class _SavedContext:
    eps: tuple[float, float]
    dscale: float
    key_mask: torch.Tensor

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def _forward_fake(args, eps_m, eps_z, p_drop):
    m = args[0]
    _, s, n, _ = m.shape
    # o is head-major [H, N, S*C] on sm_100a (cuBLAS bmm(w, v)), natural [S, N, HC] on sm_90a
    o_shape = (
        (H, n, s * C)
        if (
            torch.cuda.get_device_capability(m.device) == (10, 0)
            or (native_aux_enabled(m) and _split_contraction(n))
        )
        else (s, n, HC)
    )
    return [
        torch.empty_like(m),
        torch.empty((H, n, n), device=m.device, dtype=torch.bfloat16),
        torch.empty((H, n, s * C), device=m.device, dtype=torch.bfloat16),
        torch.empty((s, n, D), device=m.device, dtype=torch.bfloat16),
        torch.empty(o_shape, device=m.device, dtype=torch.bfloat16),
        torch.empty((n, D) if p_drop else (0,), device=m.device, dtype=torch.bfloat16),
    ]


from miniworld_engine.kernels._compile import opaque


@opaque(fake=_forward_fake, name="pwa_h100_fwd")
def _forward_op(
    args: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float
) -> list[torch.Tensor]:
    scope = _NATIVE_TRAINING_SCOPE.set(_h100_training_default(args[0]))
    try:
        ctx = _SavedContext()
        out = _PwaMath.forward(ctx, *args, eps_m, eps_z, p_drop)
        sv = ctx.saved_tensors
        return [out, *sv[2:6], sv[13] if p_drop else args[0].new_empty((0,))]
    finally:
        _NATIVE_TRAINING_SCOPE.reset(scope)


def _backward_fake(args, kept, dy, eps_m, eps_z, p_drop):
    return [torch.empty_like(args[i]) for i in (0, 1, 3, 4, 5, 6, 7, 8, 9)]


@opaque(fake=_backward_fake, name="pwa_h100_bwd")
def _backward_op(
    args: list[torch.Tensor],
    kept: list[torch.Tensor],
    dy: torch.Tensor,
    eps_m: float,
    eps_z: float,
    p_drop: float,
) -> list[torch.Tensor]:
    scope = _NATIVE_TRAINING_SCOPE.set(_h100_training_default(args[0]))
    try:
        return _backward_math(args, kept, dy, eps_m, eps_z, p_drop)
    finally:
        _NATIVE_TRAINING_SCOPE.reset(scope)


def _backward_math(args, kept, dy, eps_m, eps_z, p_drop):
    ctx = _SavedContext()
    ctx.eps = (eps_m, eps_z)
    ctx.dscale = 1 / (1 - p_drop) if p_drop else 1.0
    ctx.key_mask = args[2]
    ctx.saved_tensors = (
        args[0][0],
        args[1][0],
        *kept[:4],
        *args[3:],
        *((kept[4],) if p_drop else ()),
    )
    gradients = _PwaMath.backward(ctx, dy)
    return [gradients[i] for i in (0, 1, 3, 4, 5, 6, 7, 8, 9)]


class PwaTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = list(args[:10])
        ctx.eps_m, ctx.eps_z, ctx.p_drop = args[10:]
        out, *kept = _forward_op(tensors, ctx.eps_m, ctx.eps_z, ctx.p_drop)
        ctx.save_for_backward(*tensors, *kept)
        return out

    @staticmethod
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = _backward_op(
            list(vals[:10]), list(vals[10:]), dy, ctx.eps_m, ctx.eps_z, ctx.p_drop
        )
        if native_aux_enabled(vals[0]) and torch.cuda.get_device_capability(
            dy.device
        ) == (9, 0):
            from miniworld_engine.integrations.token_dit_train import (
                _activate_native_caller,
            )

            _activate_native_caller(force=_h100_training_default(vals[0]))
        return (*grads[:2], None, *grads[2:], None, None, None)


def _infer_fake(args, eps_m, eps_z):
    return torch.empty_like(args[0])


@opaque(fake=_infer_fake, name="pwa_h100_infer")
def _inference_op(args: list[torch.Tensor], eps_m: float, eps_z: float) -> torch.Tensor:
    from types import SimpleNamespace as NS

    msa, pair, mask, lmw, lmb, wv, wg, lzw, wb, wo = args
    module = NS(
        ln_msa=NS(weight=lmw, bias=lmb, eps=eps_m),
        ln_pair=NS(weight=lzw, eps=eps_z),
        to_value=NS(weight=wv),
        to_gate=NS(weight=wg),
        to_bias=NS(weight=wb),
        to_out=NS(weight=wo),
    )
    return _inference_math(module, msa, pair, None if mask.numel() == 0 else mask)


def pair_weighted_averaging_inference(module, msa, pair, mask):
    if _is_wide(module):
        return _wide.pair_weighted_averaging_inference(module, msa, pair, mask)
    mask = (
        mask
        if mask is not None
        else (
            torch.empty((0,), device=msa.device, dtype=torch.bool)
            if native_inference_enabled(msa.device)
            else torch.ones((1, msa.shape[2]), device=msa.device, dtype=torch.bool)
        )
    )
    args = [
        msa.contiguous(),
        pair.contiguous(),
        mask.contiguous(),
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_value.weight,
        module.to_gate.weight,
        module.ln_pair.weight,
        module.to_bias.weight,
        module.to_out.weight,
    ]
    return _inference_op(args, module.ln_msa.eps, module.ln_pair.eps)
