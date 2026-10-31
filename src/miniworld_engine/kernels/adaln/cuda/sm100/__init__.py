"""B200 (sm_100a) adaptive LayerNorm without a backward: ``y = sigmoid(LN(c) Ws^T + bs) * LN(x) + LN(c) Wb^T``. At the atom width
(cond 128: ``FUSED_KC``) in ONE launch, ``adaln_fused_sm100.cu`` (LN(cond) built in shared memory as the MMA's A operand, the weights
resident, the gate and LN(x) in the epilogue: cn never reaches HBM); elsewhere in two launches,

    ln_cond_xstats (adaln_rows.cu)     cn = LN(cond) * wc (activation dtype; fp32 on the TF32 grid) and x's (rstd, mean rstd)
    adaln_gemm_sm100 (this directory)  one tcgen05 GEMM, [cn Wb^T | cn Ws^T] as two tensor-memory accumulators, the gate and LN(x) in the
                                       epilogue; launched as a programmatic dependent of the row pass

instead of the row kernels around cuBLAS (``ln_cond_fwd`` -> a weight pack -> ``addmm`` -> ``adaln_fwd``: four launches and the
[scale | bias] pre-activations through HBM). bf16 on the bf16 tensor cores, fp32 on the TF32 ones (where torch allows TF32). The weights
are cast (fp32: rounded to the TF32 grid) once per parameter version (``_capture.param_forms``). The training route keeps the row
kernels + cuBLAS of ``kernels/adaln/cuda`` (its backward reads the pre-activations).

The kernel is built per (dtype, output tile width) as a cubin and launched through the sm_100a host binding of the Transition kernels
(``fused_sm100a._ext``: tensor maps, cubin registry, launches with the programmatic-dependent attribute).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent
#: Output-tile widths per dtype, widest first (two accumulators double-buffered: 2 x 2 x BN <= 512 tensor-memory columns; a bf16 output is
#: staged in 64-column groups)
_BNS = {False: (128, 64), True: (128, 64, 32)}
#: A/B and test override of the output-tile width (None: ``_pick_bn``)
BN_FORCE: int | None = None


def _tf32_allowed() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def supported(x: torch.Tensor, cond: torch.Tensor) -> bool:
    """sm_100 (B200), x / cond of one dtype (bf16, or fp32 with TF32 allowed), both widths multiples of 128 up to 1024, rows matching."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import _is_b200

    if not (x.is_cuda and cond.is_cuda and x.dtype == cond.dtype and x.dtype in (torch.bfloat16, torch.float32)):
        return False
    if x.dtype is torch.float32 and not _tf32_allowed():
        return False
    d, dc = x.shape[-1], cond.shape[-1]
    if d % 128 or dc % 128 or d > 1024 or dc > 1024 or x.numel() == 0 or x.shape[:-1] != cond.shape[:-1]:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    return _is_b200(idx)


#: Conditioning widths taken by the one-kernel route (``adaln_fused_sm100.cu``: LN(cond) built in shared memory, the weights resident);
#: its output-tile widths per dtype (fp32: the resident weights and the two A buffers must fit shared memory)
FUSED_KC = (128,)
_FUSED_BNS = {False: (128, 64), True: (128, 64)}   # fp32: built and tested, not routed (see ``supported``)
#: A/B override: None (the one-kernel route where it applies), False (always the two launches)
FUSED: bool | None = None


def _fused_ok(d: int, dc: int) -> bool:
    return FUSED is not False and dc in FUSED_KC and d in (128, 256)


@functools.lru_cache(maxsize=16)
def _load_fused(f32: bool, bn: int, kc: int, dx: int):
    from miniworld_engine.kernels import _nvcc
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    tag = f"{'f32' if f32 else 'bf16'}_bn{bn}_k{kc}_d{dx}"
    inc = _dir   # this directory's frozen copy of the sm_100a primitives (sm100.cuh)
    path = _nvcc.cubin(f"adaln_sm100a/fused_{tag}", ["-std=c++17", "-O3", "-arch=sm_100a", "-cubin", f"-I{inc}", f"-DF32={int(f32)}",
                                                     f"-DBN={bn}", f"-DKC={kc}", f"-DDX={dx}"], source=_dir / "adaln_fused_sm100.cu",
                       toolchain="sm100")
    ext = fused_sm100a._ext()
    ext.load_cubin(f"adaln_sm100a_fused_{tag}", str(path))
    return ext, f"adaln_sm100a_fused_{tag}"


def _fused(x2, c2, wc, wb, ws, bs, eps_x, eps_c, rna, nsm):
    """The one-kernel route (``adaln_fused_sm100.cu``)."""
    f32 = x2.dtype is torch.float32
    m, d = x2.shape
    dc = c2.shape[1]
    mt = -(-m // 128)
    widths = _FUSED_BNS[f32]
    bn = BN_FORCE if BN_FORCE in widths else next((w for w in widths if mt * (d // w) >= nsm), widths[-1])
    nt = d // bn
    ext, name = _load_fused(f32, bn, dc, d)
    y = torch.empty_like(x2)
    bke = 32 if f32 else 64
    t = ext.tmap
    grid = max(nt, min(mt * nt, (nsm // nt) * nt))
    ext.launch_kernel(name, "adaln_fused_sm100", grid, 320, 1,
                      [t(wb, dc, d, bke, bn), t(ws, dc, d, bke, bn), t(c2, dc, m, bke, 128), t(x2, d, m, bke, 128), wc, x2, y, bs, m,
                       float(eps_c), float(eps_x), int(bool(rna) and f32)])
    return y


#: staging buffers per epilogue warp of the GEMM (2: double-buffered stores; 1: a deeper operand ring), by dtype. fp32: 1 (measured,
#: CUDA-graph replay per call: [1920 / 3840, 768] c384 17.9 / 25.1 -> 17.3 / 24.2 us)
STAGING = {False: 2, True: 1}


@functools.lru_cache(maxsize=16)
def _load(f32: bool, bn: int, stg: int = 2):
    """The sm_100a host binding with this (dtype, tile width, cluster) cubin registered (loaded on the current device: a rejected cubin fails
    here). Returns (binding, registered name)."""
    from miniworld_engine.kernels import _nvcc
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    tag = f"{'f32' if f32 else 'bf16'}_bn{bn}_s{stg}"
    inc = _dir   # this directory's frozen copy of the sm_100a primitives (sm100.cuh)
    path = _nvcc.cubin(f"adaln_sm100a/gemm_{tag}", ["-std=c++17", "-O3", "-arch=sm_100a", "-cubin", f"-I{inc}", f"-DF32={int(f32)}",
                                                    f"-DBN={bn}", f"-DSTGP={stg}"], source=_dir / "adaln_gemm_sm100.cu", toolchain="sm100")
    ext = fused_sm100a._ext()
    ext.load_cubin(f"adaln_sm100a_{tag}", str(path))
    return ext, f"adaln_sm100a_{tag}"


def _pick_bn(m: int, n: int, f32: bool, nsm: int) -> int:
    """The tile width with the least (waves x per-tile operand intake): one tile reads 128 rows of cn and 2 BN rows of the weights per
    K-block, so a narrower tile only pays where it fills more of the first wave."""
    best = None
    for bn in _BNS[f32]:
        tiles = -(-m // 128) * -(-n // bn)
        cost = -(-tiles // nsm) * (128 + 2 * bn)
        if best is None or cost < best[0]:
            best = (cost, bn)
    return best[1]


def _weights(ws, bs, wb, f32: bool):
    """(Wb, Ws) in the activation dtype (fp32: on the TF32 grid), contiguous, and the gate bias fp32 (rounded to bf16 first for bf16)."""
    from miniworld_engine.kernels._tf32 import tf32_grid

    dt = torch.float32 if f32 else torch.bfloat16
    wb_, ws_ = wb.to(dt).contiguous(), ws.to(dt).contiguous()
    if f32:
        wb_, ws_ = tf32_grid(wb_), tf32_grid(ws_)
    b = bs.to(dt).float().contiguous() if bs is not None else torch.zeros(ws.shape[0], dtype=torch.float32, device=ws.device)
    return wb_, ws_, b


#: fp32 route: "cublas" (ln_cond_fwd -> cuBLAS TF32 -> adaln_fwd, the weights packed once per parameter version, every width) or "tcgen05"
#: (adaln_gemm_sm100 with the gate in its epilogue; at cond 128 the one-kernel route). Measured (B200, CUDA-graph replay per call, us,
#: tcgen05 / cublas / the Triton materialize route): [640, 768] c384 11.9 / 11.2 / 13.7, [1920, 768] c384 17.1 / 16.3 / 19.0, [3840, 768]
#: c384 24.1 / 23.1 / 23.6, c768 17.0 / 12.8 / 15.8, 22.4 / 19.6 / 22.3, 33.7 / 30.9 / 30.7: the TF32 tcgen05 GEMM at this thin K is
#: below cuBLAS's, more than the epilogue fusion gives back. The atom width (cond 128): the one-kernel route at [5120 / 40960, 128] 12.1 /
#: 39.4 against 12.4 / 46.0 for the rows around a GEMM; the row kernels around cuBLAS measured 30.6 / 40.9 in the ladder (L4096 / 8192).
F32_ROUTE = "cublas"


def _f32_cublas(x2, c2, wc, wcat, bcat, eps_x, eps_c, rna):
    """The fp32 row kernels around one cuBLAS TF32 GEMM: cn = LN(cond) wc, [scale | bias] = cn [Ws; Wb]^T + [bs | 0], y = adaln_fwd (x's
    statistics in the same pass)."""
    from miniworld_engine.kernels._tf32 import tf32_matmul

    from .. import _ext as _rows_ext

    e = _rows_ext()
    cn, _ = e.ln_cond_fwd(c2, wc, float(eps_c))
    with tf32_matmul():
        sb = torch.addmm(bcat, cn, wcat.t())
    y, _ = e.adaln_fwd(x2, sb, float(eps_x), bool(rna))
    return y


def _infer_fake(x2, c2, wc, wb, ws, bs, eps_x, eps_c, rna):
    """Output structure of ``_infer``: y [M, D] like x2."""
    return torch.empty_like(x2)


def _f32_cublas_op_fake(x2, c2, wc, wcat, bcat, eps_x, eps_c, rna):
    """Output structure of ``_f32_cublas_op``: y [M, D] like x2."""
    return torch.empty_like(x2)


@opaque(fake=_f32_cublas_op_fake, name="adaln_sm100a_f32_cublas")
def _f32_cublas_op(x2: torch.Tensor, c2: torch.Tensor, wc: torch.Tensor, wcat: torch.Tensor, bcat: torch.Tensor, eps_x: float, eps_c: float,
                   rna: bool) -> torch.Tensor:
    """``_f32_cublas`` as one opaque op."""
    return _f32_cublas(x2, c2, wc, wcat, bcat, eps_x, eps_c, rna)


@opaque(fake=_infer_fake, name="adaln_sm100a_infer")
def _infer(x2: torch.Tensor, c2: torch.Tensor, wc: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, bs: torch.Tensor, eps_x: float,
           eps_c: float, rna: bool) -> torch.Tensor:
    """x2 [M, D], c2 [M, DC] contiguous of one dtype; wb / ws [D, DC] in that dtype (fp32: TF32 grid), bs [D] fp32. Returns y [M, D]."""
    from miniworld_engine.kernels._device import num_sms

    from .. import _ext as _rows_ext

    f32 = x2.dtype is torch.float32
    m, d = x2.shape
    dc = c2.shape[1]
    if _fused_ok(d, dc):
        return _fused(x2, c2, wc.contiguous(), wb, ws, bs, eps_x, eps_c, rna, num_sms(x2.device))
    cn, st = _rows_ext().ln_cond_xstats(c2, wc.contiguous(), x2, float(eps_c), float(eps_x), f32)
    nsm = num_sms(x2.device)
    bn = BN_FORCE or _pick_bn(m, d, f32, nsm)
    ext, name = _load(f32, bn, STAGING[f32])
    bke, ocg = (32, 32) if f32 else (64, 64)
    y = torch.empty_like(x2)
    t = ext.tmap
    tiles = -(-m // 128) * -(-d // bn)
    ext.launch_kernel(
        name, "adaln_gemm_sm100", max(1, min(nsm, tiles)), 320, 1,
        [t(cn, dc, m, bke, 128), t(wb, dc, d, bke, bn), t(ws, dc, d, bke, bn), t(y, d, m, ocg, 32), x2, st, bs, m, d, dc, d,
         int(bool(rna) and f32)],
        pdl=True,
    )
    return y


def adaln_inference(x, cond, wc, ws, bs, wb, eps_x: float, eps_c: float, *, rna: bool = False) -> torch.Tensor:
    """The forward without a backward: x [..., D], cond [..., DC] of one dtype (``supported``); the projection weights in any dtype (fp32
    masters are cast once per parameter version). ``rna``: an fp32 result on the TF32 grid (its consumer is a TF32 MMA)."""
    from miniworld_engine.kernels import _capture

    f32 = x.dtype is torch.float32
    params = (ws, wb) if bs is None else (ws, bs, wb)
    d, dc = x.shape[-1], cond.shape[-1]
    if f32 and F32_ROUTE == "cublas":
        wcat, bcat = _capture.param_forms(
            "adaln_sm100a_f32_cublas", params,
            lambda: (torch.cat([ws, wb]).float().contiguous(),
                     torch.cat([bs.float() if bs is not None else ws.new_zeros(d, dtype=torch.float32), ws.new_zeros(d, dtype=torch.float32)])))
        y = _f32_cublas_op(x.reshape(-1, d).contiguous(), cond.reshape(-1, dc).contiguous(), wc.contiguous(), wcat, bcat, float(eps_x),
                           float(eps_c), bool(rna))
        return y.view(x.shape)
    wb_, ws_, bs_ = _capture.param_forms(f"adaln_sm100a_{x.dtype}", params, lambda: _weights(ws, bs, wb, f32))
    y = _infer(x.reshape(-1, d).contiguous(), cond.reshape(-1, dc).contiguous(), wc, wb_, ws_, bs_, float(eps_x), float(eps_c), bool(rna))
    return y.view(x.shape)


__all__ = ["adaln_inference", "supported"]
