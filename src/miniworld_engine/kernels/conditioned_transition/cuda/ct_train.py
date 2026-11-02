"""H100 / B200 ConditionedTransition tail, training: forward and backward (``ct_rows.cu`` + cuBLAS; the row kernels are
architecture-neutral and built for sm_90a and sm_100a).

    ab = x [Wa; Wb]^T        (cuBLAS)       h = silu(a) b                 (``swiglu_fwd``)
    out = h Ws^T             (cuBLAS)       sp = cond Wsc^T + b           (cuBLAS, bias epilogue)
    y = res + sigmoid(sp) out               (``sigmul_fwd``)

and the backward as the mirror (``sigmul_bwd``, ``swiglu_bwd`` and one cuBLAS GEMM per operand gradient). ``x`` is the output of the module's AdaLN
(the kernel is the post-AdaLN tail, as the Triton family's). bf16 activations (fp32 master weights are rounded to bf16 per call and their gradients
come back in fp32) or fp32 on TF32 GEMMs (the process's ``allow_tf32``). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels import _cast_cuda

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.adaln.cuda.infer import colsum

_dir = Path(__file__).parent
_FM = os.environ.get("CT_FM", "0") == "1"   # the feature-major backward (measured: no gain at 18432 rows, see _backward_feature_major)


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="ct_rows_sm90",
        sources=[str(_dir / "ct_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(), "-O3", "-std=c++17", *gencodes("90a", "100a"), "--expt-relaxed-constexpr", f"-I{_dir.parent.parent / 'adaln' / 'cuda'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor, cond: torch.Tensor, nd: int) -> bool:
    """sm_90 or sm_100, bf16, or fp32 under TF32 (``allow_tf32``); the hidden width a multiple of 8."""
    if not (x.is_cuda and cond.is_cuda and x.dtype == cond.dtype):
        return False
    if x.dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return False
    if x.dtype not in (torch.bfloat16, torch.float32) or nd % 8:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx) in ((9, 0), (10, 0))


def _squeeze_gate_save_fake(h, cond, ws, wsc, bsc, res):
    """(y, out, sp) [M, D]."""
    n = (h.shape[0], ws.shape[0])
    return h.new_empty(n), h.new_empty(n), h.new_empty(n)


@opaque(fake=_squeeze_gate_save_fake, name="conditioned_transition_cuda_squeeze_gate_save")
def squeeze_gate_save(h: torch.Tensor, cond: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
                      res: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 training forward of the tail's second half in one wgmma kernel (``ct_sm90.cu``): y = res + sigmoid(cond Wsc^T + b) (h Ws^T), and the two
    products the backward needs, out = h Ws^T and sp = cond Wsc^T + b, each rounded to bf16 as the unfused ops round them."""
    from .ct_inference import ext as ct_ext

    y = torch.empty(h.shape[0], ws.shape[0], device=h.device, dtype=h.dtype)
    out, sp = torch.empty_like(y), torch.empty_like(y)
    ct_ext().squeeze_gate(h, ws, cond, wsc, bsc, res, y, num_sms(h.device), 128, True, out, sp)
    return y, out, sp


def _cast_weights_fake(wa, wb, ws, wsc, bsc):
    """([Wa; Wb] [2 ND, D], Ws, Wsc, bsc) in bf16."""
    bf = torch.bfloat16
    return (wa.new_empty((2 * wa.shape[0], wa.shape[1]), dtype=bf), ws.new_empty(ws.shape, dtype=bf), wsc.new_empty(wsc.shape, dtype=bf),
            bsc.new_empty(bsc.shape, dtype=bf))


@opaque(fake=_cast_weights_fake, name="conditioned_transition_cuda_cast_weights")
def cast_weights(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The five fp32 master weights of a training step as bf16 in ONE launch; Wa and Wb land in the two halves of their concatenation."""
    bf = torch.bfloat16
    nd = wa.shape[0]
    wab = torch.empty(2 * nd, wa.shape[1], device=wa.device, dtype=bf)
    wsb, wscb, bscb = (torch.empty(t.shape, device=t.device, dtype=bf) for t in (ws, wsc, bsc))
    _cast_cuda.cast_into([wa, wb, ws, wsc, bsc], [wab[:nd], wab[nd:], wsb, wscb, bscb])
    return wab, wsb, wscb, bscb


def _swiglu_fwd_fake(ab):
    """h [M, ND] of ab [M, 2 ND]."""
    return ab.new_empty((ab.shape[0], ab.shape[1] // 2))


@opaque(fake=_swiglu_fwd_fake, name="conditioned_transition_cuda_swiglu_fwd")
def swiglu_fwd(ab: torch.Tensor) -> torch.Tensor:
    """h = silu(a) b of ab = [a | b]."""
    return ext().swiglu_fwd(ab)


def _swiglu_bwd_fake(dh, ab):
    """[dA | dB] like ab."""
    return torch.empty_like(ab)


@opaque(fake=_swiglu_bwd_fake, name="conditioned_transition_cuda_swiglu_bwd")
def swiglu_bwd(dh: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """da = dh b silu'(a), db = dh silu(a), written as [dA | dB]."""
    return ext().swiglu_bwd(dh, ab)


def _sigmul_fwd_fake(out, sp, res):
    """y like out."""
    return torch.empty_like(out)


@opaque(fake=_sigmul_fwd_fake, name="conditioned_transition_cuda_sigmul_fwd")
def sigmul_fwd(out: torch.Tensor, sp: torch.Tensor, res: torch.Tensor | None) -> torch.Tensor:
    """y = res + sigmoid(sp) out (the gated update rounded to the stream dtype before the add)."""
    return ext().sigmul_fwd(out, sp, res)


def sigmul_bwd_blocks(dy: torch.Tensor) -> int:
    """Blocks of ``sigmul_bwd``'s grid (= rows of its partial-sum buffer): a function of the shape alone, so the fake knows it."""
    v = 4 if dy.dtype is torch.float32 else 8
    ndc = dy.shape[-1] // v
    threads = ndc * max(1, 128 // ndc)
    return min(-(-(dy.numel() // v) // threads), 132 * 8)


def _sigmul_bwd_fake(dy, out, sp):
    """(dout, dscale, the block partials of the column sums of dscale)."""
    return torch.empty_like(dy), torch.empty_like(dy), dy.new_empty((sigmul_bwd_blocks(dy), dy.shape[-1]), dtype=torch.float32)


@opaque(fake=_sigmul_bwd_fake, name="conditioned_transition_cuda_sigmul_bwd")
def sigmul_bwd(dy: torch.Tensor, out: torch.Tensor, sp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dout = dy g, dscale = dy out g (1 - g), and the block partials of sum(dscale)."""
    dout, dsc, part = ext().sigmul_bwd(dy, out, sp, sigmul_bwd_blocks(dy))
    return dout, dsc, part


# Measured (M = 36864, bf16): the fused wgmma squeeze + gate kernel with the two saved outputs takes 425 us against 108 + 43 + 79 for cuBLAS + addmm + sigmul_fwd
# (a 128 x 128 tile is bound by L2 -> SM traffic at ~280 TFLOPS; cuBLAS runs 192 x 192): kept for the small-M inference tail, off for training.
_FUSED_SQUEEZE_TRAIN = False


def _fused_squeeze_ok(h, cond, ws) -> bool:
    """The squeeze + gate kernel's tiling: K slabs of 64, whole 128-column output tiles."""
    return _FUSED_SQUEEZE_TRAIN and torch.cuda.get_device_capability(h.device) == (9, 0) and ws.shape[0] % 128 == 0 and h.shape[1] % 64 == 0 and cond.shape[1] % 64 == 0 and h.stride(1) == 1 and cond.stride(1) == 1


def _pack_fp32_fake(wa, wb, ws, wsc):
    """Output structure of ``conditioned_transition_cuda_pack_fp32``."""
    nd, d = wa.shape
    dc = wsc.shape[1]
    return wa.new_empty((2 * nd, d)), wa.new_empty((d, 2 * nd)), wa.new_empty((nd, d)), wa.new_empty((dc, d))


@opaque(fake=_pack_fp32_fake, name="conditioned_transition_cuda_pack_fp32")
def pack_fp32(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 weights of a training step in ONE launch (``adaln.cuda.pack_f32``): [Wa; Wb] [2 ND, D], and the K-major (transposed) right operands of the three
    TF32 data-gradient GEMMs: [Wa; Wb]^T [D, 2 ND], Ws^T [ND, D], Wsc^T [DC, D] (cuBLAS "nt" kernels: 15 - 25 % faster than "nn"; the PyTorch transposing
    copies were 40 us a step at 18432 rows)."""
    from miniworld_engine.kernels.adaln.cuda import _ext as rows_ext

    nd, d = wa.shape
    dc = wsc.shape[1]
    dev, f32 = wa.device, torch.float32
    wab = torch.empty(2 * nd, d, device=dev, dtype=f32)
    wabT = torch.empty(d, 2 * nd, device=dev, dtype=f32)
    wsT = torch.empty(nd, d, device=dev, dtype=f32)
    wscT = torch.empty(dc, d, device=dev, dtype=f32)
    wa, wb, ws, wsc = (w.contiguous() for w in (wa, wb, ws, wsc))
    # every copy on the TF32 grid (round to nearest): the tcgen05 GEMMs (``ct_tc``) truncate their operands; cuBLAS' own rounding of them is the identity
    rows_ext().pack_f32([wa, wb, wa, wb, ws, wsc], [wab, wab, wabT, wabT, wsT, wscT],
                        [[0, 0, 0, 1], [0, nd, 0, 1], [1, 0, 0, 1], [1, 0, nd, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    return wab, wabT, wsT, wscT


_XT = os.environ.get("CT_XT", "1") != "0"   # read once at import (the routing runs inside traces)

# ---- B200 fp32: the expand GEMM with the SwiGLU in its epilogue (SWAB) and the dh GEMM with the SwiGLU backward in its epilogue (GAB), TF32 tcgen05
# (``ct_tc``). Measured (kernels alone, CUDA graph, us), cuBLAS + row pass -> fused:
#                     expand + swiglu_fwd -> SWAB          dh + swiglu_bwd -> GAB
#     [36864, 768] n2      315 -> 284 (128 x 2-CTA)          282 -> 233 (256 x 2-CTA)
#     [18432, 768]         161 -> 145                        146 -> 122
#     [6144, 768]           55 -> 55                          51 -> 50
#     [393216, 128] n2     348 -> 301 (64, 1-CTA)            415 -> 303 (64, 1-CTA)
#     [196608, 128]        180 -> 156                        211 -> 157
#     [49152, 128]          47 -> 46                          54 -> 47
# so both from TC_MIN_ROWS rows (token: above the tie at a few thousand rows).
TC = os.environ.get("CT_TC", "1") != "0"
TC_MIN_ROWS = {"token": 12288, "atom": 32768}


def tc_route(x: torch.Tensor, nd: int, xn_tf32: bool) -> tuple[bool, bool]:
    """(SWAB forward, GAB backward) for this call: B200, fp32 activations already on the TF32 grid (the AdaLN's ``rna``), the widths the builds tile."""
    m, d = x.shape
    if not (TC and xn_tf32 and x.dtype is torch.float32 and x.is_cuda and torch.backends.cuda.matmul.allow_tf32):
        return False, False
    if torch.cuda.get_device_capability(x.device) != (10, 0) or d % 32 or nd % 256:
        return False, False
    atom = d <= 128
    if m < TC_MIN_ROWS["atom" if atom else "token"]:
        return False, False
    return True, d <= 1024


def _tc_bn(d: int, mode: str):  # noqa: D103
    """Tile choice per width (measured above): the atom width on single CTAs 64 wide, the token widths on CTA pairs."""
    if d <= 128:
        return (64, False)
    return (128, True) if mode == "swab" else (256, True)


def _swab_fake(x, wab):
    """(h [M, ND], [a | b] [M, 2 ND]) like x."""
    nd = wab.shape[0] // 2
    return x.new_empty((x.shape[0], nd)), x.new_empty((x.shape[0], 2 * nd))


@opaque(fake=_swab_fake, name="conditioned_transition_cuda_swab_f32")
def swab(x: torch.Tensor, wab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, [a | b]) of x [M, D] (TF32 grid) and [Wa; Wb] [2 ND, D] (TF32 grid): one tcgen05 GEMM with the SwiGLU in its epilogue."""
    from . import ct_tc

    m, d = x.shape
    nd = wab.shape[0] // 2
    h = x.new_empty((m, nd))
    ab = x.new_empty((m, 2 * nd))
    ct_tc.gemm("swab", m, nd, x, wab[:nd], h, o1=ab, b1=wab[nd:], k0=d, ld0=nd, ld1=2 * nd, hoff=nd, force=_tc_bn(d, "swab"))
    return h, ab


def _gab_fake(dout, wsT, ab):
    """[dA | dB] like ab."""
    return torch.empty_like(ab)


# ---- B200 bf16 backward: dh = dout Ws and the SwiGLU backward in ONE tcgen05 GEMM (``tg_gemm.cu`` MODE 7, GATE_AB without h: the epilogue
# TMA-loads the forward's saved [a | b] tile and writes [dA | dB]; dh never reaches HBM). Replaces cuBLAS dh + ``swiglu_bwd`` (a write and a
# read of dh [M, ND]). GATE7_BN: the output tile width (128 or 256 columns of the hidden width); None turns the route off (A/B).
GATE7_BN: int | None = 256


#: the forward's expand on B200 bf16: h and [a | b] from one tcgen05 GEMM with the SwiGLU in its epilogue (``tg_gemm.cu`` MODE 5, CTA-pair
#: build, 128-column tiles of each of a and b) instead of cuBLAS ab + ``swiglu_fwd`` (which reads ab back). Kernels alone, bf16 d768 hid1536,
#: M 18432 / 36864 / 49152: 91.8 / 178.9 / 251.1 us -> 85.2 / 163.2 / 214.3. h is silu(a) b of the fp32 products (the saved a, b are rounded
#: to bf16 as before). False: the cuBLAS route (A/B).
SWIGLU_AB = True


def _swab_ok(x: torch.Tensor, wab: torch.Tensor) -> bool:
    nd = wab.shape[0] // 2
    return (SWIGLU_AB and x.dtype is torch.bfloat16 and wab.dtype is torch.bfloat16 and torch.cuda.get_device_capability(x.device) == (10, 0)
            and nd % 128 == 0 and x.shape[1] % 64 == 0 and wab.is_contiguous() and x.is_contiguous())


def _swiglu_ab_fake(x, wab):
    """(h [M, ND], ab [M, 2 ND])."""
    nd = wab.shape[0] // 2
    return x.new_empty((x.shape[0], nd)), x.new_empty((x.shape[0], 2 * nd))


@opaque(fake=_swiglu_ab_fake, name="conditioned_transition_cuda_swiglu_ab_sm100a")
def swiglu_ab(x: torch.Tensor, wab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """h = silu(a) b and ab = [a | b] = x [Wa; Wb]^T (bf16, ``wab`` the concatenation [2 ND, D])."""
    from .ct_tg import gemm

    m, d = x.shape
    nd = wab.shape[0] // 2
    h = x.new_empty((m, nd))
    ab = x.new_empty((m, 2 * nd))
    gemm("swiglu_ab", 128, True, m, nd, x, wab[:nd], h, b1=wab[nd:], o1=ab, k0=d, hoff=nd)
    return h, ab


def _transpose_bf16_fake(t):
    """t^T [C, R]."""
    return t.new_empty((t.shape[1], t.shape[0]))


@opaque(fake=_transpose_bf16_fake, name="conditioned_transition_cuda_transpose_bf16")
def transpose_bf16(t: torch.Tensor) -> torch.Tensor:
    """t^T (bf16, contiguous): one tiled pass (``ct_rows.cu``)."""
    return ext().transpose_bf16(t.contiguous())


def _gate7_ok(dout: torch.Tensor, ab: torch.Tensor) -> bool:
    if GATE7_BN is None or dout.dtype is not torch.bfloat16 or ab.dtype is not torch.bfloat16:
        return False
    nd = ab.shape[1] // 2
    return (torch.cuda.get_device_capability(dout.device) == (10, 0) and dout.shape[1] % 64 == 0 and nd % GATE7_BN == 0
            and ab.is_contiguous() and dout.shape[0] == ab.shape[0])


def _gate_ab_fake(dout, wsT, ab):
    """[dA | dB] like ab."""
    return torch.empty_like(ab)


@opaque(fake=_gab_fake, name="conditioned_transition_cuda_gab_f32")
def gab(dout: torch.Tensor, wsT: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """[dA | dB] = SwiGLU backward of dh = dout Ws (dout [M, D] and Ws^T [ND, D] on the TF32 grid) on the forward's ab: one tcgen05 GEMM, dh never in HBM."""
    from . import ct_tc

    m, d = dout.shape
    nd = wsT.shape[0]
    dab = torch.empty_like(ab)
    ct_tc.gemm("gab", m, nd, dout, wsT, dab, aux=ab, k0=d, ld0=2 * nd, ldaux=2 * nd, hoff=nd, force=_tc_bn(d, "gab"))
    return dab


@opaque(fake=_gate_ab_fake, name="conditioned_transition_cuda_gate_ab_sm100a")
def gate_ab(dout: torch.Tensor, wsT: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """[dA | dB] of dh = dout Ws (``wsT`` = Ws^T [ND, D], bf16) and the saved ab = [a | b] [M, 2 ND]: da = dh b silu'(a), db = dh silu(a)."""
    from .ct_tg import gemm

    m, d = dout.shape
    nd = wsT.shape[0]
    dab = torch.empty_like(ab)
    gemm("gate7", GATE7_BN, False, m, nd, dout, wsT, dab, k0=d, o1=dab, res=ab, hoff=nd)
    return dab


def _sigmul_bwd_x_fake(dy, out, sp):
    """(dout, dscale, the block partials of the column sums of dscale, dout^T padded to 32 rows)."""
    m = dy.numel() // dy.shape[-1]
    blocks = min(sigmul_bwd_blocks(dy), -(-m // 32))
    return torch.empty_like(dy), torch.empty_like(dy), dy.new_empty((blocks, dy.shape[-1])), dy.new_empty((dy.shape[-1], (m + 31) // 32 * 32))


@opaque(fake=_sigmul_bwd_x_fake, name="conditioned_transition_cuda_sigmul_bwd_x")
def sigmul_bwd_x(dy: torch.Tensor, out: torch.Tensor, sp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sigmul_bwd`` (fp32) plus dout^T [D, rows padded to 32] (the K-major left operand of dWs = dout^T h)."""
    dout, dsc, part, doutT = ext().sigmul_bwd_x(dy, out, sp, sigmul_bwd_blocks(dy))
    return dout, dsc, part, doutT


def _transpose_f32_fake(t):
    """t^T: [C, R], same dtype."""
    return t.new_empty((t.shape[1], t.shape[0]))


@opaque(fake=_transpose_f32_fake, name="conditioned_transition_cuda_transpose_f32")
def transpose_f32(t: torch.Tensor) -> torch.Tensor:
    """t^T [C, R] (fp32, contiguous), rounded to the TF32 grid to nearest. The TF32 weight-gradient GEMM dW = G^T X contracts over the ROWS of both
    operands, which makes both of them MN-major for cuBLAS (G^T X: 200 TFLOP/s at 18432 x 3072 x 768); with the SMALLER operand K-major (X^T copied once, 43 us
    for 56 MB) the GEMM runs at ~290 and the copy is paid back several times."""
    from miniworld_engine.kernels.adaln.cuda import _ext as rows_ext

    t = t.contiguous()
    o = torch.empty(t.shape[1], t.shape[0], device=t.device, dtype=t.dtype)
    rows_ext().pack_f32([t], [o], [[1, 0, 0, 1]])
    return o


#: fp32 weight gradients with thin outputs over many rows (the atom width: [128 .. 512] x [128 .. 256] outputs, K = the rows): cuBLAS picks sm_80
#: split-K kernels for G^T X; as a batched GEMM over WG_SPLIT row slices plus a sum over the slices it runs its sm_100 kernels (B200, 49152 rows,
#: kernels alone, us: dWs [128 x 256] 23.2 -> 16.1 (32 slices), [dWa; dWb] [512 x 128] 33.3 -> 28.6 (16), dWsc [128 x 128] 17.0 -> ~12). Up to
#: WG_SPLIT_MAX_ROWS rows (above it the split-K GEMMs fill the card).
WG_SPLIT_MAX_ROWS = int(os.environ.get("CT_WG_SPLIT_MAX_ROWS", "65536"))


def wgrad_f32(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """g^T x [P, Q] of fp32 g [M, P], x [M, Q] (TF32), deterministic: as row-slice batches summed in order where that is faster (see above)."""
    m, p = g.shape
    q = x.shape[1]
    s = 32 if p * q <= 32768 else 16
    if (g.dtype is torch.float32 and m <= WG_SPLIT_MAX_ROWS and m % s == 0 and m // s >= 256 and p <= 512 and q <= 512 and g.is_cuda
            and torch.cuda.get_device_capability(g.device) == (10, 0)):
        return torch.bmm(g.view(s, m // s, p).transpose(1, 2), x.view(s, m // s, q)).sum(0)
    return torch.mm(g.t(), x)


#: fp32 at a few row tiles (up to SIDE_MAX_ROWS rows): the GEMMs fill a fraction of the card, so the gate projection (forward) and the gate /
#: squeeze weight-gradient GEMMs (backward) run on a side stream beside the expand / dh / dx chain (a CUDA graph captures the fork and join as
#: parallel branches). Not inside a torch.compile trace (the eager graph regime is the deployed one for these kernels).
SIDE_MAX_ROWS = int(os.environ.get("CT_SIDE_MAX_ROWS", "12288"))
_SIDE: dict = {}


def _side(x: torch.Tensor):
    """The side stream for this call, or None."""
    if x.dtype is not torch.float32 or x.shape[0] > SIDE_MAX_ROWS or torch.compiler.is_compiling():
        return None
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    st = _SIDE.get(idx)
    if st is None:
        st = _SIDE[idx] = torch.cuda.Stream(device=idx)
    return st


_SIDE_PACK = os.environ.get("CT_SIDE_PACK", "1") != "0"


def _pack_fp32_split(wa, wb, ws, wsc, side):
    """``pack_fp32``'s outputs, the forward's [Wa; Wb] packed on the current stream and the three K-major copies only the backward reads on ``side``
    (joined by the forward's own join of that stream, before its gate pass). Eager / graph-capture only (``_side`` is None under a trace)."""
    from miniworld_engine.kernels.adaln.cuda import _ext as rows_ext

    nd, d = wa.shape
    dc = wsc.shape[1]
    dev, f32 = wa.device, torch.float32
    wa, wb, ws, wsc = (w.contiguous() for w in (wa, wb, ws, wsc))
    wab = torch.empty(2 * nd, d, device=dev, dtype=f32)
    wabT = torch.empty(d, 2 * nd, device=dev, dtype=f32)
    wsT = torch.empty(nd, d, device=dev, dtype=f32)
    wscT = torch.empty(dc, d, device=dev, dtype=f32)
    cur = torch.cuda.current_stream(dev)
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        rows_ext().pack_f32([wa, wb, ws, wsc], [wabT, wabT, wsT, wscT], [[1, 0, 0, 1], [1, 0, nd, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    rows_ext().pack_f32([wa, wb], [wab, wab], [[0, 0, 0, 1], [0, nd, 0, 1]])
    return wab, wabT, wsT, wscT


class _Tail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, res, xn_tf32=False):  # noqa: D102
        dt = x.dtype
        if dt is torch.bfloat16 and all(t.dtype is torch.float32 for t in (wa, wb, ws, wsc, bsc)):
            wab, wsb, wscb, bscb = cast_weights(wa, wb, ws, wsc, bsc)   # one launch for the five casts and the concatenation
            wabT = wsT = wscT = None
        elif dt is torch.float32 and all(t.dtype is torch.float32 for t in (wa, wb, ws, wsc)) and torch.backends.cuda.matmul.allow_tf32:
            side0 = _side(x) if _SIDE_PACK else None
            if side0 is not None:   # a few row tiles: [Wa; Wb] here, the backward's K-major copies on the side stream beside the forward GEMMs
                wab, wabT, wsT, wscT = _pack_fp32_split(wa, wb, ws, wsc, side0)
            else:
                wab, wabT, wsT, wscT = pack_fp32(wa, wb, ws, wsc)
            wsb, wscb, bscb = ws.contiguous(), wsc.contiguous(), bsc
        else:
            wab = torch.cat([wa, wb]).to(dt)
            wsb, wscb, bscb = ws.to(dt), wsc.to(dt), bsc.to(dt)
            wabT = wsT = wscT = None
        tc_f, tc_b = tc_route(x, wab.shape[0] // 2, xn_tf32) if wabT is not None else (False, False)
        ctx.tc_b = tc_b
        side = _side(x) if wabT is not None else None
        ctx.side = side is not None
        if side is not None:   # the gate projection beside the expand / squeeze chain
            cur = torch.cuda.current_stream(x.device)
            sp = torch.empty((x.shape[0], wscb.shape[0]), dtype=dt, device=x.device)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                torch.addmm(bscb, cond, wscb.t(), out=sp)
        if tc_f:
            h, ab = swab(x, wab)
        elif _swab_ok(x, wab):   # B200 bf16: the expand and the SwiGLU in one tcgen05 GEMM
            h, ab = swiglu_ab(x, wab)
        else:
            ab = torch.mm(x, wab.t())
            h = swiglu_fwd(ab)
        if dt is torch.bfloat16 and _fused_squeeze_ok(h, cond, wsb):   # one wgmma kernel for both GEMMs and the gate (+ residual)
            y, out, sp = squeeze_gate_save(h, cond, wsb, wscb, bsc.float().contiguous(), res)
        else:
            out = torch.mm(h, wsb.t())
            if side is not None:
                cur.wait_stream(side)
            else:
                sp = torch.addmm(bscb, cond, wscb.t())
            y = sigmul_fwd(out, sp, res)
        ctx.save_for_backward(x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc, *([] if wabT is None else [wabT, wsT, wscT]))
        ctx.has_res = res is not None
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc, *tw = ctx.saved_tensors
        dy = dy.contiguous()
        if x.dtype is torch.float32 and not tw:
            tw = [w.t().contiguous() for w in (wab, wsb, wscb)]
        if x.dtype is torch.float32 and _feature_major_ok(x, ws) and _FM:
            return _backward_feature_major(ctx, dy, x, cond, ab, h, out, sp, wab, wsb, wscb, tw)
        f32 = torch.float32
        o = lambda w: f32 if (w.dtype == f32 and x.dtype != f32) else None  # noqa: E731  fp32 master weights: fp32 gradients, unrounded
        xt = x.dtype is f32 and _XT and x.shape[0] % 4 == 0 and x.shape[1] >= 256   # token widths (measured at the atom width, 128: slower)
        if xt and dy.shape[-1] % 128 == 0 and dy.shape[-1] <= 1024:   # sigmul_bwd also writes dout^T (K-major left operand of dWs)
            dout, dsc, part, doutT = sigmul_bwd_x(dy, out, sp)
        else:
            dout, dsc, part = sigmul_bwd(dy, out, sp)
            doutT = transpose_f32(dout) if xt else None
        t = (lambda w: w) if x.dtype is not f32 else None   # TF32: the K-major copies packed by the forward (18432 rows: dh -21, dx -72, dcond -23 us)
        k = (lambda i, w: tw[i].t()) if x.dtype is f32 else (lambda i, w: w)
        side = _side(x) if ctx.side else None
        if side is not None:   # dWs, dWsc, dcond beside the dh / dx chain (outputs allocated here, on the backward's stream)
            cur = torch.cuda.current_stream(x.device)
            dws = x.new_empty((ws.shape[0], h.shape[1]))
            dwsc = x.new_empty((wsc.shape[0], cond.shape[1]))
            dcond = torch.empty_like(cond)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                torch.mm(doutT[:, :dout.shape[0]] if xt else dout.t(), h, out=dws)
                torch.mm(dsc.t(), cond, out=dwsc)
                torch.mm(dsc, k(2, wscb), out=dcond)
        elif xt:   # TF32 weight gradients: the smaller operand goes K-major (rows contiguous), the big gradient stays as it is (see transpose_f32)
            dws = torch.mm(doutT[:, :dout.shape[0]], h)   # left operand K-major, [D, ND] straight out (no transposing copy)
        else:
            dws = torch.mm(dout.t(), h, out_dtype=f32) if o(ws) else (wgrad_f32(dout, h) if x.dtype is f32 else torch.mm(dout.t(), h))
        if ctx.tc_b:   # dout from sigmul_bwd_x / sigmul_bwd (fp32): on the TF32 grid
            dab = gab(dout, tw[1], ab)
        elif _gate7_ok(dout, ab):   # B200 bf16: dh and the SwiGLU backward in one tcgen05 GEMM
            dab = gate_ab(dout, transpose_bf16(wsb), ab)
        else:
            dh = torch.mm(dout, k(1, wsb))
            dab = swiglu_bwd(dh, ab)
        if xt:   # (on the side stream too, beside dx, measured 5 - 10 us slower at 6144 rows)
            dwab = torch.mm(dab.t(), transpose_f32(x).t())
        else:
            dwab = torch.mm(dab.t(), x, out_dtype=f32) if o(wa) else (wgrad_f32(dab, x) if x.dtype is f32 else torch.mm(dab.t(), x))
        dx = torch.mm(dab, k(0, wab))
        if side is not None:
            cur.wait_stream(side)
        else:
            dwsc = torch.mm(dsc.t(), cond, out_dtype=f32) if o(wsc) else (wgrad_f32(dsc, cond) if x.dtype is f32 else torch.mm(dsc.t(), cond))
            dcond = torch.mm(dsc, k(2, wscb))
        dbsc = colsum(part).to(dsc.dtype if not o(wsc) else f32)   # the block partials of sum(dscale), summed in a fixed order
        nd = wab.shape[0] // 2
        return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None), None



def _feature_major_ok(x, ws) -> bool:
    """fp32 (TF32) tail with token-width gradients: the weight-gradient GEMMs run on the gradients written feature-major (tpose.cuh). Not the atom width
    (measured: no difference there)."""
    d, nd = ws.shape
    return d >= 256 and d % 128 == 0 and nd % 128 == 0 and d <= 1024


def _backward_feature_major(ctx, dy, x, cond, ab, h, out, sp, wab, wsb, wscb, tw):
    """``_Tail.backward`` for fp32 with dout / dscale / dab produced feature-major ([features, rows]): every weight gradient is a GEMM with the rows contiguous
    in its left operand (M = 18432: dWab 430 -> ~295 us, dWs 237 -> ~145, dWsc 78 -> ~61); the data-gradient GEMMs read the same buffers as column-major operands."""
    m = x.shape[0]
    doutT, dscT, part = ext().sigmul_bwd_t(dy, out, sp, sigmul_bwd_blocks(dy))
    doutT, dscT = doutT[:, :m], dscT[:, :m]
    dws = torch.mm(doutT, h)
    dh = torch.mm(doutT.t(), tw[1].t())
    dabT = ext().swiglu_bwd_t(dh, ab)[:, :m]
    dwab = torch.mm(dabT, x)
    dx = torch.mm(dabT.t(), tw[0].t())
    dwsc = torch.mm(dscT, cond)
    dcond = torch.mm(dscT.t(), tw[2].t())
    dbsc = colsum(part)
    nd = wab.shape[0] // 2
    return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None), None

# ---- D = 768, hidden 2 D, bf16, whole 128-row tiles: the Transition family's wgmma kernels (kernels/transition/cuda/wide), which keep a, b and h of the
# expand GEMM and fuse the SwiGLU backward into the dh GEMM. The tail's squeeze (h Ws^T) is their squeeze kernel with a zero residual (its rounding is the
# plain GEMM's: bf16(h Ws^T + 0)); the gate and the residual follow in ``sigmul_fwd``.
# Measured on the ladder shapes (A = 48, bf16): their expand_saved + squeeze 164 + 87 us against cuBLAS + swiglu_fwd + cuBLAS 124 + 61 + 55, their gate_saved
# 178 us against cuBLAS dh + swiglu_bwd 63 + 98: a net loss of ~10 us, so the row kernels around cuBLAS stay the route; the path is kept for a faster kernel.
_USE_WIDE = False
_ZEROS: dict = {}


def _zeros_like_rows(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, tuple(x.shape))
    z = _ZEROS.get(key)
    if z is None:
        z = _ZEROS[key] = torch.zeros_like(x)
    return z


def _wide_expand_saved_fake(xn, zeros, w1p, ws):
    """(out, h, a, b) shapes."""
    m, d = xn.shape
    h = ws.shape[1]
    return xn.new_empty((m, d)), xn.new_empty((m, h)), xn.new_empty((m, h)), xn.new_empty((m, h))


@opaque(fake=_wide_expand_saved_fake, name="conditioned_transition_cuda_wide_expand_saved")
def _wide_expand_saved(xn: torch.Tensor, zeros: torch.Tensor, w1p: torch.Tensor, ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h Ws^T, h, a, b) with h = silu(a) b, [a | b] = xn [Wa; Wb]^T (the Transition family's expand_saved + squeeze, zero residual)."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw

    out, h, a, b = fw._ext_for(xn, True, 2).fwd_split(xn, zeros, w1p, ws)
    return out, h, a, b


def _wide_gate_saved_fake(dy, wst, a, b):
    """[dA | dB] [M, 2 H]."""
    return dy.new_empty((a.shape[0], 2 * a.shape[1]))


@opaque(fake=_wide_gate_saved_fake, name="conditioned_transition_cuda_wide_gate_saved")
def _wide_gate_saved(dy: torch.Tensor, wst: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[dA | dB] = SwiGLU backward of dh = dy Ws from the saved a, b (one kernel; dh never reaches HBM)."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw

    return fw._ext_for(dy, True, 2).gate_saved(dy, wst, a, b)


def wide_supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    if x.shape[-1] != 768 or wa.shape[0] != 1536 or x.dtype is not torch.bfloat16:
        return False
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw

    return fw.available(x, wa.to(torch.bfloat16) if wa.dtype != torch.bfloat16 else wa, ws)


class _TailWide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, res):  # noqa: D102
        from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw
        from miniworld_engine.kernels.transition.cuda.fused_sm90a import _cuda_prepare

        dt = x.dtype
        wa_, wb_, ws_, wst = _cuda_prepare(wa, wb, ws, True)
        w1p = fw._pack(wa_, wb_, 128)
        out, h, a, b = _wide_expand_saved(x, _zeros_like_rows(x), w1p, ws_)
        wscb, bscb = wsc.to(dt), bsc.to(dt)
        sp = torch.addmm(bscb, cond, wscb.t())
        y = sigmul_fwd(out, sp, res)
        ctx.save_for_backward(x, cond, a, b, h, out, sp, wa_, wb_, wst, wscb, wa, ws, wsc)
        ctx.has_res = res is not None
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x, cond, a, b, h, out, sp, wa_, wb_, wst, wscb, wa, ws, wsc = ctx.saved_tensors
        dy = dy.contiguous()
        dout, dsc, part = sigmul_bwd(dy, out, sp)
        f32 = torch.float32
        o = lambda w: w.dtype == f32  # noqa: E731  fp32 master weights: fp32 gradients, unrounded
        dws = torch.mm(dout.t(), h, out_dtype=f32) if o(ws) else torch.mm(dout.t(), h)
        dab = _wide_gate_saved(dout, wst, a, b)
        dwab = torch.mm(dab.t(), x, out_dtype=f32) if o(wa) else torch.mm(dab.t(), x)
        dx = torch.mm(dab, torch.cat([wa_, wb_]))
        dwsc = torch.mm(dsc.t(), cond, out_dtype=f32) if o(wsc) else torch.mm(dsc.t(), cond)
        dcond = torch.mm(dsc, wscb)
        dbsc = colsum(part).to(dsc.dtype if not o(wsc) else f32)
        nd = a.shape[1]
        return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None)


def ct_tail_train(x, cond, wa, wb, ws, wsc, bsc, residual=None, *, xn_tf32: bool = False):
    """``residual + sigmoid(cond Wsc^T + b) * (silu(x Wa^T) * (x Wb^T)) Ws^T`` with a backward; ``x`` (M, D), ``cond`` (M, DC) of one dtype, the weights in
    any dtype (rounded to the activations' dtype per call; gradients in the weights' dtype). bf16 D = 768 / hidden 1536 on whole 128-row tiles runs the
    Transition family's wgmma expand / dh kernels, everything else the row kernels around cuBLAS."""
    x, cond = x.contiguous(), cond.contiguous()
    if _USE_WIDE and x.shape[0] % 128 == 0 and torch.cuda.get_device_capability(x.device) == (9, 0) and wide_supported(x, wa, ws):
        return _TailWide.apply(x, cond, wa, wb, ws, wsc, bsc, residual)
    return _Tail.apply(x, cond, wa, wb, ws, wsc, bsc, residual, xn_tf32)


__all__ = ["ct_tail_train", "ext", "supported"]
