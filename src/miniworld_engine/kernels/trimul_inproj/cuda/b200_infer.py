"""B200 (sm_100a) TriMul inference for every width D = 64 / 128 / 256 / 384 / 512, bidirectional or one direction, bf16, B=1.

Front (every D): ``k1w`` -- input LN + gated projections + mask -> channel-major planes (tcgen05; D <= 128 computes the LN
statistics in-kernel, D >= 256 takes them from ``k1w_stats``). Contractions: cuBLAS. Output:
  D <= 128  ``k3g`` -- output LN + output projection + gate + residual in one kernel (the D128 K3 generalised over (D, H)).
  D >= 256  ``k3w`` -- both output GEMMs on the raw operands (t as stored, channel-major; x) with the two LayerNorms folded into
            the weights (``wide_fold_prep``) and undone per token in the epilogue from ``wide_ln_stats`` / ``k1w_stats``;
            gate, dropout and residual in the same epilogue. Nothing between the contraction and y touches HBM but t.
Sources: ``b200_sources/{k1w,k3g,k3w,wide_aux}.cu``. Every call owns its buffers.
"""

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).with_name("b200_sources")
WIDTHS = (64, 128, 256, 384, 512)
EPS = 1e-5


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_v6",
        sources=[str(_SRC / f) for f in ("k1w.cu", "k3g.cu", "k3w.cu", "wide_aux.cu", "k1wb.cu", "wide_bwd.cu", "b1s.cu", "b7m.cu", "b1g.cu", "b7g.cu",
                                         "bind_b200.cpp")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--expt-relaxed-constexpr",
                           "-DNDEBUG", f"-I{_SRC}"],
        extra_cflags=["-std=c++17", f"-I{_SRC}"], verbose=False)


def supports(width: int, length: int, dropout: bool = False, hidden: int | None = None, direction: int = 0) -> bool:
    """k1w / k3g / k3w tile 128 tokens and ln_stats 256, so L is a multiple of 16; a dropout scale on the fused D <= 128 output
    goes through k3g's training epilogue, which groups tiles by column (L a multiple of 128). The hidden (contraction) width is
    the pair width, or -- one direction, D64 / D128 -- twice it (see ``hidden_ok``)."""
    if width not in WIDTHS or length <= 0 or length % 16:
        return False
    if not hidden_ok(width, width if hidden is None else hidden, direction):
        return False
    return not (dropout and width <= 128 and length % 128)


def hidden_ok(width: int, hidden: int, direction: int) -> bool:
    """hidden == width, or hidden == 2 width in one direction at D64 / D128 (Protenix's template TriMul: pair 64, hidden 128).
    The latter has, per token, exactly the bidirectional kernels' shapes -- 4 D planes out of k1w, an LN_out over 2 D channels
    into k3g, the same b1s / b1g / b7m / b7g backward -- only the cuBLAS contraction between them pairs the planes differently
    (all 2 D channels in one direction instead of D per direction), so the D64 / D128 kernels serve it unchanged."""
    return hidden == width or (direction != 0 and width <= 128 and hidden == 2 * width)


def planes_of(width: int, hidden: int, direction: int) -> int:
    """k1w's channel-major plane count: [a_out | a_in | b_out | b_in] (D each) bidirectional, [a | b] (hidden each) otherwise."""
    return 4 * width if direction == 0 else 2 * hidden


def contracted(width: int, hidden: int, direction: int) -> int:
    """Channels per contraction product: D per direction bidirectional, the hidden width in one direction."""
    return width if direction == 0 else hidden


def _front(E, wl, wlg, wr, wrg, wp=None, wpp=None):
    """k1w reads the four front matrices in place when their columns are contiguous; otherwise (the D128 bidirectional module
    stores them column-major) ``k1w_prep`` writes row-major copies, in the same launch as k3g's ``wpp`` when that is asked for."""
    ws = (wl, wlg, wr, wrg)
    wc = None if all(w.stride(1) == 1 and w.stride(0) % 8 == 0 for w in ws) else wl.new_empty((4, *wl.shape))
    if wc is not None or wp is not None:
        E.k1w_prep(wl, wlg, wr, wrg, wc, wp, wpp)
    return ws if wc is None else tuple(wc.unbind(0))


def _fold(x):
    """[c, (B,) n, n] -> [c * B, n, n]: the samples join the channel planes in cuBLAS's batch dim (a view, the planes are contiguous)."""
    n = x.shape[-1]
    return x.view(-1, n, n)


def _contract(planes, d, direction):
    """Planes [P, n, n] or [P, B, n, n] (batch b-major per plane) -> t [(2 d or d), (B,) n, n], per sample the same products as B = 1.

    The ``out=`` buffer is the 3-D folded tensor itself and the caller's shape is a view of it: ``bmm(out=<3-D view of a 4-D buffer>)``
    makes functionalization re-apply the view with the 3-D strides to the 4-D base (``as_strided`` "mismatch in length of strides and
    shape") whenever this body is traced."""
    n = planes.shape[-1]
    channels = (2 if direction == 0 else 1) * d
    samples = planes.numel() // (planes.shape[0] * n * n)
    t = planes.new_empty((channels * samples, n, n))
    pl = lambda a, b: _fold(planes[a:b])
    if direction == 0:
        torch.bmm(pl(0, d), pl(2 * d, 3 * d).transpose(1, 2), out=t[:d * samples])
        torch.bmm(pl(d, 2 * d).transpose(1, 2), pl(3 * d, 4 * d), out=t[d * samples:])
    elif direction == 1:
        torch.bmm(pl(0, d), pl(d, 2 * d).transpose(1, 2), out=t)
    else:
        torch.bmm(pl(0, d).transpose(1, 2), pl(d, 2 * d), out=t)
    return t.view(channels, *planes.shape[1:])


def _inference_fake(leaves, mask, ds, direction):
    """y only."""
    return torch.empty_like(leaves[0])


@opaque(fake=_inference_fake, name="trimul_b200_inference")
def inference(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int) -> torch.Tensor:
    """leaves = x, wl, wlg, wr, wrg, wg, wp (bf16, [out, in]; the front four may be strided), gi, bi, go, bo (fp32);
    mask [B * L] bool token mask or None (k1w forms the pair mask); ds [B * L, D] bf16 row-dropout scale or None;
    direction 0 = bidirectional, 1 = outgoing, 2 = incoming. x is [B, L, L, D]: B > 1 only for D <= 128 (the kernels take b-major
    tokens, M = B L L), D >= 256 is B = 1. The hidden width is wl's row count (``hidden_ok``)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        hid = wl.shape[0]
        p = planes_of(d, hid, direction)
        planes = x.new_empty((p, bsz, n, n) if d <= 128 else (p, n, n))
        y = torch.empty_like(x)
        if d <= 128:
            wpp = x.new_empty((d, p // 2))     # k3g's output projection, columns in its TMEM K order
            front = _front(E, wl, wlg, wr, wrg, wp, wpp)
            E.k1w_forward(x2, *front, mask, None, None, gi, bi, planes, 0, EPS)
            t = _contract(planes, contracted(d, hid, direction), direction)
            del planes
            if ds is None:
                E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 0, None, None, None, None, 0)
            else:   # the inference K3 has no dropout: the saving epilogue into throwaway buffers
                mo = x.new_empty((m,), dtype=torch.float32)
                E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 1, ds, torch.empty_like(x2), mo,
                              torch.empty_like(mo), 0)
            return y
        mean = x.new_empty((m,), dtype=torch.float32)
        rstd = torch.empty_like(mean)
        E.k1w_stats(x2, mean, rstd, EPS)
        E.k1w_forward(x2, *_front(E, wl, wlg, wr, wrg), mask, mean, rstd, gi, bi, planes, 0, EPS)
        t = _contract(planes, d, direction)
        del planes
        mo = torch.empty_like(mean)
        ro = torch.empty_like(mean)
        E.wide_ln_stats(t, mo, ro, EPS)
        wpq = torch.empty_like(wp)
        wgq = torch.empty_like(wg)
        vec = x.new_empty((4, d), dtype=torch.float32)
        E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
        E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mo, ro, mean, rstd, ds, y.view(m, d), n, 0, None, None)
    return y
