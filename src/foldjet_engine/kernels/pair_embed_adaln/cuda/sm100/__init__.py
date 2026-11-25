"""B200 (sm_100a) build and launch of the pair-embedding + adaptive-LayerNorm kernels (``pair_embed_adaln_sm100.cu``, 2026-11-16): the A100
source ``sm80/pair_embed_adaln_sm80.cu`` (``sm90/pair_embed_adaln.cu`` compiled unchanged with the mma.sync forms of its wgmma helpers: legal
on sm_100a) built for sm_100a, plus the backward with its table gradient on the B200 tensor cores (``backward_tc``: pair_embed_geom's
``pe_tbl_sm100.cuh``, tcgen05). The forward launcher is the H100 one bound to this build. Built on first use, never at import."""

from __future__ import annotations

import functools
import types
from pathlib import Path

import torch

from foldjet_engine.kernels.pair_embed_adaln.cuda import sm80 as _a100
from foldjet_engine.kernels.pair_embed_adaln.cuda import sm90 as _h100

_dir = Path(__file__).parent
NBLK_TC = 74  # pair slices of the tcgen05 table gradient per chunk (pe100::NBLK)
TC = True  # the table gradient on tcgen05 (False: the A100 build's own backward, impl _h100.BWD_IMPL; A/B)
#: at most this many pairs (B n n) the A100 build's backward instead (measured L128 x S1: 190.0 vs 194.8 us; from L256 / S5 on tcgen05 is
#: 1.03-1.11x faster)
SMALL_PAIRS = 128 * 128


@functools.lru_cache(maxsize=None)
def ext():
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="pair_embed_adaln_sm100",
        sources=[str(_dir / "pair_embed_adaln_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("100a"), f"-I{_a100._dir}"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


_G = {**vars(_h100), "ext": ext}


def _on_this_build(fn):
    return types.FunctionType(fn.__code__, _G, fn.__name__, fn.__defaults__, fn.__closure__)


forward = _on_this_build(_h100.forward)
_backward_a100 = _on_this_build(_h100.backward)


def backward(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2):
    """(dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]) from d pair = dout [B, n, n, 256] bf16 (the H100 launcher's scratch,
    the table gradient's partials per tcgen05 pair slice)."""
    if not TC or bins.shape[0] * bins.shape[1] * bins.shape[2] <= SMALL_PAIRS:
        return _backward_a100(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2)
    C = _h100.C
    B, n = bins.shape[:2]
    NT = tbl.shape[0]
    dev = tbl.device
    chunk_b = max(1, min(B, _h100.DX_CHUNK_BYTES // (n * n * C * 2)))
    nchunk = -(-B // chunk_b)
    nrb = -(-n // 8)
    dx = torch.empty(chunk_b, n, n, C, device=dev, dtype=torch.bfloat16)
    part = torch.empty(B * nrb * 4 * C, device=dev, dtype=torch.float32)
    tpart = torch.empty(nchunk * NBLK_TC * NT * C, device=dev, dtype=torch.float32)
    dtbl = torch.empty(NT, C, device=dev, dtype=torch.float32)
    dgamma = torch.empty(B, C, device=dev, dtype=torch.float32)
    dbeta = torch.empty(B, C, device=dev, dtype=torch.float32)
    dlw = torch.empty(C, device=dev, dtype=torch.float32)
    dlb = torch.empty(C, device=dev, dtype=torch.float32)
    ext().backward_tc(
        tbl,
        lnw,
        lnb,
        gamma,
        bins,
        dout,
        dx,
        part,
        tpart,
        dtbl,
        dgamma,
        dbeta,
        dlw,
        dlb,
        int(chunk_b),
        int(nS),
        int(nX),
        float(eps1),
        float(eps2),
    )
    return dtbl, dlw, dlb, dgamma, dbeta


# ---- the B200 kernels (pe_adaln_sm100.cuh, 2026-11-17, round 3) ----------------------------------------------------------------------------
#: forward: lanes a pair (8 / 16 / 32) and CTAs of 128 threads per SM at most (persistent warps)
FWD_G, FWD_CPS = 16, 4
#: backward rows: lanes a pair (16 / 32) and CTAs per SM (all samples of a launch share the card)
BWD_G, BWD_CPS = 16, 2
NR4 = 192  # most table rows of the mma.sync table gradient (pe_bwd_tbl4)
#: at most this many pairs the mma.sync table gradient under the B200 row backward (0: tcgen05 at every size)
BWD_SMALL_PAIRS = 0


def _chunk(pairs: int, cps: int, device) -> int:
    """Pairs a warp chunk: the largest of 32 / 16 / 8 that still gives every warp slot of the launch (cps CTAs of 4 warps per SM) a chunk."""
    slots = torch.cuda.get_device_properties(device).multi_processor_count * cps * 4
    for c in (32, 16):
        if pairs >= slots * c:
            return c
    return 8


def forward_b200(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2, train=True, g=None, cps=None, ch=None):
    """(pair [B, n, n, 256] bf16, bins [B, n, n] int32 -- empty unless ``train``): one launch from the fp32 table [NT, 256], fp32 / bf16
    ``gamma`` / ``beta`` and int32 / int64 ``idx`` as they are (the H100 kernel's bits)."""
    B, n = idx.shape
    out = torch.empty(B, n, n, _h100.C, device=tbl.device, dtype=torch.bfloat16)
    bins = torch.empty((B, n, n) if train else (0,), device=tbl.device, dtype=torch.int32)
    ext().forward_b200(
        tbl, lnw, lnb, gamma, beta, xt, sc, int(has_sc), idx, mask, xlim, clim, out, bins, int(nS), int(nX), float(eps1), float(eps2),
        int(g or FWD_G), int(cps or FWD_CPS), int(ch or _chunk(B * n * n, cps or FWD_CPS, tbl.device)),
    )  # fmt: skip
    return out, bins


def backward_b200(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2, g=None, cps=None, tc=None, ch=None):
    """(dW [256, NT] fp32 -- the weight's layout --, dln_w, dln_b [256] fp32, dgamma, dbeta [B, 256] in ``gamma``'s dtype) from d pair =
    ``dout`` [B, n, n, 256] bf16. The rows kernel gives every sample ``cps`` CTAs (fixed by the sizes and the card: run-to-run bitwise); the
    table gradient as in :func:`backward` (tcgen05 above :data:`SMALL_PAIRS` pairs)."""
    C = _h100.C
    B, n = bins.shape[:2]
    NT = tbl.shape[0]
    dev = tbl.device
    chunk_b = max(1, min(B, _h100.DX_CHUNK_BYTES // (n * n * C * 2)))
    nchunk = -(-B // chunk_b)
    if tc is None:
        tc = TC and (B * n * n > BWD_SMALL_PAIRS or NT > NR4)
    if ch is None:
        ch = _chunk(chunk_b * n * n, BWD_CPS, dev)
    if cps is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        cps = max(1, min(-(-(n * n) // (4 * ch)), (sms * BWD_CPS) // chunk_b))
    dx = torch.empty(chunk_b, n, n, C, device=dev, dtype=torch.bfloat16)
    part = torch.empty(B * cps * 4 * C, device=dev, dtype=torch.float32)
    tpart = torch.empty(nchunk * (NBLK_TC if tc else _h100.NBLK) * NT * C, device=dev, dtype=torch.float32)
    dw = torch.empty(C, NT, device=dev, dtype=torch.float32)
    dgamma = torch.empty(B, C, device=dev, dtype=gamma.dtype)
    dbeta = torch.empty(B, C, device=dev, dtype=gamma.dtype)
    dlw = torch.empty(C, device=dev, dtype=torch.float32)
    dlb = torch.empty(C, device=dev, dtype=torch.float32)
    ext().backward_b200(
        tbl, lnw, lnb, gamma, bins, dout, dx, part, tpart, dw, dgamma, dbeta, dlw, dlb, int(chunk_b), int(nS), int(nX), float(eps1),
        float(eps2), int(g or BWD_G), int(cps), int(ch), int(bool(tc)),
    )  # fmt: skip
    return dw, dlw, dlb, dgamma, dbeta


#: inference from the LayerNorm statistics tables (pe_adaln_sm100.cuh fwd_s): lanes a pair (8 / 16), CTAs per SM. CPS 2 (2026-11-17,
#: round 3 restart; measurer timing, S1 L128 / 256 / 384 fp32 eval): 16.3 / 28.6 / 49.1 us vs 18.3 / 34.7 / 51.1 at CPS 4 (a warp then has two
#: chunks, so the next chunk's inputs and Gram entries are in flight); CPS 1 / 3 slower
FWS_G, FWS_CPS = 8, 2
STATS = True  # inference by fwd_s (False: the training forward without the codes; A/B)


def stats_tables(weight, lnw, lnb):
    """(U [NT, 256], GG [NT, NT, 2], V [NT], BT [256], BB [1]) fp32: the inference tables of a contiguous fp32 weight [256, NT] and affine
    (two launches), made once per version of the three parameters (a capture reads the entry its warm-up made)."""

    def build():
        nt = weight.shape[1]
        dev = weight.device

        def f(*s):
            return torch.empty(s, device=dev, dtype=torch.float32)

        u, tc, gg, v, bt, bb = f(nt, 256), f(nt, 256), f(nt, nt, 2), f(nt), f(256), f(1)
        ext().stats_b200(weight.detach(), lnw.detach(), lnb.detach(), u, tc, gg, v, bt, bb)
        return (u, gg, v, bt, bb)

    from foldjet_engine.kernels import _capture

    return _capture.param_forms("pair_embed_adaln_stats_b200", (weight, lnw, lnb), build, capture_reads_eager=True)


def forward_s_b200(tables, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2, g=None, cps=None, ch=None):
    """Inference: the pair [B, n, n, 256] bf16 from :func:`stats_tables`: one launch, no reduction over the channels."""
    B, n = idx.shape
    out = torch.empty(B, n, n, _h100.C, device=xt.device, dtype=torch.bfloat16)
    cps = int(cps or FWS_CPS)
    ext().forward_s_b200(
        *tables, gamma, beta, xt, sc, int(has_sc), idx, mask, xlim, clim, out, int(nS), int(nX), float(eps1), float(eps2), int(g or FWS_G), cps,
        int(ch or _chunk(B * n * n, cps, xt.device)),
    )  # fmt: skip
    return out
