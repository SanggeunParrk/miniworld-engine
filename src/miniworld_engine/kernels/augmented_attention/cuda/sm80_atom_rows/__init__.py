"""A100 (sm_80) row stages of the AF3 local atom DiT block: the conditioning projections, the attention input stage and the attention output + conditioned transition stage,
hand-written CUDA (``mma.sync`` / ``ldmatrix`` / ``cp.async``). The window attention itself is ``kernels/augmented_attention/cuda/sm80_atom_local``. bf16, C = 128, SwiGLU hidden 256.

    cond_fwd  c -> mod [M, 1024]  = [ s1 | bi1 | skv | bikv | s2 | bi2 | so | st ]  (the eight modulation / gate tensors of the block, scales and gates already as rn(sigmoid))
    pre_fwd   x, mod -> Q, K, V, SG (+ x1, xkv)
    post_fwd  x, O, SG, mod -> out (+ u, a2, x2, t)

Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent
C = 128


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="af3_local_rows_sm80",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{_dir}",
            "-Xcompiler=-fno-gnu-unique",
        ],
        extra_cflags=["-std=c++17", "-O3", "-fno-gnu-unique"],
        verbose=False,
    )


def ready() -> None:
    _ext()


def cond_cfg(m: int) -> int:
    """Configuration of the conditioning kernel (see ops.cu): 16 rows a warp (MT = 1), 16 / 12 warps a SM."""
    return 5 if m >= 65536 else 7 if m >= 8192 else 6


def cond_fwd(c, w, bias, gamma, eps=1e-5, cross=True, cfg=None):
    """c [M, 128] bf16 -> mod [M, 1024] bf16. ``w`` [1024, 128] bf16 = Wsc1 | Wbi1 | Wsckv | Wbikv | Wsc2 | Wbi2 | Wos | Wts, ``bias`` [1024] fp32 (the scale / gate biases, zeros on the shifts),
    ``gamma`` [3, 128] fp32 (the three cond LayerNorm weights: attention AdaLN, key / value AdaLN, transition AdaLN)."""
    m = c.shape[0]
    mod = torch.empty(m, 1024, device=c.device, dtype=torch.bfloat16)
    _ext().local_cond_fwd(
        c, w, bias, gamma, mod, eps, cross, cond_cfg(m) if cfg is None else cfg
    )
    return mod


def pre_fwd(x, mod, w, bq, eps=1e-5, save=False, cross=True, cfg=2, prefetch=False):
    """x [M, 128], mod -> (Q, K, V, SG) and, with ``save``, also (x1, xkv); ``w`` [512, 128] = Wq | Wg | Wk | Wv, ``bq`` [128] fp32."""
    m = x.shape[0]
    q, k, v, sg = (torch.empty_like(x) for _ in range(4))
    x1 = torch.empty_like(x) if save else x.new_empty(0)
    xkv = torch.empty_like(x) if save and cross else x.new_empty(0)
    _ext().local_pre_fwd(
        x, mod, w, bq, q, k, v, sg, x1, xkv, eps, cross, save, prefetch, cfg
    )
    return (q, k, v, sg, x1, xkv if cross else x1) if save else (q, k, v, sg)


def post_fwd(x, o, sg, mod, wo, wu, wd, eps=1e-5, save=False, cfg=1, prefetch=False):
    """-> out [M, 128] and, with ``save``, also (u, a2, x2, t); ``wo`` [128, 128], ``wu`` [512, 128] = Wa | Wb, ``wd`` [128, 256]."""
    out = torch.empty_like(x)
    saved = [torch.empty_like(x) for _ in range(4)] if save else [x.new_empty(0)] * 4
    _ext().local_post_fwd(
        x, o, sg, mod, wo, wu, wd, out, *saved, eps, save, prefetch, cfg
    )
    return (out, *saved) if save else out


def tr_bwd(dy, t, x2, mod, wu, wst, dmod, dbts, cfg=0, prefetch=False):
    """Transition backward, first part: -> (DT [M, 128], HH [M, 256], DAB [M, 512] = da | db); writes dts into ``dmod`` columns 896 .. and accumulates d bts into ``dbts`` [128] fp32."""
    m = dy.shape[0]
    dt = torch.empty_like(dy)
    hh = torch.empty(m, 256, device=dy.device, dtype=dy.dtype)
    dab = torch.empty(m, 512, device=dy.device, dtype=dy.dtype)
    _ext().local_tr_bwd(dy, t, x2, mod, wu, wst, dt, hh, dab, dmod, dbts, prefetch, cfg)
    return dt, hh, dab


def post_bwd(
    dab,
    dy,
    a2,
    mod,
    u,
    sg,
    o,
    wut,
    wot,
    dp,
    dmod,
    dbias,
    n,
    eps=1e-5,
    cfg=0,
    prefetch=False,
):
    """Post-attention backward: -> (DA, DU, GATED, DO, D [A, 4, N] fp32); writes dg into ``dp`` columns 128 .. 255, dsc2 / dbi2 / dos into ``dmod``, accumulates d bsc2 / d bos into ``dbias``."""
    m = dy.shape[0]
    da, du, gated, do = (torch.empty_like(dy) for _ in range(4))
    dd = torch.empty(m // n, 4, n, device=dy.device, dtype=torch.float32)
    _ext().local_post_bwd(
        dab,
        dy,
        a2,
        mod,
        u,
        sg,
        o,
        wut,
        wot,
        da,
        du,
        gated,
        do,
        dp,
        dmod,
        dd,
        dbias,
        n,
        eps,
        prefetch,
        cfg,
    )
    return da, du, gated, do, dd


def pre_bwd(
    dp,
    x,
    x1,
    da,
    mod,
    wqgt,
    wkvt,
    dmod,
    dbias,
    dbq,
    eps=1e-5,
    cross=True,
    cfg=0,
    prefetch=False,
):
    """Attention-input backward: -> d single [M, 128]; writes dsc1 / dbi1 / dsckv / dbikv into ``dmod``, accumulates d bsc1 / d bskv into ``dbias`` and d bq into ``dbq``."""
    ds = torch.empty_like(x)
    _ext().local_pre_bwd(
        dp,
        x,
        x1 if cross else x,
        da,
        mod,
        wqgt,
        wkvt,
        ds,
        dmod,
        dbias,
        dbq,
        cross,
        eps,
        prefetch,
        cfg,
    )
    return ds


def cond_bwd(dmod, c, wt, gamma, dgamma, eps=1e-5, cross=True, cfg=0, prefetch=False):
    """Conditioning backward: -> (dc [M, 128], cn [3, M, 128]); accumulates the LayerNorm weight gradients into ``dgamma`` [3, 128] fp32."""
    m = c.shape[0]
    dc = torch.empty_like(c)
    cn = torch.empty(3, m, 128, device=c.device, dtype=c.dtype)
    _ext().local_cond_bwd(dmod, c, wt, gamma, dc, cn, dgamma, cross, eps, prefetch, cfg)
    return dc, cn


def wgrad(problems, segments, m, splits=None):
    """The grouped weight gradients. ``problems``: [(A, B)] bf16 [M, Ka] / [M, Kb] views (Ka, Kb multiples of 128; A, B may be column slices of a wider buffer); problem i gives D_i = A_i^T B_i
    [Ka, Kb] fp32. ``segments``: [(problem index or None, src, row0, rows, dst)]: with a problem index, the rows [row0, row0 + rows) of D_i (summed over the splits) rounded into ``dst``
    (contiguous, bf16 or fp32, rows x Kb elements); with None, ``src`` (an fp32 tensor, plain copy) rounded into ``dst``."""
    ka = [
        a.shape[1] if a.stride(1) == 1 and a.shape[1] % 128 == 0 else 0
        for a, _ in problems
    ]
    kb = [b.shape[1] for _, b in problems]
    slab = sum(x * y for x, y in zip(ka, kb, strict=True))
    tiles = sum(x // 128 * (y // 128) for x, y in zip(ka, kb, strict=True))
    if splits is None:
        splits = max(1, min(32, round(3 * 108 / tiles), m // 8192)) if m >= 8192 else 1
    dev = problems[0][0].device
    part = torch.empty(splits * slab, device=dev, dtype=torch.float32)
    offs, o = [], 0
    for x, y in zip(ka, kb, strict=True):
        offs.append(o)
        o += x * y
    srcs, soff, sn, sns, dsts = [], [], [], [], []
    for pi, src, row0, rows, dst in segments:
        if pi is None:
            srcs.append(src)
            soff.append(0)
            sn.append(dst.numel())
            sns.append(1)
        else:
            srcs.append(part)
            soff.append(offs[pi] + row0 * kb[pi])
            sn.append(rows * kb[pi])
            sns.append(splits)
        dsts.append(dst)
    _ext().local_wgrad(
        [a for a, _ in problems],
        [b for _, b in problems],
        ka,
        kb,
        m,
        splits,
        part,
        srcs,
        soff,
        sn,
        sns,
        dsts,
    )

