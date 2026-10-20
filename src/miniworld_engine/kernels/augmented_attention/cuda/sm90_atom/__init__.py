"""sm_90a (H100) row kernels of the AF3 atom block on warp-level tensor-core MMAs (mma.sync), bf16 and TF32.

  cond90.cu  c -> mod [M, 768] = [s1 | bi1 | so | s2 | bi2 | st] and mkv [M, 256] = [pre | shift]
  pre90.cu   AdaLN 1, q | gate, the K / V AdaLN, k | v
  post90.cu  gated out-projection, residual, AdaLN 2, SwiGLU, squeeze, residual

``-DTF32=1`` builds the fp32 twins (fp32 activations, operands rounded to nearest tf32, fp32 accumulation and elementwise math). The
weight packs (``pack``) fold the LayerNorm weights into the conditioning GEMMs and lay the transition weights out per hidden chunk.
"""

from __future__ import annotations

import ctypes
import functools
import re
from pathlib import Path

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture

_dir = Path(__file__).parent
DS = 128
NWARP = 4  # warps per CTA (16 rows each): 4 = 64 rows, one warpgroup, two CTAs per SM
ROWS = NWARP * 16


@functools.lru_cache(maxsize=None)
def cubin(
    stem: str, tf32: bool, nwarp: int = NWARP, minb: int = 1, defs: tuple = ()
) -> str:
    from miniworld_engine.kernels import _nvcc

    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        f"-DTF32={int(tf32)}",
        f"-DNWARP={nwarp}",
        f"-DMINB={minb}",
        *defs,
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
    )
    return str(_nvcc.cubin(f"atom_sm90/{stem}_{'tf32' if tf32 else 'bf16'}_w{nwarp}m{minb}", flags,
                           source=_dir / f"{stem}.cu"))


def _grp_bytes(nw: int, tf32: bool) -> int:
    """Shared memory of the grouped-sample slabs (grp_put / grp_flush in row90.cuh): buffers x [nw warps][16 rows][128 + pad] elements."""
    es = 4 if tf32 else 2
    return (1 if tf32 else 2) * nw * 16 * (128 + (4 if tf32 else 8)) * es


def _smem(kind: str, tf32: bool) -> int:
    es = 4 if tf32 else 2
    if kind in ("finalize", "pack"):
        return 0
    if kind in ("cond", "cond_split"):
        return 2 * 64 * 128 * es + 1024
    if kind in ("pre", "pre_split", "cond_bwd", "cond_bwd_s", "cond_bwd_sh"):
        return 2 * 64 * 128 * es + 1024
    grp = kind.endswith(
        "_g"
    )  # the grouped-sample build of a backward kernel: + the staging slabs
    kind = kind.removesuffix("_g").removesuffix("_s")
    if kind in ("post_bwd", "pre_bwd"):
        return (
            2 * 64 * 128 * es
            + 1024
            + 3 * NWARP * 128 * 4
            + (_grp_bytes(NWARP, tf32) if grp else 0)
        )
    if kind == "tr_bwd":
        return (
            2 * 3 * 128 * 128
            + 1024
            + nwarp(kind, tf32) * 128 * 4
            + (_grp_bytes(nwarp(kind, tf32), tf32) if grp else 0)
        )
    return 2 * (2 * (128 // es) * 128 * es + 128 * 128) + 1024


#: warps per CTA of a kernel (default NWARP): the fp32 transition backward measured faster with two warpgroups (565 -> 492 us)
#: CTAs per SM the register allocation targets (bf16 kernels whose shared memory allows three)
_MINB = {
    (k, False): 3 for k in ("cond", "cond_bwd", "cond_bwd_s")
}  # post_bwd / pre_bwd spill: 417 -> 496, 330 -> 567 us
_NWARP = {("tr_bwd", True): 8}


def nwarp(kind: str, tf32: bool) -> int:
    return _NWARP.get((kind, tf32), NWARP)


_KERNELS = {
    "finalize": ("finalize90", "finalize90"),
    "pack": ("pack90", "pack90"),
    "cond": ("cond90", "cond90_fwd"),
    "cond_split": ("cond90", "cond90_split"),
    "pre": ("pre90", "pre90_fwd"),
    "pre_split": ("pre90", "pre90_split"),
    "post": ("post90", "post90_fwd"),
    "tr_bwd": ("tr_bwd90", "tr_bwd90"),
    "post_bwd": ("post_bwd90", "post_bwd90"),
    "post_bwd_s": ("post_bwd90", "post_bwd90", ("-DSPLITK=1",)),
    "pre_bwd": ("pre_bwd90", "pre_bwd90"),
    "cond_bwd": ("cond_bwd90", "cond_bwd90"),
    "cond_bwd_s": ("cond_bwd90", "cond_bwd90", ("-DSPLITK=1",)),
    "cond_bwd_sh": ("cond_bwd_sh90", "cond_bwd_sh90"),
}


@functools.lru_cache(maxsize=None)
def kernel(kind: str, tf32: bool, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    base = kind.removesuffix(
        "_g"
    )  # "<kind>_g": the grouped-sample launch (same code, larger dynamic shared memory)
    stem, func, *rest = _KERNELS[base]
    defs = rest[0] if rest else ()
    with torch.cuda.device(device_index):
        return driver.Kernel(
            cubin(stem, tf32, nwarp(base, tf32), _MINB.get((base, tf32), 1), defs),
            func,
            _smem(kind, tf32),
            pdl=settings.current().local_dit_h100_pdl,
        )


def _dev(t):
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _kperm8(t: torch.Tensor) -> torch.Tensor:
    """The last axis permuted within groups of 8: [0 2 4 6 1 3 5 7] (slot t holds column 2 t, slot t + 4 column 2 t + 1). View arithmetic
    only: no index tensor, so it can run inside a CUDA-graph capture."""
    x = t.reshape(*t.shape[:-1], t.shape[-1] // 8, 4, 2)
    return torch.cat([x[..., 0], x[..., 1]], dim=-1).reshape(t.shape)


def _lperm(t: torch.Tensor, axis: int, tf32: bool) -> torch.Tensor:
    """Gather an axis into the kernels' lane layout (row90.cuh ``pcol``): out[L] = t[P(L)], L = 8 s + 2 t + e inside a group of GS,
    P = VEC t + 2 s + e. Reshape / permute only (capture-safe)."""
    gs = 16 if tf32 else 32
    tpg = gs // 8
    x = t.movedim(axis, -1)
    n = x.shape[-1]
    y = x.reshape(*x.shape[:-1], n // gs, 4, tpg, 2).transpose(-3, -2).reshape(x.shape)
    return y.movedim(-1, axis)


def _prep(t: torch.Tensor, tf32: bool) -> torch.Tensor:
    """A weight [..., rows, K] as the kernels read it: rows and K in lane order, K in TF32 slot order."""
    t = _lperm(_lperm(t, -2, tf32), -1, tf32)
    return _kperm8(t) if tf32 else t


def round_tf32(x: torch.Tensor) -> torch.Tensor:
    """x rounded to nearest tf32 (ties away from zero, as cvt.rna.tf32), still fp32."""
    i = x.contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)


# weights, in ``integrations.local_dit.WEIGHTS`` + ``CROSS_WEIGHTS`` order
_I = {
    n: i
    for i, n in enumerate(
        (
            "g1",
            "sc1_w",
            "sc1_b",
            "bi1_w",
            "wq",
            "bq",
            "wk",
            "wv",
            "ln_pair",
            "wb",
            "wg",
            "wo",
            "so_w",
            "so_b",
            "g2",
            "sc2_w",
            "sc2_b",
            "bi2_w",
            "wa",
            "wb2",
            "wd",
            "st_w",
            "st_b",
            "gkv",
            "kv_sc_w",
            "kv_sc_b",
            "kv_bi_w",
        )
    )
}


def _pack_torch(weights, tf32: bool) -> dict:
    """The kernels' weight forms (bf16, or tf32-rounded fp32) of one block's 27 parameters; cached per weight version / capture scope."""
    dt = torch.float32 if tf32 else torch.bfloat16

    def build():
        w = {n: weights[i].detach() for n, i in _I.items()}
        f = lambda t: t.float()  # noqa: E731

        def cast(t):
            return round_tf32(f(t)) if tf32 else t.to(torch.bfloat16)

        g1, g2, gk = f(w["g1"]), f(w["g2"]), f(w["gkv"])

        def kperm(
            t,
        ):  # rows / K in lane order; TF32: K columns in slot order (slot t <-> column 2 t, slot t + 4 <-> 2 t + 1)
            return _prep(t, tf32)

        cond = torch.stack(
            [
                cast(f(w["sc1_w"]) * g1),
                cast(f(w["bi1_w"]) * g1),
                cast(f(w["sc2_w"]) * g2),
                cast(f(w["bi2_w"]) * g2),
                cast(f(w["kv_sc_w"]) * gk),
                cast(f(w["kv_bi_w"]) * gk),
                cast(w["so_w"]),
                cast(w["st_w"]),
            ]
        )
        cond = kperm(cond).contiguous()
        bias = torch.cat(
            [f(w["sc1_b"]), f(w["sc2_b"]), f(w["so_b"]), f(w["st_b"])]
        ).contiguous()
        pre = kperm(cast(torch.cat([w["wq"], w["wg"], w["wk"], w["wv"]]))).contiguous()
        hb = 128 // (4 if tf32 else 2)  # hidden chunk = one 128-byte swizzle row
        nch = 256 // hb
        wa, wb = cast(w["wa"]).view(nch, hb, 128), cast(w["wb2"]).view(nch, hb, 128)
        wab = kperm(
            torch.cat([wa, wb], dim=1)
        ).contiguous()  # [nch][2 hb][128]: chunk j = [Wa_j; Wb_j]
        wd = kperm(
            cast(w["wd"]).view(128, nch, hb).permute(1, 0, 2)
        ).contiguous()  # [nch][128][hb]
        return {
            "cond": cond,
            "bias": bias,
            "pre": pre,
            "bq": f(w["bq"]).contiguous(),
            "bs": f(w["kv_sc_b"]).contiguous(),
            "wo": kperm(cast(w["wo"])).contiguous(),
            "wab": wab,
            "wd": wd,
            "gamma": f(w["ln_pair"]).contiguous(),
            "wbias": f(w["wb"]).contiguous(),
            "dtype": dt,
        }

    return _capture.param_forms(f"sm90_atom_pack_{int(tf32)}", list(weights), build)


class _Pack(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_uint64 * 27),
        ("out", ctypes.c_uint64 * 16),
        ("wbf", ctypes.c_uint32),
        ("tf32", ctypes.c_int32),
    ]


def _pack_all(weights, tf32: bool):
    """(forward pack, backward pack) of the 27 parameters from ONE kernel launch (``pack90.cu``); cached per weight version / capture scope."""
    dev = weights[0].device
    dt = torch.float32 if tf32 else torch.bfloat16
    hb = 128 // (4 if tf32 else 2)
    nch = 256 // hb

    def build():
        new = lambda *s, d=dt: torch.empty(*s, device=dev, dtype=d)  # noqa: E731
        f32 = torch.float32
        outs = [
            new(8, 128, 128),
            new(512, 128),
            new(128, 128),
            new(nch, 2 * hb, 128),
            new(nch, 128, hb),  # cond pre wo wab wd
            new(4, 2, 64, 128),
            new(2, 64, 128),
            new(nch, hb, 128),
            new(4, 2, 64, 128),
            new(8, 2, 64, 128),  # wabt wot wdt wbt wct
            new(512, d=f32),
            new(128, d=f32),
            new(128, d=f32),
            new(*weights[_I["ln_pair"]].shape, d=f32),
            new(*weights[_I["wb"]].shape, d=f32),
        ]
        s = _Pack()
        for i, w in enumerate(weights):
            s.w[i] = w.data_ptr()
            s.wbf |= int(w.dtype == torch.bfloat16) << i
        for i, o in enumerate(outs):
            s.out[i] = o.data_ptr()
        s.tf32 = int(tf32)
        kernel("pack", False, _dev(weights[0]))((64, 15, 1), (256, 1, 1), s)
        cond, pre, wo, wab, wd, wabt, wot, wdt, wbt, wct, bias, bq, bs, gamma, wbias = (
            outs
        )
        fwd = {
            "cond": cond,
            "bias": bias,
            "pre": pre,
            "bq": bq,
            "bs": bs,
            "wo": wo,
            "wab": wab,
            "wd": wd,
            "gamma": gamma,
            "wbias": wbias,
            "dtype": dt,
        }
        bwd = {"wabt": wabt, "wot": wot, "wdt": wdt, "wbt": wbt, "wct": wct}
        return fwd, bwd

    return _capture.param_forms(f"sm90_atom_packall_{int(tf32)}", list(weights), build)


def _kernel_pack_ok(weights) -> bool:
    return all(
        w.dtype in (torch.float32, torch.bfloat16) and w.is_contiguous()
        for w in weights
    )


def pack(weights, tf32: bool) -> dict:
    """The kernels' weight forms (bf16, or tf32-rounded fp32) of one block's 27 parameters; cached per weight version / capture scope."""
    return (
        _pack_all(weights, tf32)[0]
        if _kernel_pack_ok(weights)
        else _pack_torch(weights, tf32)
    )


def pack_bwd(weights, tf32: bool) -> dict:
    """Transposed (K-major B operand) forms of the weights for the backward kernels; cached per weight version / capture scope."""
    return (
        _pack_all(weights, tf32)[1]
        if _kernel_pack_ok(weights)
        else _pack_bwd_torch(weights, tf32)
    )


_SM_FILL = 132 * 2  # CTAs the split aims for (about two per SM)


def _split_for(tiles: int, choices) -> int:
    """The smallest split of the work of one 64-row tile over CTAs (``choices`` ascending) that gives about _SM_FILL CTAs; 1 for many tiles."""
    for s in choices:
        if tiles * s >= _SM_FILL:
            return s
    return choices[-1] if tiles * choices[-1] < _SM_FILL else 1


def cond_fwd(c, p, tf32: bool, eps: float = 1e-5):
    """c [M, 128] -> mod [M, 768], mkv [M, 256] (M a multiple of 128)."""
    M = c.shape[0]
    dt = torch.float32 if tf32 else torch.bfloat16
    mod = torch.empty(M, 768, device=c.device, dtype=dt)
    mkv = torch.empty(M, 256, device=c.device, dtype=dt)
    tiles = M // ROWS
    ns = _split_for(tiles, (1, 2, 4, 8, 16))
    kernel("cond_split" if ns > 1 else "cond", tf32, _dev(c))(
        (tiles, ns, 1),
        (NWARP * 32, 1, 1),
        c,
        p["cond"],
        p["bias"],
        mod,
        mkv,
        int(M),
        float(eps),
    )
    return mod, mkv


def pre_fwd(
    x, mod, mkv, p, tf32: bool, save: bool = False, eps: float = 1e-5, nmod: int = 0
):
    """-> q, sg, k, v [M, 128] (and x1, xkv when ``save``)."""
    M = x.shape[0]
    new = lambda: torch.empty(M, DS, device=x.device, dtype=x.dtype)  # noqa: E731
    q, sg, k, v = new(), new(), new(), new()
    tiles = M // ROWS
    ns = _split_for(tiles, (1, 2, 4, 8))
    # the unsplit TF32 kernel round-trips the exact x1 through memory (registers hold it rounded); the split one keeps it in registers
    x1 = new() if (save or (tf32 and ns == 1)) else None
    xkv = new() if save else None
    kernel("pre_split" if ns > 1 else "pre", tf32, _dev(x))(
        (tiles, ns, 1),
        (NWARP * 32, 1, 1),
        x,
        mod,
        mkv,
        p["pre"],
        p["bq"],
        p["bs"],
        q,
        sg,
        k,
        v,
        x1,
        xkv,
        int(M),
        int(nmod or M),
        float(eps),
        int(save),
    )
    return (q, sg, k, v, x1, xkv) if save else (q, sg, k, v)


def post_fwd(
    a, o, sg, mod, p, tf32: bool, save: bool = False, eps: float = 1e-5, nmod: int = 0
):
    """-> out [M, 128] (and u, a2, x2, t when ``save``)."""
    M = a.shape[0]
    new = lambda: torch.empty(M, DS, device=a.device, dtype=a.dtype)  # noqa: E731
    out = new()
    u, a2, x2, t = (new(), new(), new(), new()) if save else (None,) * 4
    kernel("post", tf32, _dev(a))(
        (M // ROWS, 1, 1),
        (NWARP * 32, 1, 1),
        a,
        o,
        sg,
        mod,
        p["wo"],
        p["wab"],
        p["wd"],
        out,
        u,
        a2,
        x2,
        t,
        int(M),
        int(nmod or M),
        float(eps),
        int(save),
    )
    return (out, u, a2, x2, t) if save else out


def _pack_bwd_torch(weights, tf32: bool) -> dict:
    """Transposed (K-major B operand) forms of the weights for the backward kernels; cached per weight version / capture scope."""

    def build():
        w = {n: weights[i].detach() for n, i in _I.items()}

        def cast(t):
            return round_tf32(t.float()) if tf32 else t.to(torch.bfloat16)

        def kperm(t):
            return _prep(t, tf32)

        def tiles(wt):  # [in, out] -> [2 half][64][128]
            return kperm(wt.reshape(2, 64, 128))

        f = lambda t: t.float()  # noqa: E731
        g1, g2, gk = f(w["g1"]), f(w["g2"]), f(w["gkv"])
        hb = 128 // (4 if tf32 else 2)
        nch = 256 // hb
        wab = torch.cat([cast(w["wa"]), cast(w["wb2"])])  # [512 hid, 128 in]
        wabt = kperm(
            wab.t().contiguous().view(2, 64, 4, 128).permute(2, 0, 1, 3)
        ).contiguous()  # [kb][half][64][128]
        wot = tiles(cast(w["wo"]).t().contiguous()).contiguous()  # [half][64][128]
        wdt = kperm(
            cast(w["wd"]).t().contiguous().view(nch, hb, 128)
        ).contiguous()  # [nch][hb][128]
        wbt = torch.stack(
            [tiles(cast(w[n]).t().contiguous()) for n in ("wg", "wq", "wk", "wv")]
        ).contiguous()
        cond_mats = (
            f(w["sc1_w"]) * g1,
            f(w["bi1_w"]) * g1,
            f(w["sc2_w"]) * g2,
            f(w["bi2_w"]) * g2,
            f(w["kv_sc_w"]) * gk,
            f(w["kv_bi_w"]) * gk,
            f(w["so_w"]),
            f(w["st_w"]),
        )
        wct = torch.stack(
            [tiles(cast(m).t().contiguous()) for m in cond_mats]
        ).contiguous()  # [8][half][64][128]
        return {"wabt": wabt, "wot": wot, "wdt": wdt, "wbt": wbt, "wct": wct}

    return _capture.param_forms(f"sm90_atom_pack_bwd_{int(tf32)}", list(weights), build)


def tr_bwd(dy, t, mod, x2, p, pb, tf32: bool, nmod: int = 0, shared=None):
    """-> dt [M, 128], hh [M, 256], dab [M, 512], dmod, column sum of block 5 (d st bias). ``shared`` = (spc, natoms, SP): the shared-conditioning
    grouped-sample mode (``spc`` > 0, ``natoms`` = N, M a multiple of 4 N): dmod is then [M / (4 N), natoms, 768] of the activation dtype, one
    row per four samples (their modulation gradients summed in fp32 inside the CTA)."""
    M = dy.shape[0]
    dev = dy.device
    dt = torch.empty(M, DS, device=dev, dtype=dy.dtype)
    hh = torch.empty(M, 256, device=dev, dtype=dy.dtype)
    dab = torch.empty(M, 512, device=dev, dtype=dy.dtype)
    nw = nwarp("tr_bwd", tf32)
    rows = nw * 16
    spc, natoms, sp = shared or (0, 0, 1)
    dmod = (
        torch.empty(M // (4 * natoms), natoms, 768, device=dev, dtype=dy.dtype)
        if shared
        else torch.empty(M, 768, device=dev, dtype=dy.dtype)
    )
    grid = (M // rows, 1, 1)
    nch = 256 // (128 // (4 if tf32 else 2))
    ns = _split_for(grid[0], tuple(s for s in (1, 2, 4, 8) if nch % s == 0))
    grid = (grid[0], ns, 1)
    colpart = torch.empty(grid[0] * grid[1], 128, device=dev, dtype=torch.float32)
    kernel("tr_bwd_g" if shared else "tr_bwd", tf32, _dev(dy))(
        grid,
        (nw * 32, 1, 1),
        dy,
        t,
        mod,
        x2,
        p["wab"],
        pb["wdt"],
        dt,
        dmod if not shared else None,
        hh,
        dab,
        colpart,
        int(M),
        int(nmod or M),
        dmod if shared else None,
        int(spc),
        int(natoms),
    )
    return dt, hh, dab, dmod, colpart.sum(0)


def post_bwd(
    dab,
    dy,
    a2,
    mod,
    u,
    sg,
    o,
    pb,
    dmod,
    dp,
    n: int,
    tf32: bool,
    eps: float = 1e-5,
    nmod: int = 0,
    shared=None,
):
    """-> da2, du, gated, do [M, 128]; writes dp[:, 0:128] (dg), dmod blocks 2-4 (``dmod`` as returned by ``tr_bwd``: per-sample rows, or the
    [M / 4N, N, 768] group sums when ``shared``); dd [A, 4, N] fp32."""
    M = dy.shape[0]
    dev = dy.device
    new = lambda: torch.empty(M, DS, device=dev, dtype=dy.dtype)  # noqa: E731
    da2, du, gated, do = new(), new(), new(), new()
    dd = torch.empty(M // n, 4, n, device=dev, dtype=torch.float32)
    spc, natoms, sp = shared or (0, 0, 1)
    tiles = M // ROWS
    ns = _split_for(tiles, (1, 2, 4))
    grid = (tiles, ns, 1)
    part = (
        torch.empty(tiles * ns * 16 * 512, device=dev, dtype=torch.float32)
        if ns > 1
        else None
    )
    ctr = torch.zeros(tiles, device=dev, dtype=torch.int32) if ns > 1 else None
    colpart = torch.empty(tiles, 256, device=dev, dtype=torch.float32)
    kernel(
        ("post_bwd_s" if ns > 1 else "post_bwd") + ("_g" if shared else ""),
        tf32,
        _dev(dy),
    )(
        grid,
        (NWARP * 32, 1, 1),
        dab,
        dy,
        a2,
        mod,
        u,
        sg,
        o,
        pb["wabt"],
        pb["wot"],
        da2,
        du,
        gated,
        do,
        dp,
        None if shared else dmod,
        dd,
        colpart,
        int(M),
        int(nmod or M),
        int(n),
        float(eps),
        dmod if shared else None,
        int(spc),
        int(natoms),
        part,
        ctr,
    )
    cs = colpart.sum(0)
    return da2, du, gated, do, dd, cs[:128], cs[128:]


def pre_bwd(
    dp,
    x1,
    x,
    mod,
    mkv,
    da2,
    pb,
    p,
    dmod,
    tf32: bool,
    eps: float = 1e-5,
    nmod: int = 0,
    shared=None,
):
    """-> dx [M, 128], dmkv [M, 256] (``shared``: group sums [M / 4N, N, 256]); writes dmod blocks 0, 1 (group sums when shared)."""
    M = x.shape[0]
    dx = torch.empty(M, DS, device=x.device, dtype=x.dtype)
    spc, natoms, sp = shared or (0, 0, 1)
    dmkv = (
        torch.empty(M // (4 * natoms), natoms, 256, device=x.device, dtype=x.dtype)
        if shared
        else torch.empty(M, 256, device=x.device, dtype=x.dtype)
    )
    grid = (M // ROWS, 1, 1)
    colpart = torch.empty(grid[0] * grid[1], 384, device=x.device, dtype=torch.float32)
    kernel("pre_bwd_g" if shared else "pre_bwd", tf32, _dev(x))(
        grid,
        (NWARP * 32, 1, 1),
        dp,
        x1,
        x,
        mod,
        mkv,
        da2,
        pb["wbt"],
        p["bs"],
        None if shared else dmod,
        None if shared else dmkv,
        dx,
        colpart,
        int(M),
        int(nmod or M),
        float(eps),
        dmod if shared else None,
        dmkv if shared else None,
        int(spc),
        int(natoms),
    )
    cs = colpart.sum(0)
    return dx, dmkv, cs[:128], cs[128:256], cs[256:]


def cond_bwd(dmod, dmkv, c, pb, tf32: bool, eps: float = 1e-5):
    """-> dc [M, 128], n = LN(c) [M, 128]."""
    M = c.shape[0]
    dc = torch.empty(M, DS, device=c.device, dtype=c.dtype)
    n = torch.empty(M, DS, device=c.device, dtype=c.dtype)
    tiles = M // ROWS
    ns = _split_for(tiles, (1, 4))
    part = (
        torch.empty(tiles * ns * 16 * 512, device=c.device, dtype=torch.float32)
        if ns > 1
        else None
    )
    ctr = torch.zeros(tiles, device=c.device, dtype=torch.int32) if ns > 1 else None
    kernel("cond_bwd_s" if ns > 1 else "cond_bwd", tf32, _dev(c))(
        (tiles, ns, 1),
        (NWARP * 32, 1, 1),
        dmod,
        dmkv,
        c,
        pb["wct"],
        dc,
        n,
        int(M),
        float(eps),
        part,
        ctr,
    )
    return dc, n


def cond_bwd_shared(dmodp, dmkvp, c, pb, tf32: bool, a: int, eps: float = 1e-5):
    """Shared conditioning, c [N, 128], partial modulation gradients dmodp [P, N, 768] / dmkvp [P, N, 256] (P sample groups) ->
    dc / a [N, 128], n = LN(c), the fp32 sums ds [N, 768] and dk [N, 256]; all fp32."""
    P, N = dmodp.shape[0], dmodp.shape[1]
    f = lambda *s: torch.empty(*s, device=c.device, dtype=torch.float32)  # noqa: E731
    dc, nrm, ds, dk = f(N, DS), f(N, DS), f(N, 768), f(N, 256)
    kernel("cond_bwd_sh", tf32, _dev(c))(
        (N // ROWS, 1, 1),
        (NWARP * 32, 1, 1),
        dmodp,
        dmkvp,
        c,
        pb["wct"],
        ds,
        dk,
        nrm,
        dc,
        int(N),
        int(P),
        float(eps),
        float(1.0 / a),
    )
    return dc, nrm, ds, dk


class _Fin(ctypes.Structure):
    _fields_ = (
        [
            (n, ctypes.c_uint64)
            for n in (
                "dwgq",
                "dwkv",
                "dwo",
                "dwab",
                "dwd",
                "dc1",
                "dc2",
                "dckv",
                "dwso",
                "dwst",
                "cs_sc1",
                "cs_bq",
                "cs_so",
                "cs_sc2",
                "cs_st",
                "cs_bs",
                "dgamma_pair",
                "dwb",
                "g1",
                "g2",
                "gkv",
                "sc1_w",
                "bi1_w",
                "sc2_w",
                "bi2_w",
                "kv_sc_w",
                "kv_bi_w",
            )
        ]
        + [("out", ctypes.c_uint64 * 27)]
        + [("bf", ctypes.c_uint32), ("wbf", ctypes.c_uint32)]
    )


def finalize(src: dict, weights, outs, device_index: int):
    """The 27 parameter gradients (``outs``: preallocated tensors in the parameters' dtypes, ``_NAMES`` order) from the fp32 GEMM outputs in
    ``src`` (names as in ``finalize90.cu``) and the forward parameters ``weights``."""
    f = _Fin()
    for n, t in src.items():
        setattr(f, n, t.data_ptr())
    wnames = (
        "g1",
        "g2",
        "gkv",
        "sc1_w",
        "bi1_w",
        "sc2_w",
        "bi2_w",
        "kv_sc_w",
        "kv_bi_w",
    )
    for j, n in enumerate(wnames):
        w = weights[_I[n]]
        setattr(f, n, w.data_ptr())
        f.wbf |= int(w.dtype == torch.bfloat16) << j
    for i, t in enumerate(outs):
        f.out[i] = t.data_ptr()
        f.bf |= int(t.dtype == torch.bfloat16) << i
    kernel("finalize", False, device_index)((16, 27, 1), (256, 1, 1), f)
