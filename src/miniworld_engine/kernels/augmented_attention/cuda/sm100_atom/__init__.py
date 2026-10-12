"""sm_100a (B200) row kernels of the AF3 atom transformer block (``modules/local_dit.LocalDiTBlock``; ``integrations/local_dit``), forward
and backward. The attention and the pair bias of that block are ``sm100_atom_local``'s.

    a = a + sigmoid(to_scale(s)) * to_out(sigmoid(g) * attention(q, k, v))
    a = a + ConditionedTransition(a, s)                                                          SwiGLU 128 -> 256 -> 128

d_single = d_cond = 128, d_pair = 16, 4 heads x 32; the caller pads the atoms to a multiple of 128 and masks the padding.
Page: ``docs/gpus/b200/local_dit/local_dit.md``.

The kernels (``sm100.cuh`` holds the tcgen05 / TMA / mbarrier helpers), bf16:
  cond_fwd.cu     the six conditioning projections -> mod [M, 768] = s1 | bi1 | so | s2 | bi2 | st (gates as rn(sigmoid)).
  pre_fwd.cu      AdaLN 1 + the q / k / v / gate projections (``-DCROSS=1``: the cross-attention mode's K / V AdaLN as well).
  post_fwd.cu     gate, out projection, residual, AdaLN 2, SwiGLU, squeeze, residual (+ the activations the backward reads).
  tr_bwd_gate.cu  the transition's gate backward and SwiGLU backward.
  post_bwd.cu     AdaLN 2 + LN backward, residual, output gate, the out projection's dgrad, the attention gate; dO and D.
  pre_bwd.cu      the input projections' dgrad, AdaLN 1 + LN backward -> d single (``-DCROSS=1``: through the K / V AdaLN too).
  cond_bwd.cu     the conditioning projections' dgrad and the two cond LayerNorm backwards -> d cond.
  dw_atom.cu      every dense weight gradient of a block in one grouped split-K launch (bf16 and TF32).

The fp32 path (fp32 activations, ``KERNELS_TF32``, loaded by ``atom_kernel32`` only when an fp32 call is served; used by the local
atom block, ``integrations/local_dit``):
  gemm_tf32.cu    every projection and activation-gradient product as one tcgen05 kind::tf32 GEMM launch with the block's epilogues
                  (bias + sigmoid blocks, gated residual, SwiGLU).
  rows_tf32.cu    the row-wise stages around them (LayerNorms, AdaLN and their backward, gates, SwiGLU backward, D), fp32 CUDA cores.

The cubins are built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``:
13.1 builds; 12.9's ptxas gives slightly longer code) and cached under
``MINIWORLD_ENGINE_JIT_ROOT`` keyed by the sources and the flags. They launch through the CUDA driver
(``augmented_attention.cuda.sm100.driver``) on torch's current stream, CUDA-graph capturable.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

NH, DH, DM, DP = 4, 32, 128, 16
_dir = Path(__file__).parent
_SMEM = 232448
SOURCES = ("cond_fwd", "pre_fwd", "post_fwd",
           "tr_bwd_gate", "post_bwd", "pre_bwd", "cond_bwd")


# --------------------------------------------------------------------------------------------------- build
@functools.lru_cache(maxsize=None)
def cubin(stem: str, defs: tuple = ()) -> str:
    """Path of ``<stem>.cu`` built for sm_100a with the sources' default options (+ ``-D`` ``defs``); rebuilt only when a source or a flag
    changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}", *("-D" + d for d in defs))
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "local_dit_sm100" / f"{stem}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(_dir / f"{stem}.cu"), "-o", str(tmp)], capture_output=True, text=True,
                             timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {stem}.cu:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


@functools.lru_cache(maxsize=None)
def _load_atom_kernel(stem: str, func: str, device_index: int, smem: int = _SMEM, pdl: bool = False, defs: tuple = ()):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, defs), func, smem, pdl=pdl)


#: (stem, function, dynamic smem, PDL[, -D defs]) of every kernel the block launches
KERNELS = {
    "cond": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True),
    "pre": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True),
    "prex": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1",)),
    "post": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True),
    "trg": ("tr_bwd_gate", "atom_tr_bwd_gate_sm100", _SMEM, True),
    "postb": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True),
    "preb": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True),
    "prebx": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True, ("CROSS=1",)),
    "condb": ("cond_bwd", "atom_cond_bwd_sm100", _SMEM, True),
}


def atom_kernel(name: str, device_index: int):
    stem, func, smem, pdl, *defs = KERNELS[name]
    return _load_atom_kernel(stem, func, device_index, smem, pdl, *defs)


def load_all(device_index: int) -> None:
    """Build and load every kernel on the device (raises on any failure)."""
    for name in KERNELS:
        atom_kernel(name, device_index)


@functools.lru_cache(maxsize=8)
def nsm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


# --------------------------------------------------------------------------------------------------- tensor maps
def _tm(t, rows, box_rows, cols=DM, box_cols=DH, swizzle=128):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, [cols, rows], cols * 2, [box_cols, box_rows], swizzle=swizzle)


def _map(t, dims, stride, box, swizzle=128, dtype="bf16"):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, dims, stride, box, swizzle=swizzle, dtype=dtype)


def _rows(t, cols=DM):
    """Row-major bf16 [M, cols] as 32-row x 64-column boxes (128-B swizzle)."""
    return _map(t, [cols, t.shape[0]], cols * 2, [64, 32])


def _rows16(t, cols=DM):
    return _map(t, [cols, t.shape[0]], cols * 2, [64, 16])


# --------------------------------------------------------------------------------------------------- weights
def cond_params(blk):
    """The six conditioning projections stacked for cond_fwd: Wmod [768, 128] bf16 = [Wsc1; Wbi1; Wos; Wsc2; Wbi2; Wts],
    bias [768] fp32 (zero for the shifts), the two cond LayerNorm weights g1, g2 [128] fp32."""
    att, tr = blk.attention, blk.transition
    a1, a2 = att.ada_ln_in, tr.ada_ln_in
    W = torch.cat([a1.to_scale.weight, a1.to_bias.weight, att.to_scale.weight, a2.to_scale.weight, a2.to_bias.weight,
                   tr.to_scale.weight])
    z = torch.zeros(DM, device=W.device)
    b = torch.cat([a1.to_scale.bias.float(), z, att.to_scale.bias.float(), a2.to_scale.bias.float(), z, tr.to_scale.bias.float()])
    return (W.to(torch.bfloat16).contiguous(), b.contiguous(), a1.ln_cond.weight.float().contiguous(),
            a2.ln_cond.weight.float().contiguous())


def pre_params(blk):
    """The attention's four input projections stacked for pre_fwd: W [512, 128] bf16 = [Wq; Wk; Wv; Wg], bq [128] fp32."""
    att = blk.attention
    W = torch.cat([att.to_query.weight, att.to_key.weight, att.to_value.weight, att.to_gate.weight])
    return W.to(torch.bfloat16).contiguous(), att.to_query.bias.float().contiguous()


def post_params(blk):
    """post_fwd's weights: Wo [128, 128], Wu = [Wa; Wb] [512, 128], Ws [128, 256] (bf16)."""
    att, tr = blk.attention, blk.transition
    def bf(t):
        return t.to(torch.bfloat16).contiguous()
    return bf(att.to_out.weight), bf(torch.cat([tr.expand_a.weight, tr.expand_b.weight])), bf(tr.squeeze.weight)


# --------------------------------------------------------------------------------------------------- launches
def cond_fwd(c, W, b, g1, g2, eps=1e-5):
    """c [M, 128] bf16 -> mod [M, 768] bf16 = [s1 | bi1 | so | s2 | bi2 | st] (the gates as rn(sigmoid))."""
    M = c.shape[0]
    out = torch.empty(M, 6 * DM, device=c.device, dtype=torch.bfloat16)
    maps = (_rows(c), _map(W, [DM, 6 * DM], DM * 2, [32, 128], swizzle=64), _rows(out, 6 * DM))
    ntile = M // 32
    d = _dev(c)
    atom_kernel("cond", d)((min(nsm(d) // 2, ntile), 2, 1), (384, 1, 1), *maps, g1, g2, b, int(ntile), float(eps))
    return out


def pre_fwd(a, mod, W, bq, save=False, eps=1e-5):
    """a [M, 128], mod [M, 768] -> Q, K, V, G [M, 128] bf16 (+ x1 = AdaLN 1 output when ``save``)."""
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    X = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    maps = (_rows(a), _map(W, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64), _rows(mod, 6 * DM),
            _rows(Q), _rows(K), _rows(V), _rows(G), _rows(X), _rows(G), _rows(G))
    ntile = M // 32
    d = _dev(a)
    atom_kernel("pre", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, bq, bq, int(ntile), float(eps), int(save))
    return Q, K, V, G, X


def pre_fwd_kv(a, mod, W, bq, mkv, bs, save=False, eps=1e-5):
    """The cross-attention mode's ``pre_fwd``: K and V from xkv = LN(x1) sigmoid(mkv[:, :128] + bs) + mkv[:, 128:] (a second AdaLN of x1)
    instead of from x1. -> Q, K, V, G [M, 128] bf16 and, when ``save``, x1 and xkv (else X = G, XK = None)."""
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    X = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    XK = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    maps = (_rows(a), _map(W, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64), _rows(mod, 6 * DM),
            _rows(Q), _rows(K), _rows(V), _rows(G), _rows(X), _rows(mkv, 2 * DM), _rows(XK))
    ntile = M // 32
    d = _dev(a)
    atom_kernel("prex", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, bq, bs.float().contiguous(), int(ntile), float(eps), int(save))
    return Q, K, V, G, X, (XK if save else None)


def post_fwd(a, o, g, mod, Wo, Wu, Wd, save=False, eps=1e-5):
    """a, o, g [M, 128], mod [M, 768] -> the block output [M, 128] bf16 (+ u, a2, x2, t when ``save``)."""
    M = a.shape[0]
    def new():
        return torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    out = new()
    sv = tuple(new() for _ in range(4)) if save else (out,) * 4
    maps = (_rows16(a), _rows16(o), _rows16(g), _rows16(mod, 6 * DM), _map(Wo, [DM, DM], DM * 2, [64, 128]),
            _map(Wu, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64), _map(Wd, [2 * DM, DM], 4 * DM, [32, 128], swizzle=64),
            _rows16(out), *(_rows16(t) for t in sv))
    ntile = M // 16
    d = _dev(a)
    atom_kernel("post", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, int(ntile), float(eps), int(save))
    return (out, *sv)


def tr_bwd_gate(dy, t, mod, x2, Wu, WsT, dmod, dbts):
    """dy, t, mod (st), x2 -> DT [M, 128], HH [M, 256], DAB [M, 512]; dts into dmod[:, 640:768]; d bts += the column sums."""
    M = dy.shape[0]
    dev = dy.device
    DT = torch.empty(M, DM, device=dev, dtype=torch.bfloat16)
    HH = torch.empty(M, 2 * DM, device=dev, dtype=torch.bfloat16)
    DAB = torch.empty(M, 4 * DM, device=dev, dtype=torch.bfloat16)
    maps = (_rows16(dy), _rows16(t), _rows16(mod, 6 * DM), _rows16(x2), _map(Wu, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64),
            _map(WsT, [DM, 2 * DM], DM * 2, [32, 128], swizzle=64), _rows16(DT), _rows16(dmod, 6 * DM), _rows16(HH, 2 * DM),
            _rows16(DAB, 4 * DM))
    ntile = M // 16
    d = _dev(dy)
    atom_kernel("trg", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, dbts, int(ntile))
    return DT, HH, DAB


def post_bwd(DAB, dy, a2, mod, u, sg, o, WuT, WoT, dmod, dP, Dd, DBIAS, N, eps=1e-5):
    """-> DA, DU, GATED, DO [M, 128]; dmod blocks 2-4, dP block 3 (dg), D [A, 4, N]; d bsc2 / d bos into DBIAS."""
    M = dy.shape[0]
    def new():
        return torch.empty(M, DM, device=dy.device, dtype=torch.bfloat16)
    DA, DU, GATED, DO = new(), new(), new(), new()
    maps = (_rows16(DAB, 4 * DM), _rows16(dy), _rows16(a2), _rows16(mod, 6 * DM), _rows16(u), _rows16(sg), _rows16(o),
            _map(WuT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64), _map(WoT, [DM, DM], DM * 2, [32, 128], swizzle=64),
            _rows16(DA), _rows16(dmod, 6 * DM), _rows16(DU), _rows16(GATED), _rows16(DO), _rows16(dP, 4 * DM))
    ntile = M // 16
    d = _dev(dy)
    atom_kernel("postb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, Dd, DBIAS, int(N), int(ntile), float(eps))
    return DA, DU, GATED, DO


def pre_bwd(dP, a, mod, DA, WT, dmod, DBIAS, DBQ, eps=1e-5):
    """dP [M, 512], a, mod (s1), DA -> d single [M, 128]; dmod blocks 0-1; d bsc1 into DBIAS[0:128], d bq into DBQ."""
    M = a.shape[0]
    DS = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    maps = (_rows16(dP, 4 * DM), _rows16(a), _rows16(mod, 6 * DM), _rows16(DA), _map(WT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64),
            _rows16(DS), _rows16(dmod, 6 * DM), _rows16(DA), _rows16(DA), _rows16(DA))
    ntile = M // 16
    d = _dev(a)
    atom_kernel("preb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, DBIAS, DBQ, DBQ, DBQ, int(ntile), float(eps))
    return DS


def pre_bwd_kv(dP, a, mod, DA, WT, dmod, DBIAS, DBQ, x1, mkv, bs, DBS, eps=1e-5):
    """The cross-attention mode's ``pre_bwd``: dP = [dq | dk | dv | dg] with WT the real [Wq; Wk; Wv; Wg]^T; k and v come from xkv = LN(x1)
    sigmoid(mkv[:, :128] + bs) + mkv[:, 128:], whose backward (dxkv = dk Wk + dv Wv, in the kernel) is added to dx1 before AdaLN 1's.
    -> d single [M, 128] and dmkv [M, 256] = [d pre-sigmoid scale | dxkv]; d bs (fp32 [128]) is added into DBS."""
    M = a.shape[0]
    DS = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    dmkv = torch.empty(M, 2 * DM, device=a.device, dtype=torch.bfloat16)
    maps = (_rows16(dP, 4 * DM), _rows16(a), _rows16(mod, 6 * DM), _rows16(DA), _map(WT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64),
            _rows16(DS), _rows16(dmod, 6 * DM), _rows16(x1), _rows16(mkv, 2 * DM), _rows16(dmkv, 2 * DM))
    ntile = M // 16
    d = _dev(a)
    atom_kernel("prebx", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, DBIAS, DBQ, DBS, bs.float().contiguous(), int(ntile), float(eps))
    return DS, dmkv


def cond_bwd(dmod, c, WmodT, g1, g2, DG, eps=1e-5):
    """dmod [M, 768] (backward layout), c [M, 128] -> dc, cn1, cn2 [M, 128]; d g1 / d g2 into DG [256]."""
    M = c.shape[0]
    def new():
        return torch.empty(M, DM, device=c.device, dtype=torch.bfloat16)
    dc, cn1, cn2 = new(), new(), new()
    maps = (_rows16(dmod, 6 * DM), _rows16(c), _map(WmodT, [6 * DM, DM], 12 * DM, [32, 128], swizzle=64),
            _rows16(dc), _rows16(cn1), _rows16(cn2))
    ntile = M // 16
    d = _dev(c)
    atom_kernel("condb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, g1, g2, DG, int(ntile), float(eps))
    return dc, cn1, cn2


# --------------------------------------------------------------------------------------------------- fp32 path (TF32 tensor cores)
SOURCES_TF32 = ("gemm_tf32", "rows_tf32")
F32 = torch.float32


@functools.lru_cache(maxsize=None)
def cubin_tf32(stem: str, defs: tuple = ()) -> str:
    """``cubin`` for the fp32-path sources, which include ``../sm100/sm100.cuh`` (the kind::tf32 helpers): that header is hashed."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}", *("-D" + d for d in defs))
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir.parent / "sm100" / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "local_dit_sm100_tf32" / f"{stem}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(_dir / f"{stem}.cu"), "-o", str(tmp)], capture_output=True, text=True,
                             timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {stem}.cu:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


#: (stem, function, dynamic smem) of every fp32-path kernel (gemm_tf32's size is static_asserted in the source)
KERNELS_TF32 = {
    "gemm": ("gemm_tf32", "atom_gemm_tf32", 213248),
    "gemm_rg": ("gemm_rg_tf32", "atom_gemm_rg_tf32", 229632),
    "ln": ("rows_tf32", "f32_ln_aff", 0),
    "adaln": ("rows_tf32", "f32_adaln", 0),
    "gate": ("rows_tf32", "f32_gate", 0),
    "tail_b": ("rows_tf32", "f32_tail_bwd", 0),
    "swiglu_b": ("rows_tf32", "f32_swiglu_bwd", 0),
    "adaln_b": ("rows_tf32", "f32_adaln_bwd", 0),
    "gate_b": ("rows_tf32", "f32_gate_bwd", 0),
    "ln_b": ("rows_tf32", "f32_ln_bwd", 0),
    "colsum": ("rows_tf32", "f32_colsum", 0),
}

#: Programmatic dependent launch for the fp32 path's kernels (``MINIWORLD_TF32_PDL``, default on). Every one of them executes
#: ``griddepcontrol.wait`` before its first global-memory access (after its smem / TMEM setup), so it may be scheduled while the previous
#: kernel drains: the launch latency and the setup overlap the previous kernel's tail. ``set_pdl32`` flips the loaded kernels (A/B timing).
PDL32 = [os.environ.get("MINIWORLD_TF32_PDL", "1") != "0"]
_PDL32_LOADED: list = []


def set_pdl32(on: bool) -> bool:
    """Turn programmatic dependent launch of the loaded (and later loaded) fp32-path kernels on or off; returns the previous setting."""
    prev, PDL32[0] = PDL32[0], bool(on)
    for k in _PDL32_LOADED:
        k.pdl = PDL32[0]
    return prev


@functools.lru_cache(maxsize=None)
def atom_kernel32(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_TF32[name]
    with torch.cuda.device(device_index):
        k = driver.Kernel(cubin_tf32(stem), func, smem, pdl=PDL32[0])
    _PDL32_LOADED.append(k)
    return k


#: the fused inference kernels of the fp32 path (``fused_tf32.cu``; loaded apart from KERNELS_TF32 so a failure costs the fusion only)
KERNELS_FUSED32 = {
    "pre": ("fused_tf32", "atom_pre_tf32", 229632),
    "post": ("fused_tf32", "atom_post_tf32", 229632),
}


@functools.lru_cache(maxsize=None)
def fused_kernel32(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_FUSED32[name]
    with torch.cuda.device(device_index):
        k = driver.Kernel(cubin_tf32(stem), func, smem, pdl=PDL32[0])
    _PDL32_LOADED.append(k)
    return k


def load_fused32(device_index: int) -> None:
    for name in KERNELS_FUSED32:
        fused_kernel32(name, device_index)


def _nfull(tiles, nsm_):
    """How many of the fused kernels' 128-row tiles run as full units; the rest run as two half units each (``fused_tf32.cu``). A CTA's unit
    is a serial latency chain (ncu: ~5 % issue utilisation), so the time is ~ the waves of units over the SMs times a unit's cost; a half unit
    costs 0.7-0.8 of a full one (round 5, 3072 / 6144 atoms). So: every tile split when the halves fit one wave; otherwise the tail beyond the
    whole waves split when ITS halves fit one wave (4096 atoms on 148 SMs: 148 full + 24 halves, 1.75 waves instead of 2); else none.
    MINIWORLD_TF32_FUSED_UNITS=full / half forces either."""
    mode = os.environ.get("MINIWORLD_TF32_FUSED_UNITS", "auto")
    if mode in ("full", "half"):
        return tiles if mode == "full" else 0
    if 2 * tiles <= nsm_:
        return 0
    full = tiles // nsm_ * nsm_
    tail = tiles - full
    return full if 0 < tail and 2 * tail <= nsm_ else tiles


def pre32(s, s1, b1, wp, bq, rndmask, mks=None, mkb=None, eps=1e-5, nfull=None):
    """Fused AdaLN 1 (+ the cross mode's K / V AdaLN, when mks / mkb are given) and the four input projections (``atom_pre_tf32``):
    s [M, 128] fp32 contiguous, s1 / b1 (and mks / mkb) fp32 [M, 128] views with a common row stride each, wp the tf32-rounded [512, 128]
    pack ([q; g; k; v] with mks, else [q; k; v; g]), bq [128] -> P [M, 512] fp32 (the blocks in rndmask tf32-rounded). ``nfull`` (default
    ``_nfull``): the 128-row tiles run whole; the others as two 128 x 256 column halves."""
    M = s.shape[0]
    assert M % 128 == 0 and s.is_contiguous() and s.shape[1] == DM and tuple(wp.shape) == (4 * DM, DM), (s.shape, wp.shape)
    assert s1.stride(1) == b1.stride(1) == 1 and s1.stride(0) == b1.stride(0)
    cross = mks is not None
    if cross:
        assert mks.stride(1) == mkb.stride(1) == 1 and mks.stride(0) == mkb.stride(0)
    p = torch.empty(M, 4 * DM, device=s.device, dtype=F32)
    d = _dev(s)
    tiles = M // 128
    nfull = _nfull(tiles, nsm(d)) if nfull is None else int(nfull)
    units = nfull + 2 * (tiles - nfull)
    fused_kernel32("pre", d)((min(nsm(d), units), 1, 1), (384, 1, 1), _tm32(wp, (32, 256)), _tm32(p, (32, 32)), s, s1, b1,
                             int(s1.stride(0)), mks, mkb, int(mks.stride(0)) if cross else 0, bq, int(cross), int(rndmask), int(M), float(eps),
                             int(nfull))
    return p


def post32(o, g, s, so, s2, b2, st, wo, wu, ws, eps=1e-5, nfull=None):
    """Fused gate, out projection + gated residual, AdaLN 2, SwiGLU, squeeze + gated residual (``atom_post_tf32``): o, s [M, 128] fp32
    contiguous, g a [M, 128] view, so / s2 / b2 / st [M, 128] views with one row stride, wo / wu / ws the tf32-rounded packs -> out [M, 128].
    ``nfull`` (default ``_nfull``): the 128-row tiles run whole; the others as two 64-row halves."""
    M = s.shape[0]
    assert M % 128 == 0 and s.is_contiguous() and o.is_contiguous() and o.shape == s.shape, (o.shape, s.shape)
    assert g.stride(1) == 1 and all(t.stride(1) == 1 and t.stride(0) == so.stride(0) for t in (s2, b2, st))
    out = torch.empty(M, DM, device=s.device, dtype=F32)
    d = _dev(s)
    tiles = M // 128
    nfull = _nfull(tiles, nsm(d)) if nfull is None else int(nfull)
    units = nfull + 2 * (tiles - nfull)
    fused_kernel32("post", d)((min(nsm(d), units), 1, 1), (384, 1, 1), _tm32(wo, (32, 128)), _tm32(wu, (32, 256)), _tm32(ws, (32, 128)),
                              o, g, int(g.stride(0)), s, so, s2, b2, st, int(so.stride(0)), out, int(M), float(eps), int(nfull))
    return out


def load_all32(device_index: int) -> None:
    """Build and load every fp32-path kernel on the device (raises on any failure)."""
    for name in KERNELS_TF32:
        atom_kernel32(name, device_index)


_TMAPS32: dict = {}


def _tm32(t, box, swizzle=128):
    """TMA map of a row-major fp32 [rows, cols] tensor (unit column stride, any row stride), cached by everything it encodes (address,
    geometry, box, swizzle; encoding one costs ~5-10 us of host time). The cache does not hold the tensor."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    assert t.dtype == F32 and t.dim() == 2 and t.stride(1) == 1, (t.dtype, t.shape, t.stride())
    rows, cols = t.shape
    key = (t.data_ptr(), cols, rows, t.stride(0), tuple(box), swizzle)
    m = _TMAPS32.get(key)
    if m is None:
        if len(_TMAPS32) >= 512:
            _TMAPS32.clear()
        m = driver.TensorMap(t, [cols, rows], t.stride(0) * 4, list(box), swizzle=swizzle, dtype="f32")
        m.keep = None
        _TMAPS32[key] = m
    return m


#: gemm32 epilogues
STORE, RESID_GATE, SWIGLU = 0, 1, 2


def gemm32(a, w, out=None, *, k=None, n=None, a_col0=0, b_row0=0, b_col0=0, ocol0=0, bias=None, sigmask=0, rndmask=0, mode=STORE,
           res=None, gate=None, out2=None, a_grp=0):
    """The fp32 path's TF32 product (``gemm_tf32.cu``): acc = a[:, a_col0 : a_col0 + k] @ w[b_row0 : b_row0 + n, b_col0 : b_col0 + k]^T.

    ``a_grp`` > 0 makes it a grouped product: output columns [g a_grp, (g + 1) a_grp) read the A columns a_col0 + g k .. (several small
    products with their own A blocks and consecutive B row blocks in one launch -- the conditioning tables).

    STORE       out[:, ocol0 + j] = acc[:, j] (+ bias[b_row0 + j]), sigmoid on the 128-column blocks (b_row0 + j) // 128 set in sigmask,
                rounded to tf32 (RNA) on the blocks set in rndmask (columns that only feed later MMAs)
    RESID_GATE  out[:, j] = res[:, j] + gate[:, j] * acc[:, j]; out2 (optional) = acc
    SWIGLU      w is the interleaved [Wa_0; Wb_0; Wa_1; Wb_1] pack (n = 512): out [M, 256] = silu(a) * b; out2 (optional) = acc [M, 512]
    a, w, out, out2 are fp32 [rows, cols] with unit column stride (out allocated [M, n] (SWIGLU: [M, n / 2]) when None); res and gate
    fp32 [M, >= n] views. M is a multiple of 128, k of 32, n of 128 (SWIGLU: 512). The kernel rounds the A slices to tf32 (RNA) in shared
    memory before the MMAs; w must already be tf32-rounded (``round_tf32``): kind::tf32 truncates whatever it is given."""
    M = a.shape[0]
    k = a.shape[1] - a_col0 if k is None else k
    n = w.shape[0] - b_row0 if n is None else n
    d = _dev(a)
    # 256-column tiles halve the A traffic, but at small M they leave most SMs idle: 128 when the 256-wide tiling has fewer than one wave
    wide = mode == SWIGLU or (n % 256 == 0 and (M // 128) * (n // 256) >= nsm(d))
    nt = 256 if wide else 128
    assert M % 128 == 0 and k % 32 == 0 and n % nt == 0, (M, k, n, nt)
    assert mode != SWIGLU or n == 512
    assert a_grp == 0 or (a_grp % nt == 0 and n % a_grp == 0), (a_grp, nt, n)
    if out is None:
        out = torch.empty(M, n // 2 if mode == SWIGLU else n, device=a.device, dtype=F32)
    ma, mb = _tm32(a, (32, 128)), _tm32(w, (32, nt))
    mo = _tm32(out, (32, 32))
    mo2 = mo if out2 is None else _tm32(out2, (32, 32))
    tiles = (M // 128) * (n // nt)
    if mode == RESID_GATE and n == DM and a_col0 == 0 and b_row0 == 0 and b_col0 == 0 and ocol0 == 0 and bias is None:
        # the gated residual with res and gate by TMA (gemm_rg_tf32.cu): 144 -> ~60 us at 196608 rows
        atom_kernel32("gemm_rg", d)((min(nsm(d), tiles), 1, 1), (512, 1, 1), ma, mb, mo, mo2, _tm32(res, (32, 32)), _tm32(gate, (32, 32)),
                                    int(M), int(k), int(out2 is not None))
        return out
    atom_kernel32("gemm", d)((min(nsm(d), tiles), 1, 1), (512, 1, 1), ma, mb, mo, mo2, bias, res, gate, int(M), int(n), int(k), int(nt),
                             int(a_col0), int(b_row0), int(b_col0), int(ocol0), int(sigmask), int(mode), int(out2 is not None),
                             int(res.stride(0)) if res is not None else 0, int(gate.stride(0)) if gate is not None else 0, int(rndmask),
                             int(a_grp))
    return out


def round_tf32(t):
    """t (fp32) rounded to tf32, to nearest with ties away from zero (cvt.rna.tf32.f32): add half a tf32 ulp to the magnitude bits, drop
    the low 13. For the weight operands of gemm32, once per pack (kind::tf32 MMAs truncate)."""
    i = t.detach().float().contiguous().view(torch.int32)
    return ((i + 0x1000) & -0x2000).view(torch.float32)


def _rowgrid(M):
    return ((M + 7) // 8, 1, 1)


def _new(t, cols=DM):
    return torch.empty(t.shape[0], cols, device=t.device, dtype=F32)


def ln32(c, g1, g2, g3=None, eps=1e-5, out=None):
    """cn_k = LN(c) * g_k  (c [M, 128] fp32; g3 optional) -> (cn1, cn2, cn3 or None). With ``out`` (fp32 [M, 384] or, with g3, [M, 512],
    unit column stride): one table [cn1 | cn2 | c | cn3] instead (c copied), the operand of the grouped conditioning GEMM; returns its
    column views."""
    M = c.shape[0]
    if out is None:
        o1, o2 = _new(c), _new(c)
        o3 = _new(c) if g3 is not None else None
        oc, ld = None, DM
    else:
        assert out.dtype == F32 and out.stride(1) == 1 and out.shape == (M, (4 if g3 is not None else 3) * DM), (out.shape, out.stride())
        o1, o2, oc = out[:, :DM], out[:, DM:2 * DM], out[:, 2 * DM:3 * DM]
        o3 = out[:, 3 * DM:] if g3 is not None else None
        ld = out.stride(0)
    atom_kernel32("ln", _dev(c))(_rowgrid(M), (256, 1, 1), c, g1, o1, g2, o2, g3, o3, oc, int(ld), int(M), float(eps))
    return o1, o2, o3


def adaln32(a, scale, shift, eps=1e-5):
    """LN(a) * scale + shift: a [M, 128] fp32, scale / shift fp32 [M, 128] views (the stored sigmoid scale)."""
    out = _new(a)
    atom_kernel32("adaln", _dev(a))(_rowgrid(a.shape[0]), (256, 1, 1), a, scale, int(scale.stride(0)), shift, int(shift.stride(0)), out,
                                    int(a.shape[0]), float(eps))
    return out


def gate32(g, o):
    """sigmoid(g) * o: g a fp32 [M, 128] view, o [M, 128]."""
    out = _new(o)
    atom_kernel32("gate", _dev(o))(_rowgrid(o.shape[0]), (256, 1, 1), g, int(g.stride(0)), o, out, int(o.shape[0]))
    return out


def tail_bwd32(dy, t, st, dst):
    """-> dt = dy * st; dst (a view, written) = dy * t * st (1 - st)."""
    dt = _new(dy)
    atom_kernel32("tail_b", _dev(dy))(_rowgrid(dy.shape[0]), (256, 1, 1), dy, t, st, int(st.stride(0)), dt, dst, int(dst.stride(0)),
                                      int(dy.shape[0]))
    return dt


def swiglu_bwd32(dh, u):
    """dh [M, 256], u [M, 512] (interleaved [a_0 | b_0 | a_1 | b_1]) -> d[a | b] [M, 512] (same layout), h = silu(a) b [M, 256]."""
    dab, hh = _new(u, 4 * DM), _new(u, 2 * DM)
    atom_kernel32("swiglu_b", _dev(u))(_rowgrid(u.shape[0]), (256, 1, 1), dh, u, dab, hh, int(u.shape[0]))
    return dab, hh


def adaln_bwd32(a, scale, dx, dres, dscale, dshift, y=None, so=None, dso=None, eps=1e-5):
    """AdaLN backward: -> da = dres + LN backward of dx * scale; dscale (view) = dx LN(a) scale (1 - scale), dshift (view) = dx. With y:
    also d so (view) = da y so (1 - so) and the returned dyo = da so (the attention output gate upstream of a1 = a + so y)."""
    M = a.shape[0]
    da = _new(a)
    dyo = _new(a) if y is not None else None
    atom_kernel32("adaln_b", _dev(a))(_rowgrid(M), (256, 1, 1), a, scale, int(scale.stride(0)), dx, dres, da, dscale, int(dscale.stride(0)),
                                      dshift, int(dshift.stride(0)), y, so, int(so.stride(0)) if so is not None else 0, dso,
                                      int(dso.stride(0)) if dso is not None else 0, dyo, int(M), float(eps))
    return da, dyo


def gate_bwd32(dgated, g, o, dg, A, n):
    """dgated [M, 128], g a view, o [M, 128] -> dO [M, 128], D [A, 4, n]; dg (view, written) = dgated o s (1 - s)."""
    M = o.shape[0]
    do = _new(o)
    D = torch.empty(A, NH, n, device=o.device, dtype=F32)
    atom_kernel32("gate_b", _dev(o))(_rowgrid(M), (256, 1, 1), dgated, g, int(g.stride(0)), o, do, dg, int(dg.stride(0)), D, int(M), int(n))
    return do, D


def colsum32(x):
    """The column sums of a fp32 [M, 128 g] view (unit column stride, any row pitch) -> [128 g] fp32: one launch writing G partial rows (a
    persistent grid, deterministic) and a tiny sum of those."""
    M, cols = x.shape
    assert x.dtype == F32 and x.stride(1) == 1 and cols % DM == 0, (x.dtype, x.shape, x.stride())
    d = _dev(x)
    g = max(1, min(2 * nsm(d), (M + 7) // 8))
    part = torch.empty(g, cols, device=x.device, dtype=F32)
    atom_kernel32("colsum", d)((g, cols // DM, 1), (256, 1, 1), x, int(x.stride(0)), part, int(cols), int(M))
    return part.sum(0)


def ln_bwd32(c, dc, dcn1, g1, dcn2, g2, dcn3=None, g3=None, eps=1e-5):
    """dc [M, 128] += the LayerNorm backward of dcn_k * g_k (k = 1, 2, 3); -> dg [384] fp32 = [dg1 | dg2 | dg3] (sum_rows dcn_k LN(c))."""
    M = c.shape[0]
    d = _dev(c)
    dg = torch.zeros(3 * DM, device=c.device, dtype=F32)
    atom_kernel32("ln_b", d)((max(1, min(4 * nsm(d), (M + 7) // 8)), 1, 1), (256, 1, 1), c, dc, dcn1, g1, dcn2, g2, dcn3, g3, dg, int(M),
                             float(eps))
    return dg


# ------------------------------------------------------------------------------------------------------------ weight gradients (one launch)
import ctypes  # noqa: E402

_DW_MAXP, _DW_MAXO = 12, 24
_DW_NS, _DW_STG = 5, 40 * 1024


class _DwTab(ctypes.Structure):
    _fields_ = [("np", ctypes.c_int), ("mb", ctypes.c_int * _DW_MAXP), ("nn", ctypes.c_int * _DW_MAXP), ("ns", ctypes.c_int * _DW_MAXP),
                ("outbase", ctypes.c_int * _DW_MAXP), ("wsoff", ctypes.c_longlong * _DW_MAXP), ("e4off", ctypes.c_longlong * _DW_MAXP),
                ("e4total", ctypes.c_longlong)]


class _DwOut(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p * _DW_MAXO), ("bf16", ctypes.c_int * _DW_MAXO)]


class _DwMaps(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * (2 * _DW_MAXP * 128))]


class AtomDw:
    """The block backward's weight gradients D_p = A_p^T B_p (A_p [R, 128 mb_p], B_p [R, nn_p], column views of the activations / gradients) in
    one split-K launch (``dw_atom.cu``) and one reduction, written straight to the weights' own gradient buffers in their dtypes. bf16 operands
    (``f32`` False) or fp32 operands on the TF32 tensor cores. A CTA takes a row slice of one problem; the CTAs go to the problems in
    proportion to their bytes."""

    def __init__(self, device_index: int, f32: bool):
        from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

        self.f32, self.dev = f32, device_index
        bkr, ns = (32, 2) if f32 else (16, 5)
        defs = (f"DW_BKR={bkr}", f"NS={ns}", "DW_F32") if f32 else (f"NS={ns}",)
        cu = cubin_tf32("dw_atom", defs)
        stg = 20 * bkr * 128 if f32 else 40 * 1024
        with torch.cuda.device(device_index):
            self.kd = driver.Kernel(cu, "atom_dw", ns * stg + 256, pdl=False)
            self.kr = driver.Kernel(cu, "atom_dw_reduce", 0, pdl=False)

    def __call__(self, problems, R: int) -> None:
        """problems: list of (A, B, dests) -- A [R, 128 mb], B [R, nn] (unit column stride views of one dtype), dests the mb contiguous
        [128, nn] destination tensors (fp32 or bf16) of the 128-row blocks of D."""
        from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

        assert 0 < len(problems) <= _DW_MAXP
        f32 = self.f32
        dt = F32 if f32 else torch.bfloat16
        esz = 4 if f32 else 2
        box, sw, dname = ([32, 32], "128a32", "f32") if f32 else ([64, 32], 128, "bf16")
        nsm_ = nsm(self.dev)
        tab, out, raw = _DwTab(), _DwOut(), bytearray(2 * _DW_MAXP * 128)
        weights = [A.shape[1] + B.shape[1] for A, B, _ in problems]
        total = sum(weights)
        keep, wsoff, e4 = [], 0, 0
        nouts = 0
        tab.np = len(problems)
        for p, (A, B, dests) in enumerate(problems):
            assert A.dtype == dt and B.dtype == dt and A.shape[0] == R and B.shape[0] == R and A.shape[1] % 128 == 0
            assert A.stride(1) == 1 and B.stride(1) == 1 and B.shape[1] in (128, 256) and len(dests) == A.shape[1] // 128
            for t, slot in ((A, p), (B, _DW_MAXP + p)):
                m = driver.TensorMap(t, [t.shape[1], R], t.stride(0) * esz, box, swizzle=sw, dtype=dname)
                keep.append(m)
                raw[slot * 128:(slot + 1) * 128] = ctypes.string_at(m.addr, 128)
            mb, nn = A.shape[1] // 128, B.shape[1]
            tab.mb[p], tab.nn[p] = mb, nn
            tab.ns[p] = max(1, round(nsm_ * weights[p] / total))
            tab.wsoff[p], tab.e4off[p], tab.outbase[p] = wsoff, e4, nouts
            wsoff += tab.ns[p] * mb * 128 * nn
            e4 += mb * 128 * nn // 4
            for d in dests:
                assert d.is_contiguous() and d.shape == (128, nn) and d.dtype in (F32, torch.bfloat16), (d.shape, d.dtype)
                out.ptr[nouts], out.bf16[nouts] = d.data_ptr(), int(d.dtype == torch.bfloat16)
                nouts += 1
        assert nouts <= _DW_MAXO
        tab.e4total = e4
        maps = _DwMaps()
        ctypes.memmove(ctypes.addressof(maps), bytes(raw), len(raw))
        dev = problems[0][0].device
        ws = torch.empty(wsoff, device=dev, dtype=F32)
        ctas = sum(tab.ns[p] for p in range(tab.np))
        self.kd((ctas, 1, 1), (256, 1, 1), maps, tab, ws, int(R))
        self.kr(((e4 + 255) // 256, 1, 1), (256, 1, 1), ws, tab, out)
        self._keep = (keep, ws, problems)


@functools.lru_cache(maxsize=None)
def atom_dw(device_index: int, f32: bool) -> AtomDw:
    return AtomDw(device_index, f32)

