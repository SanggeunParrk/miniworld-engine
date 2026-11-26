"""B200-only (sm_100a, card ``b200_sm100``) row kernels of the AF3 atom transformer block (``kernels/integrations/local_dit_b200``),
forward and backward: the tcgen05 / TMEM / TMA kernels of ``origin/b200/ladder-final`` (dc425caa5, ``sm100_atom``), kept in their own
package so that ``sm100_atom`` (whose sources also carry the H100 ``-DATOM_HOPPER`` half the H100 build compiles) stays untouched.
The attention and the pair bias of that block are ``b200_atom_local``'s. Nothing here is built or loaded on another card.

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

On sm_90a (H100) the same sources carry a second half under ``-DATOM_HOPPER=1`` (``*_sm90`` kernels, wgmma, bf16; ``_HOPPER_KERNELS``):
``atom_kernel`` / ``load_all`` and the launchers below pick it by ``_hopper(device)``; the fp32 (TF32) path is sm_100a only.

The fp32 path (fp32 activations, ``KERNELS_TF32``, loaded by ``atom_kernel32`` only when an fp32 call is served; used by the local
atom block, ``integrations/local_dit``):
  gemm_tf32.cu    every projection and activation-gradient product as one tcgen05 kind::tf32 GEMM launch with the block's epilogues
                  (bias + sigmoid blocks, gated residual, SwiGLU).
  rows_tf32.cu    the row-wise stages around them (LayerNorms, AdaLN and their backward, gates, SwiGLU backward, D), fp32 CUDA cores.

The cubins are built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``:
13.1 builds; 12.9's ptxas gives slightly longer code) and cached under
``settings.engine_jit_root`` keyed by the sources and the flags. They launch through the CUDA driver
(``augmented_attention.cuda.sm100.driver``) on torch's current stream, CUDA-graph capturable.
"""

from __future__ import annotations

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)

import functools
from pathlib import Path

import torch

from foldjet_engine.kernels._device import num_sms
from foldjet_engine.kernels._tf32 import tf32_grid

NH, DH, DM, DP = 4, 32, 128, 16
_dir = Path(__file__).parent
_SMEM = 232448
SOURCES = (
    "cond_fwd",
    "pre_fwd",
    "post_fwd",
    "tr_bwd_gate",
    "post_bwd",
    "pre_bwd",
    "cond_bwd",
)


# --------------------------------------------------------------------------------------------------- build
@functools.lru_cache(maxsize=None)
def cubin(stem: str, defs: tuple = (), hopper: bool = False) -> str:
    """Path of ``<stem>.cu`` built for sm_100a (``hopper``: sm_90a with ``-DATOM_HOPPER=1``, the sources' Hopper half) with the sources'
    default options (+ ``-D`` ``defs``); rebuilt only when a source or a flag changes."""
    from foldjet_engine.kernels import _nvcc

    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_90a" if hopper else "-arch=sm_100a",
        *(("-DATOM_HOPPER=1",) if hopper else ()),
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
        *("-D" + d for d in defs),
    )
    family = "local_dit_sm90" if hopper else "local_dit_b200"
    return str(_nvcc.cubin(f"{family}/{stem}", flags, source=_dir / f"{stem}.cu", toolchain="torch" if hopper else "sm100"))


@functools.lru_cache(maxsize=None)
def _load_atom_kernel(
    stem: str,
    func: str,
    device_index: int,
    smem: int = _SMEM,
    pdl: bool = False,
    defs: tuple = (),
):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, defs, _hopper(device_index)), func, smem, pdl=pdl)


#: (stem, function, dynamic smem, PDL[, -D defs]) of every kernel the block launches
KERNELS = {
    "cond": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True),
    "cond_kv": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True, ("KV=1",)),  # + the K / V AdaLN's raw tables (inference: one launch)
    "pre": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True),
    "prex": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1",)),
    "prexp": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1")),
    # round 3 part 12: + the windowed pair bias made by the compute warps before their first tile (``-DEB=1``, inference; ``eb=``)
    "prex_eb": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "EB=1")),
    "prexp_eb": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "EB=1")),
    "post":("post_fwd", "atom_post_fwd_sm100", _SMEM, True),
    "trg": ("tr_bwd_gate", "atom_tr_bwd_gate_sm100", _SMEM, True),
    "postb": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True),
    "preb": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True),
    "prebx": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True, ("CROSS=1",)),
    "prebxp": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1")),
    "condb": ("cond_bwd", "atom_cond_bwd_sm100", _SMEM, True),
}

#: The B200 variant path's forward builds (Boltz-2 / RoseTTAFold3 / Chai-1 on ``integrations/local_dit_b200``, 2026-11-17), loaded apart
#: (``atom_kernel``): ``-DIDENT=1`` Chai-1's identity AdaLN tables, ``-DKQLN=1`` RoseTTAFold3's flat q / k LayerNorm, ``-DPARS=1`` the parallel
#: schedule (RoseTTAFold3, Chai-1), ``-DUP=1`` Boltz-2's up-gated SwiGLU
KERNELS_VAR = {
    "cond_id": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True, ("IDENT=1", "PF=1")),
    "cond_pf": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True, ("PF=1",)),  # (round 3b: PF = the first input tile prefetched into L2)
    "pre_kq": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("KQLN=1",)),
    "post_pars": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True, ("PARS=1",)),
    "post_up": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True, ("UP=1",)),
    # round 3b: ADA1 = the AdaLN outputs x1 / x2 with one rounding (the bf16-autocast reference's), RES1 = the residual updates as fp32
    # multiply-adds with one rounding each; PF = the first input tile prefetched into L2
    "pre_a1": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("ADA1=1", "PF=1")),
    "pre_kq_a1": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("KQLN=1", "ADA1=1", "PF=1")),
    "pre_a1_eb": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("ADA1=1", "PF=1", "EB=1")),  # (round 3 part 12: + the pair bias, ``eb=``)
    "pre_kq_a1_eb": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("KQLN=1", "ADA1=1", "PF=1", "EB=1")),
    "post_pars_a1": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True, ("PARS=1", "ADA1=1", "RES1=1")),
    "post_up_a1": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True, ("UP=1", "ADA1=1", "RES1=1")),
    # ... and their training builds: the backward row kernels with the same options
    "postb_pars": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True, ("PARS=1",)),
    "postb_pars_id": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True, ("PARS=1", "IDENT=1")),
    "preb_id": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True, ("IDENT=1",)),
    "condb_id": ("cond_bwd", "atom_cond_bwd_sm100", _SMEM, True, ("IDENT=1",)),
    "trg_up": ("tr_bwd_gate", "atom_tr_bwd_gate_sm100", _SMEM, True, ("UP=1",)),
    "postb_up": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True, ("UP=1",)),
}

#: ESMFold2's builds (``-DRMS=1``: RMSNorm AdaLNs, tables of silu(c), raw gates; inference), loaded apart from KERNELS (``load_esm``)
KERNELS_ESM = {
    "cond_rms": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True, ("RMS=1",)),
    "pre_rms": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "RMS=1")),
    "post_rms": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True, ("RMS=1",)),
    # inference: the AdaLNs in fp32 with one rounding and the raw scale tables (the reference module's arithmetic; ``raw=True``); round 3
    # part 4: + the q / k RMSNorm and rotary on the staged Q / K rows (``rope``; local_esm_qkrope's launch removed from inference)
    "pre_rms_raw_rope": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "RMS=1", "RAWSC=1", "QKROPE=1")),
    "post_rms_raw": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True, ("RMS=1", "RAWSC=1", "RES3=1")),
}
#: ... and the backward builds of ESMFold2's training block (``-DRMS=1``: RMSNorm backward, raw scales and gates), loaded apart (``KERNELS_ESM_TRAIN``)
KERNELS_ESM_TRAIN = {
    "trg_rms": ("tr_bwd_gate", "atom_tr_bwd_gate_sm100", _SMEM, True, ("RMS=1",)),
    "postb_rms": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True, ("RMS=1",)),
    "prebxp_rms": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "RMS=1")),
    # round 3 part 7 (2026-11-17): the AdaLNs with the raw scale tables (``-DRAWSC=1``: (1 + s) in fp32, one rounding), forward and backward
    "pre_rms_raw": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "RMS=1", "RAWSC=1")),
    "postb_rms_raw": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True, ("RMS=1", "RAWSC=1")),
    "prebxp_rms_raw": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "RMS=1", "RAWSC=1")),
    # ... and the projection kernel with the q / k norm + rotary that also saves the raw q | k (``-DQKSAVE=1``: local_esm_qkrope's launch removed)
    "pre_rms_raw_rope_save": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True, ("CROSS=1", "PARALLEL=1", "RMS=1", "RAWSC=1", "QKROPE=1", "QKSAVE=1")),
}


@functools.lru_cache(maxsize=8)
def _hopper(device_index: int) -> bool:  # noqa: ARG001  -- B200-only package: never the Hopper half
    return False


#: (stem, function, dynamic smem, PDL) of every kernel the block launches on sm_90a (the ``ATOM_HOPPER`` half of the same sources)
_HOPPER_KERNELS = {
    "cond": ("cond_fwd", "atom_cond_fwd_sm90", 131136, False),
    "cond1": ("cond_fwd", "atom_cond_fwd_single_sm90", 114752, False),
    "pre": ("pre_fwd", "atom_pre_fwd_sm90", 180288, False),
    "pre_cross": ("pre_fwd", "atom_pre_cross_fwd_sm90", 114752, False),
    "post": ("post_fwd", "atom_post_fwd_sm90", 147520, False),
    "post2": ("post_fwd", "atom_post_fwd_2stage_sm90", 114752, False),
    "post2_packed": ("post_fwd", "atom_post_fwd_2stage_packed_sm90", 114752, False),
    "pack": ("cond_fwd", "atom_pack_sm90", 0, False),
    "pack_t": ("tr_bwd_gate", "atom_pack_transpose_sm90", 0, False),
    "trg": ("tr_bwd_gate", "atom_tr_bwd_sm90", 131136, False),
    "postb": ("post_bwd", "atom_post_bwd_sm90", 98368, False),
    "preb": ("pre_bwd", "atom_pre_bwd_sm90", 98368, False),
    "condb": ("cond_bwd", "atom_cond_bwd_sm90", 168000, False),
    "grad_pack": ("pre_bwd", "atom_grad_pack_sm90", 0, False),
    "col_part": ("cond_bwd", "atom_column_partials_sm90", 0, False),
    "col_fin": ("cond_bwd", "atom_column_finish_sm90", 0, False),
}


def atom_kernel(name: str, device_index: int):
    if _hopper(device_index) and name not in _HOPPER_KERNELS:
        raise NotImplementedError(f"Atom Hopper kernel {name} is not implemented")
    stem, func, smem, pdl, *defs = (
        _HOPPER_KERNELS
        if _hopper(device_index)
        else KERNELS_ESM
        if name in KERNELS_ESM
        else KERNELS_ESM_TRAIN
        if name in KERNELS_ESM_TRAIN
        else KERNELS_VAR
        if name in KERNELS_VAR
        else KERNELS
    )[name]
    return _load_atom_kernel(stem, func, device_index, smem, pdl, *defs)


def load_all(device_index: int) -> None:
    """Build and load every kernel on the device (raises on any failure)."""
    for name in _HOPPER_KERNELS if _hopper(device_index) else KERNELS:
        atom_kernel(name, device_index)


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


# --------------------------------------------------------------------------------------------------- tensor maps
def _tm(t, rows, box_rows, cols=DM, box_cols=DH, swizzle=128):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, [cols, rows], cols * 2, [box_cols, box_rows], swizzle=swizzle)


def _map(t, dims, stride, box, swizzle=128, dtype="bf16"):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, dims, stride, box, swizzle=swizzle, dtype=dtype)


def _rows(t, cols=DM):
    """Row-major bf16 [M, cols] (rows t.stride(0) apart: a column view of a wider tensor works) as 32-row x 64-column boxes (128-B swizzle)."""
    assert t.stride(-1) == 1
    return _map(t, [cols, t.shape[0]], t.stride(0) * 2, [64, 32])


def _pre_maps(a, W, mod, Q, K, V, G, X, mkv, XK):
    """The ten tensor maps of the sm_100a pre_fwd builds."""
    return (
        _rows(a),
        _map(W, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64),
        _rows(mod, 6 * DM),
        *(_rows(t) for t in (Q, K, V, G, X)),
        _rows(mkv, 2 * DM) if mkv is not None else _rows(G),
        _rows(XK),
    )


def _rows16(t, cols=DM):
    assert t.stride(-1) == 1
    return _map(t, [cols, t.shape[0]], t.stride(0) * 2, [64, 16])


# --------------------------------------------------------------------------------------------------- weights
def cond_params(blk):
    """The six conditioning projections stacked for cond_fwd: Wmod [768, 128] bf16 = [Wsc1; Wbi1; Wos; Wsc2; Wbi2; Wts],
    bias [768] fp32 (zero for the shifts), the two cond LayerNorm weights g1, g2 [128] fp32."""
    att, tr = blk.attention, blk.transition
    a1, a2 = att.ada_ln_in, tr.ada_ln_in
    W = torch.cat(
        [
            a1.to_scale.weight,
            a1.to_bias.weight,
            att.to_scale.weight,
            a2.to_scale.weight,
            a2.to_bias.weight,
            tr.to_scale.weight,
        ]
    )
    z = torch.zeros(DM, device=W.device)
    b = torch.cat(
        [
            a1.to_scale.bias.float(),
            z,
            att.to_scale.bias.float(),
            a2.to_scale.bias.float(),
            z,
            tr.to_scale.bias.float(),
        ]
    )
    return (
        W.to(torch.bfloat16).contiguous(),
        b.contiguous(),
        a1.ln_cond.weight.float().contiguous(),
        a2.ln_cond.weight.float().contiguous(),
    )


def pre_params(blk):
    """The attention's four input projections stacked for pre_fwd: W [512, 128] bf16 = [Wq; Wk; Wv; Wg], bq [128] fp32."""
    att = blk.attention
    W = torch.cat(
        [
            att.to_query.weight,
            att.to_key.weight,
            att.to_value.weight,
            att.to_gate.weight,
        ]
    )
    return W.to(torch.bfloat16).contiguous(), att.to_query.bias.float().contiguous()


def post_params(blk):
    """post_fwd's weights: Wo [128, 128], Wu = [Wa; Wb] [512, 128], Ws [128, 256] (bf16)."""
    att, tr = blk.attention, blk.transition

    def bf(t):
        return t.to(torch.bfloat16).contiguous()

    return (
        bf(att.to_out.weight),
        bf(torch.cat([tr.expand_a.weight, tr.expand_b.weight])),
        bf(tr.squeeze.weight),
    )


# --------------------------------------------------------------------------------------------------- launches
def cond_fwd(c, W, b, g1, g2, eps=1e-5, *, hopper_waves=2.0, rms=False, g3=None, ident=False, pf=None):
    """c [M, 128] bf16 -> mod [M, 768] bf16 = [s1 | bi1 | so | s2 | bi2 | st] (the gates as rn(sigmoid)).
    ``ident`` (Chai-1, B200 variant path): the identity AdaLN's tables of the raw c (no LayerNorm; the scales rn(c W^T + b) raw, b = 1).
    ``rms`` (ESMFold2, sm_100a): every table from rn(silu(c)), no sigmoid; W [1152, 128] = [Wmod; Wkv_scale; Wkv_shift; 0], b [1152] (the
    scales' bias 1) -> (mod, mkv [M, 256] = the K / V AdaLN's [scale | shift]).
    ``g3`` (the chained / parallel K / V AdaLN's cond LayerNorm weight, fp32 [128]; sm_100a, the ``-DKV=1`` build): W [1152, 128] =
    [Wmod; Wkv_scale; Wkv_shift; 0], b [1152] (zero past 768) -> (mod, mkv [M, 256] = rn(rn(LN_g3(c)) [Wkv_scale; Wkv_shift]^T), raw) in the
    one launch."""
    M = c.shape[0]
    out = torch.empty(M, 6 * DM, device=c.device, dtype=torch.bfloat16)
    if rms or g3 is not None:
        assert tuple(W.shape) == (9 * DM, DM) and b.numel() == 9 * DM, (W.shape, b.shape)
        mkv = torch.empty(M, 2 * DM, device=c.device, dtype=torch.bfloat16)
        ntile = M // 32
        d = _dev(c)
        gxc = min(num_sms(d) // 3, ntile)
        common = (_rows(c), _map(W, [DM, 9 * DM], DM * 2, [32, 128], swizzle=64), _rows(out, 6 * DM), _rows(mkv, 2 * DM), g1, g2, b, int(ntile), float(eps))
        atom_kernel("cond_rms" if rms else "cond_kv", d)((gxc, 3, 1), (384, 1, 1), *common, *(() if rms else (g3,)))
        return out, mkv
    if _hopper(_dev(c)):
        d = _dev(c)
        maps = (
            _map(c, [DM, M], DM * 2, [64, 64]),
            _map(W, [DM, 6 * DM], DM * 2, [64, 128]),
        )
        wide = M > 64 * num_sms(d)
        waves = float(hopper_waves) if wide else 1.0
        if waves not in (0.5, 1.0, 2.0, 4.0):
            raise ValueError("Hopper conditioning waves must be 0.5, 1, 2 or 4")
        atom_kernel("cond1" if wide else "cond", d)(
            (min(int(waves * num_sms(d)), (M + 63) // 64), 2, 1),
            (160, 1, 1),
            *maps,
            g1,
            g2,
            b,
            out,
            int(M),
            float(eps),
        )
        return out
    maps = (
        _rows(c),
        _map(W, [DM, 6 * DM], DM * 2, [32, 128], swizzle=64),
        _rows(out, 6 * DM),
        _rows(out, 6 * DM),
    )
    ntile = M // 32
    d = _dev(c)
    # pf (B200 variant path, round 3b): True / False = the PF build with / without its L2 prefetch of c; None = the plain build
    atom_kernel("cond_id" if ident else "cond" if pf is None else "cond_pf", d)(
        (min(num_sms(d) // 2, ntile), 2, 1),
        (384, 1, 1),
        *maps,
        g1,
        g2,
        b,
        int(ntile),
        float(eps),
        *(((c if pf else None),) if (ident or pf is not None) else ()),
    )
    return out


def pre_fwd(a, mod, W, bq, save=False, eps=1e-5, cross=False, ada1=False, eb=None):
    """a [M, 128], mod [M, 768] -> Q, K, V, G [M, 128] bf16 (+ x1 = AdaLN 1 output when ``save``). ``cross`` (sm_90a only): the cross-attention
    mode's variant of the Hopper kernel."""
    return _pre_fwd(a, mod, W, bq, save, eps, cross, None, 1e-5, ada1, eb)[:5]


def pre_fwd_kq(a, mod, W, bq, kq, kq_eps=1e-5, save=False, eps=1e-5, ada1=False, eb=None):
    """``pre_fwd`` with RoseTTAFold3's flat q / k LayerNorm (B200 variant path): ``kq`` fp32 [4, 128] = [wq, bq, wk, bk] of the LayerNorm
    (eps ``kq_eps``) applied to Q and K. -> Q, K, V, G, X and QKR [M, 256] bf16, the raw projections q, k (written when ``save``)."""
    q, k, v, g, x, qkr = _pre_fwd(a, mod, W, bq, save, eps, False, kq, kq_eps, ada1, eb)
    assert qkr is not None
    return q, k, v, g, x, qkr


def _pre_fwd(a, mod, W, bq, save, eps, cross, kq, kq_eps, ada1=False, eb=None):
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    X = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    if _hopper(_dev(a)):
        d = _dev(a)
        maps = (
            _map(a, [DM, M], DM * 2, [64, 64]),
            _map(mod, [6 * DM, M], 12 * DM, [64, 64]),
            _map(W, [DM, 4 * DM], DM * 2, [64, 128]),
        )
        atom_kernel("pre_cross" if cross else "pre", d)(
            (min((2 if cross else 1) * num_sms(d), (M + 63) // 64), 1, 1),
            (160, 1, 1),
            *maps,
            bq,
            Q,
            K,
            V,
            G,
            X,
            int(M),
            float(eps),
            int(save),
        )
        return Q, K, V, G, X, None
    assert not cross, "cross is the Hopper kernel's mode; sm_100a uses pre_fwd_kv"
    maps = _pre_maps(a, W, mod, Q, K, V, G, X, None, G)
    ntile = M // 32
    d = _dev(a)
    extra = ()
    if kq is not None:
        qkr = torch.empty(M if save else 0, 2 * DM, device=a.device, dtype=torch.bfloat16)
        extra = (kq, float(kq_eps), qkr)
    assert eb is None or ada1, "eb: the ADA1 builds (pre_a1_eb / pre_kq_a1_eb)"
    atom_kernel(("pre_kq" if kq is not None else "pre") + ("_a1" if ada1 else "") + ("_eb" if eb is not None else ""), d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        bq,
        bq,
        int(ntile),
        float(eps),
        int(save),
        int(mod.shape[0]),
        *extra,
        *((a,) if ada1 else ()),  # (the ADA1 builds are PF builds: the input tile prefetched into L2)
        *(eb if eb is not None else ()),
    )
    return Q, K, V, G, X, (extra[2] if kq is not None else None)


def pre_fwd_kv(a, mod, W, bq, mkv, bs, save=False, eps=1e-5, parallel=False, rms=False, raw=False, rope=None, eb=None):
    # (every sm_100a row kernel reads the conditioning tables mod / mkv at row r % rows(mod): per-row tables [M, ..], or [N, ..] shared
    # by the samples)
    """The cross-attention mode's ``pre_fwd``: K and V from xkv = LN(x1) sigmoid(mkv[:, :128] + bs) + mkv[:, 128:] (a second AdaLN of x1)
    instead of from x1; ``parallel`` (AlphaFold 3): of the raw atoms, xkv = LN(a) sigmoid(..) + .. -> Q, K, V, G [M, 128] bf16 and, when
    ``save``, x1 and xkv (else X = G, XK = None). ``rms`` (ESMFold2, with ``parallel``): RMSNorm statistics, the K / V scale raw.
    ``rope`` = (wq, wk, cos, sin, eps) (ESMFold2 inference, with ``rms`` and ``raw``; round 3 part 4): Q and K leave RMS-normalised per head
    (weights fp32 [32]) and rotated (tables fp32 [n, 16], row r is atom r % n), as ``esm.esm_qkrope`` would make them (the ``-DQKROPE=1`` build)."""
    assert not rms or parallel
    assert rope is None or (rms and raw), "rope: an ESMFold2 build with the raw scale tables"
    assert not raw or rms, "raw: an ESMFold2 (RMS) build"
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    X = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    XK = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    maps = _pre_maps(a, W, mod, Q, K, V, G, X, mkv, XK)
    ntile = M // 32
    d = _dev(a)
    extra = ()
    if rope is not None:
        wq, wk, cs, sn, qeps = rope
        assert cs.dtype == sn.dtype == wq.dtype == wk.dtype == torch.float32 and cs.is_contiguous() and sn.is_contiguous() and cs.shape[1] == 16
        extra = (wq.contiguous(), wk.contiguous(), cs, sn, int(cs.shape[0]), float(qeps))
    qkr = None
    if rope is not None and save:  # (training, round 3 part 7: Q / K leave normalised and rotated, the raw q | k go to QKR [M, 256])
        qkr = torch.empty(M, 2 * DM, device=a.device, dtype=torch.bfloat16)
        extra = (*extra, qkr)
    name = (("pre_rms_raw_rope_save" if qkr is not None else "pre_rms_raw_rope" if rope is not None else "pre_rms_raw") if raw else "pre_rms") if rms else "prexp" if parallel else "prex"
    if eb is not None:  # (round 3 part 12: the pair bias in this launch -- the prex / prexp builds)
        assert name in ("prex", "prexp"), "eb: the prex_eb / prexp_eb builds"
        name += "_eb"
    atom_kernel(name, d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        bq,
        bs.float().contiguous(),
        int(ntile),
        float(eps),
        int(save),
        int(mod.shape[0]),
        *extra,
        *(eb if eb is not None else ()),
    )
    if qkr is not None:
        return Q, K, V, G, X, XK, qkr
    return Q, K, V, G, X, (XK if save else None)


def post_fwd(a, o, g, mod, Wo, Wu, Wd, save=False, eps=1e-5, *, packed_hopper=False, rms=False, raw=False, pars=False, wup=None, ada1=False):
    """a, o, g [M, 128], mod [M, 768] -> the block output [M, 128] bf16 (+ u, a2, x2, t when ``save``). B200 variant path: ``pars`` the
    parallel schedule (the transition normalises a), ``wup`` [256, 128] bf16 Boltz-2's up-gate weight."""
    M = a.shape[0]

    def new():
        return torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)

    out = new()
    sv = tuple(new() for _ in range(4)) if save else (out,) * 4
    if _hopper(_dev(a)):
        d = _dev(a)
        maps = (
            _map(Wo, [DM, DM], DM * 2, [64, 128]),
            _map(Wu, [DM, 4 * DM], DM * 2, [64, 128]),
            _map(Wd, [2 * DM, DM], DM * 4, [64, 128]),
        )
        wide = M > 64 * num_sms(d)
        packed = wide and packed_hopper
        name = "post2_packed" if packed else "post2" if wide else "post"
        atom_kernel(name, d)(
            (min((2 if wide else 1) * num_sms(d), (M + 63) // 64), 1, 1),
            (160, 1, 1),
            a,
            o,
            g,
            mod,
            *maps,
            out,
            *sv,
            int(M),
            float(eps),
            int(save),
        )
        return (out, *sv)
    import os

    box3 = os.environ.get("MW_LD10_B3", "1") != "0"  # (round 3 part 10: 3-D tensor maps, fewer TMA boxes; BOX3 builds)
    if box3:
        r3 = lambda t, cols=DM: _map(t, [64, t.shape[0], cols // 64], [t.stride(0) * 2, 128], [64, 16, 2 if cols == DM else 8])  # noqa: E731
        maps = (
            r3(a),
            r3(o),
            r3(g),
            r3(mod, 6 * DM),
            _map(Wo, [64, DM, 2], [DM * 2, 128], [64, 128, 2]),
            _map(Wu, [32, 4 * DM, 4], [DM * 2, 64], [32, 128, 4], swizzle=64),
            _map(Wd, [32, DM, 8], [4 * DM, 64], [32, 128, 4], swizzle=64),
            r3(out),
            *(r3(t) for t in sv),
        )
    else:
        maps = (
            _rows16(a),
            _rows16(o),
            _rows16(g),
            _rows16(mod, 6 * DM),
            _map(Wo, [DM, DM], DM * 2, [64, 128]),
            _map(Wu, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64),
            _map(Wd, [2 * DM, DM], 4 * DM, [32, 128], swizzle=64),
            _rows16(out),
            *(_rows16(t) for t in sv),
        )
    ntile = M // 16
    d = _dev(a)
    assert not (pars and wup is not None), "PARS and UP: no model has both (no build)"
    assert not (rms and (pars or wup is not None))
    name = ("post_rms_raw" if raw else "post_rms") if rms else "post_up" if wup is not None else "post_pars" if pars else "post"
    if ada1:  # (B200 variant path, round 3b: the AdaLN output and the residual updates with one rounding each)
        assert name in ("post_up", "post_pars"), name
        name += "_a1"
    # (round 3 part 10, MW_LD10_PG) at most two tiles a CTA when they fit: the two warpgroups each take one, and half as many CTAs wait for
    # an SM the attention frees (the trace: post CTAs that start after the attention's end spend ~3.5 us in setup + weight staging)
    import os

    # (measured: 2048 atoms 43.0 -> 38.9-41.0 us on Protenix / RoseTTAFold3 / Boltz-2; at <= 64 tiles level or slower -- ESMFold2Pair 1024
    # 38.9 -> 41.0 -- so only above half the SMs)
    half = os.environ.get("MW_LD10_PG", "1") != "0" and 2 * ntile > num_sms(d)
    grid = (ntile + 1) // 2 if half else min(num_sms(d), ntile)
    (_box3_kernel if box3 else atom_kernel)(name, d)(
        (min(num_sms(d), grid), 1, 1),
        (384, 1, 1),
        *maps,
        int(ntile),
        float(eps),
        int(save),
        int(mod.shape[0]),
        *((_map(wup, [DM, 2 * DM], DM * 2, [64, 128]),) if wup is not None else ()),
    )
    return (out, *sv)


def _tsum(part, which, mod, M):
    """The table-sum arguments of a backward row kernel (``tsum.cuh``): with tables shared by the samples (mod has fewer rows than M) the
    kernel sums their gradient over the samples into ``part`` (a ``sm100_atom_local.TableSums``) instead of storing it per sample row."""
    if mod.shape[0] == M:
        return (None, 0)
    if part is None:
        raise ValueError("shared tables (fewer rows than the activations) need the table-sum partials (sm100_atom_local.TableSums)")
    return (getattr(part, which), int(part.J))


def tr_bwd_gate(dy, t, mod, x2, Wu, WsT, dmod, dbts, rms=False, part=None, wup=None):
    """dy, t, mod (st), x2 -> DT [M, 128], HH [M, 256], DAB [M, 512]; dts into dmod[:, 640:768]; d bts += the column sums. ``rms`` (ESMFold2):
    the gate st is raw (no sigmoid backward). dmod may be a column view of a wider tensor (rows dmod.stride(0) apart). ``wup`` (Boltz-2,
    B200 variant path): the up-gate weight [256, 128] bf16; DAB is then [M, 768] = [da | db | du]."""
    M = dy.shape[0]
    dev = dy.device
    DT = torch.empty(M, DM, device=dev, dtype=torch.bfloat16)
    HH = torch.empty(M, 2 * DM, device=dev, dtype=torch.bfloat16)
    wab = 6 * DM if wup is not None else 4 * DM
    DAB = torch.empty(M, wab, device=dev, dtype=torch.bfloat16)
    if _hopper(_dev(dy)):
        d = _dev(dy)
        maps = (
            _map(Wu, [DM, 4 * DM], DM * 2, [64, 128]),
            _map(WsT, [DM, 2 * DM], DM * 2, [64, 128]),
        )
        atom_kernel("trg", d)(
            (min(num_sms(d), (M + 63) // 64), 1, 1),
            (160, 1, 1),
            dy,
            t,
            mod,
            x2,
            *maps,
            DT,
            HH,
            DAB,
            dmod,
            int(M),
        )
        return DT, HH, DAB
    maps = (
        _rows16(dy),
        _rows16(t),
        _rows16(mod, 6 * DM),
        _rows16(x2),
        _map(Wu, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64),
        _map(WsT, [DM, 2 * DM], DM * 2, [32, 128], swizzle=64),
        _rows16(DT),
        _rows16(dmod, 6 * DM),
        _rows16(HH, 2 * DM),
        _rows16(DAB, wab),
    )
    ntile = M // 16
    d = _dev(dy)
    extra = (_map(wup, [DM, 2 * DM], DM * 2, [64, 128]),) if wup is not None else ()
    atom_kernel("trg_rms" if rms else "trg_up" if wup is not None else "trg", d)(
        (min(num_sms(d), ntile), 1, 1), (384, 1, 1), *maps, dbts, int(ntile), int(mod.shape[0]), *_tsum(part, "tr", mod, M), *extra
    )
    return DT, HH, DAB


def post_bwd(DAB, dy, a2, mod, u, sg, o, WuT, WoT, dmod, dP, Dd, DBIAS, N, eps=1e-5, rms=False, part=None, pars=False, ident=False, wupT=None, raw=False):
    """-> DA, DU, GATED, DO [M, 128]; dmod blocks 2-4, dP block 3 (dg), D [A, 4, N]; d bsc2 / d bos into DBIAS. ``rms`` (ESMFold2): AdaLN 2 an
    RMSNorm with a raw scale, the output gate raw. B200 variant path: ``pars`` the parallel schedule (``a2`` is then the block input a),
    ``ident`` Chai-1's identity AdaLN (raw scale), ``wupT`` Boltz-2's up-gate weight transposed [128, 256] bf16 (DAB is then [M, 768])."""
    M = dy.shape[0]

    def new():
        return torch.empty(M, DM, device=dy.device, dtype=torch.bfloat16)

    DA, DU, GATED, DO = new(), new(), new(), new()
    if _hopper(_dev(dy)):
        d = _dev(dy)
        maps = (
            _map(WuT, [4 * DM, DM], 8 * DM, [64, 128]),
            _map(WoT, [DM, DM], DM * 2, [64, 128]),
        )
        atom_kernel("postb", d)(
            (min(num_sms(d), (M + 63) // 64), 1, 1),
            (160, 1, 1),
            DAB,
            dy,
            a2,
            mod,
            u,
            sg,
            o,
            *maps,
            DA,
            DU,
            GATED,
            DO,
            dmod,
            dP,
            Dd,
            int(M),
            int(N),
            float(eps),
        )
        return DA, DU, GATED, DO
    maps = (
        _rows16(DAB, 6 * DM if wupT is not None else 4 * DM),
        _rows16(dy),
        _rows16(a2),
        _rows16(mod, 6 * DM),
        _rows16(u),
        _rows16(sg),
        _rows16(o),
        _map(WuT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64),
        _map(WoT, [DM, DM], DM * 2, [32, 128], swizzle=64),
        _rows16(DA),
        _rows16(dmod, 6 * DM),
        _rows16(DU),
        _rows16(GATED),
        _rows16(DO),
        _rows16(dP, 4 * DM),
    )
    ntile = M // 16
    d = _dev(dy)
    assert not ident or pars, "IDENT without PARS: no model has it (no build)"
    assert wupT is None or not (pars or ident), "UP with PARS / IDENT: no model has it (no build)"
    extra = (_map(wupT, [2 * DM, DM], 4 * DM, [64, 128]),) if wupT is not None else ()
    assert not raw or rms, "raw: the ESMFold2 (RMS) build with the raw scale tables (round 3 part 7)"
    atom_kernel(("postb_rms_raw" if raw else "postb_rms") if rms else "postb_up" if wupT is not None else "postb_pars_id" if ident else "postb_pars" if pars else "postb", d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        Dd,
        DBIAS,
        int(N),
        int(ntile),
        float(eps),
        int(mod.shape[0]),
        *_tsum(part, "post", mod, M),
        *extra,
    )
    return DA, DU, GATED, DO


def pre_bwd(dP, a, mod, DA, WT, dmod, DBIAS, DBQ, eps=1e-5, part=None, ident=False):
    """dP [M, 512], a, mod (s1), DA -> d single [M, 128]; dmod blocks 0-1; d bsc1 into DBIAS[0:128], d bq into DBQ. ``ident`` (B200 variant
    path): Chai-1's identity AdaLN (raw scale)."""
    M = a.shape[0]
    DS = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    if _hopper(_dev(a)):
        d = _dev(a)
        atom_kernel("preb", d)(
            (min(num_sms(d), (M + 63) // 64), 1, 1),
            (160, 1, 1),
            dP,
            a,
            mod,
            DA,
            _map(WT, [4 * DM, DM], 8 * DM, [64, 128]),
            DS,
            dmod,
            int(M),
            float(eps),
        )
        return DS
    maps = (
        _rows16(dP, 4 * DM),
        _rows16(a),
        _rows16(mod, 6 * DM),
        _rows16(DA),
        _map(WT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64),
        _rows16(DS),
        _rows16(dmod, 6 * DM),
        _rows16(DA),
        _rows16(DA),
        _rows16(DA),
    )
    ntile = M // 16
    d = _dev(a)
    atom_kernel("preb_id" if ident else "preb", d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        DBIAS,
        DBQ,
        DBQ,
        DBQ,
        int(ntile),
        float(eps),
        int(mod.shape[0]),
        *_tsum(part, "pre", mod, M),
    )
    return DS


def pre_bwd_kv(dP, a, mod, DA, WT, dmod, DBIAS, DBQ, x1, mkv, bs, DBS, eps=1e-5, parallel=False, rms=False, dmkv=None, part=None, raw=False):
    """The cross-attention mode's ``pre_bwd``: dP = [dq | dk | dv | dg] with WT the real [Wq; Wk; Wv; Wg]^T; k and v come from xkv = LN(x1)
    sigmoid(mkv[:, :128] + bs) + mkv[:, 128:], whose backward (dxkv = dk Wk + dv Wv, in the kernel) is added to dx1 before AdaLN 1's.
    -> d single [M, 128] and dmkv [M, 256] = [d pre-sigmoid scale | dxkv]; d bs (fp32 [128]) is added into DBS. ``parallel`` (AlphaFold 3,
    xkv = LN(a) ..): the K / V AdaLN's LN backward goes into d single (x1 is not read). ``rms`` (ESMFold2, with ``parallel``): RMSNorm
    statistics, the scales raw. ``dmkv``: where dmkv goes (a [M, 256] view, e.g. columns of a wider gradient tensor), else a fresh tensor.
    ``raw`` (ESMFold2 training, with ``rms``; round 3 part 7): AdaLN 1's scale table holds the raw s (``-DRAWSC=1``: (1 + s) in fp32)."""
    assert not rms or parallel
    assert not raw or rms
    M = a.shape[0]
    DS = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    if dmkv is None:
        dmkv = torch.empty(M, 2 * DM, device=a.device, dtype=torch.bfloat16)
    maps = (
        _rows16(dP, 4 * DM),
        _rows16(a),
        _rows16(mod, 6 * DM),
        _rows16(DA),
        _map(WT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64),
        _rows16(DS),
        _rows16(dmod, 6 * DM),
        _rows16(x1),
        _rows16(mkv, 2 * DM),
        _rows16(dmkv, 2 * DM),
    )
    ntile = M // 16
    d = _dev(a)
    atom_kernel(("prebxp_rms_raw" if raw else "prebxp_rms") if rms else "prebxp" if parallel else "prebx", d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        DBIAS,
        DBQ,
        DBS,
        bs.float().contiguous(),
        int(ntile),
        float(eps),
        int(mod.shape[0]),
        *_tsum(part, "pre", mod, M),
    )
    return DS, dmkv


def cond_bwd(dmod, c, WmodT, g1, g2, DG, eps=1e-5, ident=False):
    """dmod [M, 768] (backward layout), c [M, 128] -> dc, cn1, cn2 [M, 128]; d g1 / d g2 into DG [256]. ``ident`` (B200 variant path):
    Chai-1's identity AdaLN (tables of the raw c: cn1 = cn2 = c, no LayerNorm backward)."""
    M = c.shape[0]

    def new():
        return torch.empty(M, DM, device=c.device, dtype=torch.bfloat16)

    dc, cn1, cn2 = new(), new(), new()
    if _hopper(_dev(c)):
        d = _dev(c)
        atom_kernel("condb", d)(
            (min(num_sms(d), (M + 63) // 64), 1, 1),
            (160, 1, 1),
            dmod,
            c,
            _map(WmodT, [6 * DM, DM], 12 * DM, [64, 128]),
            g1,
            g2,
            dc,
            cn1,
            cn2,
            DG,
            int(M),
            float(eps),
        )
        return dc, cn1, cn2
    maps = (
        _rows16(dmod, 6 * DM),
        _rows16(c),
        _map(WmodT, [6 * DM, DM], 12 * DM, [32, 128], swizzle=64),
        _rows16(dc),
        _rows16(cn1),
        _rows16(cn2),
    )
    ntile = M // 16
    d = _dev(c)
    atom_kernel("condb_id" if ident else "condb", d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        g1,
        g2,
        DG,
        int(ntile),
        float(eps),
    )
    return dc, cn1, cn2


# --------------------------------------------------------------------------------------------------- fp32 path (TF32 tensor cores)
SOURCES_TF32 = ("gemm_tf32", "rows_tf32")
F32 = torch.float32


@functools.lru_cache(maxsize=None)
def cubin_tf32(stem: str, defs: tuple = ()) -> str:
    """``cubin`` for the fp32-path sources, which include ``../sm100/sm100.cuh`` (the kind::tf32 helpers): that header is hashed."""
    from foldjet_engine.kernels import _nvcc

    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
        *("-D" + d for d in defs),
    )
    return str(_nvcc.cubin(f"local_dit_b200_tf32/{stem}", flags, source=_dir / f"{stem}.cu", toolchain="sm100"))


#: (stem, function, dynamic smem) of every fp32-path kernel (gemm_tf32's size is static_asserted in the source)
KERNELS_TF32 = {
    "gemm": ("gemm_tf32", "atom_gemm_tf32", 229632),
    "gemm_rg": ("gemm_rg_tf32", "atom_gemm_rg_tf32", 229632),
    "gemm_rg_gb": ("gemm_rg_tf32", "atom_gemm_rg_tf32", 229632, ("GATEB=1",)),  # (round 3 part 13: dgated + the output gate's backward)
    "gemm_sb": ("gemm_sb_tf32", "atom_gemm_sb_tf32", 229632),  # (round 3 part 11: dh product + SwiGLU backward)
    "gemm_k64": ("gemm_tf32", "atom_gemm_tf32", 229632, ("K64=1",)),  # (round 3 part 11: K = 64 stages from 3-D boxes)
    "ln": ("rows_tf32", "f32_ln_aff", 0),
    "adaln": ("rows_tf32", "f32_adaln", 0),
    "gate": ("rows_tf32", "f32_gate", 0),
    "tail_b": ("rows_tf32", "f32_tail_bwd", 0),
    "swiglu_b": ("rows_tf32", "f32_swiglu_bwd", 0),
    "adaln_b": ("rows_tf32", "f32_adaln_bwd", 0),
    "gate_b": ("rows_tf32", "f32_gate_bwd", 0),
    "ln_b": ("rows_tf32", "f32_ln_bwd", 0),
    "colsum": ("rows_tf32", "f32_colsum", 0),
    "adaln_rms": ("rows_tf32", "f32_adaln_rms", 0),
    "adaln_b_rms": ("rows_tf32", "f32_adaln_bwd_rms", 0),
    "tail_b_rms": ("rows_tf32", "f32_tail_bwd_rms", 0),
    "silu": ("rows_tf32", "f32_silu", 0),
    "silu_b": ("rows_tf32", "f32_silu_bwd", 0),
}

# The fp32 path's kernels launch with programmatic dependent launch: every one of them executes ``griddepcontrol.wait`` before its first
# global-memory access (after its smem / TMEM setup), so it may be scheduled while the previous kernel drains -- the launch latency and the
# setup overlap the previous kernel's tail.


@functools.lru_cache(maxsize=None)
def atom_kernel32(name: str, device_index: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem, *defs = KERNELS_TF32[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin_tf32(stem, *defs), func, smem, pdl=True)


#: the fused inference kernels of the fp32 path (``fused_tf32.cu``; loaded apart from KERNELS_TF32 so a failure costs the fusion only)
KERNELS_FUSED32 = {
    "pre": ("fused_tf32", "atom_pre_tf32", 229632),
    "post": ("fused_tf32", "atom_post_tf32", 229632),
    # ESMFold2's builds (-DRMS=1: RMSNorm AdaLNs), loaded only by its fp32 inference (``kernels/integrations/local_dit_esm_b200``)
    "pre_rms": ("fused_tf32", "atom_pre_tf32", 229632, ("RMS=1",)),
    "post_rms": ("fused_tf32", "atom_post_tf32", 229632, ("RMS=1",)),
    # the post kernel on a CTA pair, one hidden half each (round 3 part 6; post32 pair=True)
    "post_pair": ("fused_tf32", "atom_post_tf32", 229632, ("PAIR=1",)),
    "post_rms_pair": ("fused_tf32", "atom_post_tf32", 229632, ("RMS=1", "PAIR=1")),
    # (round 3 part 10) the quarter units' post with transposed products (atom_post_tf32t: the same cubins, another entry point)
    "pre_t": ("fused_tf32", "atom_pre_tf32t", 229632),
    "pre_rms_t": ("fused_tf32", "atom_pre_tf32t", 229632, ("RMS=1",)),
    "post_t": ("fused_tf32", "atom_post_tf32t", 229632),
    "post_rms_t": ("fused_tf32", "atom_post_tf32t", 229632, ("RMS=1",)),
    "post_pair_t": ("fused_tf32", "atom_post_tf32t", 229632, ("PAIR=1",)),
    "post_rms_pair_t": ("fused_tf32", "atom_post_tf32t", 229632, ("RMS=1", "PAIR=1")),
    # (round 3 part 11) the same on 64-row units (-DTU64=1: 38..74 tiles, b200_atom._wide)
    "pre_w": ("fused_tf32", "atom_pre_tf32t", 229632, ("TU64=1",)),
    "pre_rms_w": ("fused_tf32", "atom_pre_tf32t", 229632, ("RMS=1", "TU64=1")),
    "post_w": ("fused_tf32", "atom_post_tf32t", 229632, ("TU64=1",)),
    "post_rms_w": ("fused_tf32", "atom_post_tf32t", 229632, ("RMS=1", "TU64=1")),
    # (round 3 part 14) the transposed units with fp16-exact kind::f16 products (-DF16=1; _h: 32-row units, _hw: 64-row units)
    "pre_h": ("fused_tf32", "atom_pre_f16t", 229632, ("F16=1",)),
    "pre_rms_h": ("fused_tf32", "atom_pre_f16t", 229632, ("RMS=1", "F16=1")),
    "post_h": ("fused_tf32", "atom_post_f16t", 229632, ("F16=1",)),
    "post_rms_h": ("fused_tf32", "atom_post_f16t", 229632, ("RMS=1", "F16=1")),
    "post_pair_h": ("fused_tf32", "atom_post_f16t", 229632, ("PAIR=1", "F16=1")),
    "post_rms_pair_h": ("fused_tf32", "atom_post_f16t", 229632, ("RMS=1", "PAIR=1", "F16=1")),
    "pre_hw": ("fused_tf32", "atom_pre_f16t", 229632, ("TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
    "pre_rms_hw": ("fused_tf32", "atom_pre_f16t", 229632, ("RMS=1", "TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
    "post_hw": ("fused_tf32", "atom_post_f16t", 229632, ("TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
    "post_rms_hw": ("fused_tf32", "atom_post_f16t", 229632, ("RMS=1", "TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
}
#: ... and their B200 variant-path builds (2026-11-17): RoseTTAFold3's flat q / k LayerNorm (pre), the parallel schedule and Boltz-2's
#: up-gated SwiGLU (post)
KERNELS_FUSED32_VAR = {
    "pre_kq": ("fused_tf32", "atom_pre_tf32", 229632, ("KQLN=1",)),
    # (round 3b: the PARS build runs its transition before the attention's end -- EARLY, its default)
    "post_pars": ("fused_tf32", "atom_post_tf32", 229632, ("PARS=1",)),
    "post_up": ("fused_tf32", "atom_post_tf32", 229632, ("UP=1",)),
    # (round 3 part 7) the UP post on a CTA pair: rank r takes the hidden quarters 2 r, 2 r + 1 (post32: MW_LD7_UPPAIR)
    "post_up_pair": ("fused_tf32", "atom_post_tf32", 229632, ("UP=1", "PAIR=1")),
    # (round 3 part 10) the transposed quarter-unit post of the variant builds (EARLY parallel schedule; UP, its granules two 64-row boxes)
    "post_pars_t": ("fused_tf32", "atom_post_tf32t", 229632, ("PARS=1",)),
    "pre_kq_t": ("fused_tf32", "atom_pre_tf32t", 229632, ("KQLN=1",)),
    "post_up_t": ("fused_tf32", "atom_post_tf32t", 229632, ("UP=1",)),
    "post_up_pair_t": ("fused_tf32", "atom_post_tf32t", 229632, ("UP=1", "PAIR=1")),
    # (round 3 part 11) on 64-row units
    "post_pars_w": ("fused_tf32", "atom_post_tf32t", 229632, ("PARS=1", "TU64=1")),
    "pre_kq_w": ("fused_tf32", "atom_pre_tf32t", 229632, ("KQLN=1", "TU64=1")),
    "post_up_w": ("fused_tf32", "atom_post_tf32t", 229632, ("UP=1", "TU64=1")),
    # (round 3 part 14) fp16-exact kind::f16 products
    "post_pars_h": ("fused_tf32", "atom_post_f16t", 229632, ("PARS=1", "F16=1")),
    "pre_kq_h": ("fused_tf32", "atom_pre_f16t", 229632, ("KQLN=1", "F16=1")),
    "post_up_h": ("fused_tf32", "atom_post_f16t", 229632, ("UP=1", "F16=1")),
    "post_up_pair_h": ("fused_tf32", "atom_post_f16t", 229632, ("UP=1", "PAIR=1", "F16=1")),
    "post_pars_hw": ("fused_tf32", "atom_post_f16t", 229632, ("PARS=1", "TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
    "pre_kq_hw": ("fused_tf32", "atom_pre_f16t", 229632, ("KQLN=1", "TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
    "post_up_hw": ("fused_tf32", "atom_post_f16t", 229632, ("UP=1", "TU64=1", "F16=1", "PRE1R=1", "POST1R=1")),
}


@functools.lru_cache(maxsize=None)
def fused_kernel32(name: str, device_index: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem, *defs = KERNELS_FUSED32_VAR[name] if name in KERNELS_FUSED32_VAR else KERNELS_FUSED32[name]
    import os

    if os.environ.get("MW_LD14_XB") == "1" and defs and "PAIR=1" in defs[0]:  # (A / B: the tf32 PAIR build's t exchange by one bulk copy)
        defs = [tuple(defs[0]) + ("XBULK=1",)]
    if os.environ.get("MW_LD15_1R") == "0" and defs:  # (A / B: the 64-row F16 units' row loads round by round, part 14's form)
        defs = [tuple(d for d in defs[0] if d not in ("PRE1R=1", "POST1R=1"))]
    with torch.cuda.device(device_index):
        pair = any("PAIR=1" in d for d in defs[0]) if defs else False  # (the CTA-pair post builds launch as clusters of 2)
        return driver.Kernel(cubin_tf32(stem, *defs), func, smem, cluster=2 if pair else None, pdl=True)


def load_fused32(device_index: int) -> None:
    for name in ("pre", "post", "post_pair", "pre_t", "post_t", "post_pair_t", "pre_w", "post_w", "pre_h", "post_h", "post_pair_h", "post_hw"):
        fused_kernel32(name, device_index)


def _nfull(tiles, nsm_):
    """How many of the fused kernels' 128-row tiles run as full units; the rest run as two half units each (``fused_tf32.cu``). A CTA's unit
    is a serial latency chain (ncu: ~5 % issue utilisation), so the time is ~ the waves of units over the SMs times a unit's cost; a half unit
    costs 0.7-0.8 of a full one (round 5, 3072 / 6144 atoms). So: every tile split when the halves fit one wave; otherwise the tail beyond the
    whole waves split when ITS halves fit one wave (4096 atoms on 148 SMs: 148 full + 24 halves, 1.75 waves instead of 2); else none.
    settings.tf32_fused_units=full / half forces either."""
    mode = "auto"  # (the branch's settings.tf32_fused_units; 2.5.x has no settings: the measured rule)
    if mode in ("full", "half"):
        return tiles if mode == "full" else 0
    if 2 * tiles <= nsm_:
        return 0
    full = tiles // nsm_ * nsm_
    tail = tiles - full
    return full if 0 < tail and 2 * tail <= nsm_ else tiles


def _quarters(var: str) -> bool:
    """The fused kernels' quarter units when the quarters of every tile fit one wave (round 3 part 5, 2026-11-17): ``atom_post_tf32``
    32-row units, ``atom_pre_tf32`` 64-row x 256-column units. MW_LD5_PQ=0 / MW_LD5_PRQ=0 turn them off (A / B measurements)."""
    import os

    return os.environ.get(var, "1") != "0"


def _wide(tiles, nsm_, var: str) -> bool:
    """The fused kernels' 64-row transposed units (``-DTU64=1``, round 3 part 11): when the quarter units of every tile do not fit one wave
    but the 64-row units do (38..74 tiles on 148 SMs: 1024 atoms x 5 samples, 5120 - 9472 rows). MW_LD11_PW=0 / MW_LD11_PRW=0 (post / pre)
    turn them off (A / B: the row-form half units)."""
    import os

    return 4 * tiles > nsm_ and 2 * tiles <= nsm_ and os.environ.get(var, "1") != "0"


def _pair_units(tiles, nsm_, var="MW_LD6_PAIR"):
    """The post kernel on CTA pairs (``-DPAIR=1``, round 3 part 6): -1 (quarter units, a pair each) when the pairs of every tile's quarters
    fit one wave (<= 18 tiles, 2304 rows), else None. MW_LD6_PAIR=0 turns it off (A / B)."""
    import os

    mode = os.environ.get(var, "1")
    if mode == "0":
        return None
    if _quarters("MW_LD5_PQ") and 8 * tiles <= nsm_:  # quarter units only (halves on pairs: 4096 atoms 71.7 -> 73.5 us)
        return -1
    return None


#: (round 3 part 14) fp16-exact forms of the fused kernels' tf32 weight packs, by the pack objects (held: their addresses stay theirs)
_F16PK: dict = {}


def _f16pack(w, blocks=1):
    """The fp16-exact form of a tf32-rounded fp32 weight pack for the ``-DF16=1`` fused kernels (``fused_tf32.cu``, round 3 part 14): w's rows
    in ``blocks`` equal blocks, block b scaled by 2^s_b so that its largest magnitude lies in [2^14, 2^15) -- a tf32 value (11 significant
    bits) is an fp16 value whenever it lies in fp16's normal range, so the fp16 pack holds the same numbers times 2^s_b. Returns (w16,
    [2^-s_b]) or None when some element is not an fp16 value bit for bit after the scaling (a block spanning more than ~2^29 in magnitude, or
    a non-finite one): the caller then runs the tf32 kernels."""
    wb = w.detach().reshape(blocks, -1)
    _, ex = torch.frexp(wb.abs().amax(dim=1))  # max = m 2^ex, m in [0.5, 1): floor(log2 max) = ex - 1
    s = (15 - ex).to(F32)
    sc = wb * torch.pow(2.0, s)[:, None]  # (pow2 factors: exact)
    w16 = sc.half()
    if not (bool(torch.isfinite(sc).all()) and torch.equal(w16.float(), sc)):
        return None
    return w16.reshape(w.shape).contiguous(), [2.0 ** -int(v) for v in s.tolist()]


def _f16cached(objs, build):
    """``build()`` cached on the weight-pack objects ``objs`` (and their versions). A CUDA-graph capture never builds one (the packs a
    capture makes are not the eager ones): it gets None, i.e. the tf32 kernels. MW_LD14_F16=0 turns the fp16-exact kernels off (A / B)."""
    import os

    if os.environ.get("MW_LD14_F16", "1") == "0":
        return None
    key = tuple(id(o) for o in objs)
    hit = _F16PK.get(key)
    if hit is not None and all(a is b for a, b in zip(hit[0], objs)) and hit[1] == tuple(o._version for o in objs):
        return hit[2]
    if torch.cuda.is_current_stream_capturing():
        return None
    out = build()
    if len(_F16PK) >= 64:
        _F16PK.clear()
    _F16PK[key] = (tuple(objs), tuple(o._version for o in objs), out)
    return out


def _f16pre(wp):
    """pre's fp16-exact pack: Wp [512, 128] in its four 128-row blocks (q / k / v / g, one scale each) -> (wp16, [4 factors]) or None."""
    return _f16cached((wp,), lambda: _f16pack(wp, 4))


def _f16post(wo, wu, ws, up):
    """post's fp16-exact packs and factors (wo16, wu16, ws16, (f_o, f_u, f_s), (A1, B1, U1)) or None: A1 / B1 / U1 the largest L1 norm of a
    Wa / Wb / Wu row (the kernel's bound of |h| per row, from its x2 maximum)."""

    def build():
        a, b, c = _f16pack(wo), _f16pack(wu), _f16pack(ws)
        if a is None or b is None or c is None:
            return None
        r = wu.detach().abs().reshape(*((4, 3, 64) if up else (2, 2, 128)), DM).sum(-1)  # (UP: [Wa_q; Wb_q; Wu_q]; else [Wa_h; Wb_h])
        bnd = tuple(float(r[:, p].max()) for p in range(3 if up else 2)) + ((0.0,) if not up else ())
        return a[0], b[0], c[0], (a[1][0], b[1][0], c[1][0]), bnd

    return _f16cached((wo, wu, ws), build)


def _tm16(t, box):
    """TMA map of a contiguous fp16 [rows, cols] pack (2-byte elements: the bf16 encoding, no conversion), cached as ``_tm32``."""
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    assert t.dtype == torch.float16 and t.dim() == 2 and t.is_contiguous(), (t.dtype, t.shape)
    rows, cols = t.shape
    key = ("f16", t.data_ptr(), cols, rows, tuple(box))
    m = _TMAPS32.get(key)
    if m is None:
        if len(_TMAPS32) >= 512:
            _TMAPS32.clear()
        m = driver.TensorMap(t, [cols, rows], cols * 2, list(box), swizzle=128, dtype="bf16")
        m.keep = None
        _TMAPS32[key] = m
    return m


def pre32(s, s1, b1, wp, bq, rndmask, mks=None, mkb=None, eps=1e-5, nfull=None, parallel=False, rms=False, kq=None, kq_eps=1e-5, samples=1):
    """Fused AdaLN 1 (+ the cross mode's K / V AdaLN, when mks / mkb are given) and the four input projections (``atom_pre_tf32``):
    s [M, 128] fp32 contiguous, s1 / b1 (and mks / mkb) fp32 [M, 128] views with a common row stride each, wp the tf32-rounded [512, 128]
    pack ([q; g; k; v] with mks, else [q; k; v; g]), bq [128] -> P [M, 512] fp32 (the blocks in rndmask tf32-rounded). ``nfull`` (default
    ``_nfull``): the 128-row tiles run whole; the others as two 128 x 256 column halves. ``parallel`` (AlphaFold 3): the K / V AdaLN
    normalises s instead of x1. The tables may have mrows < M rows (a conditioning shared by the M / mrows samples): row r reads r % mrows. ``rms``
    (ESMFold2): RMSNorm AdaLNs (the ``-DRMS=1`` build)."""
    M = s.shape[0]
    assert M % 128 == 0 and s.is_contiguous() and s.shape[1] == DM and tuple(wp.shape) == (4 * DM, DM), (s.shape, wp.shape)
    assert s1.stride(1) == b1.stride(1) == 1 and s1.stride(0) == b1.stride(0)
    mrows = s1.shape[0]
    assert M % mrows == 0 and mrows % 128 == 0 and b1.shape[0] == mrows, (M, mrows)
    cross = mks is not None
    if cross:
        assert mks.stride(1) == mkb.stride(1) == 1 and mks.stride(0) == mkb.stride(0)
    p = torch.empty(M, 4 * DM, device=s.device, dtype=F32)
    d = _dev(s)
    tiles = M // 128
    # (round 3 part 11) 64-row transposed units, one a CTA: with a shared conditioning (1024 x 5: 69.6 -> 63.5 us with the wide post) or
    # from 52 tiles (7168 rows 90.1 -> 88.1); at 40 / 48 unshared tiles the row form's half units were 2 us faster (5120, 6144 rows)
    import os

    # (round 3 part 14) the fp16-exact kind::f16 kernels for the transposed units when the pack allows (_f16pre; MW_LD14_F16=0: tf32), the
    # 64-row ones by the part-11 rule (MW_LD14_PRW=1: at any 38..74 tiles -- measured slower at 40 / 48 unshared tiles: the early pair bias
    # beside it ran 2 us longer)
    hp = _f16pre(wp) if nfull is None else None
    wide_any = hp is not None and os.environ.get("MW_LD14_PRW", "0") == "1"
    big = hp is not None and nfull is None and 2 * tiles > num_sms(d) and os.environ.get("MW_LD14_BIG", "3") in ("1", "2")
    if big or (
        nfull is None
        and (wide_any or samples > 1 or mrows < M or tiles >= 52 or os.environ.get("MW_LD11_PRW") == "2")
        and _wide(tiles, num_sms(d), "MW_LD11_PRW")
    ):
        name = "pre_rms" if rms else "pre_kq" if kq is not None else "pre"
        fused_kernel32(name + ("_hw" if hp is not None else "_w"), d)(
            (min(num_sms(d), 2 * tiles), 1, 1),
            (384, 1, 1),
            _tm16(hp[0], (64, 128)) if hp is not None else _tm32(wp, (32, 128)),
            _tm32(p, (32, 32)),
            s,
            s1,
            b1,
            int(s1.stride(0)),
            mks,
            mkb,
            int(mks.stride(0)) if cross else 0,
            bq,
            int(2 if cross and parallel else cross),
            int(rndmask),
            int(M),
            float(eps),
            -1,
            int(mrows),
            *((kq, float(kq_eps)) if kq is not None else ()),
            *(hp[1] if hp is not None else ()),
        )
        return p
    if nfull is None and 4 * tiles <= num_sms(d) and _quarters("MW_LD5_PRQ"):
        nfull = -1  # every tile as four 64-row x 256-column quarter units (round 3 part 5)
    nfull = _nfull(tiles, num_sms(d)) if nfull is None else int(nfull)
    units = 4 * tiles if nfull < 0 else nfull + 2 * (tiles - nfull)
    assert not (rms and kq is not None)
    # (round 3 part 10) quarter units without the flat q / k LayerNorm: the transposed kernel (32-row units over all 512 columns, one CTA
    # a unit; MW_LD10_T=0: the row form)
    tr = nfull < 0 and _quarters("MW_LD10_T")
    assert kq is None or mks is None
    name = "pre_rms" if rms else "pre_kq" if kq is not None else "pre"
    if not tr:
        hp = None
    fused_kernel32((name + ("_h" if hp is not None else "_t")) if tr else name, d)(
        (min(num_sms(d), units), 1, 1),
        (384, 1, 1),
        _tm16(hp[0], (64, 128)) if hp is not None else _tm32(wp, (32, 128 if tr else 256)),
        _tm32(p, (32, 32)),
        s,
        s1,
        b1,
        int(s1.stride(0)),
        mks,
        mkb,
        int(mks.stride(0)) if cross else 0,
        bq,
        int(2 if cross and parallel else cross),
        int(rndmask),
        int(M),
        float(eps),
        int(nfull),
        int(mrows),
        *((kq, float(kq_eps)) if kq is not None else ()),
        *(hp[1] if hp is not None else ()),
    )
    return p


def post32(o, g, s, so, s2, b2, st, wo, wu, ws, eps=1e-5, nfull=None, rms=False, pars=False, up=False):
    """Fused gate, out projection + gated residual, AdaLN 2, SwiGLU, squeeze + gated residual (``atom_post_tf32``): o, s [M, 128] fp32
    contiguous, g a [M, 128] view, so / s2 / b2 / st [mrows, 128] views with one row stride (mrows = M, or fewer rows shared by the samples:
    row r reads r % mrows), wo / wu / ws the tf32-rounded packs -> out [M, 128].
    ``nfull`` (default ``_nfull``): the 128-row tiles run whole; the others as two 64-row halves. ``rms`` (ESMFold2): RMSNorm AdaLN 2."""
    M = s.shape[0]
    assert M % 128 == 0 and s.is_contiguous() and o.is_contiguous() and o.shape == s.shape, (o.shape, s.shape)
    assert g.stride(1) == 1 and all(t.stride(1) == 1 and t.stride(0) == so.stride(0) for t in (s2, b2, st))
    mrows = so.shape[0]
    assert M % mrows == 0 and mrows % 128 == 0 and all(t.shape[0] == mrows for t in (s2, b2, st)), (M, mrows)
    out = torch.empty(M, DM, device=s.device, dtype=F32)
    d = _dev(s)
    tiles = M // 128
    pair = None
    if nfull is None and not pars:  # (round 3 part 7: UP too, its hidden quarters 2 r, 2 r + 1 on rank r)
        pair = _pair_units(tiles, num_sms(d), "MW_LD7_UPPAIR" if up else "MW_LD6_PAIR")
        if pair is not None:
            nfull = pair
    assert not (pars and up) and not (rms and (pars or up))
    # (round 3 part 14) the transposed units with fp16-exact kind::f16 products when the packs allow (_f16post; MW_LD14_F16=0: tf32)
    hp = _f16post(wo, wu, ws, up) if (nfull is None or pair is not None) else None
    import os

    big = hp is not None and nfull is None and pair is None and 2 * tiles > num_sms(d) and os.environ.get("MW_LD14_BIG", "3") in ("1", "3")
    if big or (nfull is None and pair is None and _wide(tiles, num_sms(d), "MW_LD11_PW")):  # (round 3 part 11) 64-row transposed units
        name = "post_rms" if rms else "post_up" if up else "post_pars" if pars else "post"
        fused_kernel32(name + ("_hw" if hp is not None else "_w"), d)(
            (min(num_sms(d), 2 * tiles), 1, 1),
            (384, 1, 1),
            _tm16(hp[0], (64, 128)) if hp is not None else _tm32(wo, (32, 128)),
            _tm16(hp[1], (64, 64 if up else 128)) if hp is not None else _tm32(wu, (32, 64 if up else 128)),
            _tm16(hp[2], (64, 128)) if hp is not None else _tm32(ws, (32, 128)),
            o,
            g,
            int(g.stride(0)),
            s,
            so,
            s2,
            b2,
            st,
            int(so.stride(0)),
            out,
            int(M),
            float(eps),
            -1,
            int(mrows),
            *((*hp[3], *hp[4]) if hp is not None else ()),
        )
        return out
    if nfull is None and 4 * tiles <= num_sms(d) and _quarters("MW_LD5_PQ"):
        nfull = -1  # every tile as four 32-row quarter units (round 3 part 5)
    nfull = _nfull(tiles, num_sms(d)) if nfull is None else int(nfull)
    units = 4 * tiles if nfull < 0 else nfull + 2 * (tiles - nfull)
    # (round 3 part 10) quarter units of the default / RMS builds: the transposed kernel (atom_post_tf32t; MW_LD10_T=0: the row form)
    tr = nfull < 0 and _quarters("MW_LD10_T")
    if pair is not None:  # a CTA pair per unit (clusters of 2), one hidden half each
        name = "post_rms_pair" if rms else "post_up_pair" if up else "post_pair"
        grid = 2 * min(num_sms(d) // 2, units)
    else:
        name = "post_rms" if rms else "post_up" if up else "post_pars" if pars else "post"
        grid = min(num_sms(d), units)
    if not tr:
        hp = None
    if tr:
        name += "_h" if hp is not None else "_t"
    fused_kernel32(name, d)(
        (grid, 1, 1),
        (384, 1, 1),
        _tm16(hp[0], (64, 128)) if hp is not None else _tm32(wo, (32, 128)),
        _tm16(hp[1], (64, 64 if up else 128)) if hp is not None else _tm32(wu, (32, (64 if tr else 192) if up else 128 if tr else 256)),  # (up: the [Wa_q; Wb_q; Wu_q] quarter pack [768, 128]; tr: halves)
        _tm16(hp[2], (64, 128)) if hp is not None else _tm32(ws, (32, 128)),
        o,
        g,
        int(g.stride(0)),
        s,
        so,
        s2,
        b2,
        st,
        int(so.stride(0)),
        out,
        int(M),
        float(eps),
        int(nfull),
        int(mrows),
        *((*hp[3], *hp[4]) if hp is not None else ()),
    )
    return out


def load_all32(device_index: int) -> None:
    """Build and load every fp32-path kernel on the device (raises on any failure)."""
    for name in KERNELS_TF32:
        atom_kernel32(name, device_index)


_TMAPS32: dict = {}


def _tm32(t, box, swizzle=128):
    """TMA map of a row-major fp32 [rows, cols] tensor (unit column stride, any row stride), cached by everything it encodes (address,
    geometry, box, swizzle; encoding one costs ~5-10 us of host time). The cache does not hold the tensor."""
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    assert t.dtype == F32 and t.dim() == 2 and t.stride(1) == 1, (
        t.dtype,
        t.shape,
        t.stride(),
    )
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


def _tm32k(t):
    """3-D TMA map of a row-major fp32 [rows, cols] tensor as (32 columns, rows, cols / 32 K-slices), box [32, 128, 2] (two K-slices of 128
    rows: ``gemm_tf32.cu -DK64``), cached as ``_tm32``."""
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    assert t.dtype == F32 and t.dim() == 2 and t.stride(1) == 1 and t.shape[1] % 32 == 0, (t.dtype, t.shape, t.stride())
    rows, cols = t.shape
    key = ("k64", t.data_ptr(), cols, rows, t.stride(0))
    m = _TMAPS32.get(key)
    if m is None:
        if len(_TMAPS32) >= 512:
            _TMAPS32.clear()
        m = driver.TensorMap(t, [32, rows, cols // 32], [t.stride(0) * 4, 128], [32, 128, 2], swizzle=128, dtype="f32")
        m.keep = None
        _TMAPS32[key] = m
    return m


#: gemm32 epilogues
STORE, RESID_GATE, SWIGLU = 0, 1, 2


def gemm32(
    a,
    w,
    out=None,
    *,
    k=None,
    n=None,
    a_col0=0,
    b_row0=0,
    b_col0=0,
    ocol0=0,
    bias=None,
    sigmask=0,
    rndmask=0,
    mode=STORE,
    res=None,
    gate=None,
    out2=None,
    a_grp=0,
):
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
    wide = mode == SWIGLU or (n % 256 == 0 and (M // 128) * (n // 256) >= num_sms(d))
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
        atom_kernel32("gemm_rg", d)(
            (min(num_sms(d), tiles), 1, 1),
            (512, 1, 1),
            ma,
            mb,
            mo,
            mo2,
            _tm32(res, (32, 32)),
            _tm32(gate, (32, 32)),
            int(M),
            int(k),
            int(out2 is not None),
            int(gate.shape[0]),
        )
        return out
    # (round 3 part 11) K = 64 stages from 3-D TMA boxes (-DK64=1) for the K >= 256, 128-wide products of many tiles: one issuing thread
    # moves ~one box per ~600 clk, so two 16-KB boxes a K = 32 stage held them at ~3.5 TB/s. MW_LD11_K64=0: off (A / B)
    import os

    k64 = (
        nt == 128
        and k % 64 == 0
        and k >= 256
        and tiles >= num_sms(d)
        and a_col0 % 32 == 0
        and b_col0 % 32 == 0
        and a.shape[1] % 32 == 0
        and w.shape[1] % 32 == 0
        and os.environ.get("MW_LD11_K64", "1") != "0"
    )
    if k64:
        ma, mb = _tm32k(a), _tm32k(w)
    atom_kernel32("gemm_k64" if k64 else "gemm", d)(
        (min(num_sms(d), tiles), 1, 1),
        (512, 1, 1),
        ma,
        mb,
        mo,
        mo2,
        bias,
        res,
        gate,
        int(M),
        int(n),
        int(k),
        int(nt),
        int(a_col0),
        int(b_row0),
        int(b_col0),
        int(ocol0),
        int(sigmask),
        int(mode),
        int(out2 is not None),
        int(res.stride(0)) if res is not None else 0,
        int(gate.stride(0)) if gate is not None else 0,
        int(rndmask),
        int(a_grp),
        int(gate.shape[0]) if gate is not None else int(M),
    )
    return out


def gemm_gate_bwd32(dyo, woT, g, o, dg, A, n):
    """dgated = dyo @ woT^T (K = N = 128) with ``gate_bwd32`` as its epilogue (round 3 part 13: dgated never leaves TMEM): -> dO [M, 128]
    (tf32-rounded), D [A, 4, n]; dg (view, written) = dgated o s (1 - s), s = sigmoid(g). Bit for bit gemm32 + gate_bwd32."""
    M = dyo.shape[0]
    d = _dev(dyo)
    assert M % 128 == 0 and dyo.shape[1] == DM and woT.shape == (DM, DM) and o.is_contiguous() and g.stride(1) == 1 and dg.stride(1) == 1
    do = _new(o)
    D = torch.empty(A, NH, n, device=o.device, dtype=F32)
    tiles = M // 128
    atom_kernel32("gemm_rg_gb", d)(
        (min(num_sms(d), tiles), 1, 1),
        (512, 1, 1),
        _tm32(dyo, (32, 128)),
        _tm32(woT, (32, 128)),
        _tm32(do, (32, 32)),
        _tm32(dg, (32, 32)),
        _tm32(o, (32, 32)),
        _tm32(g, (32, 32)),
        int(M),
        DM,
        0,
        int(M),
        D,
        int(n),
    )
    return do, D


def round_tf32(t):
    """t (fp32) rounded to tf32, to nearest with ties away from zero (cvt.rna.tf32.f32): add half a tf32 ulp to the magnitude bits, drop
    the low 13. For the weight operands of gemm32, once per pack (kind::tf32 MMAs truncate)."""
    return tf32_grid(t.detach().float().contiguous())


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
        o1, o2, oc = out[:, :DM], out[:, DM : 2 * DM], out[:, 2 * DM : 3 * DM]
        o3 = out[:, 3 * DM :] if g3 is not None else None
        ld = out.stride(0)
    atom_kernel32("ln", _dev(c))(
        _rowgrid(M),
        (256, 1, 1),
        c,
        g1,
        o1,
        g2,
        o2,
        g3,
        o3,
        oc,
        int(ld),
        int(M),
        float(eps),
    )
    return o1, o2, o3


def adaln32(a, scale, shift, eps=1e-5, rms=False):
    """LN(a) * scale + shift: a [M, 128] fp32, scale / shift fp32 [M or mrows, 128] views (the stored sigmoid scale; mrows < M: tables
    shared by the M / mrows samples, row r reads table row r % mrows). ``rms`` (ESMFold2): RMS(a)."""
    out = _new(a)
    atom_kernel32("adaln_rms" if rms else "adaln", _dev(a))(
        _rowgrid(a.shape[0]),
        (256, 1, 1),
        a,
        scale,
        int(scale.stride(0)),
        shift,
        int(shift.stride(0)),
        out,
        int(a.shape[0]),
        float(eps),
        int(scale.shape[0]),
    )
    return out


def silu32(c):
    """silu(c): c [M, 128] fp32 (contiguous) -> [M, 128] fp32 (ESMFold2's conditioning, the operand of every table)."""
    out = _new(c)
    atom_kernel32("silu", _dev(c))(_rowgrid(c.shape[0]), (256, 1, 1), c, out, int(c.shape[0]))
    return out


def silu_bwd32(c, g):
    """g silu'(c): c, g [M, 128] fp32 (contiguous) -> [M, 128] fp32."""
    out = _new(c)
    atom_kernel32("silu_b", _dev(c))(_rowgrid(c.shape[0]), (256, 1, 1), c, g, out, int(c.shape[0]))
    return out


def gate32(g, o):
    """sigmoid(g) * o: g a fp32 [M, 128] view, o [M, 128]."""
    out = _new(o)
    atom_kernel32("gate", _dev(o))(_rowgrid(o.shape[0]), (256, 1, 1), g, int(g.stride(0)), o, out, int(o.shape[0]))
    return out


def tail_bwd32(dy, t, st, dst, rms=False):
    """-> dt = dy * st; dst (a view, written) = dy * t * st (1 - st) (``rms``, ESMFold2's raw gate: dy * t). st / dst with fewer rows than dy
    (tables shared by the samples): dst summed over the samples (fixed order)."""
    dt = _new(dy)
    atom_kernel32("tail_b_rms" if rms else "tail_b", _dev(dy))(
        _rowgrid(st.shape[0]),
        (256, 1, 1),
        dy,
        t,
        st,
        int(st.stride(0)),
        dt,
        dst,
        int(dst.stride(0)),
        int(dy.shape[0]),
        int(st.shape[0]),
    )
    return dt


def dh_swiglu_bwd32(dt, wsT, u):
    """``swiglu_bwd32(gemm32(dt, wsT), u)`` in one launch (``gemm_sb_tf32.cu``, round 3 part 11: dh stays in TMEM; the same bits):
    dt [M, K] fp32, wsT the tf32-rounded [256, K] -> d[a | b] [M, 512], h [M, 256]. None when MW_LD11_SB=0 (A / B: the two launches)."""
    import os

    if os.environ.get("MW_LD11_SB", "1") == "0":
        dab, hh = swiglu_bwd32(gemm32(dt, wsT), u)
        return dab, hh
    M, K = dt.shape
    assert M % 128 == 0 and K % 32 == 0 and tuple(wsT.shape) == (2 * DM, K) and tuple(u.shape) == (M, 4 * DM), (dt.shape, wsT.shape, u.shape)
    dab, hh = _new(u, 4 * DM), _new(u, 2 * DM)
    d = _dev(dt)
    atom_kernel32("gemm_sb", d)(
        (min(num_sms(d), M // 128), 1, 1),
        (512, 1, 1),
        _tm32(dt, (32, 128)),
        _tm32(wsT, (32, 256)),
        _tm32(u, (32, 32)),
        _tm32(dab, (32, 32)),
        _tm32(hh, (32, 32)),
        int(M),
        int(K),
    )
    return dab, hh


def swiglu_bwd32(dh, u):
    """dh [M, 256], u [M, 512] (interleaved [a_0 | b_0 | a_1 | b_1]) -> d[a | b] [M, 512] (same layout), h = silu(a) b [M, 256]."""
    dab, hh = _new(u, 4 * DM), _new(u, 2 * DM)
    atom_kernel32("swiglu_b", _dev(u))(_rowgrid(u.shape[0]), (256, 1, 1), dh, u, dab, hh, int(u.shape[0]))
    return dab, hh


def adaln_bwd32(a, scale, dx, dres, dscale, dshift, y=None, so=None, dso=None, eps=1e-5, rms=False):
    """AdaLN backward: -> da = dres + LN backward of dx * scale; dscale (view) = dx LN(a) scale (1 - scale), dshift (view) = dx. With y:
    also d so (view) = da y so (1 - so) and the returned dyo = da so (the attention output gate upstream of a1 = a + so y). scale (and so,
    dscale, dshift, dso) with fewer rows than a: tables shared by the samples, their gradients summed over the samples (fixed order)."""
    M = a.shape[0]
    da = _new(a)
    dyo = _new(a) if y is not None else None
    atom_kernel32("adaln_b_rms" if rms else "adaln_b", _dev(a))(
        _rowgrid(scale.shape[0]),
        (256, 1, 1),
        a,
        scale,
        int(scale.stride(0)),
        dx,
        dres,
        da,
        dscale,
        int(dscale.stride(0)),
        dshift,
        int(dshift.stride(0)),
        y,
        so,
        int(so.stride(0)) if so is not None else 0,
        dso,
        int(dso.stride(0)) if dso is not None else 0,
        dyo,
        int(M),
        float(eps),
        int(scale.shape[0]),
    )
    return da, dyo


def gate_bwd32(dgated, g, o, dg, A, n):
    """dgated [M, 128], g a view, o [M, 128] -> dO [M, 128], D [A, 4, n]; dg (view, written) = dgated o s (1 - s)."""
    M = o.shape[0]
    do = _new(o)
    D = torch.empty(A, NH, n, device=o.device, dtype=F32)
    atom_kernel32("gate_b", _dev(o))(
        _rowgrid(M),
        (256, 1, 1),
        dgated,
        g,
        int(g.stride(0)),
        o,
        do,
        dg,
        int(dg.stride(0)),
        D,
        int(M),
        int(n),
    )
    return do, D


def colsum32(x, partials=False):
    """The column sums of a fp32 [M, 128 g] view (unit column stride, any row pitch) -> [128 g] fp32: one launch writing G partial rows (a
    persistent grid, deterministic) and a tiny sum of those."""
    M, cols = x.shape
    assert x.dtype == F32 and x.stride(1) == 1 and cols % DM == 0, (
        x.dtype,
        x.shape,
        x.stride(),
    )
    d = _dev(x)
    g = max(1, min(2 * num_sms(d), (M + 7) // 8))
    part = torch.empty(g, cols, device=x.device, dtype=F32)
    atom_kernel32("colsum", d)((g, cols // DM, 1), (256, 1, 1), x, int(x.stride(0)), part, int(cols), int(M))
    return part if partials else part.sum(0)  # (``partials``: the G partial rows, for ``col_jobs``)


def ln_bwd32(c, dc, dcn1, g1, dcn2, g2, dcn3=None, g3=None, eps=1e-5):
    """dc [M, 128] += the LayerNorm backward of dcn_k * g_k (k = 1, 2, 3); -> dg [384] fp32 = [dg1 | dg2 | dg3] (sum_rows dcn_k LN(c))."""
    M = c.shape[0]
    d = _dev(c)
    nb = max(1, min(4 * num_sms(d), (M + 7) // 8))
    dg = torch.zeros(nb, 3 * DM, device=c.device, dtype=F32)  # a partial row per block (rows_tf32.cu f32_ln_bwd), summed in order below
    atom_kernel32("ln_b", d)(
        (nb, 1, 1),
        (256, 1, 1),
        c,
        dc,
        dcn1,
        g1,
        dcn2,
        g2,
        dcn3,
        g3,
        dg,
        int(M),
        float(eps),
    )
    return dg.sum(0)


# ------------------------------------------------------------------------------------------------------------ weight gradients (one launch)
import ctypes  # noqa: E402

_DW_MAXP, _DW_MAXO = 12, 24
_DW_NS, _DW_STG = 5, 40 * 1024


class _DwTab(ctypes.Structure):
    _fields_ = [
        ("np", ctypes.c_int),
        ("mb", ctypes.c_int * _DW_MAXP),
        ("nn", ctypes.c_int * _DW_MAXP),
        ("ns", ctypes.c_int * _DW_MAXP),
        ("outbase", ctypes.c_int * _DW_MAXP),
        ("wsoff", ctypes.c_longlong * _DW_MAXP),
        ("e4off", ctypes.c_longlong * _DW_MAXP),
        ("e4total", ctypes.c_longlong),
    ]


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
        from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

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
        from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

        assert 0 < len(problems) <= _DW_MAXP
        f32 = self.f32
        dt = F32 if f32 else torch.bfloat16
        esz = 4 if f32 else 2
        box, sw, dname = ([32, 32], "128a32", "f32") if f32 else ([64, 32], 128, "bf16")
        nsm_ = num_sms(self.dev)
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
                raw[slot * 128 : (slot + 1) * 128] = ctypes.string_at(m.addr, 128)
            mb, nn = A.shape[1] // 128, B.shape[1]
            tab.mb[p], tab.nn[p] = mb, nn
            tab.ns[p] = max(1, round(nsm_ * weights[p] / total))
            tab.wsoff[p], tab.e4off[p], tab.outbase[p] = wsoff, e4, nouts
            wsoff += tab.ns[p] * mb * 128 * nn
            e4 += mb * 128 * nn // 4
            for d in dests:
                assert d.is_contiguous() and d.shape == (128, nn) and d.dtype in (F32, torch.bfloat16), (d.shape, d.dtype)
                out.ptr[nouts], out.bf16[nouts] = (
                    d.data_ptr(),
                    int(d.dtype == torch.bfloat16),
                )
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

    def release(self) -> None:
        """Drop the references to the last call's operands, destination views and workspace (the launches are issued: the tensor maps were
        read, and the stream's order keeps the memory until the kernels ran). A destination view holds its base tensor, so a gradient
        buffer this object still references cannot be taken over by autograd's AccumulateGrad (it is copied instead)."""
        self._keep = None


@functools.lru_cache(maxsize=None)
def atom_dw(device_index: int, f32: bool) -> AtomDw:
    return AtomDw(device_index, f32)


# --------------------------------------------------------------------------------------------------- sm_90a (H100) helpers
# PyTorch may erase its cached cuBLASLt workspace when a different graph is reset.
# A capture-local tensor keeps each retained graph's scratch storage alive.
# This is an opt-in lifetime repair; all projection operands and CUDA kernels stay unchanged.
_OWNED_WGRAD_CPP = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
torch::Tensor owned_wgrad(torch::Tensor g, torch::Tensor x) {
  TORCH_CHECK(g.is_cuda() && x.is_cuda() && g.device()==x.device()
    && g.scalar_type()==at::kBFloat16 && x.scalar_type()==at::kBFloat16
    && g.dim()==2 && x.dim()==2 && g.size(0)==x.size(0)
    && g.stride(1)==1 && x.stride(1)==1 && g.size(0)>0
    && g.stride(0)>=g.size(1) && x.stride(0)>=x.size(1),
    "owned cuBLASLt wgrad requires BF16 row-major or row-padded operands");
  c10::cuda::CUDAGuard guard(g.device());
  const int64_t reduction=g.size(0), outputs=g.size(1), inputs=x.size(1);
  auto out=torch::empty({outputs,inputs},g.options().dtype(at::kFloat));
  const size_t workspace_bytes=at::cuda::getCUDABlasLtWorkspaceSize();
  auto workspace=torch::empty({int64_t(workspace_bytes)},g.options().dtype(at::kByte));
  auto checked=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"owned cuBLASLt wgrad status ",int(s));};
  struct Descriptors {
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulPreference_t preference=nullptr;
    ~Descriptors(){
      if(preference)cublasLtMatmulPreferenceDestroy(preference);
      if(c)cublasLtMatrixLayoutDestroy(c);
      if(b)cublasLtMatrixLayoutDestroy(b);
      if(a)cublasLtMatrixLayoutDestroy(a);
      if(operation)cublasLtMatmulDescDestroy(operation);
    }
  } d;
  checked(cublasLtMatmulDescCreate(&d.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t op_a=CUBLAS_OP_N,op_b=CUBLAS_OP_T;
  checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSA,&op_a,sizeof(op_a)));
  checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSB,&op_b,sizeof(op_b)));
  checked(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_16BF,inputs,reduction,x.stride(0)));
  checked(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_16BF,outputs,reduction,g.stride(0)));
  checked(cublasLtMatrixLayoutCreate(&d.c,CUDA_R_32F,inputs,outputs,inputs));
  checked(cublasLtMatmulPreferenceCreate(&d.preference));
  checked(cublasLtMatmulPreferenceSetAttribute(d.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
  const void* ptrs[]={x.data_ptr(),g.data_ptr(),out.data_ptr(),out.data_ptr()};
  cublasLtMatmulPreferenceAttributes_t attrs[]={CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
  for(int i=0;i<4;++i){
    uint32_t alignment=256;
    while(reinterpret_cast<uintptr_t>(ptrs[i])%alignment)alignment>>=1;
    checked(cublasLtMatmulPreferenceSetAttribute(d.preference,attrs[i],&alignment,sizeof(alignment)));
  }
  cublasLtMatmulHeuristicResult_t algo{};int found=0;
  auto handle=at::cuda::getCurrentCUDABlasLtHandle();
  checked(cublasLtMatmulAlgoGetHeuristic(handle,d.operation,d.a,d.b,d.c,d.c,d.preference,1,&algo,&found));
  TORCH_CHECK(found==1,"owned cuBLASLt wgrad has no valid algorithm");
  float alpha=1.f,beta=0.f;
  checked(cublasLtMatmul(handle,d.operation,&alpha,x.data_ptr(),d.a,g.data_ptr(),d.b,
    &beta,out.data_ptr(),d.c,out.data_ptr(),d.c,&algo.algo,workspace.data_ptr(),workspace_bytes,at::cuda::getCurrentCUDAStream()));
  return out;
}

"""


@functools.lru_cache(maxsize=1)
def _owned_wgrad_extension():
    from foldjet_engine.kernels._nvcc import load_inline_extension as load_inline
    from foldjet_engine.kernels._nvcc import ensure_cuda_home

    from foldjet_engine.kernels._codegraph import CUDA_HOME_STANDIN

    cuda_home = ensure_cuda_home() or CUDA_HOME_STANDIN  # the toolkit's headers are needed only to build
    if tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) < (12, 0):
        raise RuntimeError("Atom Hopper cuBLASLt gradients require CUDA 12 or newer")
    return load_inline(  # built (once per content key) under engine_jit_root/torch_extensions
        name="atom_hopper_owned_wgrad",
        cpp_sources=_OWNED_WGRAD_CPP,
        functions=["owned_wgrad"],
        with_cuda=True,
        extra_include_paths=[str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )


def owned_wgrad_hopper(gradient, activation):
    """BF16 GEMM to FP32 with cuBLASLt scratch owned by the captured graph."""
    return _owned_wgrad_extension().owned_wgrad(gradient, activation)


def pack_hopper(weights, backward=False):
    """Live CUDA weight packing; identical launch count for BF16 and FP32 masters."""
    if len(weights) != 23:
        raise ValueError("Atom weight pack requires the 23 canonical weights")
    d, dev = _dev(weights[0]), weights[0].device
    shapes = ((768, 128), (512, 128), (128, 128), (512, 128), (128, 256))
    wc, wp, wo, wu, wd = (torch.empty(s, device=dev, dtype=torch.bfloat16) for s in shapes)
    b = torch.empty(768, device=dev, dtype=torch.float32)
    g1, g2, bq = (torch.empty(128, device=dev, dtype=torch.float32) for _ in range(3))
    if any(w.dtype not in (torch.bfloat16, torch.float32) or not w.is_contiguous() for w in weights):
        raise ValueError("Atom Hopper pack requires contiguous BF16/FP32 parameters")
    bfmask = sum((w.dtype == torch.bfloat16) << i for i, w in enumerate(weights))
    atom_kernel("pack", d)((384, 1, 1), (256, 1, 1), *weights, wc, wp, wo, wu, wd, b, g1, g2, bq, bfmask)
    wb = {}
    if backward:
        wt = [torch.empty(tuple(reversed(s)), device=dev, dtype=torch.bfloat16) for s in shapes]
        atom_kernel("pack_t", d)((272, 1, 1), (32, 8, 1), wc, wp, wo, wu, wd, *wt)
        wb = dict(zip(("WmodT", "WpT", "WoT", "WuT", "WsT"), wt, strict=True))
    return (wc, b, g1, g2), (wp, bq), (wo, wu, wd), wb


def finish_columns_hopper(dmod, dp, dgparts, dbias, dbq, dg):
    part = torch.empty(4, 896, device=dmod.device, dtype=torch.float32)
    d, m = _dev(dmod), dmod.shape[0]
    atom_kernel("col_part", d)((56, 4, 1), (256, 1, 1), dmod, dp, part, int(m))
    atom_kernel("col_fin", d)((4, 1, 1), (256, 1, 1), part, dgparts, dbias, dbq, dg, int((m + 63) // 64))


def grad_pack_hopper(grads, weights):
    if any(g.dtype != torch.float32 or not g.is_contiguous() for g in grads):
        raise ValueError("Hopper gradient pack requires contiguous FP32 gradient views")
    out = [torch.empty_like(w) for w in weights]
    bfmask = sum((w.dtype == torch.bfloat16) << i for i, w in enumerate(weights))
    atom_kernel("grad_pack", _dev(weights[0]))((128, 1, 1), (256, 1, 1), *grads, *out, bfmask)
    return out


def kqln_bwd(dP, qkr, kq, eps=1e-5, partials=False):
    """The backward of RoseTTAFold3's flat q / k LayerNorm (``pre_fwd_kq``) on dP's dq | dk [M, 512] bf16, in place, one launch
    (``kqln_bwd.cu``; B200 variant path): ``qkr`` [M, 256] bf16 the raw projections, ``kq`` fp32 [4, 128] = [wq | bq | wk | bk]. -> the
    column sums [d wq | d bq | d wk | d bk] fp32 [512] (one partial row per CTA, summed in order: deterministic)."""
    M = dP.shape[0]
    assert dP.dtype == torch.bfloat16 and dP.is_contiguous() and dP.shape[1] == 4 * DM and tuple(qkr.shape) == (M, 2 * DM) and qkr.is_contiguous()
    assert kq.dtype == torch.float32 and kq.is_contiguous() and kq.numel() == 4 * DM
    d = _dev(dP)
    grid = min(num_sms(d), (M + 7) // 8)
    parts = torch.empty(grid, 4 * DM, device=dP.device, dtype=torch.float32)
    _load_atom_kernel("kqln_bwd", "atom_kqln_bwd", d, 0, False)((grid, 1, 1), (256, 1, 1), dP, qkr, kq, parts, int(M), float(eps))
    return parts if partials else parts.sum(0)  # (``partials``: the per-CTA rows [grid, 512], for ``col_jobs``)


def var_pack(mats, vecs, backward=False):
    """The variant block's packs in one launch (``var_pack.cu``; B200 variant path): ``mats`` the 14 (15 with Boltz-2's up-gate) matrices in
    the kernel's order, ``vecs`` up to 13 [128] vectors (None: zeros); fp32 or bf16 sources. -> (BF bf16 flat forward pack, VEC fp32
    [len(vecs), 128], BT bf16 flat transposed pack or None)."""
    assert len(mats) in (14, 15) and len(vecs) <= 13
    srcs = [t if t is None or t.dtype in (torch.float32, torch.bfloat16) else t.float() for t in (*mats, *vecs)]
    srcs = [t if t is None or t.is_contiguous() else t.contiguous() for t in srcs]
    m, v = srcs[: len(mats)], srcs[len(mats) :]
    dev = m[0].device
    nf = (2432 if len(m) == 15 else 2176) * DM
    bf = torch.empty(nf, device=dev, dtype=torch.bfloat16)
    vec = torch.empty(len(v), DM, device=dev, dtype=torch.float32)
    bt = torch.empty(nf, device=dev, dtype=torch.bfloat16) if backward else None
    mmask = sum(1 << i for i, t in enumerate(m) if t.dtype == torch.bfloat16)
    vmask = sum(1 << i for i, t in enumerate(v) if t is not None and t.dtype == torch.bfloat16)
    d = _dev(m[0])
    _load_atom_kernel("var_pack", "atom_var_pack", d, 0, False)(
        (8 * (19 if len(m) == 15 else 17) + 1, 1, 1),  # a CTA per 16-column strip of a [128][128] unit of the sources, one for the vectors
        (256, 1, 1),
        *m,
        *([None] * (15 - len(m))),
        *v,
        *([None] * (13 - len(v))),
        bf,
        vec,
        bt,
        int(mmask),
        int(vmask),
        len(v),
        int(backward),
    )
    return bf, vec, bt


def _var_row32(func: str, d: int):
    return _load_atom_kernel("var_rows_tf32", func, d, 0, False)


def adaln_bwd32_id(a, scale, dx, dres, dscale, dshift, eps=1e-5):
    """``adaln_bwd32`` for Chai-1's identity AdaLN (B200 variant path; ``var_rows_tf32.cu``): the scale is raw, so dscale (view) = dx LN(a);
    -> da = dres + LN backward of dx * scale; dshift (view) = dx. Fewer table rows than a: their gradients summed over the samples."""
    M = a.shape[0]
    da = _new(a)
    _var_row32("f32_adaln_bwd_id", _dev(a))(
        _rowgrid(scale.shape[0]),
        (256, 1, 1),
        a,
        scale,
        int(scale.stride(0)),
        dx,
        dres,
        da,
        dscale,
        int(dscale.stride(0)),
        dshift,
        int(dshift.stride(0)),
        int(M),
        float(eps),
        int(scale.shape[0]),
    )
    return da


def ln_bwd32_grp(c, dgrp, g1, g2, eps=1e-5):
    """The conditioning LayerNorms' backward from one grouped product (``var_rows_tf32.cu`` f32_ln_bwd2; B200 variant path, fp32 training):
    ``dgrp`` [M, 384] fp32 = [dcn1, dcn2, dc0] -> (dc [M, 128] = dc0 + the two LayerNorm backwards, the per-CTA partial rows [grid, 256] of
    [d g1, d g2] -- sum them in order)."""
    M = c.shape[0]
    assert c.is_contiguous() and dgrp.is_contiguous() and tuple(dgrp.shape) == (M, 3 * DM) and dgrp.dtype == F32
    d = _dev(c)
    grid = min(num_sms(d), (M + 7) // 8)
    dc = _new(c)
    parts = torch.empty(grid, 2 * DM, device=c.device, dtype=F32)
    _var_row32("f32_ln_bwd2", d)((grid, 1, 1), (256, 1, 1), c, dgrp, g1, g2, dc, parts, int(M), float(eps))
    return dc, parts


def swiglu_up_bwd32(dh, u, up):
    """Boltz-2's up-gated SwiGLU backward in one launch (``var_rows_tf32.cu`` f32_swiglu_up_bwd; B200 variant path, fp32 training): dh
    [M, 256], u [M, 512] (interleaved [a_0, b_0, a_1, b_1]), up [M, 256] -> dabu [M, 768] = [d a_0, d b_0, d a_1, d b_1, d up], h =
    silu(a) b up [M, 256]."""
    M = u.shape[0]
    assert all(t.is_contiguous() and t.dtype == F32 for t in (dh, u, up)) and u.shape[1] == 4 * DM and dh.shape == up.shape == (M, 2 * DM)
    dabu, hh = _new(u, 6 * DM), _new(u, 2 * DM)
    _var_row32("f32_swiglu_up_bwd", _dev(u))(_rowgrid(M), (256, 1, 1), dh, u, up, dabu, hh, int(M))
    return dabu, hh


def kqln_fwd32(p, kq, eps=1e-5):
    """RoseTTAFold3's flat q / k LayerNorm on the fp32 projection p [M, 512] = [q | k | v | g], in place (q / k leave tf32-rounded); -> the
    raw q | k [M, 256] (B200 variant path, fp32 training)."""
    M = p.shape[0]
    assert p.dtype == F32 and p.is_contiguous() and p.shape[1] == 4 * DM and kq.dtype == F32 and kq.is_contiguous() and kq.numel() == 4 * DM
    qkr = _new(p, 2 * DM)
    _var_row32("f32_kqln_fwd", _dev(p))(_rowgrid(M), (256, 1, 1), p, kq, qkr, int(M), float(eps))
    return qkr


def kqln_bwd32(dP, qkr, kq, eps=1e-5, partials=False):
    """``kqln_bwd`` on fp32 operands: dP [M, 512] fp32 (dq | dk rewritten in place), qkr [M, 256] fp32. -> [d wq | d bq | d wk | d bk] [512]."""
    M = dP.shape[0]
    assert dP.dtype == F32 and dP.is_contiguous() and dP.shape[1] == 4 * DM and tuple(qkr.shape) == (M, 2 * DM) and qkr.is_contiguous()
    d = _dev(dP)
    grid = min(num_sms(d), (M + 7) // 8)
    parts = torch.empty(grid, 4 * DM, device=dP.device, dtype=F32)
    _var_row32("f32_kqln_bwd", d)((grid, 1, 1), (256, 1, 1), dP, qkr, kq, parts, int(M), float(eps))
    return parts if partials else parts.sum(0)


def var_pack32(mats, vecs):
    """The variant block's fp32 training packs in one launch (``var_pack32.cu``; B200 variant path): ``mats`` the 14 (15 with Boltz-2's
    up-gate) matrices in the fp32 path's order, ``vecs`` up to 16 [128] vectors (None: zeros); fp32 or bf16 sources. -> (BF fp32 flat
    forward pack on the TF32 grid, VEC fp32 [len(vecs), 128], BT its transposed operands, BG [384, 256] the conditioning backward's grouped
    operand, BU [128, 768] = [Wu^T, Wup^T] with the up-gate (else None))."""
    assert len(mats) in (14, 15) and len(vecs) <= 16
    srcs = [t if t is None or t.dtype in (torch.float32, torch.bfloat16) else t.float() for t in (*mats, *vecs)]
    srcs = [t if t is None or t.is_contiguous() else t.contiguous() for t in srcs]
    m, v = srcs[: len(mats)], srcs[len(mats) :]
    dev = m[0].device
    nf = (2432 if len(m) == 15 else 2176) * DM
    bf = torch.empty(nf, device=dev, dtype=F32)
    bt = torch.empty(nf, device=dev, dtype=F32)
    bg = torch.empty(3 * DM, 2 * DM, device=dev, dtype=F32)
    bu = torch.empty(DM, 6 * DM, device=dev, dtype=F32) if len(m) == 15 else None
    vec = torch.empty(len(v), DM, device=dev, dtype=F32)
    mmask = sum(1 << i for i, t in enumerate(m) if t.dtype == torch.bfloat16)
    vmask = sum(1 << i for i, t in enumerate(v) if t is not None and t.dtype == torch.bfloat16)
    d = _dev(m[0])
    _load_atom_kernel("var_pack32", "atom_var_pack32", d, 0, False)(
        (8 * (19 if len(m) == 15 else 17) + 1, 1, 1),
        (256, 1, 1),
        *m,
        *([None] * (15 - len(m))),
        *v,
        *([None] * (16 - len(v))),
        bf,
        vec,
        bt,
        bg,
        int(mmask),
        int(vmask),
        len(v),
        bu,
    )
    return bf, vec, bt, bg, bu


class _EsmUnit(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p if n == "src" else ctypes.c_int32) for n in ("src", "sp", "sbf", "d0", "p0", "d1", "p1", "dt", "pt", "pad_")]


class _EsmUnits(ctypes.Structure):
    _fields_ = [("u", _EsmUnit * 24), ("vsrc", ctypes.c_void_p * 4), ("vbf", ctypes.c_int32 * 4)]


def esm_pack(units, vecs, nbf, nbt, f32):
    """ESMFold2's training weight packs in one launch (``esm_pack.cu``; ``integrations/local_dit_esm_b200``, round 3 part 7): ``units`` up to
    24 (src, row0, pitch, d0, p0, d1, p1, dt, pt) -- a [128][128] block of ``src`` (fp32 or bf16, contiguous; None: zeros) from element
    ``row0`` with row pitch ``pitch``, copied to the flat forward pack at element d0 (row pitch p0) and d1 (p1; d1 < 0: no second copy) and,
    dt >= 0, transposed into the flat pack BT at dt (row pitch pt); ``vecs`` up to 4 [128] vectors (None: zeros). ``f32``: fp32 outputs on
    the TF32 grid, else bf16. -> (BF [nbf], BT [nbt] or None, VEC fp32 [len(vecs), 128])."""
    assert 0 < len(units) <= 24 and len(vecs) <= 4
    tab, keep = _EsmUnits(), []
    dev = None
    for i, (src, row0, pitch, d0, p0, d1, p1, dt, pt) in enumerate(units):
        u = tab.u[i]
        if src is not None:
            assert src.is_contiguous() and src.dtype in (F32, torch.bfloat16), (src.dtype, src.shape)
            u.src, u.sbf = src.data_ptr() + row0 * src.element_size(), int(src.dtype == torch.bfloat16)
            keep.append(src)
            dev = src.device
        u.sp, u.d0, u.p0, u.d1, u.p1, u.dt, u.pt = pitch, d0, p0, d1, p1, dt, pt
    for i, v in enumerate(vecs):
        if v is not None:
            assert v.is_contiguous() and v.numel() == DM and v.dtype in (F32, torch.bfloat16)
            tab.vsrc[i], tab.vbf[i] = v.data_ptr(), int(v.dtype == torch.bfloat16)
    assert dev is not None
    odt = F32 if f32 else torch.bfloat16
    bf = torch.empty(nbf, device=dev, dtype=odt)
    bt = torch.empty(nbt, device=dev, dtype=odt) if nbt else None
    vec = torch.empty(max(1, len(vecs)), DM, device=dev, dtype=F32)
    _load_atom_kernel("esm_pack", "atom_esm_pack", _dev(bf), 0, False, ("F32=1",) if f32 else ())(
        (8 * len(units) + (1 if vecs else 0), 1, 1), (256, 1, 1), tab, len(units), len(vecs), bf, bt, vec
    )
    return bf, bt, vec

class _ColJob(ctypes.Structure):
    _fields_ = [("src", ctypes.c_void_p), ("dst", ctypes.c_void_p), ("rows", ctypes.c_int32), ("pitch", ctypes.c_int32), ("n", ctypes.c_int32), ("bf16", ctypes.c_int32)]


class _ColJobs(ctypes.Structure):
    _fields_ = [("j", _ColJob * 16)]


def col_jobs(jobs):
    """The small gradients of a backward in one launch (``col_jobs.cu``; B200 variant path): ``jobs`` up to 16 (src, col0, like) -- src fp32
    [rows, cols] partial rows (unit column stride; or a [cols] vector), summed over the rows in order at columns col0 .. col0 + like.numel()
    into a fresh tensor of ``like``'s shape and dtype (fp32 or bf16). -> the fresh tensors, in order."""
    assert 0 < len(jobs) <= 16
    tab, outs, keep = _ColJobs(), [], []
    for i, (src, col0, like) in enumerate(jobs):
        s2 = src if src.ndim == 2 else src.reshape(1, -1)
        if s2.dtype != F32:
            s2 = s2.float()
        n = like.numel()
        assert s2.stride(1) == 1 and col0 + n <= s2.shape[1] and like.dtype in (F32, torch.bfloat16), (s2.shape, s2.stride(), col0, n, like.dtype)
        out = torch.empty(like.shape, device=src.device, dtype=like.dtype)
        j = tab.j[i]
        j.src, j.dst = s2.data_ptr() + 4 * col0, out.data_ptr()
        j.rows, j.pitch, j.n, j.bf16 = s2.shape[0], (s2.stride(0) if s2.shape[0] > 1 else 0), n, int(like.dtype == torch.bfloat16)
        outs.append(out)
        keep.append(s2)
    _load_atom_kernel("col_jobs", "atom_col_jobs", _dev(outs[0]), 0, False)((len(jobs), 1, 1), (256, 1, 1), tab)
    return outs


def conf_mask_(bias, uid, kmask, n, slide):
    """Chai-1's conformer mask on the windowed pair bias [4, n / 32, 32, 128] fp32, in place, one launch (``conf_mask.cu``; B200 variant
    path): -inf at the slots whose key is of another conformer than a real query. ``uid`` int64 [n] (padded with -1), ``kmask`` bool [n] or
    None, ``slide``: the slid window's keys in the windows the tcgen05 attention moves."""
    assert uid.dtype == torch.int64 and uid.is_contiguous() and uid.numel() == n and n % 128 == 0
    assert bias.is_contiguous() and tuple(bias.shape) == (NH, n // 32, 32, 128), bias.shape
    d = _dev(bias)
    _load_atom_kernel("conf_mask", "atom_conf_mask", d, 0, False)((n // 32, 1, 1), (256, 1, 1), bias, uid, kmask, int(n), int(slide))
    return bias


def _box3_kernel(name: str, device_index: int):
    """``atom_kernel`` of a ``post_fwd`` build with ``-DBOX3=1`` added (round 3 part 10: 3-D tensor maps, the tile inputs in 4 TMA boxes)."""
    stem, func, smem, pdl, *defs = (KERNELS_ESM if name in KERNELS_ESM else KERNELS_VAR if name in KERNELS_VAR else KERNELS)[name]
    assert stem == "post_fwd", name
    return _load_atom_kernel(stem, func, device_index, smem, pdl, (*(defs[0] if defs else ()), "BOX3=1"))
