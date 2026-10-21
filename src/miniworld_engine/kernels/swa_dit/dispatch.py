"""The fused SWA atom DiT block's host side: which implementation serves each stage, and its launches.

Required H100 shapes retain the accurate legacy composition until native candidates pass all acceptance gates.
settings.swa_dit_h100_native=True forces handwritten CUDA and cuBLAS (False disables it); the Hopper implementation of the B200
fusion boundaries follows the shape default.
Unsupported native contracts and failed builds raise. Candidate timings do not describe the default path.

Moved from team-gm ``swa_fused_triton.block_fwd`` / ``block_bwd`` (commit 14f2c73). The launch sequence, the buffers and
the arguments are team-gm's; what differs is the engine plumbing:

* the CUDA-vs-Triton choice reads ``settings`` instead of team-gm's environment switches
  (``SWA_QKVG_FWD`` / ``SWA_FFN_FWD`` / ``SWA_FFN_BWD`` -> ``swa_dit_{qkvg_fwd,ffn_fwd,ffn_bwd}_cuda``,
  ``SWA_FFN_DW`` -> ``swa_dit_ffn_dw``, ``SWA_DQ1`` -> ``swa_dit_dq1``; same defaults), and
  ``engine_backend="triton"`` turns every CUDA stage off, as it does for the other families;
* the CUDA kernels build through ``cuda/loader.py`` (``settings.engine_jit_root``) and any failure falls back to Triton;
* each Triton launch passes ``shape_key=atom_key(S, A=..., C=C[, NHID=NHID])`` where team-gm passed ``SB = log2(rows)``:
  the atom count S bucketed like every other atom-level kernel, and the AUGMENT COUNT A = N // B exactly, always -- the
  input feature embedder calls with A = 1 and diffusion with its num_augment (48 in training), and the row tiles of the
  modulation-reducing backward kernels (SP augments x AT atoms) want different shapes for the two;
* the forward and the backward are each ONE opaque op (``kernels._compile.opaque``): the per-stage choice is as
  untraceable as the launches, so it sits inside the same op. ``autograd.SWADiTBlockFunction`` owns the saved tensors.

bf16 stages -- every one hand-CUDA on sm_90 (each with its own switch; Triton on other cards, on a failed build, or with
``engine_backend="triton"``). Forward: qkvg -> window attention (``cuda/swa_attn.cu``) -> out-projection + gated residual
+ FFN. Backward: FFN (materialised dW operands; Triton with ``swa_dit_ffn_dw="fused"``) -> out-projection -> attention dq,
dk, dv in one kernel -> qkvg, then the weight gradients as cuBLAS GEMMs over the saved operands. The qkvg forward, the
out-projection + FFN forward and the FFN backward have two CUDA implementations: team-gm's 64-row wgmma kernels
(``swa_qkvg_fwd.cu``, ``swa_ffn_fwd.cu``, ``swa_ffn_bwd.cu``) and the slab kernels of ``swa_slab.cu`` (16-row slabs per
4-warp group, built for the small row counts the block mostly runs at; the out-projection and qkvg backwards are slab
kernels too); ``swa_dit_rows_impl`` picks, "auto" by row count and augment count (``_WGMMA_MIN_ROWS``). On the CUDA
backward the attention side shares one [M, 4C] buffer (see ``_bwd_attention_side``).

B200 (sm_100a), bf16, the served widths (``cuda/sm100.supported``): every stage on the hand-written sm_100a kernels of
``cuda/sm100`` (``_sm100``); the atom count must be a multiple of 128 there -- the block refuses otherwise.
``settings.swa_dit_sm100=0`` keeps the Triton path; a build or load failure warns once and keeps it too.

fp32 stages, B200 (sm_100a) at the served widths with S a multiple of 128 (``cuda/sm100/tf32_fwd.supported_tf32``): every stage on
the hand-written TF32 tensor-core kernels (``tcgen05.mma kind::tf32``) -- ``tf32_fwd.block_fwd_tf32`` (``_tf32``), the window
attention included (TF32 Q / K / V / P, fp32 softmax) -- for inference. A training forward (``save``) would take them only with a TF32
backward, and there is none at the moment (``_tf32_bwd_ready``: the first one was removed for giving different gradients on identical
steps), so fp32 training runs the Triton fp32 path. ``settings.swa_dit_tf32=0`` (or ``settings.swa_dit_sm100=0`` /
``engine_backend="triton"``) keeps the Triton fp32 path for inference too, and so does a build or load failure (warned once). The
hoisted modulation has the same split (``swa_dit_mod_fwd_tf32`` without a gradient).

fp32 stages elsewhere (q fp32): on sm_90 with ``swa_dit_fp32_cuda`` (default on) and TF32 allowed (matmul precision "high" or
"medium"), every stage is the hand-CUDA extension ``cuda/swa_fp32.cu`` -- one ``fwd`` and one ``bwd`` call, tf32 wgmma
with rna-rounded operands (split weights in the FFN) and a bf16 window attention; otherwise ``triton/forward_fp32.py``
and ``triton/backward_fp32.py`` for qkvg, out-projection + FFN and their backwards, around the SAME window-attention
kernels on bf16 operands (what FlashAttention-4 runs in the per-op fp32 path). Both write the same saved tensors and the
same dW operands. ``swa_dit_ffn_dw`` and ``swa_dit_dq1`` are bf16 knobs; the fp32 backward always materialises the dW
operands and keeps dq1 in fp32.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import torch
import triton

from miniworld_engine import settings
from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd
from miniworld_engine.kernels.swa_dit.interface import (
    D_ATOM,
    FP32_EPS,
    HALF_WINDOW,
    N_HEAD,
    N_HIDDEN,
    _matmul_precision,
    is_global,
)
from miniworld_engine.kernels.swa_dit.triton.backward import (
    _swa_attn_bwd_dkv_kernel,
    _swa_attn_bwd_dq_kernel,
    _swa_ffn_bwd_kernel,
    _swa_ffn_dw_kernel,
    _swa_oproj_bwd_kernel,
    _swa_qkvg_bwd_kernel,
)
from miniworld_engine.kernels.swa_dit.triton.backward_fp32 import (
    _swa_ffn_bwd_fp32_kernel,
    _swa_oproj_bwd_fp32_kernel,
    _swa_qkvg_bwd_fp32_kernel,
)
from miniworld_engine.kernels.swa_dit.triton.forward import (
    _swa_attn_fwd_kernel,
    _swa_oproj_ffn_fwd_kernel,
    _swa_qkvg_fwd_kernel,
)
from miniworld_engine.kernels.swa_dit.triton.forward_fp32 import (
    _swa_oproj_ffn_fwd_fp32_kernel,
    _swa_qkvg_fwd_fp32_kernel,
)

#: The one-launch weight-gradient kernel (``cuda/swa_wgrad.cu``) serves up to this many rows (N*S): measured on H100 against the
#: five cuBLAS GEMMs it replaces (graph replay): 1.6x at M = 4096, a tie at 20480 (55 us), cuBLAS ahead from there (L2-resident
#: operands and split-K at 50k+ rows).
_WGRAD_MAX_ROWS = 16384
#: The same kernel on fp32 operands (bf16 hi / lo planes, ~30x closer to fp32 than cuBLAS's TF32 GEMMs): 34.6 us against 47.4 us at
#: M = 4096, behind cuBLAS from ~20000 rows.
_WGRAD_F32_MAX_ROWS = 8192
#: The TMA + wgmma fp32 kernel (``swa_wgrad_wg_f32.cu``): 25.5 us against cuBLAS 46.8 at M = 4096, 178.6 against 185.4 at 49152, behind
#: it at 196608 (664 against 572).
_WGRAD_F32_WG_MAX_ROWS = 65536

#: ``pack`` keys an axis only below 4096; an augment count at or above it shares the top key.
_A_KEY_MAX = 4095

def _native_b200_default(q, training=None):
    """Cells that passed native gates and both paired and archived speed bars."""
    if q.ndim != 3 or q.shape[2] != 128:
        return False
    a, n = q.shape[:2]
    training = torch.is_grad_enabled() if training is None else training
    if training:
        return ((q.dtype == torch.bfloat16 and (a, n) in ((1, 8192), (5, 4096), (5, 8192)))
                or (q.dtype == torch.float32 and (a, n) == (5, 8192)))
    return q.dtype == torch.bfloat16 and ((a == 1 and n in (4096, 8192))
                                        or (a == 5 and n in (1024, 2048, 4096, 8192)))


def native_h100_enabled(q, training=None):
    """Whether H100 serves this fused-block call (every sm_90 call unless ``settings.swa_dit_h100_native`` is False); which
    stages run it is :func:`native_h100_stages`."""
    from miniworld_engine.kernels.swa_dit.cuda.loader import is_sm90
    override = settings.current().swa_dit_h100_native
    if override is not None:
        return override and is_sm90(q.device)
    return is_sm90(q.device)


def native_h100_stages(q) -> bool:
    """Whether an H100 call runs the native stages (``_native_h100_fwd`` / ``_bwd``) rather than the composed ones.

    bf16 takes the composed stages: every one hand-CUDA on sm_90 (02073e40), no Triton launch anywhere on the grid A 1 - 48 x
    256 - 16384 atoms (ragged included), inference and training, shared and per-sample conditioning, and 1.1 - 6.7x faster than
    the native stages (A 48, 2048 atoms, training: 1.25 against 2.82 ms; A 5, 8192 atoms, inference: 0.22 against 0.31 ms).
    A stage whose CUDA build is missing would fall to its Triton kernel, which H100 does not run: the native stages then.
    fp32 under TF32 (matmul precision "high" / "medium") takes the composed stages too, ``swa_fp32.cu`` on tf32 wgmma: 1.3 - 1.5x
    faster than the native stages and no less accurate (A 5, 1024 - 24 576 atoms, inference: 0.069 against 0.104 ms, 1.23 against
    1.65 ms). Under "highest" fp32 keeps the native stages (exact fp32 cuBLAS projections). ``settings.swa_dit_h100_native=True``
    forces the native stages.
    """
    if not native_h100_enabled(q):
        return False
    if settings.current().swa_dit_h100_native is True:
        return True
    if q.dtype == torch.float32:
        # _fp32_cuda's test, with the precision read as a constant under torch.compile (this runs in the autograd.Function's forward)
        composed = (settings.current().swa_dit_fp32_cuda and _matmul_precision() != "highest"
                    and _composed_cuda(q.device, "fp32"))
    else:
        composed = q.dtype == torch.bfloat16 and _composed_cuda(q.device)
    return not composed


def _composed_cuda(device: torch.device, extra: str | None = None) -> bool:
    """Whether every hand-CUDA extension the composed bf16 stages use (and ``extra``: "fp32" for the fp32 ones) builds / loads on
    ``device`` (the prebuilt bundle or the JIT); ``engine_backend="triton"`` asks for their Triton kernels instead."""
    return settings.current().engine_backend != "triton" and _composed_builds(device, extra)


@functools.lru_cache(maxsize=8)
def _composed_builds(device: torch.device, extra: str | None) -> bool:
    from miniworld_engine.kernels.swa_dit.cuda.loader import extension

    stages = ("qkvg", "fwd", "bwd", "attn", "slab", "pack", "mod", "wgrad") + ((extra,) if extra else ())
    return all(extension(which) is not None for which in stages)


def _h100_global(q, half_window):
    """Global attention on H100 always takes the native sm_90a stages (FlashAttention-4 between them): the composed path's
    global attention is the sm_100a stages, which an H100 cannot run, whatever the per-shape defaults pick for the window."""
    from miniworld_engine.kernels.swa_dit.cuda.loader import is_sm90

    return is_global(half_window) and q.is_cuda and is_sm90(q.device)


def _native_b200_fusion(q=None, training=None, B=None):
    """The B200-style fused row stages read one conditioning row per sequence row; a call whose samples share their conditioning
    (``q.shape[0] != B``: a sampling step) runs the plain native stages instead."""
    if q is None or (B is not None and q.ndim == 3 and q.shape[0] != B):
        return False
    return _native_b200_default(q, training)


def _tiled_serves(qh, d_out=None):
    """Whether the tiled window attention (``swa_tiled``: TMA tiles, wgmma, persistent waves) takes this call: bf16 heads
    [N, 4, S, 32] with S a multiple of 64 (and a bf16 dO for the backward). It is the attention of every native call it serves,
    whatever the row stages around it are: tying it to the B200-style row fusion (``_native_b200_fusion``) sent a call whose
    samples share their conditioning -- every diffusion sampling step -- and every fp32 call to the untiled ``native_attention``,
    37x slower (A 5, 8192 atoms: 1520 against 41 us)."""
    return qh.shape[2] % 64 == 0 and (d_out is None or d_out.dtype == torch.bfloat16)


def _native_h100_extension(which="qkvg"):
    from miniworld_engine.kernels.swa_dit.cuda.loader import ERRORS, extension
    if settings.current().engine_backend == "triton":
        raise RuntimeError("native H100 SWA rejects a forced Triton backend")
    ext = extension(which)
    if ext is None:
        raise RuntimeError(f"native H100 SWA {which} extension unavailable: {ERRORS.get(which)}")
    return ext


def _native_mm(a, b):
    # Private cuBLAS state preserves the incumbent GEMM choice. The captured
    # graph owns its scratch tensor independently of other graphs' lifetimes.
    return _native_h100_extension().native_owned_mm(a, b, torch.backends.cuda.matmul.allow_tf32)


def _native_contract(q, mod, cs, sn, seqused, weights, B, half_window, training):
    if (q.dtype not in (torch.bfloat16, torch.float32) or q.ndim != 3 or q.shape[-1] != 128
            or not q.is_cuda or not q.is_contiguous() or q.numel() == 0
            or (half_window != 64 and not is_global(half_window)) or B <= 0 or q.shape[0] % B
            or mod.dtype != torch.float32 or not mod.is_contiguous()
            or seqused.dtype != torch.int32 or not seqused.is_contiguous()
            or tuple(mod.shape) != (B * q.shape[1], 768)
            or tuple(seqused.shape) != (q.shape[0],)):
        raise ValueError("native H100 SWA requires contiguous BF16/FP32 [A*B,S,128], FP32 modulation and window64; got "
                         f"q {tuple(q.shape)} {q.dtype} contiguous={q.is_contiguous()}, mod {tuple(mod.shape)} {mod.dtype}, "
                         f"seqused {tuple(seqused.shape)} {seqused.dtype}, B={B}, half_window={half_window}")
    if any(t.device != q.device for t in (mod, cs, sn, seqused, *weights)):
        raise ValueError("native H100 SWA tensors must be on the same CUDA device")
    if any(t.dtype != torch.float32 or not t.is_contiguous()
           or tuple(t.shape) != (B * q.shape[1], 16) for t in (cs, sn)):
        raise ValueError("native H100 SWA requires contiguous FP32 RoPE [B*S,16]")
    shapes = ((384, 128), (128, 128), (128, 128), (512, 128), (128, 256))
    if any(t.dtype != q.dtype or not t.is_contiguous() or tuple(t.shape) != shape
           for t, shape in zip(weights, shapes, strict=True)):
        raise ValueError("native H100 SWA requires canonical packed projection weights")


def _native_h100_fwd(q, mod, cs, sn, sequ, wqkv, wg, wo, wu, wd, B, hw, eps, save):
    _native_contract(q, mod, cs, sn, sequ, (wqkv, wg, wo, wu, wd), B, hw, save)
    ext = _native_h100_extension()
    N, S, _ = q.shape
    A, M = N // B, N * S
    bf = q.dtype == torch.bfloat16
    fusion = _native_b200_fusion(q, save, B)
    small = fusion and bf and A == 1 and M >= 4096
    def empty(*shape, dtype=None):
        return torch.empty(shape, device=q.device, dtype=q.dtype if dtype is None else dtype)
    def rows(stage, ins, outs):
        ext.native_rows(ins, outs, stage, A, B, S, bf, eps)
    wstack = empty(512, 128)
    ext.native_copy([wqkv, wg], [wstack[:384], wstack[384:]], 0)
    if bf:
        qkvg_ext = _native_h100_extension("qkvg_small") if small else ext
        qh, kh, vh, g, x, pq, pk = qkvg_ext.qkvg_fwd(q.reshape(M, 128), mod, cs, sn, wstack, A, B, S, eps, FP32_EPS, save)
    else:
        x = empty(M, 128)
        rows(0, [q, mod], [x])
        p = _native_mm(x, wstack.t())
        qh, kh, vh = [empty(N, 4, S, 32, dtype=torch.bfloat16) for _ in range(3)]
        g, pq, pk = [empty(M, 128) for _ in range(3)]
        rows(1, [p, cs, sn], [qh, kh, vh, g, pq, pk])
    o, lse = empty(M, 128, dtype=torch.bfloat16), empty(N, 4, S, dtype=torch.float32)
    if is_global(hw):
        # Global attention (the input feature embedder's atom encoder): the native stages around FlashAttention-4, which skips
        # rows past seqused -- they must read zero.
        from miniworld_engine.kernels.swa_dit.cuda.sm100 import attn_global_fwd

        o.zero_()
        attn_global_fwd(qh, kh, vh, sequ, o, lse)
    else:
        attention = ext.tiled_attention if _tiled_serves(qh) else ext.native_attention
        attention([qh, kh, vh, sequ], [o, lse], 0)
    if bf:
        ffn_ext = _native_h100_extension("fwd_small" if small else "fwd")
        packed_wu = torch.empty_like(wu)
        ext.native_copy([wu], [packed_wu], 2)
        out, q1, att, y, ffn = ffn_ext.ffn_fwd(q.reshape(M, 128), g, o, mod, wo, packed_wu, wd, A, B, S, eps, save)
    else:
        gated = empty(M, 128)
        rows(2, [o, g], [gated])
        att = _native_mm(gated, wo.t())
        q1, y = empty(M, 128), empty(M, 128)
        rows(3, [q, att, mod], [q1, y])
        ab = _native_mm(y, wu.t())
        hh = empty(M, 256)
        rows(4, [ab], [hh])
        ffn = _native_mm(hh, wd.t())
        out = empty(M, 128)
        rows(5, [q1, ffn, mod], [out])
    outputs = [out.view(N, S, 128)]
    if save:
        outputs += [qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn]
    return outputs


def _native_h100_bwd(dy, q, mod, cs, sn, sequ, wqkv, wg, wo, wu, wd,
                     qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B, hw, eps):
    _native_contract(q, mod, cs, sn, sequ, (wqkv, wg, wo, wu, wd), B, hw, True)
    dy = dy.contiguous()  # e.g. the expanded gradient of a sum
    ext = _native_h100_extension()
    N, S, _ = q.shape
    A, M, bf = N // B, N * S, q.dtype == torch.bfloat16
    fusion = _native_b200_fusion(q, True, B)
    def empty(*shape, dtype=None):
        return torch.empty(shape, device=q.device, dtype=q.dtype if dtype is None else dtype)
    def rows(stage, ins, outs):
        ext.native_rows(ins, outs, stage, A, B, S, bf, eps)
    dmod = torch.empty_like(mod)
    if bf and fusion:
        overwrite = fusion and A == 1
        ffn_ext = _native_h100_extension("bwd_overwrite" if overwrite else "bwd")
        wab, wdt, wabt = ext.native_ffn_pack(wu, wd, dmod, not overwrite)
        dq1, dffn, hh, dab = ffn_ext.ffn_bwd(
            dy.reshape(M, 128), q1, y, ffn, mod, dmod, wab, wdt, wabt,
            A, B, S, eps, 8 if A % 8 == 0 else 0)
    else:
        dffn = empty(M, 128)
        rows(6, [dy, mod, ffn], [dffn, dmod])
        ab, dh = _native_mm(y, wu.t()), _native_mm(dffn, wd)
        dab, hh = empty(M, 512), empty(M, 256)
        rows(7, [ab, dh], [dab, hh])
        dY = _native_mm(dab, wu)
        dq1 = empty(M, 128)
        rows(8, [dy, dY, q1, mod], [dq1, dmod])
    datt, gated = empty(M, 128), empty(M, 128)
    dO, dG, delta = empty(M, 128, dtype=torch.bfloat16), empty(M, 128), empty(N, 4, S, dtype=torch.float32)
    if bf and A == 1 and fusion:
        ext.fused_oproj_bwd([dq1, mod, att, o, g, wo], [datt, gated, dO, dG, delta, dmod], S)
    else:
        rows(9, [dq1, mod, att, o, g], [datt, gated, dmod])
        dgated = _native_mm(datt, wo)
        rows(10, [dgated, o, g], [dO, dG, delta])
    if is_global(hw):
        from miniworld_engine.kernels.swa_dit.cuda.sm100 import attn_global_bwd

        dqh, dkh, dvh = [torch.zeros_like(qh) for _ in range(3)]   # FA4 leaves the rows past seqused alone
        attn_global_bwd(qh, kh, vh, o, dO, lse, sequ, dqh, dkh, dvh)
    else:
        dqh, dkh, dvh = [torch.empty_like(qh) for _ in range(3)]
        ai, ao = [qh, kh, vh, sequ, dO, lse, delta], [dqh, dkh, dvh]
        attention = ext.tiled_attention if _tiled_serves(qh, dO) else ext.native_attention
        attention(ai, ao, 1)
        attention(ai, ao, 2)
    dp = empty(M, 512)
    wstack = empty(512, 128)
    ext.native_copy([wqkv, wg], [wstack[:384], wstack[384:]], 0)
    dq = empty(M, 128)
    if fusion and bf and A == 1:
        ext.fused_qkvg_bwd([q, mod, cs, sn, pq, pk, dqh, dkh, dvh, dG, dq1, wstack],
                           [dq, dp, dmod], S, eps)
    else:
        rows(11, [pq, pk, dqh, dkh, dvh, dG, cs, sn], [dp])
        dX = _native_mm(dp, wstack)
        rows(12, [dq1, dX, q, mod], [dq, dmod])
    def wgrad(gradient, activation):
        return _native_mm(gradient.t(), activation)
    return [dq.view_as(q), dmod, wgrad(dp[:, :384], x), wgrad(dp[:, 384:], x),
            wgrad(datt, gated), wgrad(dab, y), wgrad(dffn, hh)]


def _native_h100_cast_fake(values: list[torch.Tensor], templates: list[torch.Tensor]) -> list[torch.Tensor]:
    """One empty tensor per template, in the template's shape, dtype and device."""
    return [torch.empty_like(t) for t in templates]


@opaque(fake=_native_h100_cast_fake, name="swa_dit_h100_native_cast")
def native_h100_cast(values: list[torch.Tensor], templates: list[torch.Tensor]) -> list[torch.Tensor]:
    """Copy-cast each of ``values`` into a new tensor shaped and typed like the matching template."""
    out = [torch.empty_like(t) for t in templates]
    _native_h100_extension().native_copy(values, out, 0)
    return out


def _native_h100_mod_fwd_fake(c: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[R, 768] fp32 modulation, c as fp32, w as fp32."""
    return (c.new_empty((c.shape[0], 768), dtype=torch.float32),
            c.new_empty(c.shape, dtype=torch.float32), w.new_empty(w.shape, dtype=torch.float32))


@opaque(fake=_native_h100_mod_fwd_fake, name="swa_dit_h100_native_mod_fwd")
def native_h100_mod_fwd(c: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(silu-free fp32 c @ w^T, c as fp32, w as fp32) on the H100 native copy kernels and cuBLAS."""
    ext = _native_h100_extension()
    a, w32 = torch.empty_like(c, dtype=torch.float32), torch.empty_like(w, dtype=torch.float32)
    ext.native_copy([c], [a], 1)
    ext.native_copy([w], [w32], 0)
    with torch.autocast("cuda", enabled=False):
        if c.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            # A bf16 block: bf16 operands -- the module's own GEMM precision -- on the tensor cores, fp32 accumulation and output.
            # The fp32 copies ran an FFMA GEMM (A 5, 8192 atoms: 49 us shared, 228 us per-sample conditioning rows).
            return torch.mm(a.to(torch.bfloat16), w.t(), out_dtype=torch.float32), a, w32
        return _native_mm(a, w32.t()), a, w32


def _native_h100_mod_bwd_fake(dy: torch.Tensor, c: torch.Tensor, a: torch.Tensor, w32: torch.Tensor,
                              w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """dc like c, dweight like w."""
    return torch.empty_like(c), torch.empty_like(w)


@opaque(fake=_native_h100_mod_bwd_fake, name="swa_dit_h100_native_mod_bwd")
def native_h100_mod_bwd(dy: torch.Tensor, c: torch.Tensor, a: torch.Tensor, w32: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(dc, dweight) of :func:`native_h100_mod_fwd`; dc and dweight are cast back to c's and w's dtype."""
    ext = _native_h100_extension()
    with torch.autocast("cuda", enabled=False):
        da, dw = _native_mm(dy, w32), _native_mm(dy.t(), a)
    dc, dweight = torch.empty_like(c), torch.empty_like(w)
    ext.native_rows([c, da], [dc], 14, 1, 1, c.shape[0], c.dtype == torch.bfloat16, FP32_EPS)
    ext.native_copy([dw], [dweight], 0)
    return dc, dweight



def _augments(N: int, B: int) -> int:
    """The augment count as the cache key carries it: A = N // B, exactly (1 included), capped to what ``pack`` keys."""
    return min(N // B, _A_KEY_MAX)


def _cuda(which: str, wanted: bool, device: torch.device) -> Any:
    """The hand-CUDA extension for stage ``which``, or None -> the Triton kernel serves the stage."""
    if not wanted or settings.current().engine_backend == "triton":
        return None
    from miniworld_engine.kernels.swa_dit.cuda.loader import extension, is_sm90

    if not is_sm90(device):
        return None
    return extension(which)


_SM100_FAILED = False


def _sm100(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> Any:
    """The sm_100a kernels (``cuda/sm100``) when they serve this bf16 call, else None -> the Triton path."""
    global _SM100_FAILED
    if (
        _SM100_FAILED
        or (not settings.current().swa_dit_sm100)
        or settings.current().engine_backend == "triton"
    ):
        return None
    from miniworld_engine.kernels.swa_dit.cuda import sm100

    if not sm100.supported(q, nhid, half_window, eps):
        return None
    try:
        sm100.kernels(
            q.device.index
            if q.device.index is not None
            else torch.cuda.current_device()
        )
    except Exception as exc:  # a toolchain or driver problem keeps the Triton path
        _SM100_FAILED = True
        warnings.warn(
            f"sm_100a SWA atom DiT kernels unavailable, keeping the Triton path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return sm100


_TF32_FAILED = False


def _tf32_off() -> bool:
    """The switches that keep the Triton fp32 path: ``settings.swa_dit_tf32=0``, ``settings.swa_dit_sm100=0``, engine_backend triton."""
    return (
        (not settings.current().swa_dit_tf32)
        or (not settings.current().swa_dit_sm100)
        or settings.current().engine_backend == "triton"
    )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@device_constant
def _tf32_bwd_ready(device: torch.device) -> bool:
    """Whether an sm_100a TF32 backward serves fp32 training calls on ``device``. None does: the first one (``cuda/sm100/tf32_bwd``)
    gave different gradients for identical steps and faulted on poisoned memory, so it was removed; until its replacement lands, an
    fp32 training call (forward with saves, the hoisted modulation with a gradient) keeps the Triton fp32 path. Inference keeps the
    TF32 forward kernels."""
    return False


@device_constant
def _tf32_ready(device: torch.device, save: bool) -> bool:
    """Whether the sm_100a TF32 kernels serve fp32 calls on ``device``: the switches, B200, and the forward kernels build and load
    -- with ``save`` (a training forward, whose fp32 saves only the TF32 backward reads) the backward's too. Build and load happen
    here, once per device; a failure warns once and keeps the Triton fp32 path. A constant to the compiled graph
    (``device_constant``): Dynamo bakes the answer in and never traces the build."""
    global _TF32_FAILED
    if (
        _TF32_FAILED
        or device.type != "cuda"
        or _tf32_off()
        or not tf32_fwd.is_b200(device)
    ):
        return False
    try:
        tf32_fwd.kernels_tf32(_device_index(device))
    except Exception as exc:  # a toolchain or driver problem keeps the Triton path
        _TF32_FAILED = True
        warnings.warn(
            f"sm_100a TF32 SWA atom DiT kernels unavailable, keeping the Triton fp32 path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return not save or _tf32_bwd_ready(device)


def _tf32(q: torch.Tensor, nhid: int, half_window: int, eps: float, save: bool) -> Any:
    """The sm_100a TF32 forward module (``cuda/sm100/tf32_fwd``) when it serves this fp32 call, else None -> the Triton fp32 path.
    Traceable: tensor metadata (``tf32_fwd.supported_tf32``) and the per-device constant ``_tf32_ready``. A deterministic function
    of the call (shapes, dtype, device, switches, earlier failures), so the op's fake reaches the same answer as the launch."""
    if q.dtype != torch.float32 or not tf32_fwd.supported_tf32(
        q, nhid, half_window, eps
    ):
        return None
    if not _tf32_ready(q.device, save):
        return None
    return tf32_fwd




#: ``swa_dit_rows_impl="auto"``: the team-gm wgmma kernels serve a stage from this many rows (N*S) on, and only when the
#: augment count is a multiple of 8 (their forward tiles are 8 augments x 8 atoms: at A = 1 seven eighths of every tile is
#: padding); below, and at every other A, the slab kernels (``cuda/swa_slab.cu``). Measured on H100 (graph replay, per call):
#: qkvg / out-projection + FFN forward cross over between M = 4096 (A = 8: slab 10.8 / 13.3 us, wgmma 12.6 / 15.1) and 6144
#: (A = 48: slab 16.5 / 19.4, wgmma 12.8 / 15.5); the FFN backward between 16384 (A = 8, 16: 50 vs 52) and 24576 (A = 48:
#: 69 vs 66).
_WGMMA_MIN_ROWS = {"qkvg": 5120, "fwd": 5120, "bwd": 20480}


def _rows_cuda(which: str, wanted: bool, M: int, A: int, device: torch.device) -> tuple[str, Any]:
    """Which hand-CUDA kernel serves the qkvg forward ("qkvg"), the out-projection + FFN forward ("fwd") or the FFN
    backward ("bwd") at M rows and A augments: ("slab", ext), ("wgmma", ext) or ("", None) -> Triton. A slab extension that
    does not build falls back to the wgmma one."""
    if not wanted:
        return "", None
    impl = settings.current().swa_dit_rows_impl
    if impl == "auto":
        impl = "wgmma" if A % 8 == 0 and _WGMMA_MIN_ROWS[which] <= M else "slab"
    if impl == "slab":
        ext = _cuda("slab", True, device)
        if ext is not None:
            return "slab", ext
    ext = _cuda(which, True, device)
    return ("wgmma", ext) if ext is not None else ("", None)


def _attn_served(C: int, H: int, half_window: int) -> bool:
    """The hand-CUDA window attention's contract: 4 heads of 32, half window 64 (bf16 Q / K / V, which both precisions pass)."""
    return C == D_ATOM and H == N_HEAD and half_window == HALF_WINDOW


@functools.lru_cache(maxsize=16)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _pack_ffn64(wu: torch.Tensor) -> torch.Tensor:
    """rows per 64-wide hidden chunk j: [Wu[64j..64j+63] ; Wu[NHID + 64j..]] (the forward kernel's a|b tile)."""
    NH = wu.shape[0] // 2
    C = wu.shape[1]
    return (
        wu.view(2, NH // 64, 64, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()
    )


def _pack_ffn(
    wu: torch.Tensor, wd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """wab: per 32-wide hidden chunk j rows [Wu[32j..] ; Wu[NHID + 32j..]]; wdt = Wd^T; wabt = wab^T."""
    NH = wd.shape[1]
    C = wd.shape[0]
    wab = (
        wu.view(2, NH // 32, 32, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()
    )
    return wab, wd.t().contiguous(), wab.t().contiguous()


_PACK_CACHE: dict[int, tuple] = {}


def _ffn_packs(wu: torch.Tensor, wd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(wab64, wab32, wdt, wabt) of the wgmma FFN kernels: one launch (``cuda/swa_pack.cu``), kept for the next call on the same
    weight tensors at the same version (the backward of the step the forward packed for) -- not while a CUDA graph is captured,
    where every replay has to repack the weights an optimizer step changed. The torch copies where the extension does not build."""
    import weakref

    ext = _cuda("pack", True, wu.device) if wu.dtype == torch.bfloat16 and wu.is_contiguous() and wd.is_contiguous() else None
    if ext is None:
        wab = _pack_ffn(wu, wd)
        return (_pack_ffn64(wu), *wab)
    if torch.cuda.is_current_stream_capturing():
        return tuple(ext.pack_ffn(wu, wd))
    dev = wu.device.index if wu.device.index is not None else torch.cuda.current_device()
    hit = _PACK_CACHE.get(dev)
    if hit is not None and hit[0]() is wu and hit[1]() is wd and hit[2] == (wu._version, wd._version):
        return hit[3]
    packs = tuple(ext.pack_ffn(wu, wd))
    _PACK_CACHE[dev] = (weakref.ref(wu), weakref.ref(wd), (wu._version, wd._version), packs)
    return packs


def _swa_dit_block_fwd_fake(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save):
    """Shapes of the forward's outputs: [out [N, S, C] in q's dtype] alone, or with ``save`` the 13 tensors the backward
    reads -- Q / K / V head-major [N, 4, S, C/4] (the attention operands: bf16, except fp32 on the TF32 path), G [N*S, C] in
    q's dtype, O [N*S, C] (the attention output: bf16, except fp32 on the TF32 path), lse [N, 4, S] fp32, then q1, x, pq, pk,
    att, y, ffn [N*S, C] in q's dtype. The structure depends on ``save`` alone; the heads' / O's dtype on the path, chosen by the
    same ``_tf32`` the launch asks."""
    N, S, C = q.shape
    H = N_HEAD
    M = N * S
    outputs = [q.new_empty((N, S, C))]
    if save:
        tf32 = (
            q.dtype == torch.float32
            and not is_global(half_window)
            and _tf32(q, wd.shape[1], half_window, eps, save) is not None
        )
        hd = torch.float32 if tf32 else torch.bfloat16
        heads = [q.new_empty((N, H, S, C // H), dtype=hd) for _ in range(3)]
        saved = [q.new_empty((M, C)) for _ in range(7)]
        outputs += [
            *heads,
            q.new_empty((M, C)),
            q.new_empty((M, C), dtype=hd),
            q.new_empty((N, H, S), dtype=torch.float32),
            *saved,
        ]
    return outputs


@opaque(fake=_swa_dit_block_fwd_fake, name="swa_dit_block_fwd")
def swa_dit_block_fwd(
    q: torch.Tensor,
    mod: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqused: torch.Tensor,
    wqkv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    B: int,
    half_window: int,
    eps: float,
    save: bool,
) -> list[torch.Tensor]:
    """The block forward. q [N, S, C] bf16 or fp32 (the weights in the same dtype); mod [B*S, 6C] fp32 (hoisted adaLN
    modulation of this block); cos / sin [B*S, D/2] fp32; seqused [N] int32. Returns [out] or, with ``save``,
    [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF] (the tensors the backward reads)."""
    if native_h100_stages(q) or _h100_global(q, half_window):
        # The native stages take an fp32 modulation; a bf16 one comes from the composed bf16 hoist (global attention, which only the
        # native stages serve on H100, under a bf16 call that the composed stages otherwise take).
        return _native_h100_fwd(q, mod.float(), cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save)
    if is_global(half_window):
        sm100 = _sm100(q, wd.shape[1], half_window, eps)
        if sm100 is None:
            raise RuntimeError(
                "swa_dit_block: global attention runs on the sm_100a stages (bf16, B200) only -- check "
                "interface.refusal first"
            )
        return sm100.block_fwd(
            q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window
        )
    if q.dtype == torch.float32:
        tf32 = _tf32(q, wd.shape[1], half_window, eps, save)
        if tf32 is not None:  # B200: every stage on the TF32 tensor-core kernels
            return tf32.block_fwd_tf32(
                q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window
            )
        return _swa_dit_fwd_fp32_launch(
            q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save
        )
    sm100 = _sm100(q, wd.shape[1], half_window, eps)
    if sm100 is not None:
        return sm100.block_fwd(
            q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window
        )
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = _augments(N, B)
    NHID = wd.shape[1]
    policy = settings.current()
    qf = q.reshape(M, C)
    served = C == D_ATOM and H == N_HEAD and NHID == N_HIDDEN and q.dtype == torch.bfloat16
    impl_q = _rows_cuda("qkvg", policy.swa_dit_qkvg_fwd_cuda and served, M, N // B, q.device)
    impl_f = _rows_cuda("fwd", policy.swa_dit_ffn_fwd_cuda and served, M, N // B, q.device)
    if impl_q[0] == "slab":
        Qh, Kh, Vh, G, Xs, PQs, PKs = impl_q[1].qkvg_fwd(qf, mod, cos, sin, wqkv, wg, N, B, S, eps, FP32_EPS, save)
    elif impl_q[0] == "wgmma":
        Qh, Kh, Vh, G, Xs, PQs, PKs = impl_q[1].qkvg_fwd(qf, mod, cos, sin, torch.cat([wqkv, wg]), N // B, B, S, eps, FP32_EPS,
                                                         save)
    else:
        Qh = torch.empty(N, H, S, D, device=q.device, dtype=q.dtype)
        Kh = torch.empty_like(Qh)
        Vh = torch.empty_like(Qh)
        G = torch.empty_like(qf)
        r1 = torch.empty(M, device=q.device, dtype=torch.float32)
        if save:
            Xs = torch.empty_like(qf)
            PQs = torch.empty_like(qf)
            PKs = torch.empty_like(qf)
        else:
            Xs = PQs = PKs = G
        _swa_qkvg_fwd_kernel[lambda m: (triton.cdiv(M, m["BR"]),)](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
            qf, mod, cos, sin, wqkv, wg, Qh, Kh, Vh, G, r1, Xs, PQs, PKs, M, S, B, eps, FP32_EPS,
            shape_key=atom_key(S, A=A, C=C), C=C, H=H, D=D, MODW=6 * C, SAVE=save)
    ext = _cuda("attn", policy.swa_dit_attn_cuda and _attn_served(C, H, half_window), q.device)
    if ext is not None:
        O, lse = ext.attn_fwd(Qh, Kh, Vh, seqused, D ** -0.5)
    else:
        O = torch.empty_like(qf)
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        _swa_attn_fwd_kernel[lambda m: (triton.cdiv(S, m["BM"]), N * H)](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
            Qh, Kh, Vh, seqused, O, lse, S, D ** -0.5, shape_key=atom_key(S, A=A, C=C), C=C, H=H, D=D, HW=half_window)
    if impl_f[0] == "slab":
        q1, Att = impl_f[1].oproj_fwd(qf, G, O, mod, wo, N, B, S, save)
        q2, Ys, FFs = impl_f[1].ffn_fwd(q1, mod, wu, wd, N, B, S, eps, save)
    elif impl_f[0] == "wgmma":
        q2, q1, Att, Ys, FFs = impl_f[1].ffn_fwd(qf, G, O, mod, wo, _ffn_packs(wu, wd)[0], wd, N // B, B, S, eps, save)
    else:
        q2 = torch.empty_like(qf)
        q1 = torch.empty_like(qf) if save else q2
        r2 = torch.empty(M, device=q.device, dtype=torch.float32)
        if save:
            Att = torch.empty_like(qf)
            Ys = torch.empty_like(qf)
            ABs = Ys
            FFs = torch.empty_like(qf)
        else:
            Att = Ys = ABs = FFs = q2
        _swa_oproj_ffn_fwd_kernel[
            lambda m: (triton.cdiv(M, m["BR"]),)
        ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
            qf,
            O,
            G,
            mod,
            wo,
            wu,
            wd,
            q1,
            q2,
            r2,
            Att,
            Ys,
            ABs,
            FFs,
            M,
            S,
            B,
            eps,
            shape_key=atom_key(S, A=A, C=C, NHID=NHID),
            C=C,
            NHID=NHID,
            MODW=6 * C,
            SAVE=save,
        )
    outputs = [q2.view(N, S, C)]
    if save:
        outputs += [Qh, Kh, Vh, G, O, lse, q1, Xs, PQs, PKs, Att, Ys, FFs]
    return outputs


def _fp32_cuda(C: int, NHID: int, half_window: int, device: torch.device) -> Any:
    """The fp32 hand-CUDA extension (``cuda/swa_fp32.cu``) when it serves this call, else None -> the fp32 Triton.

    It serves only while the process allows TF32 (``torch.get_float32_matmul_precision()`` "high" or "medium" -- what
    MiniWorld's trainer sets, and what the per-op path's cuBLAS projections then run at): its FFN splits the weights but
    not the activations, which matches the per-op path under TF32, not under "highest", where the per-op FFN is exact
    fp32. There the Triton kernels' tf32x3 FFN serves instead, off H100 / B200 (there "highest" never reaches this: the callers run the
    PyTorch statements, interface.highest_fp32_on_hopper_blackwell). Read on every call, like the other settings.
    """
    wanted = (settings.current().swa_dit_fp32_cuda and torch.get_float32_matmul_precision() != "highest"
              and C == D_ATOM and NHID == N_HIDDEN and half_window == HALF_WINDOW)
    return _cuda("fp32", wanted, device)


def _swa_dit_fwd_fp32_launch(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save):
    """The fp32 forward: qkvg -> window attention on bf16 operands -> out-proj + FFN, as the CUDA extension (sm_90,
    ``swa_dit_fp32_cuda``) or ``forward_fp32``'s Triton kernels around the shared attention kernel. Same outputs as the
    bf16 forward; everything but the attention operands (Q / K / V, O) is fp32."""
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = _augments(N, B)
    NHID = wd.shape[1]
    dev = q.device
    ext = _fp32_cuda(C, NHID, half_window, dev)
    if ext is not None:
        outputs = ext.fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, FP32_EPS, save)
        outputs[0] = outputs[0].view(N, S, C)
        return outputs
    qf = q.reshape(M, C)
    Qh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
    Kh = torch.empty_like(Qh)
    Vh = torch.empty_like(Qh)
    G = torch.empty_like(qf)
    r1 = torch.empty(M, device=dev, dtype=torch.float32)
    if save:
        Xs = torch.empty_like(qf)
        PQs = torch.empty_like(qf)
        PKs = torch.empty_like(qf)
    else:
        Xs = PQs = PKs = G
    _swa_qkvg_fwd_fp32_kernel[
        lambda m: (triton.cdiv(M, m["BR"]),)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        qf,
        mod,
        cos,
        sin,
        wqkv,
        wg,
        Qh,
        Kh,
        Vh,
        G,
        r1,
        Xs,
        PQs,
        PKs,
        M,
        S,
        B,
        eps,
        FP32_EPS,
        shape_key=atom_key(S, A=A, C=C),
        C=C,
        H=H,
        D=D,
        MODW=6 * C,
        SAVE=save,
    )
    O = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    _swa_attn_fwd_kernel[
        lambda m: (triton.cdiv(S, m["BM"]), N * H)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        Qh,
        Kh,
        Vh,
        seqused,
        O,
        lse,
        S,
        D**-0.5,
        shape_key=atom_key(S, A=A, C=C),
        C=C,
        H=H,
        D=D,
        HW=half_window,
    )
    q2 = torch.empty_like(qf)
    q1 = torch.empty_like(qf) if save else q2
    r2 = torch.empty(M, device=dev, dtype=torch.float32)
    if save:
        Att = torch.empty_like(qf)
        Ys = torch.empty_like(qf)
        FFs = torch.empty_like(qf)
    else:
        Att = Ys = FFs = q2
    _swa_oproj_ffn_fwd_fp32_kernel[
        lambda m: (triton.cdiv(M, m["BR"]),)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        qf,
        O,
        G,
        mod,
        wo,
        wu,
        wd,
        q1,
        q2,
        r2,
        Att,
        Ys,
        FFs,
        M,
        S,
        B,
        eps,
        shape_key=atom_key(S, A=A, C=C, NHID=NHID),
        C=C,
        NHID=NHID,
        MODW=6 * C,
        SAVE=save,
    )
    outputs = [q2.view(N, S, C)]
    if save:
        outputs += [Qh, Kh, Vh, G, O, lse, q1, Xs, PQs, PKs, Att, Ys, FFs]
    return outputs


def _mm32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with an fp32 result: a weight gradient off bf16 operands keeps cuBLAS's fp32 accumulator instead of rounding it."""
    return (
        torch.mm(a, b, out_dtype=torch.float32) if a.dtype == torch.bfloat16 else a @ b
    )


def _swa_dit_block_bwd_fake(
    dy,
    q,
    mod,
    cos,
    sin,
    seqused,
    wqkv,
    wg,
    wo,
    wu,
    wd,
    qh,
    kh,
    vh,
    g,
    o,
    lse,
    q1,
    x,
    pq,
    pk,
    att,
    y,
    ffn,
    B,
    half_window,
    eps,
):
    """Shapes of the backward's outputs: dq like q ([N, S, C], q's dtype), dmod like mod (fp32 [B*S, 6C]), and each
    weight gradient in its weight's shape, fp32."""
    return [
        torch.empty_like(q),
        torch.empty_like(mod),
        *(
            torch.empty(w.shape, dtype=torch.float32, device=w.device)
            for w in (wqkv, wg, wo, wu, wd)
        ),
    ]


@opaque(fake=_swa_dit_block_bwd_fake, name="swa_dit_block_bwd")
def swa_dit_block_bwd(
    dy: torch.Tensor,
    q: torch.Tensor,
    mod: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqused: torch.Tensor,
    wqkv: torch.Tensor,
    wg: torch.Tensor,
    wo: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    qh: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    g: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    q1: torch.Tensor,
    x: torch.Tensor,
    pq: torch.Tensor,
    pk: torch.Tensor,
    att: torch.Tensor,
    y: torch.Tensor,
    ffn: torch.Tensor,
    B: int,
    half_window: int,
    eps: float,
) -> list[torch.Tensor]:
    """Backward of :func:`swa_dit_block_fwd` from its saved tensors. dy [N, S, C] in q's dtype. Returns [dq [N, S, C] in q's
    dtype, dmod [B*S, 6C] fp32, dWqkv, dWg, dWo, dWu, dWd fp32]: the weight gradients leave their fp32 accumulators unrounded
    (the parameters may be an fp32 master; ``autograd.SWADiTBlockFunction`` hands them back in the parameters' dtype)."""
    if native_h100_stages(q) or _h100_global(q, half_window):
        grads = _native_h100_bwd(dy, q, mod.float(), cos, sin, seqused, wqkv, wg, wo, wu, wd,
                                 qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B, half_window, eps)
        grads[1] = grads[1].to(mod.dtype)   # dmod like mod (the contract), whatever the native stages computed it in
        return grads
    if is_global(half_window):
        sm100 = _sm100(q, wd.shape[1], half_window, eps)
        if sm100 is None:
            raise RuntimeError(
                "swa_dit_block: global attention runs on the sm_100a stages (bf16, B200) only -- check "
                "interface.refusal first"
            )
        return sm100.block_bwd(
            dy,
            q,
            mod,
            cos,
            sin,
            seqused,
            wqkv,
            wg,
            wo,
            wu,
            wd,
            qh,
            kh,
            vh,
            g,
            o,
            lse,
            q1,
            x,
            pq,
            pk,
            att,
            y,
            ffn,
            B,
            half_window,
        )
    if q.dtype == torch.float32:
        if (
            qh.dtype == torch.float32
        ):  # fp32 heads: saves of a TF32 training forward, which no backward serves
            raise RuntimeError(
                "swa_dit_block: fp32 saves of the TF32 forward have no backward (_tf32_bwd_ready is False)"
            )
        return _swa_dit_bwd_fp32_launch(
            dy,
            q,
            mod,
            cos,
            sin,
            seqused,
            wqkv,
            wg,
            wo,
            wu,
            wd,
            qh,
            kh,
            vh,
            g,
            o,
            lse,
            q1,
            x,
            pq,
            pk,
            att,
            y,
            ffn,
            B,
            half_window,
            eps,
        )
    sm100 = _sm100(q, wd.shape[1], half_window, eps)
    if sm100 is not None:
        return sm100.block_bwd(
            dy,
            q,
            mod,
            cos,
            sin,
            seqused,
            wqkv,
            wg,
            wo,
            wu,
            wd,
            qh,
            kh,
            vh,
            g,
            o,
            lse,
            q1,
            x,
            pq,
            pk,
            att,
            y,
            ffn,
            B,
            half_window,
        )
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = N // B
    Ak = _augments(N, B)
    NHID = wd.shape[1]
    dev = q.device
    policy = settings.current()
    fused_dw = policy.swa_dit_ffn_dw == "fused"
    dq1_dtype = torch.float32 if policy.swa_dit_dq1 == "fp32" else torch.bfloat16
    dq1 = torch.empty(M, C, device=dev, dtype=dq1_dtype)
    dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dG = torch.empty_like(dO)
    Dv = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    if fused_dw:
        dab = hh = dffn = dO
    else:
        dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.bfloat16)
        hh = torch.empty(M, NHID, device=dev, dtype=torch.bfloat16)
        dffn = torch.empty_like(dO)
    datt = torch.empty_like(dO)
    gated = torch.empty_like(dO)
    grid_t = (
        lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    )  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
    dy2 = dy.reshape(M, C)
    dwu: torch.Tensor | None = None
    dwd: torch.Tensor | None = None
    wanted = policy.swa_dit_ffn_bwd_cuda and not fused_dw and C == D_ATOM and NHID == N_HIDDEN and q.dtype == torch.bfloat16
    impl_f = _rows_cuda("bwd", wanted, M, A, dev)
    # the attention side on hand-CUDA (sm_90) -- see _bwd_attention_side
    served = C == D_ATOM and H == N_HEAD and q.dtype == torch.bfloat16 and dq1_dtype == torch.bfloat16
    ext_o = _cuda("slab", policy.swa_dit_oproj_bwd_cuda and served, dev)
    ext_a = _cuda("attn", policy.swa_dit_attn_cuda and _attn_served(C, H, half_window) and q.dtype == torch.bfloat16, dev)
    ext_q = _cuda("slab", policy.swa_dit_qkvg_bwd_cuda and served, dev)
    # dmod: the slab kernels own its column blocks and store (no accumulation) when one slab holds every augment (A in 1, 2, 4, 8,
    # 16: swa_slab.cu pick_sp), so only the other paths need it zeroed
    plain = A in (1, 2, 4, 8, 16) and impl_f[0] == "slab" and ext_o is not None and ext_q is not None
    dmod = (torch.empty if plain else torch.zeros)(B * S, 6 * C, device=dev, dtype=torch.float32)
    if impl_f[0] == "slab":
        dq1, dffn, hh, dab = impl_f[1].ffn_bwd(dy2, q1, y, ffn, mod, dmod, wu, wd, A, B, S, eps)
    elif impl_f[0] == "wgmma":
        dq1, dffn, hh, dab = impl_f[1].ffn_bwd(dy2, q1, y, ffn, mod, dmod, *_ffn_packs(wu, wd)[1:], A, B, S, eps,
                                               8 if A % 8 == 0 else 0)
    else:
        _swa_ffn_bwd_kernel[grid_t](
            dy2,
            q1,
            mod,
            wu,
            wd,
            y,
            ffn,
            dq1,
            dab,
            hh,
            dffn,
            dmod,
            S,
            A,
            B,
            eps,
            shape_key=atom_key(S, A=Ak, C=C, NHID=NHID),
            C=C,
            NHID=NHID,
            MODW=6 * C,
            DWOPS=not fused_dw,
        )
        if fused_dw:
            dwu32 = torch.zeros(2 * NHID, C, device=dev, dtype=torch.float32)
            dwd32 = torch.zeros(C, NHID, device=dev, dtype=torch.float32)
            sms = _sm_count(
                dev.index if dev.index is not None else torch.cuda.current_device()
            )
            ns = (
                lambda m: max(
                    1, min(triton.cdiv(M, m["BR"]), 2 * sms // (NHID // m["HS"]))
                )
            )  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
            _swa_ffn_dw_kernel[lambda m: (ns(m), NHID // m["HS"])](
                dy2, mod, wu, wd, y, dwu32, dwd32, M, S, B,
                shape_key=atom_key(S, A=Ak, C=C, NHID=NHID), NSPLIT=0, C=C, NHID=NHID, MODW=6 * C)
            dwu, dwd = dwu32, dwd32
    if ext_o is not None or ext_a is not None or ext_q is not None:
        served_w = (policy.swa_dit_wgrad_cuda and ext_q is not None and dwu is None and C == D_ATOM and NHID == N_HIDDEN
                    and q.dtype == torch.bfloat16)
        # TMA + wgmma kernel at every row count (bit-identical to the mma.sync one, faster than cuBLAS up to 196k rows); the
        # mma.sync kernel serves where it does not build, cuBLAS above its row limit
        ext_w = _cuda("wgwg", served_w, dev) or _cuda("wgrad", served_w and M <= _WGRAD_MAX_ROWS, dev)
        side = _bwd_attention_side(ext_o, ext_a, ext_q, dq1, q, mod, cos, sin, seqused, wqkv, wg, wo, qh, kh, vh,
                                   g, o, lse, x, pq, pk, att, dmod, B, half_window, eps, Ak, raw=ext_w is not None)
        if ext_w is not None:
            dq, dP, datt, gated = side
            dwqkv, dwg, dwo, dwu, dwd = ext_w.wgrad(dP, x.reshape(M, C), datt, gated, dab, y.reshape(M, C), dffn, hh)
            # the weight gradients leave as fp32, as the contract (and the fake) says; the autograd node casts them
            return [dq.view(N, S, C), dmod, *(w.float() for w in (dwqkv, dwg, dwo, dwu, dwd))]
        dq, dwqkv, dwg, dwo = side
        if dwu is None or dwd is None:
            dwu = dab.t() @ y
            dwd = dffn.t() @ hh
        return [dq.view(N, S, C), dmod, *(w.float() for w in (dwqkv, dwg, dwo, dwu, dwd))]
    _swa_oproj_bwd_kernel[grid_t](
        dq1,
        o,
        g,
        mod,
        wo,
        att,
        dO,
        dG,
        Dv,
        datt,
        gated,
        dmod,
        S,
        A,
        B,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        MODW=6 * C,
    )
    dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
    dKh = torch.empty_like(dQh)
    dVh = torch.empty_like(dQh)
    _swa_attn_bwd_dq_kernel[
        lambda m: (triton.cdiv(S, m["BM"]), N * H)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        qh,
        kh,
        vh,
        dO,
        lse,
        Dv,
        seqused,
        dQh,
        S,
        D**-0.5,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        HW=half_window,
    )
    _swa_attn_bwd_dkv_kernel[
        lambda m: (triton.cdiv(S, m["BN"]), N * H)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        qh,
        kh,
        vh,
        dO,
        lse,
        Dv,
        seqused,
        dKh,
        dVh,
        S,
        D**-0.5,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        HW=half_window,
    )
    dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16)
    _swa_qkvg_bwd_kernel[grid_t](
        q.reshape(M, C),
        mod,
        cos,
        sin,
        wqkv,
        wg,
        pq,
        pk,
        dQh,
        dKh,
        dVh,
        dG,
        dq1,
        dq,
        dP,
        dmod,
        S,
        A,
        B,
        eps,
        FP32_EPS,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        MODW=6 * C,
    )
    dwqkv = _mm32(dP[:, : 3 * C].t(), x)
    dwg = _mm32(dP[:, 3 * C :].t(), x)
    dwo = _mm32(datt.t(), gated)
    if dwu is None or dwd is None:
        dwu = _mm32(dab.t(), y)
        dwd = _mm32(dffn.t(), hh)
    return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd]


def _bwd_attention_side(ext_o, ext_a, ext_q, dq1, q, mod, cos, sin, seqused, wqkv, wg, wo, qh, kh, vh, g, o, lse, x, pq, pk,
                        att, dmod, B, half_window, eps, Ak, raw=False):
    """The bf16 backward from dq1 down to dq when at least one of its three stages is hand-CUDA: out-projection
    (``ext_o``), window attention dq / dk / dv (``ext_a``), qkvg (``ext_q``); a None extension runs that stage's Triton
    kernel(s). Returns (dq [M, C], dWqkv, dWg, dWo); with ``raw`` (and a CUDA qkvg stage) the three weight gradients are not
    computed: returns (dq, dP [M, 4C], datt, gated) -- their operands, for the one-launch weight-gradient kernel.

    The CUDA stages work in one [M, 4C] bf16 buffer ``dP``: the attention backward writes dq | dk | dv row-major into its
    column blocks 0 | 1 | 2, the out-projection backward writes dG into block 3, and the qkvg backward turns dq, dk into
    dp_q, dp_k in place -- so the Triton path's dP copies of dv and dG never happen. A Triton stage next to a CUDA one
    exchanges its tensors in its own layout (head-major attention gradients, row-major [M, C] dG), converted here.
    """
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = N // B
    dev = q.device
    bf = torch.bfloat16
    scale = D ** -0.5
    dP = torch.empty(M, 4 * C, device=dev, dtype=bf)
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
    if ext_o is not None:
        dO, datt, gated, Dv = ext_o.oproj_bwd(dq1, g, o, att, mod, dmod, wo, dP, A, B, S)
        dG = dP[:, 3 * C:]
    else:
        dO = torch.empty(M, C, device=dev, dtype=bf)
        dG = torch.empty_like(dO)
        datt = torch.empty_like(dO)
        gated = torch.empty_like(dO)
        Dv = torch.empty(N, H, S, device=dev, dtype=torch.float32)
        _swa_oproj_bwd_kernel[grid_t](
            dq1, o, g, mod, wo, att, dO, dG, Dv, datt, gated, dmod, S, A, B,
            shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
        if ext_q is not None:
            dP[:, 3 * C:] = dG
    heads: list[torch.Tensor] | None = None
    if ext_a is not None:
        if ext_q is not None:
            ext_a.attn_bwd(qh, kh, vh, dO, lse, Dv, seqused, dP, scale)
        else:
            heads = list(ext_a.attn_bwd(qh, kh, vh, dO, lse, Dv, seqused, None, scale))
    else:
        dQh = torch.empty(N, H, S, D, device=dev, dtype=bf)
        dKh = torch.empty_like(dQh)
        dVh = torch.empty_like(dQh)
        _swa_attn_bwd_dq_kernel[lambda m: (triton.cdiv(S, m["BM"]), N * H)](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
            qh, kh, vh, dO, lse, Dv, seqused, dQh, S, scale, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D,
            HW=half_window)
        _swa_attn_bwd_dkv_kernel[lambda m: (triton.cdiv(S, m["BN"]), N * H)](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
            qh, kh, vh, dO, lse, Dv, seqused, dKh, dVh, S, scale, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D,
            HW=half_window)
        heads = [dQh, dKh, dVh]
        if ext_q is not None:
            for i, t in enumerate(heads):                          # head-major [N, H, S, D] -> row-major column block i
                dP[:, i * C:(i + 1) * C].view(N, S, H, D).copy_(t.transpose(1, 2))
    if ext_q is not None:
        dq = ext_q.qkvg_bwd(q.reshape(M, C), pq, pk, dq1, mod, dmod, cos, sin, wqkv, wg, dP, A, B, S, eps, FP32_EPS)
        if raw:
            return dq, dP, datt, gated
        dwqkv = dP[:, :3 * C].t() @ x
        dwg = dP[:, 3 * C:].t() @ x
    else:
        assert heads is not None
        dQh, dKh, dVh = heads
        dq = torch.empty(M, C, device=dev, dtype=bf)
        dPt = torch.empty(M, 4 * C, device=dev, dtype=bf)
        _swa_qkvg_bwd_kernel[grid_t](
            q.reshape(M, C), mod, cos, sin, wqkv, wg, pq, pk, dQh, dKh, dVh, dG.contiguous(), dq1, dq, dPt, dmod, S, A, B,
            eps, FP32_EPS, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
        dwqkv = dPt[:, :3 * C].t() @ x
        dwg = dPt[:, 3 * C:].t() @ x
    dwo = datt.t() @ gated
    return dq, dwqkv, dwg, dwo


def _swa_dit_bwd_fp32_launch(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk,
                             att, y, ffn, B, half_window, eps):
    """The fp32 backward: FFN -> out-projection -> the attention backward on bf16 operands (dO, and dQ / dK / dV, in
    bf16 as FlashAttention-4 has them) -> qkvg, as the CUDA extension (sm_90, ``swa_dit_fp32_cuda``) or
    ``backward_fp32``'s Triton kernels around the shared attention kernels; fp32 weight gradients from torch GEMMs over
    the fp32 operands either way."""
    N, S, C = q.shape
    NHID = wd.shape[1]
    ext = _fp32_cuda(C, NHID, half_window, q.device)
    if ext is not None:
        dq, dmod, dP, datt, gated, dab, hh, dffn = ext.bwd(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh,
                                                           g, o, lse, q1, pq, pk, att, y, ffn, B, half_window, eps, FP32_EPS)
    else:
        dq, dmod, dP, datt, gated, dab, hh, dffn = _swa_dit_bwd_fp32_triton(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu,
                                                                            wd, qh, kh, vh, g, o, lse, q1, pq, pk, att, y,
                                                                            ffn, B, half_window, eps)
    served_w = settings.current().swa_dit_wgrad_cuda and C == D_ATOM and NHID == N_HIDDEN
    # TMA + wgmma on bf16 hi / lo planes up to _WGRAD_F32_WG_MAX_ROWS (ties cuBLAS's TF32 at 49k rows, ahead below, 60x closer to
    # fp32); the mma.sync kernel where it does not build; above, cuBLAS's TF32 GEMMs are faster
    ext_w = _cuda("wgwg32", served_w and N * S <= _WGRAD_F32_WG_MAX_ROWS, q.device) or _cuda(
        "wgrad", served_w and N * S <= _WGRAD_F32_MAX_ROWS, q.device)
    if ext_w is not None:
        M = N * S
        dwqkv, dwg, dwo, dwu, dwd = ext_w.wgrad_f32(dP, x.reshape(M, C), datt, gated, dab, y.reshape(M, C), dffn, hh)
        return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd]
    dwqkv = dP[:, :3 * C].t() @ x
    dwg = dP[:, 3 * C:].t() @ x
    dwo = datt.t() @ gated
    dwu = dab.t() @ y
    dwd = dffn.t() @ hh
    return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd]


def _swa_dit_bwd_fp32_triton(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, pq, pk, att,
                             y, ffn, B, half_window, eps):
    """The fp32 backward's Triton kernels: [dq, dmod, dP, datt, gated, dab, hh, dffn] (the weight-gradient operands)."""
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = N // B
    Ak = _augments(N, B)
    NHID = wd.shape[1]
    dev = q.device
    f32 = torch.float32
    dmod = torch.zeros(B * S, 6 * C, device=dev, dtype=f32)
    dq1 = torch.empty(M, C, device=dev, dtype=f32)
    dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dG = torch.empty(M, C, device=dev, dtype=f32)
    Dv = torch.empty(N, H, S, device=dev, dtype=f32)
    dab = torch.empty(M, 2 * NHID, device=dev, dtype=f32)
    hh = torch.empty(M, NHID, device=dev, dtype=f32)
    dffn = torch.empty(M, C, device=dev, dtype=f32)
    datt = torch.empty(M, C, device=dev, dtype=f32)
    gated = torch.empty(M, C, device=dev, dtype=f32)
    grid_t = (
        lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    )  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
    _swa_ffn_bwd_fp32_kernel[grid_t](
        dy.reshape(M, C),
        q1,
        mod,
        wu,
        wd,
        y,
        ffn,
        dq1,
        dab,
        hh,
        dffn,
        dmod,
        S,
        A,
        B,
        eps,
        shape_key=atom_key(S, A=Ak, C=C, NHID=NHID),
        C=C,
        NHID=NHID,
        MODW=6 * C,
    )
    _swa_oproj_bwd_fp32_kernel[grid_t](
        dq1,
        o,
        g,
        mod,
        wo,
        att,
        dO,
        dG,
        Dv,
        datt,
        gated,
        dmod,
        S,
        A,
        B,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        MODW=6 * C,
    )
    dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
    dKh = torch.empty_like(dQh)
    dVh = torch.empty_like(dQh)
    _swa_attn_bwd_dq_kernel[
        lambda m: (triton.cdiv(S, m["BM"]), N * H)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        qh,
        kh,
        vh,
        dO,
        lse,
        Dv,
        seqused,
        dQh,
        S,
        D**-0.5,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        HW=half_window,
    )
    _swa_attn_bwd_dkv_kernel[
        lambda m: (triton.cdiv(S, m["BN"]), N * H)
    ](  # ty: ignore[invalid-argument-type]  # ty cannot bind triton's self-typed __call__
        qh,
        kh,
        vh,
        dO,
        lse,
        Dv,
        seqused,
        dKh,
        dVh,
        S,
        D**-0.5,
        shape_key=atom_key(S, A=Ak, C=C),
        C=C,
        H=H,
        D=D,
        HW=half_window,
    )
    dq = torch.empty(M, C, device=dev, dtype=f32)
    dP = torch.empty(M, 4 * C, device=dev, dtype=f32)
    _swa_qkvg_bwd_fp32_kernel[grid_t](
        q.reshape(M, C), mod, cos, sin, wqkv, wg, pq, pk, dQh, dKh, dVh, dG, dq1, dq, dP, dmod, S, A, B, eps, FP32_EPS,
        shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
    return [dq, dmod, dP, datt, gated, dab, hh, dffn]


def _swa_dit_mod_fwd_sm100_fake(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """Shape of the output: [R, 6C] fp32."""
    return c.new_empty((c.shape[0], wmod.shape[0]), dtype=torch.float32)


@opaque(fake=_swa_dit_mod_fwd_sm100_fake, name="swa_dit_mod_fwd_sm100")
def swa_dit_mod_fwd_sm100(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """silu(c) Wmod^T [R, 6C] fp32 on the sm_100a mod_fwd kernel; c [R, C] bf16 contiguous, Wmod [6C, C] bf16."""
    from miniworld_engine.kernels.swa_dit.cuda import sm100

    return sm100.mod_fwd(c, wmod)


def _swa_dit_mod_bwd_sm100_fake(
    g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor
) -> list[torch.Tensor]:
    """Shapes of the outputs: dc like c, dWmod fp32 in Wmod's shape."""
    return [
        torch.empty_like(c),
        torch.empty(wmod.shape, dtype=torch.float32, device=wmod.device),
    ]


@opaque(fake=_swa_dit_mod_bwd_sm100_fake, name="swa_dit_mod_bwd_sm100")
def swa_dit_mod_bwd_sm100(
    g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor
) -> list[torch.Tensor]:
    """[dc (bf16), dWmod (fp32)] of :func:`swa_dit_mod_fwd_sm100` from g = d mod [R, 6C] fp32; R a multiple of 128."""
    from miniworld_engine.kernels.swa_dit.cuda import sm100

    dc, dw = sm100.mod_bwd(g, c, wmod)
    return [dc, dw]


def _swa_dit_mod_fwd_tf32_fake(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """Shape of the output: [R, 6C] fp32."""
    return c.new_empty((c.shape[0], wmod.shape[0]), dtype=torch.float32)


@opaque(fake=_swa_dit_mod_fwd_tf32_fake, name="swa_dit_mod_fwd_tf32")
def swa_dit_mod_fwd_tf32(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """silu(c) Wmod^T [R, 6C] fp32 on the sm_100a TF32 mod_fwd_tf32 kernel; c [R, C] fp32 contiguous, Wmod [6C, C] fp32."""
    from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd

    return tf32_fwd.mod_fwd_tf32(c, wmod)


def _swa_dit_mod_fwd_sm90_fake(c: torch.Tensor, wmod: torch.Tensor, save: bool) -> list[torch.Tensor]:
    """Shapes of the outputs: [R, 6C] fp32 and silu(c) ([R, C] bf16 when ``save``, else [0])."""
    return [c.new_empty((c.shape[0], wmod.shape[0]), dtype=torch.float32),
            c.new_empty((c.shape[0], c.shape[1]) if save else (0,), dtype=torch.bfloat16)]


@opaque(fake=_swa_dit_mod_fwd_sm90_fake, name="swa_dit_mod_fwd_sm90")
def swa_dit_mod_fwd_sm90(c: torch.Tensor, wmod: torch.Tensor, save: bool) -> list[torch.Tensor]:
    """[silu(c) Wmod^T [R, 6C] fp32, silu(c)] on the sm_90a ``mod_fwd`` kernel; c [R, C] and Wmod [6C, C] contiguous, bf16.
    ``save`` also returns silu(c) (bf16) for the backward, else an empty tensor. Call :func:`mod_sm90` first."""
    ext = _cuda("modwg", True, c.device) or _cuda("mod", True, c.device)     # TMA + wgmma forward (10.4 -> 8.3 us at R = 4096)
    mod, a = ext.mod_fwd(c, wmod, save)
    return [mod, a]


def _swa_dit_mod_bwd_sm90_fake(g: torch.Tensor, c: torch.Tensor, a: torch.Tensor, wmod: torch.Tensor) -> list[torch.Tensor]:
    """Shapes of the outputs: dc like c, dWmod like Wmod."""
    return [torch.empty_like(c), torch.empty_like(wmod)]


@opaque(fake=_swa_dit_mod_bwd_sm90_fake, name="swa_dit_mod_bwd_sm90")
def swa_dit_mod_bwd_sm90(g: torch.Tensor, c: torch.Tensor, a: torch.Tensor, wmod: torch.Tensor) -> list[torch.Tensor]:
    """[dc, dWmod] of :func:`swa_dit_mod_fwd_sm90` from g = d mod [R, 6C] fp32 and the saved silu(c) ``a``."""
    dc, dw = _cuda("mod", True, c.device).mod_bwd(g, c, a, wmod, True, True)
    return [dc, dw]


def mod_sm90(c: torch.Tensor, wmod: torch.Tensor) -> bool:
    """Whether the hoisted modulation runs on the sm_90a kernels: sm_90, d_cond = C = 128, Wmod [6C, C], c and Wmod both bf16
    (any matmul precision: the products are exact) and the extension builds. fp32 c stays on torch's GEMMs: the kernels'
    fp32 variant (bf16 hi / lo planes, 2^-17) is the more accurate but not the faster one next to cuBLAS's TF32."""
    policy = settings.current()
    if not policy.swa_dit_mod_cuda or policy.engine_backend == "triton" or not c.is_cuda:
        return False
    if c.shape[-1] != D_ATOM or tuple(wmod.shape) != (6 * D_ATOM, D_ATOM) or c.dtype != wmod.dtype or c.dtype != torch.bfloat16:
        return False
    return _cuda("mod", True, c.device) is not None
