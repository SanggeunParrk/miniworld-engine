"""A100 (sm_80) hand-CUDA triangle attention, inference and training (heads of 32 channels, the pair bias shared by every pair row).

The attention core:

    out[b, i, j, h, :] = softmax_k( q[b, i, j, h, :] . k[b, i, k, h, :] / sqrt(32) + bias[b, h, j, k] ) @ v[b, i, k, h, :]

``kernels/triangle_attention/cuda/sm80/attn_fwd_sm80.cuh``: one CTA = (head, 128 queries) x R = 2 pair rows, so the bias tile of a key tile
is staged once for the rows and K | V stream through a cp.async ring; fp32 online softmax in base 2, ``mma.sync`` for both products. Two CTAs share an SM (at most 128 registers per thread).
Masked keys carry the bias bf16-min (the module's ``masked_fill``): weight 0; a query whose keys are all masked gets a zero output, as the
Triton kernel.

Large masked inference (L >= 768, divisible by 384) uses 48-key tiles, a two-stage K/V ring and a stable device-side key index. Only bias planes are gathered;
K/V stay in place and padded key loads are zero-filled. Mask packing is part of the measured call. ``MINIWORLD_TRIATTN_SM80_COMPACT=0``
retains the dense core. Calls saving log-sum-exp for backward keep the dense schedule.

The front (``f1_sm80.cuh``): the input LayerNorm and the q | k | v | g and pair-bias projections in one pass over the pair tensor (the
LayerNorm scale folded into the bf16 weights, the shift into an fp32 bias), writing the token-major ``[T, 512]`` buffer the core reads with a
row stride and the bias planes.

The back (``f3_sm80.cuh``): the sigmoid gate, the output projection, the module's broadcast dropout scale and the residual add in one pass, the
result in the module's layout (the ending node reads and writes its transposed positions).

Training (``trainable``, an autograd function over two opaque ops): the forward saves the LayerNorm statistics, the normalised input, the attention
output and the log-sum-exp.  The backward is ``b3_sm80.cuh`` (gate and output-projection gradients, the attention backward's row term),
``attn_bwd_dkv_sm80.cuh`` and ``attn_bwd_dq_sm80.cuh`` (dK dV; dQ and the bias gradient: bf16 partials over groups of four rows and a fixed-order
reduction), ``b1_sm80.cuh`` (projection dgrad, LayerNorm backward and the residual), and the weight / LayerNorm-parameter gradients from one
cuBLAS GEMM over the tokens (``front_weight_grads``).  The gradients of q | k | v | g share one ``[T, 512]`` buffer, each slice written by its producer.

**Ampere-only and shape-specific by construction**: sm_80, bf16, head dim 32, d_pair 128, L a multiple of 128, one square pair stack per batch
element.  ``supports()`` / ``supports_front()`` / ``supports_back()`` / ``supports_train()`` are the whole gate; everything they reject keeps the
Triton path.
"""

import functools
import hashlib
import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

D = 32
#: pair rows per CTA task: the bias tile is shared by this many rows
ROWS = 2
#: keys per tile (64 or 32) and CTAs per SM the register budget is bounded for (the schedule knobs of the kernel)
BN, MINB = 32, 2
_L2E = 1.4426950408889634


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_TRIATTN_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_TRIATTN_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"triattn_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@functools.lru_cache(maxsize=8)
def _is_ampere(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def _token_major(t: torch.Tensor) -> torch.Tensor | None:
    """The token-major ``[B, L, L2, H * D]`` tensor a ``[B, H, L, L2, D]`` view was made from (``to_query(pair)`` followed by
    ``rearrange``, or a 128-column slice of the front's ``[T, 512]`` q | k | v | g buffer: the row stride ``sj`` may exceed ``H * D``),
    as a view, or None when ``t`` is laid out otherwise (16-byte aligned rows are what the kernel's loads need: the storage offset is a multiple of
    8 elements; an allocation itself is 512-byte aligned, and ``data_ptr`` is not readable on a fake tensor)."""
    b, h, length, length2, d = t.shape
    sb, sh, sl, sj, sd = t.stride()
    if sd != 1 or sh != d or sj < h * d or sj % 8 or sl != sj * length2 or (b > 1 and sb != sl * length) or t.storage_offset() % 8:
        return None
    return t.permute(0, 2, 3, 1, 4).reshape(b, length, length2, h * d)


def supports(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor) -> bool:
    """The kernel's own requirements: sm_80, bf16, head dim 32, a square pair stack with L a multiple of 128, q / k / v token-major
    (the layout ``to_query`` / ``to_key`` / ``to_value`` give), bias ``[B, H, L, L]``."""
    if os.environ.get("MINIWORLD_TRIATTN_SM80", "1") == "0":
        return False
    for t in (query, key, value, bias):
        if not t.is_cuda or t.dtype is not torch.bfloat16:
            return False
    if query.ndim != 5 or query.shape != key.shape or query.shape != value.shape:
        return False
    b, h, length, length2, d = query.shape
    if d != D or length != length2 or length % 128 != 0 or bias.shape != (b, h, length, length):
        return False
    if any(_token_major(t) is None for t in (query, key, value)):
        return False
    return _is_ampere(query.device.index if query.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


def _built(*tensors: torch.Tensor) -> bool:
    """A successful (cached) build of the extension; a build failure warns once and keeps the existing path.  Inside a trace the build is
    left to the first call of the opaque ops."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    if torch.compiler.is_compiling() or _is_fake(*tensors):
        return True
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"sm80 triangle attention unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=3)
        return False
    return True


def available(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor) -> bool:
    """``supports()`` plus a successful (cached) build."""
    return supports(query, key, value, bias) and _built(query)


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


def _attention_fake(q, k, v, bias, sm_scale, save_lse):
    """[out token-major [B, L, L2, H * D] (contiguous), lse [B, H, L, L2] fp32 when ``save_lse`` else empty]."""
    b, length, length2, c = q.shape
    lse = q.new_empty((b, c // D, length, length2), dtype=torch.float32) if save_lse else q.new_empty((0,), dtype=torch.float32)
    return [q.new_empty((b, length, length2, c)), lse]


@opaque(fake=_attention_fake, name="triangle_attention_sm80_forward")
def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, sm_scale: float, save_lse: bool,
               ) -> list[torch.Tensor]:
    """The attention core on token-major ``[B, L, L2, H * D]`` q / k / v and ``[B, H, L, L]`` bias; returns [out, lse]
    (lse = m + log2 l in the base-2 domain of the scaled logits, per (b, h, i, j); empty unless ``save_lse``)."""
    b, length, _, c = q.shape
    h = c // D
    out = torch.empty((b, length, length, c), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, length, length), dtype=torch.float32, device=q.device) if save_lse else q.new_empty((0,), dtype=torch.float32)
    _ext().attn_fwd(q, k, v, bias, out, lse, length, h, b, q.stride(2), k.stride(2), v.stride(2), c, sm_scale * _L2E, ROWS, BN, MINB)
    return [out, lse]


def _compact_attention_fake(q, k, v, bias, mask, sm_scale):
    """Fake of the compact (key-masked) triangle-attention forward: the same outputs as the full op."""
    return _attention_fake(q, k, v, bias, sm_scale, False)


@opaque(fake=_compact_attention_fake, name="triangle_attention_sm80_compact_forward")
def _compact_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                       mask: torch.Tensor, sm_scale: float) -> list[torch.Tensor]:
    """Gather the small bias planes and index K/V in place; all counts stay on device."""
    b, length, _, c = q.shape
    h = c // D
    out = q.new_empty((b, length, length, c))
    indices = torch.empty((b, length), dtype=torch.int32, device=q.device)
    counts = torch.empty((b,), dtype=torch.int32, device=q.device)
    compact_bias = torch.empty_like(bias)
    _ext().compact_attention(q, k, v, bias, mask, out, indices, counts, compact_bias, sm_scale * _L2E)
    return [out, q.new_empty((0,), dtype=torch.float32)]


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor, *, save_lse: bool = False,
              compact_key_mask: torch.Tensor | None = None,
              ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``query`` / ``key`` / ``value``: ``[B, H, L, L2, D]`` views of token-major ``[B, L, L2, H * D]`` tensors; ``bias`` ``[B, H, L, L2]``
    (masked keys = bf16 min).  Returns ``(out, lse)``: ``out`` ``[B, H, L, L2, D]`` as a view of a fresh token-major tensor, so the
    module's ``rearrange(out, "B H L L2 D -> B L L2 (H D)")`` is free; ``lse`` ``[B, H, L, L2]`` fp32 or None.
    ``compact_key_mask`` describes the global key exclusions ALREADY encoded in bias; it is an inference optimization hint, not an additional mask. Call ``available()`` first."""
    b, h, length, length2, d = query.shape
    # The availability gate (or the module's front contract) has established the
    # layout. Keep this view conversion traceable: storage_offset() is a Python
    # integer query that Dynamo cannot trace for slices of an opaque-op result.
    q, k, v = (t.permute(0, 2, 3, 1, 4).reshape(b, length, length2, h * d) for t in (query, key, value))
    if (compact_key_mask is not None and not save_lse and length >= 768 and length % 384 == 0
            and os.environ.get("MINIWORLD_TRIATTN_SM80_COMPACT", "1") != "0"):
        out, lse = _compact_attention(q, k, v, bias.contiguous(), compact_key_mask.contiguous(), 1.0 / math.sqrt(d))
    else:
        out, lse = _attention(q, k, v, bias.contiguous(), 1.0 / math.sqrt(d), save_lse)
    return out.view(b, length, length2, h, d).permute(0, 3, 1, 2, 4), (lse if save_lse else None)


# ------------------------------------------------------------------------------------------------------------------- the front
#: columns of the front's token-major output: q | k | v | g of 128 channels (4 heads x 32) each
QKVG = 512
#: warps of the front's persistent CTA (one per SM; 8, 12 or 16)
FRONT_WARPS = 8


def supports_front(pair: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor,
                   mask: torch.Tensor | None = None) -> bool:
    """The front's requirements: sm_80, a bf16 ``[B, L, L, 128]`` pair stack with L a multiple of 128, ``weights`` = (q, k, v, g) bf16
    ``[128, 128]`` and the bias weight bf16 ``[4, 128]``, 128-channel LayerNorm parameters, a ``[B, L]`` bool key mask if any."""
    if os.environ.get("MINIWORLD_TRIATTN_SM80", "1") == "0":
        return False
    if not pair.is_cuda or pair.dtype is not torch.bfloat16 or pair.ndim != 4 or not pair.is_contiguous():
        return False
    b, length, length2, c = pair.shape
    if c != 128 or length != length2 or length % 128 != 0 or b * length * length >= 2**31:
        return False
    if len(weights) != 5 or any(w.dtype is not torch.bfloat16 or w.device != pair.device for w in weights):
        return False
    if any(w.shape != (128, 128) for w in weights[:4]) or weights[4].shape != (4, 128):
        return False
    if any(t.shape != (128,) or t.device != pair.device for t in (ln_w, ln_b)):
        return False
    if mask is not None and (mask.dtype is not torch.bool or mask.shape != (b, length) or mask.device != pair.device):
        return False
    return _is_ampere(pair.device.index if pair.device.index is not None else torch.cuda.current_device())


def _mask_u8(mask: torch.Tensor | None, device) -> torch.Tensor:
    """The ``[B, L]`` bool key mask as the kernel's uint8 (reinterpreted in place, no cast kernel); empty = no mask."""
    if mask is None:
        return torch.empty(0, dtype=torch.uint8, device=device)
    return mask.contiguous().view(torch.uint8)


def _pack_front(ext, weights, ln_w, ln_b):
    """The front's weights in its layout, one launch (``f1_sm80.cuh``): W' = bf16(W diag gamma) ``[520, 128]`` (the LayerNorm scale folded in,
    rows 516-519 zero) and b = W beta ``[520]`` fp32 (the shift).  Never cached: a captured CUDA graph must repack after an optimizer step."""
    wp = torch.empty((520, 128), dtype=torch.bfloat16, device=ln_w.device)
    bvec = torch.empty((520,), dtype=torch.float32, device=ln_w.device)
    ext.front_pack(*(w.contiguous() for w in weights), ln_w.detach().float().contiguous(), ln_b.detach().float().contiguous(), wp, bvec)
    return wp, bvec


def _front_fake(x, weights, ln_w, ln_b, mask, eps, transposed, save_stats, save_xh):
    """[qkvg [B, L, L, 512], bias [B, 4, L, L], stats [B * L * L, 2] fp32 when ``save_stats`` else empty, xh [B, L, L, 136] when ``save_xh`` else empty]."""
    b, length = x.shape[0], x.shape[1]
    stats = x.new_empty((b * length * length, 2), dtype=torch.float32) if save_stats else x.new_empty((0,), dtype=torch.float32)
    xh = x.new_empty((b, length, length, 136)) if save_xh else x.new_empty((0,))
    return [x.new_empty((b, length, length, QKVG)), x.new_empty((b, 4, length, length)), stats, xh]


@opaque(fake=_front_fake, name="triangle_attention_sm80_front")
def _front(x: torch.Tensor, weights: list[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor, mask: torch.Tensor, eps: float,
           transposed: bool, save_stats: bool, save_xh: bool) -> list[torch.Tensor]:
    """LayerNorm + q | k | v | g + bias projections of the contiguous pair stack ``x`` ``[B, L, L, 128]`` (``weights``: q, k, v, g, bias;
    ``mask``: ``[B, L]`` uint8 or empty); returns [qkvg, bias, stats, xh] (``stats`` = per-token (mean, rstd), empty unless ``save_stats``; ``xh`` = the
    normalised input bf16(xh) followed by a column of ones and zeros, ``[B, L, L, 136]``, empty unless ``save_xh``).
    """
    ext = _ext()
    b, length = x.shape[0], x.shape[1]
    wp, bvec = _pack_front(ext, weights, ln_w, ln_b)
    qkvg = torch.empty((b, length, length, QKVG), dtype=torch.bfloat16, device=x.device)
    bias = torch.empty((b, 4, length, length), dtype=torch.bfloat16, device=x.device)
    stats = torch.empty((b * length * length, 2), dtype=torch.float32, device=x.device) if save_stats else x.new_empty((0,), dtype=torch.float32)
    xh = torch.empty((b, length, length, 136), dtype=torch.bfloat16, device=x.device) if save_xh else x.new_empty((0,))
    ext.front(x, wp, bvec, mask, qkvg, bias, stats, xh, length, int(transposed), eps, FRONT_WARPS)
    return [qkvg, bias, stats, xh]


def front(pair: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor, eps: float,
          mask: torch.Tensor | None = None, *, transposed: bool = False, save_stats: bool = False,
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The input LayerNorm of ``pair`` ``[B, L, L, 128]`` and its five projections (``weights`` = q, k, v, g, bias) in one pass over the tensor.
    Returns ``(qkvg, bias, stats)``: ``qkvg`` ``[B, L, L, 512]`` token-major (q | k | v | g, 128 channels = 4 heads of 32 each; ``qkv_views``
    slices it), ``bias`` ``[B, 4, L, L]`` (the pair bias, masked keys = bf16 min as the module's ``masked_fill``), ``stats`` ``[B L L, 2]``
    fp32 (mean, rstd of the LayerNorm) or None.  ``transposed``: token (a, b) is read from ``pair[:, b, a]`` (the ending node, with no
    transposing copy).  Call ``supports_front()`` first."""
    qkvg, bias, stats, _ = _front(pair, list(weights), ln_w, ln_b, _mask_u8(mask, pair.device), float(eps), bool(transposed), bool(save_stats), False)
    return qkvg, bias, (stats if save_stats else None)


def front_train(pair: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor, eps: float,
                mask: torch.Tensor | None = None, *, transposed: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``front()`` for training: ``(qkvg, bias, stats, xh)`` with the LayerNorm statistics and the normalised input ``xh`` ``[B, L, L, 136]`` bf16
    (columns 0 .. 127; column 128 is 1, the rest 0: the operand of the projections' weight-gradient GEMM, whose ones column gives the column sums) saved
    for the backward."""
    return tuple(_front(pair, list(weights), ln_w, ln_b, _mask_u8(mask, pair.device), float(eps), bool(transposed), True, True))


def qkv_views(qkvg: torch.Tensor, n_head: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v, g of the front's ``[B, L, L2, 512]`` output as ``[B, H, L, L2, D]`` views (no copies) -- the layout the module's attention takes."""
    b, length, length2, _ = qkvg.shape
    c = n_head * D
    return tuple(qkvg[..., i * c:(i + 1) * c].view(b, length, length2, n_head, D).permute(0, 3, 1, 2, 4) for i in range(4))  # ty: ignore[invalid-return-type]


# -------------------------------------------------------------------------------------------------------------------- the back
#: warps of the back's persistent CTA (one per SM; 8, 12 or 16)
BACK_WARPS = 8


def supports_back(res: torch.Tensor, wo: torch.Tensor) -> bool:
    """The back's requirements: sm_80, the residual ``res`` bf16 contiguous ``[B, L, L, 128]`` with L a multiple of 128 (the attention output and
    the front's gate columns are shaped like it), ``wo`` bf16 ``[128, 128]``."""
    if os.environ.get("MINIWORLD_TRIATTN_SM80", "1") == "0":
        return False
    if not res.is_cuda or res.dtype is not torch.bfloat16 or wo.dtype is not torch.bfloat16 or wo.device != res.device:
        return False
    if res.ndim != 4 or res.shape[1] != res.shape[2] or res.shape[-1] != 128 or res.shape[1] % 128 != 0 or wo.shape != (128, 128):
        return False
    if not res.is_contiguous() or res.numel() // 128 >= 2**31:
        return False
    return _is_ampere(res.device.index if res.device.index is not None else torch.cuda.current_device())


def available_module(pair: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor, wo: torch.Tensor,
                     mask: torch.Tensor | None = None) -> bool:
    """The whole module (front, core, back): the gates of the front and the back, and a successful (cached) build.  The core takes what the
    front writes, so its own gate holds when the front's does."""
    return supports_front(pair, weights, ln_w, ln_b, mask) and supports_back(pair, wo) and _built(pair)


def _back_fake(o, qkvg, wo, res, ds, transposed):
    """The result ``[B, L, L, 128]``, a fresh tensor like ``res``."""
    return res.new_empty(res.shape)


@opaque(fake=_back_fake, name="triangle_attention_sm80_back")
def _back(o: torch.Tensor, qkvg: torch.Tensor, wo: torch.Tensor, res: torch.Tensor, ds: torch.Tensor, transposed: bool) -> torch.Tensor:
    """``res + bf16(bf16(sigmoid(g) o) @ wo^T)`` (the product times the dropout scale ``ds`` ``[B, L, 128]`` when it is not empty, in bf16) for the
    token-major attention output ``o`` and the gate columns of the front's ``qkvg``; with ``transposed`` the token (a, b) reads and writes row (b, a)
    of ``res`` / the result (the ending node)."""
    b, length = res.shape[0], res.shape[1]
    out = torch.empty_like(res)
    g = qkvg.view(b * length * length, QKVG)[:, 3 * 128:]
    _ext().back(o, g, wo.contiguous(), res, out, ds, length, int(transposed), BACK_WARPS)
    return out


def back(o: torch.Tensor, qkvg: torch.Tensor, wo: torch.Tensor, res: torch.Tensor, *, transposed: bool = False, ds: torch.Tensor | None = None,
         ) -> torch.Tensor:
    """The gate, the output projection and the residual in one pass: ``res + to_out(sigmoid(g) * o)``, every statement rounded to bf16 as the
    module's.  ``o``: the attention output as token-major ``[B, L, L, 128]`` (the module's ``rearrange(out, "B H L L2 D -> B L L2 (H D)")`` of
    ``attention()``'s result is such a view), ``qkvg``: the front's output, ``wo``: ``to_out.weight``, ``res``: the pair tensor, ``ds``: the dropout
    scale ``[B, L, 128]`` indexed by the token's second index (training; the module multiplies the projection by it before the residual add).
    Call ``supports_back()`` first."""
    return _back(o, qkvg, wo, res, ds if ds is not None else o.new_empty((0,)), bool(transposed))


# ------------------------------------------------------------------------------------------------------------------ the backward's weights
#: warps of the persistent CTAs of the two stream-style backward kernels (the back's and the front's backward; one CTA per SM; 8 or 12)
BACK_BWD_WARPS = 8


def _bwd_pack_fake(weights, wo, ln_w):
    """[wt ``[4, 128, 128]``, wbt ``[128, 4]``, wot ``[128, 128]`` bf16, gamma32 ``[128]`` fp32]."""
    return [wo.new_empty((4, 128, 128)), wo.new_empty((128, 4)), wo.new_empty((128, 128)), wo.new_empty((128,), dtype=torch.float32)]


@opaque(fake=_bwd_pack_fake, name="triangle_attention_sm80_bwd_pack")
def _bwd_pack(weights: list[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor) -> list[torch.Tensor]:
    """The weights in the backward kernels' layouts, one launch (``wgrad_sm80.cuh``): ``wt[p][c][o] = W_p[o][c]`` for q, k, v, g (the front's input
    gradient is D W: the B operand is W^T), ``wbt[c][h] = Wb[h][c]``, ``wot[c][o] = Wo[o][c]`` (the back's backward ``da = dy Wo``) and the LayerNorm scale
    as fp32."""
    dev = wo.device
    wt = torch.empty((4, 128, 128), dtype=torch.bfloat16, device=dev)
    wbt = torch.empty((128, 4), dtype=torch.bfloat16, device=dev)
    wot = torch.empty((128, 128), dtype=torch.bfloat16, device=dev)
    gamma32 = torch.empty((128,), dtype=torch.float32, device=dev)
    _ext().bwd_pack(*(w.contiguous() for w in weights), wo.contiguous(), ln_w.detach().contiguous(), wt, wbt, wot, gamma32)
    return [wt, wbt, wot, gamma32]


def bwd_pack(weights: Sequence[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(wt, wbt, wot, gamma32)``: the weights of ``weights`` = (q, k, v, g, bias), ``wo`` and the LayerNorm scale in the layouts of ``back_bwd`` /
    ``front_bwd`` (pass them as ``packed``; the training backward packs once for both)."""
    wt, wbt, wot, gamma32 = _bwd_pack(list(weights), wo, ln_w)
    return wt, wbt, wot, gamma32


# ------------------------------------------------------------------------------------------------------------------- the back's backward
def _back_bwd_fake(dout, ds, o, qkvg, dqkvg, wot, transposed):
    """[dov, dy, a, delta]: fresh tensors (token-major ``[B, L, L, 128]`` x 3 and the fp32 ``[B, 4, L, L]`` row term; ``dqkvg``'s gate columns are written in place)."""
    b, length = o.shape[0], o.shape[1]
    return [o.new_empty(o.shape), o.new_empty(o.shape), o.new_empty(o.shape), o.new_empty((b, 4, length, length), dtype=torch.float32)]


@opaque(fake=_back_bwd_fake, name="triangle_attention_sm80_back_bwd", mutates_args=("dqkvg",))
def _back_bwd(dout: torch.Tensor, ds: torch.Tensor, o: torch.Tensor, qkvg: torch.Tensor, dqkvg: torch.Tensor, wot: torch.Tensor,
              transposed: bool) -> list[torch.Tensor]:
    """The backward of ``back()`` (``f3_sm80.cuh``'s forward) in one pass: with ``dy = bf16(dout ds)`` (``dout`` in the module's layout, read at the
    transposed positions when ``transposed``; ``ds`` ``[B, L, 128]`` or empty), ``da = bf16(dy Wo)`` (``wot`` = Wo^T, from ``bwd_pack``), ``s = sigmoid(g)``:
    writes the gate gradient ``bf16(da o s (1 - s))`` into the gate columns (384 .. 511) of ``dqkvg`` ``[B, L, L, 512]`` and returns
    ``[dov = bf16(da s), dy, a = bf16(s o), delta]`` (starting frame; ``dy`` and ``a`` are the operands of ``dWo = dy^T a``, ``delta[b, h]`` the head's sum
    over its 32 channels of ``o * dov``: the attention backward's row term).
    """
    b, length = o.shape[0], o.shape[1]
    dov, dy, a = (torch.empty_like(o) for _ in range(3))
    delta = torch.empty((b, 4, length, length), dtype=torch.float32, device=o.device)
    t = b * length * length
    g = qkvg.view(t, QKVG)[:, 3 * 128:]
    dg = dqkvg.view(t, QKVG)[:, 3 * 128:]
    _ext().back_bwd(dout, ds, o, g, wot, dg, dov, dy, a, delta, length, int(transposed), BACK_BWD_WARPS)
    return [dov, dy, a, delta]


def back_bwd(dout: torch.Tensor, o: torch.Tensor, qkvg: torch.Tensor, dqkvg: torch.Tensor, wo: torch.Tensor, *, transposed: bool = False,
             ds: torch.Tensor | None = None, wot: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward of ``back()``.  ``dout``: the output's gradient ``[B, L, L, 128]`` in the module's layout, ``o``: the attention output (token-major),
    ``qkvg``: the front's output, ``dqkvg``: the ``[B, L, L, 512]`` gradient buffer whose gate columns this fills, ``wo``: ``to_out.weight`` (``wot``: its
    transpose, when the caller packed it), ``ds``: the dropout scale the forward used.  Returns ``(dov, dy, a, delta)`` (the attention output's gradient,
    the operands of ``dWo = dy^T a``, the row term)."""
    wot = wot if wot is not None else wo.t().contiguous()
    dov, dy, a, delta = _back_bwd(dout.contiguous(), ds if ds is not None else o.new_empty((0,)), o, qkvg, dqkvg, wot, bool(transposed))
    return dov, dy, a, delta


# ------------------------------------------------------------------------------------------------------------- the attention core's backward
#: schedules of the two backward kernels: query side (rows per CTA, K | V ring stages, keys per tile, CTAs per SM), key side (query-tile ring stages, CTAs per SM).
#: The query side's rows per CTA set the number of bias-gradient partials (L / rows planes that ``db_reduce`` sums): four rows halve what two rows write and
#: read back, which pays for the one CTA per SM that four rows' shared memory leaves (dq + reduction, B = 1, A100: L256 -8 %, L384 -8 %, L768 -12 %
#: of the time; L128 +2 %, one microsecond).
DQ_ROWS, DQ_NKV, DQ_BN, DQ_MINB = 4, 3, 32, 1
DKV_NST, DKV_MINB = 4, 2


def _attention_bwd_fake(q, k, v, dov, bias, lse, delta, dqkvg):
    """The bias gradient ``[B, H, L, L]`` bf16 (dq / dk / dv are written into ``dqkvg``'s first 384 columns)."""
    return bias.new_empty(bias.shape)


@opaque(fake=_attention_bwd_fake, name="triangle_attention_sm80_attention_bwd", mutates_args=("dqkvg",))
def _attention_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dov: torch.Tensor, bias: torch.Tensor, lse: torch.Tensor, delta: torch.Tensor,
                   dqkvg: torch.Tensor) -> torch.Tensor:
    """The core's backward on token-major ``[B, L, L, 128]`` q / k / v (views of the front's buffer), the output's gradient ``dov``, the pair bias
    ``[B, H, L, L]`` bf16 (masked keys = bf16 min), the forward's ``lse`` and the row term ``delta`` (``[B, H, L, L]`` fp32): writes dq | dk | dv into
    the first 384 columns of ``dqkvg`` ``[B, L, L, 512]`` and returns the bias gradient (the sum over the pair rows of dS) ``[B, H, L, L]`` bf16."""
    ext = _ext()
    b, length, _, c = q.shape
    h = c // D
    t = b * length * length
    groups = length // DQ_ROWS
    dbp = torch.empty((groups, b * h, length, length), dtype=torch.bfloat16, device=q.device)
    buf = dqkvg.view(t, QKVG)
    sm_scale = 1.0 / math.sqrt(D)
    scl = sm_scale * _L2E
    ext.attn_bwd_dq(q, k, v, dov, bias, lse, delta, buf[:, 0:128], dbp, length, h, b, q.stride(2), k.stride(2), v.stride(2), c, QKVG, scl, sm_scale,
                    DQ_ROWS, DQ_NKV, DQ_BN, DQ_MINB)
    db = torch.empty((b, h, length, length), dtype=torch.bfloat16, device=q.device)
    ext.db_reduce(dbp, db, groups)
    bias_t = torch.empty_like(bias)
    ext.bias_transpose(bias, bias_t, length)
    ext.attn_bwd_dkv(q, k, v, dov, bias_t, lse, delta, buf[:, 128:256], buf[:, 256:384], length, h, b, q.stride(2), k.stride(2), v.stride(2), c, QKVG, QKVG,
                     scl, sm_scale, DKV_NST, 32, DKV_MINB)
    return db


def attention_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dov: torch.Tensor, bias: torch.Tensor, lse: torch.Tensor, delta: torch.Tensor,
                       dqkvg: torch.Tensor) -> torch.Tensor:
    """The backward of ``attention()``: dq | dk | dv go into the first 384 columns of ``dqkvg`` ``[B, L, L, 512]`` (token-major, head h = columns
    [32 h, 32 h + 32) of each 128), the return value is the pair bias' gradient ``[B, H, L, L]``.  ``q`` / ``k`` / ``v``: token-major ``[B, L, L, 128]``
    views of the front's buffer (``_token_major`` of ``qkv_views``), ``dov``: the output's gradient (token-major, contiguous), ``lse``: ``attention``'s
    ``save_lse``, ``delta``: ``sum_d o do`` per (b, h, i, j)."""
    return _attention_bwd(q, k, v, dov, bias, lse, delta, dqkvg)


# ------------------------------------------------------------------------------------------------------------------ the front's backward
def _front_bwd_fake(dqkvg, db, x, stats, dout, wt, wbt, gamma32, transposed):
    """The pair tensor's gradient ``[B, L, L, 128]``, a fresh tensor like ``x``."""
    return x.new_empty(x.shape)


@opaque(fake=_front_bwd_fake, name="triangle_attention_sm80_front_bwd")
def _front_bwd(dqkvg: torch.Tensor, db: torch.Tensor, x: torch.Tensor, stats: torch.Tensor, dout: torch.Tensor, wt: torch.Tensor, wbt: torch.Tensor,
               gamma32: torch.Tensor, transposed: bool) -> torch.Tensor:
    """The backward of the front's input gradient (``b1_sm80.cuh``): ``dpair = dout + LayerNorm_backward([dq | dk | dv | dg | db] . W)``.  ``dqkvg``
    ``[B, L, L, 512]`` (dq | dk | dv | dg), ``db`` ``[B, 4, L, L]`` bf16, ``x`` the pair tensor, ``stats`` the forward's (mean, rstd) per token,
    ``dout`` the output's gradient (``x`` and ``dout`` in the module's layout, read at the transposed positions when ``transposed``), ``wt`` / ``wbt`` /
    ``gamma32`` from ``bwd_pack``."""
    b, length = x.shape[0], x.shape[1]
    dpair = torch.empty_like(x)
    _ext().front_bwd(dqkvg.view(b * length * length, QKVG), db, x, stats, dout, dpair, wt, wbt, gamma32, length, int(transposed), BACK_BWD_WARPS)
    return dpair


def front_bwd(dqkvg: torch.Tensor, db: torch.Tensor, x: torch.Tensor, stats: torch.Tensor, dout: torch.Tensor, weights: Sequence[torch.Tensor],
              ln_w: torch.Tensor, *, transposed: bool = False, packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """The pair tensor's gradient: ``dout`` (the residual) plus the gradient through the LayerNorm and the five projections, in one pass.  The
    parameters' gradients are ``front_weight_grads``'.  ``packed`` = ``(wt, wbt, gamma32)`` of ``bwd_pack`` when the caller packed them."""
    if packed is None:
        wt, wbt, _, gamma32 = bwd_pack(weights, weights[0], ln_w)                      # the output weight is not read here: any [128, 128] bf16 tensor will do
        packed = (wt, wbt, gamma32)
    return _front_bwd(dqkvg, db, x, stats, dout.contiguous(), *packed, bool(transposed))


def _wgrad_fake(dqkvg, db, xh, weights, ln_w, ln_b):
    """[dwq, dwk, dwv, dwg, dwb, dgamma, dbeta] in the parameters' shapes and dtypes."""
    return [*(w.new_empty(w.shape) for w in weights), ln_w.new_empty(ln_w.shape), ln_b.new_empty(ln_b.shape)]


@opaque(fake=_wgrad_fake, name="triangle_attention_sm80_wgrad")
def _wgrad(dqkvg: torch.Tensor, db: torch.Tensor, xh: torch.Tensor, weights: list[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor) -> list[torch.Tensor]:
    """The parameters' gradients (see ``front_weight_grads``): the GEMM over the tokens ``G = D^T [xh | 1]`` (cuBLAS, fp32 accumulation and output) and one
    finalize launch (``wgrad_sm80.cuh``)."""
    t = dqkvg.shape[0] * dqkvg.shape[1] * dqkvg.shape[2]
    xh2 = xh.view(t, 136)
    g = torch.cat([torch.mm(dqkvg.view(t, QKVG).T, xh2, out_dtype=torch.float32),                       # [512, 136]
                   torch.mm(db.permute(1, 0, 2, 3).reshape(4, t), xh2, out_dtype=torch.float32)])        # [4, 136]
    dws = [torch.empty_like(w) for w in weights]
    d_gamma, d_beta = torch.empty_like(ln_w), torch.empty_like(ln_b)
    _ext().wgrad_finalize(g, *(w.contiguous() for w in weights), *dws, ln_w.detach().contiguous(), ln_b.detach().contiguous(), d_gamma, d_beta)
    return [*dws, d_gamma, d_beta]


def front_weight_grads(dqkvg: torch.Tensor, db: torch.Tensor, xh: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor,
                       ln_b: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """The gradients of the five projection weights and of the LayerNorm affine.  With D = [dq | dk | dv | dg | db] (``[T, 516]``) and the saved
    ``xh`` (the normalised input followed by a column of ones), G = D^T xh is one GEMM over the tokens (fp32 accumulation, ``[516, 136]``):

        dW = G[:, :128] diag(gamma) + s beta^T      (s = G[:, 128] = the column sums of D)
        dgamma = sum_o W[o, :] G[o, :128]           (dxn = D W, and dgamma_c = sum_t dxn[t, c] xh[t, c])
        dbeta  = sum_o W[o, :] s[o]

    Returns ``([dWq, dWk, dWv, dWg, dWb], dgamma, dbeta)`` in the parameters' dtypes."""
    *grads, d_gamma, d_beta = _wgrad(dqkvg, db, xh, list(weights), ln_w, ln_b)
    return grads, d_gamma, d_beta


# ------------------------------------------------------------------------------------------------------------------------- training
def supports_train(pair: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor, wo: torch.Tensor,
                   mask: torch.Tensor | None = None) -> bool:
    """The training path's requirements: those of the forward (front and back), a floating-point LayerNorm affine (its gradient is returned in its dtype)."""
    return (supports_front(pair, weights, ln_w, ln_b, mask) and supports_back(pair, wo) and ln_w.dtype in (torch.float32, torch.bfloat16)
            and ln_b.dtype == ln_w.dtype)


def _forward_train_fake(leaves, mask, ds, eps, transposed):
    """[out, qkvg, bias, stats, xh, o, lse]: the result and what the backward needs."""
    pair = leaves[0]
    b, length = pair.shape[0], pair.shape[1]
    return [pair.new_empty(pair.shape), pair.new_empty((b, length, length, QKVG)), pair.new_empty((b, 4, length, length)),
            pair.new_empty((b * length * length, 2), dtype=torch.float32), pair.new_empty((b, length, length, 136)), pair.new_empty(pair.shape),
            pair.new_empty((b, 4, length, length), dtype=torch.float32)]


@opaque(fake=_forward_train_fake, name="triangle_attention_sm80_train_fwd")
def forward_train(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, eps: float, transposed: bool) -> list[torch.Tensor]:
    """``pair + drop(TriangleAttention(pair))`` with what the backward needs: ``leaves`` = [pair, wq, wk, wv, wg, wb, wo, ln_w, ln_b], ``mask``
    ``[B, L]`` uint8 or empty, ``ds`` the dropout scale ``[B, L, 128]`` or empty.  Returns [out, qkvg, bias, stats, xh, o, lse]."""
    pair, wq, wk, wv, wg, wb, wo, ln_w, ln_b = leaves
    qkvg, bias, stats, xh = _front(pair, [wq, wk, wv, wg, wb], ln_w, ln_b, mask, eps, transposed, True, True)
    q, k, v, _ = qkv_views(qkvg)
    out, lse = _attention(_token_major(q), _token_major(k), _token_major(v), bias, 1.0 / math.sqrt(D), True)
    res = _back(out, qkvg, wo, pair, ds, transposed)
    return [res, qkvg, bias, stats, xh, out, lse]


def _backward_train_fake(leaves, mask, ds, saved, dout, transposed):
    """[dpair, dwq, dwk, dwv, dwg, dwb, dwo, dln_w, dln_b] in the leaves' shapes and dtypes."""
    return [leaf.new_empty(leaf.shape) for leaf in leaves]


@opaque(fake=_backward_train_fake, name="triangle_attention_sm80_train_bwd")
def backward_train(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, saved: list[torch.Tensor], dout: torch.Tensor,
                   transposed: bool) -> list[torch.Tensor]:
    """The gradients of ``forward_train``'s leaves: the weights packed once for the backward kernels, the back's backward (``back_bwd``), the attention
    core's backward (``attention_backward``), the front's backward (``front_bwd``) and the weight-gradient GEMMs (``front_weight_grads``)."""
    pair, wq, wk, wv, wg, wb, wo, ln_w, ln_b = leaves
    qkvg, bias, stats, xh, o, lse = saved
    b, length = pair.shape[0], pair.shape[1]
    t = b * length * length
    weights = [wq, wk, wv, wg, wb]
    wt, wbt, wot, gamma32 = bwd_pack(weights, wo, ln_w)
    dqkvg = torch.empty((b, length, length, QKVG), dtype=torch.bfloat16, device=pair.device)
    dov, dy, a, delta = back_bwd(dout, o, qkvg, dqkvg, wo, transposed=transposed, ds=ds if ds.numel() else None, wot=wot)
    d_wo = torch.mm(dy.view(t, 128).T, a.view(t, 128), out_dtype=torch.float32).to(wo.dtype)
    q, k, v, _ = qkv_views(qkvg)
    db = _attention_bwd(_token_major(q), _token_major(k), _token_major(v), dov, bias, lse, delta, dqkvg)
    dpair = front_bwd(dqkvg, db, pair, stats, dout, weights, ln_w, transposed=transposed, packed=(wt, wbt, gamma32))
    grads, d_gamma, d_beta = front_weight_grads(dqkvg, db, xh, weights, ln_w, ln_b)
    return [dpair, *grads, d_wo, d_gamma, d_beta]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:9])
        mask, ds, eps, transposed = args[9:]
        out, *saved = forward_train(leaves, mask, ds, eps, transposed)
        ctx.save_for_backward(*leaves, mask, ds, *saved)
        ctx.cfg = (transposed,)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        vals = ctx.saved_tensors
        grads = backward_train(list(vals[:9]), vals[9], vals[10], list(vals[11:]), dout.contiguous(), *ctx.cfg)
        return (*grads, None, None, None, None)


def trainable(pair: torch.Tensor, weights: Sequence[torch.Tensor], ln_w: torch.Tensor, ln_b: torch.Tensor, wo: torch.Tensor, eps: float,
              mask: torch.Tensor | None = None, *, transposed: bool = False, ds: torch.Tensor | None = None) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` with autograd: ``weights`` = (q, k, v, g, bias), ``wo`` the output projection, ``ds`` the dropout scale
    ``[B, L, 128]`` (indexed by the token's second index) or None.  Call ``supports_train()`` first."""
    leaves = (pair, *weights, wo, ln_w, ln_b)
    mk = _mask_u8(mask, pair.device)
    d = ds if ds is not None else pair.new_empty((0,))
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return forward_train(list(leaves), mk, d, float(eps), bool(transposed))[0]
    return _Training.apply(*leaves, mk, d, float(eps), bool(transposed))
