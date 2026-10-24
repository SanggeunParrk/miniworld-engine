"""CUDA row kernels of the fused token DiT (``token_dit_rows.cu``): the passes between the GEMMs, one block per row.

Same entry points and semantics as the Triton module ``kernels/conditioned_transition/triton/token_dit_kernels.py``
(``adaln_rows``, ``resgate_adaln_rows``, ``gate_rows``, ``swiglu_rows``, ``pair_bias_all``; ``qknorm_rows`` has no
Triton twin: QK-norm on the fused step needs CUDA rows on H100 or B200), so the runner can take
either. ``pair_bias_all`` is a CUDA LayerNorm of the pair rows and one cuBLAS GEMM that writes every block's bias
head-major (a fused WMMA LayerNorm + projection kernel measured 3.4-3.6x slower on B200 and was dropped). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
STAT_W = (
    128  # kept for the runner's buffer layout (the Triton module's statistics width)
)


def _ext():
    arch = "90a" if torch.cuda.get_device_capability() == (9, 0) else "100"
    return _arch_ext(arch)


@functools.lru_cache(maxsize=None)
def _arch_ext(arch):
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=f"token_dit_rows_cuda_sm{arch}",
        sources=[str(_dir / "token_dit_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes(arch, ptx=() if arch == "90a" else ("100",)),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def available() -> bool:
    """The extension builds (a failure is reported once by the caller, which keeps the Triton rows)."""
    _ext()
    return True


def token_pack(weights, heads, qk, qs, dtype=torch.bfloat16):
    if dtype is torch.float32 or torch.cuda.get_device_capability(
        weights[0].device
    ) == (9, 0):
        return _ext().token_pack_vector(
            weights, int(heads), bool(qk), float(qs), dtype is torch.float32
        )
    return _ext().token_pack(weights, int(heads), bool(qk), float(qs))


# Private candidate (default 0 = synchronous pack; 1 / 2 = asynchronous pack schedules): kept reachable here instead of being
# deleted (formerly MINIWORLD_TOKEN_DIT_H100_INFER_ASYNC).
_CANDIDATE_INFER_ASYNC = 0
_TOKEN_PACKS: dict = {}
_TOKEN_STREAMS: dict = {}


@functools.lru_cache(maxsize=8)
def _token_swiglu(device_index):
    from .gemm_swiglu import GemmSwigluSm90

    return GemmSwigluSm90(device_index)


#: Hopper fp32 token inference: the expand GEMM with the SwiGLU in its epilogue on TF32 wgmma (False: cuBLAS + ``swiglu_rows``)
_FP32_SWIGLU_GEMM = True  # expand + SwiGLU as one TF32 wgmma GEMM; xa is RNA-rounded by its row kernel (the TMA-fed operands truncate)
_TF32_WAB: dict = {}
_PAIR_BIAS: dict = {}  # hoisted pair biases, keyed on the pair / mask / weight pack (kernels/_capture.lookup_inputs)


def _swiglu_gemm_ok(xa, wab) -> bool:
    return (
        torch.cuda.get_device_capability(xa.device) == (9, 0)
        and xa.shape[0] % 128 == 0
        and xa.shape[1] % 32 == 0
        and (wab.shape[0] // 2) % 128 == 0
        and wab.shape[1] == xa.shape[1]
    )


def _swiglu_ht(M: int, H: int, nsm: int) -> int:
    """Hidden units per tile of the fp32 fused expand: 128 (128 x 256 tiles) or 64 (twice the tiles): the one whose last wave of the
    persistent grid is fuller (ties: 128, the larger tile)."""
    def eff(ht):
        tiles = (M // 128) * (H // ht)
        return tiles / (-(-tiles // nsm) * nsm)
    # Measured: HT = 64 raises the last wave's fill (M = 1920: 68 -> 91 %; M = 3200: 76 -> 91 %) but loses at both (24 x 32 L384
    # 1.07 -> 1.04, L640 1.00 -> 0.96 of Anthropic's block): the narrower tile's MMA / TMA ratio costs more than the wave. Kept, not chosen.
    return 128


def _tf32_wab(wab, ht=128):
    """[Wa; Wb] -> the SwiGLU GEMM's per-tile packing ([Wa ht rows | Wb ht rows] per tile, ht = 128 or 64), rounded to TF32 (to
    nearest, ties away), once per weight pack (per ht)."""
    from ... import _capture

    def build():
        H, K = wab.shape[0] // 2, wab.shape[1]
        w = torch.stack([wab[:H].view(H // ht, ht, K), wab[H:].view(H // ht, ht, K)], 1).reshape(2 * H, K)
        return ((w.contiguous().view(torch.int32) + 0x1000) & ~0x1FFF).view(torch.float32)

    # the entry holds ``wab``: a freed pack's address cannot alias another tensor while the entry lives
    return _capture.lookup(
        _TF32_WAB,
        (wab.data_ptr(), wab._version, tuple(wab.shape), ht),
        lambda: (build(), wab),
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[0]


def token_inference(single, cond, pair, mask, weights, qk, eq, ek, heads, extra=None):
    """Hopper port of the B200 token block's fusion and saved-buffer boundaries.

    Live CUDA weight packing, fused pair normalization/bias, gated TMA/WGMMA
    attention, fused residual/AdaLN rows and WGMMA SwiGLU. All other products
    are cuBLAS. Unsupported calls raise and never enter a portable backend.

    ``extra`` is an optional [L, L] logit bias added to every head of every block's pair bias (the structural-token
    family's ``structural_pair_attn_bias``). It is folded into the hoisted bias, so it costs one add per pair, not per step.
    BF16 only: the FP32 bias is key-permuted by its fused kernel.
    """
    import math
    from ... import _capture
    from ...augmented_attention import cuda as attention
    from ...augmented_attention.cuda import apb

    A, B, L, D = single.shape
    if (
        torch.is_grad_enabled()
        or single.dtype not in (torch.bfloat16, torch.float32)
        or pair.dtype != single.dtype
        or B != 1
        or (heads, D) not in ((16, 768), (24, 768), (12, 768), (16, 1024))
        or L % 128
        or cond.shape[-1] != 384
        or tuple(pair.shape) != (1, L, L, 128)
        or not pair.is_contiguous()
        or torch.cuda.get_device_capability(single.device) != (9, 0)
    ):
        raise ValueError(
            "Hopper token CUDA candidate requires inference BF16/FP32 bucketed 768 (12 / 16 / 24 heads) or 1024 (16 heads) / 384 / 128"
        )
    if extra is not None and (
        single.dtype is not torch.bfloat16 or tuple(extra.shape) != (L, L)
    ):
        raise ValueError("Hopper token extra pair bias requires BF16 and an [L, L] bias")
    qs = math.log2(math.e) / math.sqrt(D // heads)
    fp32 = single.dtype is torch.float32
    asynchronous = _CANDIDATE_INFER_ASYNC
    key = (
        asynchronous,
        qk,
        heads,
        single.dtype,
        *((w.data_ptr(), w._version) for w in weights),
    )

    def prepare():
        if not asynchronous:
            return (
                token_pack(weights, heads, qk, qs, single.dtype),
                weights,
                None,
                None,
                None,
            )
        idx = (
            single.device.index
            if single.device.index is not None
            else torch.cuda.current_device()
        )
        if idx not in _TOKEN_STREAMS:
            _TOKEN_STREAMS[idx] = torch.cuda.Stream(device=single.device)
        stream = _TOKEN_STREAMS[idx]
        out = _ext().token_alloc90(weights, heads, qk, fp32)
        if asynchronous == 2:
            # Keep the bandwidth-heavy pair projection separate from large
            # weight casts; release the latter beside conditioning GEMMs.
            _ext().token_stage90(weights, out, heads, qk, qs, fp32, 0, 5)
            return (
                out,
                weights,
                None,
                torch.cuda.Event(),
                {"stream": stream, "pending": True},
            )
        stream.wait_stream(torch.cuda.current_stream(single.device))
        cond_ready, matrix_ready = torch.cuda.Event(), torch.cuda.Event()
        with torch.cuda.stream(stream):
            _ext().token_stage90(weights, out, heads, qk, qs, fp32, 0, 4)
            cond_ready.record(stream)
            _ext().token_stage90(weights, out, heads, qk, qs, fp32, 5, 13)
            matrix_ready.record(stream)
        _ext().token_stage90(weights, out, heads, qk, qs, fp32, 4, 5)
        return out, weights, cond_ready, matrix_ready, None

    packed, _, cond_ready, matrix_ready, late_pack = _capture.lookup(
        _TOKEN_PACKS, key, prepare, limit=_capture.BLOCK_CACHE_LIMIT
    )
    w1, b1, w2, b2, wf, wqkvg, bqkvg, wo, wab_fp32, ws, nq, nk, wab = packed
    mask = None if mask is None else mask.reshape(L)
    # The augmented-attention module fills fully masked rows with a finite
    # negative value, so its reference softmax is uniform on that edge.
    # AttentionPairBias has a separate zero-probability reference contract.
    def make_bias():
        if fp32:
            if _FUSED_PAIR_BIAS_F32 and heads in (12, 16, 24) and pair.is_contiguous() and wf.is_contiguous():
                # LayerNorm + the projections + the key permutation + the mask in ONE pass over the pair (the LN, a cuBLAS GEMM and
                # the mask fill read / wrote it three times: 302 MB each at fp32); masked keys take the same -1e30 as token_mask90
                from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

                bias = single.new_empty((heads, L, L))
                rows32().pair_bias_tok_cuda(pair.view(L * L, 128), wf, bias, mask, int(L), 1e-5, True, -1e30)
            else:
                zh = single.new_empty((L * L, 128))
                _ext().layernorm128_rows(pair.view(L * L, 128), zh, 1e-5, True)
                bias = torch.mm(wf, zh.t()).view(heads, L, L)
                if mask is not None:
                    _ext().token_mask90(bias, mask)
        else:
            bias = apb.pair_bias(pair.view(L * L, 128), wf, mask, L, 1e-5, -1e30)
            if extra is not None:
                # the bias is in the attention's base-2 domain (``wf`` carries log2 e); a masked key stays at -1e30
                bias += extra.float() * 1.4426950408889634
        # the entry keeps its key tensors alive: a freed address cannot alias another tensor
        return bias, pair, mask, wf, extra

    # The pair bias depends on the pair, the key mask and the weights only: the same at every solver step. Served from the
    # per-input cache (``static_inputs()``: a replay re-runs no producing kernel, as the harness and a sampler declare);
    # a capture without it records the build once and every replay rebuilds it from the pair as it is then.
    bias = _capture.lookup_inputs(
        _PAIR_BIAS,
        (
            pair.data_ptr(),
            pair._version,
            tuple(pair.shape),
            None if mask is None else (mask.data_ptr(), mask._version),
            wf.data_ptr(),
            wf._version,
            heads,
            fp32,
            None if extra is None else (extra.data_ptr(), extra._version),
        ),
        make_bias,
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[0]
    shared = cond.shape[0] == 1 or cond.stride(0) == 0
    c = cond[0, 0] if shared else cond.reshape(-1, 384)
    P = c.shape[0]
    cn = single.new_empty((P, 384))
    cc = torch.empty_like(cn)
    cond_rows(c, cn, cc, 1e-5)
    if late_pack is not None and late_pack["pending"]:
        stream = late_pack["stream"]
        stream.wait_stream(torch.cuda.current_stream(single.device))
        with torch.cuda.stream(stream):
            _ext().token_stage90(weights, packed, heads, qk, qs, fp32, 5, 13)
            matrix_ready.record(stream)
        late_pack["pending"] = False
    if cond_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(cond_ready)
    g1 = torch.addmm(b1, cn, w1.t()).view(P, 4, D)
    g2 = torch.addmm(b2, cc, w2.t()).view(P, 2, D)
    M = A * L
    x = single.new_empty((M, D), dtype=torch.float32)
    xa = single.new_empty((M, D))
    adaln_in_rows(single.view(M, D), x, g1[:, 0], g1[:, 1], xa, P, 1e-5)
    if matrix_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(matrix_ready)
    qkvg = (
        torch.addmm(bqkvg[: 3 * D], xa, wqkvg[: 3 * D].t())
        if fp32
        else torch.addmm(bqkvg, xa, wqkvg.t())
    )
    if qk:
        qknorm_rows(qkvg, nq, nk, eq, ek, D)
    if fp32:
        vt = torch.mm(wqkvg[3 * D :], xa.t())
        attention.tf32_gated_inference(qkvg, vt, bias, 0, A, 2 * D, heads, D // heads)
    else:
        attention.gated_inference(qkvg, bias, 0, A, heads, D // heads)
    y = torch.mm(qkvg[:, :D], wo.t())
    # the fused kernel loses to cuBLAS + the SwiGLU rows below ~768 rows (its 128 x 256 tiles leave SMs idle): measured per cell
    fused_expand = fp32 and _FP32_SWIGLU_GEMM and M >= 768 and _swiglu_gemm_ok(xa, wab_fp32)
    resgate_adaln_rows(x, y, g2[:, 0], g1[:, 2], g1[:, 3], xa, P, 1e-5, tf32_round=fused_expand)
    h = single.new_empty((M, ws.shape[1]))
    if fused_expand:
        # expand + SwiGLU as one TF32 wgmma GEMM: xa is TF32-rounded by its row kernel and the weights once per pack, so the
        # TMA-fed (truncating) operands carry cuBLAS's rounding; [a | b] never reaches HBM
        op = _token_swiglu(single.device.index)
        ht = _swiglu_ht(M, ws.shape[1], op.nsm)
        op.ext.gemm_swiglu(xa, _tf32_wab(wab_fp32, ht), h, op.nsm, ht)
    elif fp32:
        ab = torch.mm(xa, wab_fp32.t())
        swiglu_rows(ab, h)
    else:
        op = _token_swiglu(single.device.index)
        op.ext.gemm_swiglu(xa, wab, h, op.nsm)
    y = torch.mm(h, ws.t())
    out = torch.empty_like(xa)
    resgate_out_rows(x, y, g2[:, 1], out, P)
    return out.view(A, B, L, D)


def adaln_rows(x, ms, mb, out, L, eps=1e-5):
    """out = LN(x) * sigmoid(ms[row % L]) + mb[row % L]; x fp32 [M, D] (D 768 or 1024), ms / mb per-token tables."""
    _ext().adaln_rows_cuda(x, ms, mb, out, int(L), float(eps))


def resgate_adaln_rows(x, y, gl, ms, mb, out, L, eps=1e-5, tf32_round=False):
    """x += sigmoid(gl[row % L]) * y in place (fp32); then, when ms is given, out = AdaLN(x) for the next half-block."""
    if tf32_round:
        _ext().resgate_adaln_rows_r_cuda(x, y, gl, ms, mb, out, int(L), float(eps))
    else:
        _ext().resgate_adaln_rows_cuda(x, y, gl, ms, mb, out, int(L), float(eps))


def gate_rows(o, g, out):
    """out = o * sigmoid(g)."""
    _ext().gate_rows_cuda(o, g, out)


def swiglu_rows(ab, out):
    """out = silu(a) * b for ab = [a | b]."""
    _ext().swiglu_rows_cuda(ab, out)


def adaln_in_rows(xin, x, ms, mb, out, L, eps=1e-5):
    """x = xin (the step's input rows, fp32 or bf16, into the fp32 residual [M, 768]) and out = LN(x) * sigmoid(ms) + mb,
    one pass (adaln_rows after a copy, without the copy)."""
    _ext().adaln_in_rows_cuda(xin, x, ms, mb, out, int(L), float(eps))


def resgate_out_rows(x, y, gl, out, L):
    """out = x + sigmoid(gl[row % L]) * y in out's dtype: the block's last residual, straight into the step's output."""
    _ext().resgate_out_rows_cuda(x, y, gl, out, int(L))


def layernorm_rows(z, out, eps=1e-5):
    """out = LN(z) over 384 columns, no affine (the conditioning rows), z / out fp32 or bf16."""
    _ext().layernorm_rows_cuda(z, out, float(eps))


def cond_rows(c, cn, cc, eps=1e-5):
    """cn = LN(c), cc = c, both in the GEMM dtype, in one warp-per-row pass over 384 conditioning columns."""
    _ext().cond_rows_cuda(c, cn, cc, float(eps))


def qknorm_rows(qk, wq, wk, eq, ek, d=768):
    """QK-norm in place: q = qk[:, 0:d], k = qk[:, d:2d] of every row, RMSNorm per head (head dim = len(wq): 48, 32 or 64 at
    d 768, 64 at d 1024) times wq / wk (fp32, any logit scale folded in), eps eq / ek."""
    _ext().qknorm_rows_cuda(qk, wq, wk, float(eq), float(ek), int(d))


#: ``pair_bias_all`` applies the key mask itself (the runner then skips its own -inf fill).
PAIR_BIAS_MASK = True


def key_perm(L, device):
    """Position p of each group of 8 keys holds key 4 i + j for p = 2 j + i: the key order of the H100 TF32 attention core
    (``augmented_attention/cuda/attn_tf32.cu``), in which its bias and key mask are laid out."""
    p = torch.arange(L, device=device)
    r = p % 8
    return p - r + 4 * (r % 2) + r // 2


#: Hopper fp32 pair bias in one pass (False: LayerNorm rows + a cuBLAS GEMM)
_FUSED_PAIR_BIAS_F32 = True


def pair_bias_all(z2d, wt, out, L, eps=1e-5, mask=None, perm=False):
    """Every block's pair bias, head-major: out [NB, L, L] = (LN(z) @ wt)^T with wt [C, NB] (the LayerNorm weights and
    any log2 e scale already folded in), -inf on the columns of masked keys (``mask`` [L] bool). z2d [L L, 128].
    A CUDA LayerNorm and one cuBLAS GEMM (a fused one-pass LayerNorm + projection kernel, one thread per pair row, measured
    slower -- 450 against 266 us for a whole L = 768 block call -- and was dropped). ``perm``: the key columns in
    ``key_perm`` order (the TF32 core's; the LayerNorm writes the pair rows there, so it costs nothing)."""
    nb = wt.shape[1]
    if (
        _FUSED_PAIR_BIAS_F32
        and wt.dtype is torch.float32
        and z2d.dtype is torch.float32
        and nb % 16 == 0
        and z2d.is_contiguous()
        and torch.cuda.get_device_capability(z2d.device) == (9, 0)
    ):
        # Hopper fp32: LN + the 16 projections of a block in one pass over the pair (LN, then a GEMM, read it three times)
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32 as _rows32

        rows32 = _rows32()
        wf = wt.t().contiguous()
        m0 = None if mask is None else mask.reshape(-1).to(torch.bool).contiguous()
        o2 = out.view(nb, -1)
        for b in range(nb // 16):
            rows32.pair_bias_tok_cuda(z2d, wf[16 * b : 16 * b + 16], o2[16 * b : 16 * b + 16], m0, int(L), float(eps), bool(perm), float("-inf"))
        return
    m = None if mask is None else mask.reshape(-1).to(torch.bool)
    if m is not None and perm:
        m = m[key_perm(L, m.device)]
    m = None if m is None else m.contiguous()
    zh = torch.empty(z2d.shape[0], 128, device=z2d.device, dtype=wt.dtype)
    _ext().layernorm128_rows(z2d, zh, float(eps), bool(perm))
    torch.mm(wt.t(), zh.t(), out=out.view(nb, -1))
    if m is not None:
        out.view(nb, L, L)[:, :, ~m] = float("-inf")


__all__ = [
    "STAT_W",
    "adaln_in_rows",
    "adaln_rows",
    "available",
    "cond_rows",
    "gate_rows",
    "key_perm",
    "layernorm_rows",
    "pair_bias_all",
    "qknorm_rows",
    "resgate_adaln_rows",
    "resgate_out_rows",
    "swiglu_rows",
]
