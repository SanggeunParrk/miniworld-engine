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
from torch.multiprocessing.reductions import StorageWeakRef

from miniworld_engine.kernels import _capture

_dir = Path(__file__).parent
STAT_W = 128  # kept for the runner's buffer layout (the Triton module's statistics width)


def _ext():
    return _device_ext(torch.cuda.current_device())


@functools.lru_cache(maxsize=None)
def _device_ext(index: int):
    """The current device's build, resolved once per device (a token step asks several times per call: the capability query is not free)."""
    return _arch_ext("90a" if torch.cuda.get_device_capability(index) == (9, 0) else "100")


@functools.lru_cache(maxsize=None)
def _capability(device: torch.device) -> tuple[int, int]:
    """``torch.cuda.get_device_capability`` of a tensor's device (always indexed), once per device."""
    return torch.cuda.get_device_capability(device)


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
    if dtype is torch.float32 or torch.cuda.get_device_capability(weights[0].device) == (9, 0):
        return _ext().token_pack_vector(weights, int(heads), bool(qk), float(qs), dtype is torch.float32)
    return _ext().token_pack(weights, int(heads), bool(qk), float(qs))


# Private candidate (default 0 = synchronous pack; 1 / 2 = asynchronous pack schedules): kept reachable here instead of being
# deleted (formerly MINIWORLD_TOKEN_DIT_H100_INFER_ASYNC).
_CANDIDATE_INFER_ASYNC = 0
_TOKEN_PACKS: dict = _capture.weight_cache("conditioned_transition.token_packs")
_TOKEN_STREAMS: dict = {}


@functools.lru_cache(maxsize=8)
def _token_swiglu(device_index):
    from .gemm_swiglu import GemmSwigluSm90

    return GemmSwigluSm90(device_index)


#: Hopper fp32 token inference: the expand GEMM with the SwiGLU in its epilogue on TF32 wgmma (False: cuBLAS + ``swiglu_rows``)
_FP32_SWIGLU_GEMM = True  # expand + SwiGLU as one TF32 wgmma GEMM; xa is RNA-rounded by its row kernel (the TMA-fed operands truncate)
_TF32_WAB: dict = _capture.weight_cache("conditioned_transition.tf32_wab")
_BF16_WAB64: dict = _capture.weight_cache("conditioned_transition.bf16_wab64")  # the bf16 pack's SwiGLU rows per 64 (``_bf16_wab64``)
_PAIR_BIAS: dict = _capture.weight_cache(
    "conditioned_transition.pair_bias"
)  # hoisted pair biases, keyed on the pair / mask / weight pack (kernels/_capture.lookup_inputs)


def _swiglu_gemm_ok(xa, wab) -> bool:
    return (
        torch.cuda.get_device_capability(xa.device) == (9, 0)
        and xa.shape[0] % 128 == 0
        and xa.shape[1] % 32 == 0
        and (wab.shape[0] // 2) % 128 == 0
        and wab.shape[1] == xa.shape[1]
    )


def _swiglu_ht(M: int, H: int, nsm: int, bf16: bool = False) -> int:
    """Hidden units per tile of the fused expand: 128 (128 x 256 tiles) or 64 (twice the tiles). bf16: the one whose last wave of the
    persistent grid is fuller (ties: 128, the larger tile); fp32: 128."""

    def eff(ht):
        tiles = (M // 128) * (H // ht)
        return tiles / (-(-tiles // nsm) * nsm)

    # fp32, measured: HT = 64 raises the last wave's fill (M = 1920: 68 -> 91 %; M = 3200: 76 -> 91 %) but loses at both (24 x 32 L384
    # 1.07 -> 1.04, L640 1.00 -> 0.96 of Anthropic's block): the narrower tile's MMA / TMA ratio costs more than the wave. Kept, not chosen.
    # bf16 (H = 1536, H100 132 SMs, graph replay): M = 640 10.3 -> 6.6 us, 1920 19.0 -> 16.4; ties at 1280 / 2560 / 3840 keep 128
    # (10.9 / 10.8, 19.5 / 22.3, 28.5 / 31.6): the fill rule picks each of these
    if bf16 and eff(64) > eff(128):
        return 64
    return 128


def _bf16_wab64(wab, up=False):
    """The SwiGLU GEMM's bf16 operand re-tiled per 64 hidden units, once per weight pack: the pack's [Wa 128 | Wb 128] per-tile rows
    (``token_pack``) -> [Wa 64 | Wb 64] per tile (the 128 x 128-tile form, ``_swiglu_ht``); ``up``: [W_a; W_b; W_u] (the up-gate
    transition's ``wab3``) -> [Wa 64 | Wb 64 | Wu 64] per tile (the 128 x 192-tile up-gate form)."""
    from ... import _capture

    def build():
        K = wab.shape[1]
        if up:
            return wab.view(3, -1, 64, K).transpose(0, 1).reshape(wab.shape).contiguous()
        return wab.view(-1, 2, 2, 64, K).transpose(1, 2).reshape(wab.shape).contiguous()

    # the entry holds ``wab``: a freed pack's address cannot alias another tensor while the entry lives
    return _capture.lookup(
        _BF16_WAB64,
        (wab.data_ptr(), wab._version, tuple(wab.shape), up),
        lambda: (build(), wab),
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[0]


def _tf32_wab(wab, ht=128):
    """[Wa; Wb] -> the SwiGLU GEMM's per-tile packing ([Wa ht rows | Wb ht rows] per tile, ht = 128 or 64), rounded to TF32 (to
    nearest, ties away), once per weight pack (per ht)."""
    from ... import _capture
    from ..._tf32 import tf32_grid

    def build():
        H, K = wab.shape[0] // 2, wab.shape[1]
        w = torch.stack([wab[:H].view(H // ht, ht, K), wab[H:].view(H // ht, ht, K)], 1).reshape(2 * H, K)
        return tf32_grid(w.contiguous())

    # the entry holds ``wab``: a freed pack's address cannot alias another tensor while the entry lives
    return _capture.lookup(
        _TF32_WAB,
        (wab.data_ptr(), wab._version, tuple(wab.shape), ht),
        lambda: (build(), wab),
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[0]


def _pair_bias_wide(z2d, wf, mask, L):
    """``apb.pair_bias`` for a pair wider than its 128-wide ring at the token DiT's 12 / 16 / 24 heads (256: Protenix-v2's diffusion
    pair; 512: IntelliFold2's): the CUDA LayerNorm rows, one cuBLAS GEMM, the masked keys at -1e30. Built once per trajectory (the hoisted bias), so the
    two passes and the mask fill cost nothing a solver step."""
    zh = torch.empty_like(z2d)
    layernorm_rows(z2d, zh)
    bias = torch.mm(wf, zh.t()).view(wf.shape[0], L, L)
    if mask is not None:
        bias.masked_fill_(~mask.reshape(1, 1, L), -1e30)
    return bias


#: The token DiT's model options (``variant`` of :func:`token_inference`; the defaults are AF3 Alg. 23 / Protenix):
#:   parallel  out = a + dA(a) + dT(a) (RoseTTAFold3, Chai-1) instead of a' = a + dA(a), out = a' + dT(a')
#:   sig       the sigmoid AdaLN (else Chai-1's identity AdaLN (1 + W_s c) LN(a) + W_b c of the raw conditioning)
#:   eps       the epsilon of the AdaLNs' LayerNorm of a (Chai-1's release: 0.1)
#:   gated     the gating query sigmoid(W_g a) (Chai-1: none -- the core's gate columns are +1e4, sigmoid = 1)
#:   up        the up-gate transition silu(W_a h) (W_b h) (W_u h) (Boltz-2; else SwiGLU)
#:   kq_ln     the affine LayerNorm of the flat q / k over all heads (RoseTTAFold3; ``qk`` is the per-head RMSNorm, MiniWorld)
#:   kq_eps    its epsilon
#:   qk_rna    fp32: q and k rounded to TF32 to nearest before the TF32 core, which truncates its operands (Boltz-2, RoseTTAFold3, Chai-1:
#:             the identity AdaLN's unbounded scale makes the truncation bias visible; one fused pass, or none where a q / k norm runs)
#: A bias-free query (RoseTTAFold3) is a zero to_query.bias leaf, the identity AdaLN a ones ln_cond / to_scale.bias leaf, no gating
#: query a zero to_gate leaf: the caller's leaf list (``kernels/integrations/token_dit.leaves``).
VARIANT_DEFAULT = (0.0, 1.0, 1e-5, 1.0, 0.0, 0.0, 1e-5, 0.0)
_COND_FOLD: dict = _capture.weight_cache("conditioned_transition.cond_fold")  # [W1; W2], W1 1, b1, b2 per weight pack (``_cond_fold_pack``)


def _cond_fold_pack(w1, b1, w2, b2):
    """The folded conditioning's operands of a bf16 weight pack, once per pack: ``[W1; W2]`` (bf16 [6 D, d_cond], the ONE conditioning
    GEMM's weight), ``W1 1`` (fp32 row sums of the packed W1: the LayerNorm of c applied after the GEMM), b1 and b2 in fp32. The entry
    holds the pack's tensors: a freed pack's address cannot alias another while it lives."""
    from ... import _capture

    def build():
        return (torch.cat([w1, w2]).contiguous(), w1.float().sum(1), b1.float().contiguous(), b2.float().contiguous(), (w1, b1, w2, b2))

    key = tuple((t.data_ptr(), t._version, tuple(t.shape), t.dtype) for t in (w1, b1, w2, b2))
    return (*_capture.lookup(_COND_FOLD, key, build, limit=_capture.BLOCK_CACHE_LIMIT)[:4], key)


#: The folded conditioning GEMMs of a stack, grouped and off the blocks' critical path. A stack's blocks read the SAME conditioning
#: tensor (one sampler step: FoldForge's and the engine's TokenDiTStack pass one object to every block), and a block's conditioning
#: GEMM reads nothing else. So:
#:   - the blocks that read one conditioning are learned in call order (per first block: ``_COND_GROUPS``, the last pass's sequence);
#:   - a block whose successors are known issues ONE GEMM for the next chunk of them ([P, d_cond] x [d_cond, n 6 D]: the stack-level
#:     conditioning hoist of the variant stacks, at most ``_COND_CHUNK_BYTES`` of output a chunk) on a side stream, from its own start --
#:     it runs under this block's kernels, which leave SMs idle at a sampler step's few rows -- and each of those blocks joins it before
#:     its first row pass, reading its [P, 6 D] column slice;
#:   - a block not covered by a chunk (the first of a pass, a stack seen once, a capture whose chunk weight was never built eagerly) runs
#:     its own GEMM on the side stream from the start of the previous block that read this conditioning.
#: Per (capture, stream, conditioning tensor) entry (``_COND_PIPE``): the pass's sequence, the chunks issued for later positions, the
#: last block's start event. Capture-scoped by the capture's own id, whatever is declared static: a sampler step's conditioning changes
#: between replays (FoldForge makes it inside the captured denoiser step), so nothing made from it outlives a capture or a replay -- only
#: the ORDER and the grouping of the GEMMs change, never what a block computes (a chunk's columns are the block's own [W1; W2] rows, a
#: block's key is checked before it is served). The chunk weights ([W1; W2] of n blocks, ``_COND_CHUNKS``) are made outside captures
#: only (a capture would record the concatenation into every replay).
#: A capture that reads a chunk holds it (``_capture.hold``: the LRU of 4 below evicts the chunks of other shapes, and a graph kept per
#: shape still reads its own). Off for big tables (P rows over ``_COND_PIPE_ROWS``).
_COND_PIPE: dict = {}
_COND_PIPE_ROWS = 1024
_COND_CHUNK_BYTES = 32 << 20
_COND_GROUPS: dict = _capture.weight_cache("conditioned_transition.cond_groups")
_COND_CHUNKS: dict = _capture.weight_cache("conditioned_transition.cond_chunks")
_COND_STREAMS: dict = {}
#: bf16 token steps: the CUDA row passes and the SwiGLU GEMM launched as programmatic dependent launches (each waits on
#: griddepcontrol.wait before its first read, so its launch overlaps the end of the kernel before it)
_PDL = True


def _cond_gemm(c, wc, fkey):
    """G = c [W1; W2]^T in fp32 ([P, 6 D]) of the block whose fold pack key is ``fkey``: grouped and pipelined when possible
    (``_COND_PIPE``)."""
    P = c.shape[0]
    cid = _capture._graph_capture_id()
    if P > _COND_PIPE_ROWS or cid == 0:
        return torch.mm(c, wc.t(), out_dtype=torch.float32)
    dev = c.device
    cur = torch.cuda.current_stream(dev)
    side = _COND_STREAMS.get((dev.index, cur.cuda_stream))
    if side is None:
        side = _COND_STREAMS[(dev.index, cur.cuda_stream)] = torch.cuda.Stream(device=dev)
    storage = StorageWeakRef(c.untyped_storage())
    key = (cid, cur.cuda_stream, c.data_ptr(), c._version, tuple(c.shape), c.stride())
    E = _COND_PIPE.get(key)
    if E is not None and (E["storage"].expired() or E["storage"].cdata != storage.cdata):
        E = None
    if E is None or E["seq"][0][0] == fkey:  # a new conditioning, or a new pass over the same one (its first block again)
        if E is None:
            for k in [k for k, e in _COND_PIPE.items() if e["storage"].expired() or k[0] != cid]:
                del _COND_PIPE[k]  # gone conditionings, other captures' entries (a capture's events are its own)
            while len(_COND_PIPE) >= 16:
                _COND_PIPE.pop(next(iter(_COND_PIPE)))
        start = torch.cuda.Event()
        start.record(cur)  # the conditioning is ready here
        E = _COND_PIPE[key] = {"storage": storage, "seq": [], "pending": {}, "last": start, "group": _COND_GROUPS.get(fkey)}
    pos = len(E["seq"])
    E["seq"].append((fkey, wc))
    _COND_GROUPS[E["seq"][0][0]] = tuple(E["seq"])  # the next pass's group (the longest prefix seen so far in this one)
    hit = E["pending"].pop(pos, None)
    if hit is not None and hit[0] == fkey:
        _, G, done, whole = hit
        cur.wait_event(done)
        whole.record_stream(cur)  # allocated on the side stream, read here: not reused before this stream's readers ran
    else:
        side.wait_event(E["last"])
        with torch.cuda.stream(side):
            G = torch.mm(c, wc.t(), out_dtype=torch.float32)
            done = torch.cuda.Event()
            done.record(side)
        cur.wait_event(done)
        G.record_stream(cur)
    # look ahead: one GEMM for the next chunk of this pass's learned group, from this block's start
    group, nxt = E["group"], pos + 1
    if group is not None and nxt < len(group) and nxt not in E["pending"] and group[pos][0] == fkey:
        n = max(1, min(len(group) - nxt, _COND_CHUNK_BYTES // (P * wc.shape[0] * 4)))
        keys = tuple(k for k, _ in group[nxt : nxt + n])
        W = _COND_CHUNKS.get(keys)
        if W is None and cid is None:
            while len(_COND_CHUNKS) >= 4:
                _COND_CHUNKS.pop(next(iter(_COND_CHUNKS)))
            W = _COND_CHUNKS[keys] = torch.cat([w for _, w in group[nxt : nxt + n]])
        if W is not None:
            _capture.hold(W)  # a capture reads the chunk: another shape's chunks (an LRU of 4) must not free it under the graph
            fork = torch.cuda.Event()
            fork.record(cur)
            side.wait_event(fork)
            with torch.cuda.stream(side):
                Gc = torch.mm(c, W.t(), out_dtype=torch.float32)  # [P, n 6 D]
                ready = torch.cuda.Event()
                ready.record(side)
            w6 = wc.shape[0]
            for j, k in enumerate(keys):
                E["pending"][nxt + j] = (k, Gc[:, j * w6 : (j + 1) * w6], ready, Gc)
    mine = torch.cuda.Event()
    mine.record(cur)  # this block's start: an uncovered next block's GEMM forks here
    E["last"] = mine
    return G


_STATIC_PACKS: dict = _capture.weight_cache(
    "conditioned_transition.static_packs"
)  # the weight packs of the blocks with model options (``token_inference``)
_NO_GATE = 1.0e4  # gate logits of a block without a gating query: sigmoid(1e4) == 1.0 in fp32


def token_inference(single, cond, pair, mask, weights, qk, eq, ek, heads, extra=None, variant=None):
    """Hopper port of the B200 token block's fusion and saved-buffer boundaries.

    ``variant``: the model options (:data:`VARIANT_DEFAULT`'s order), None for the defaults. ``weights``: the canonical leaves
    (``kernels/integrations/token_dit.WEIGHTS``, + ``QK_WEIGHTS`` with ``qk`` or ``kq_ln``), then ``transition.up.weight`` (``up``),
    then the flat q / k LayerNorm's weight / bias of q and k (``kq_ln``).

    Live CUDA weight packing, fused pair normalization/bias, gated TMA/WGMMA
    attention, fused residual/AdaLN rows and WGMMA SwiGLU. All other products
    are cuBLAS. Unsupported calls raise and never enter a portable backend.

    ``extra`` is an optional [L, L] logit bias added to every head of every block's pair bias (the structural-token
    family's ``structural_pair_attn_bias``). It is folded into the hoisted bias, so it costs one add per pair, not per step.
    BF16 and FP32 (the FP32 bias is key-permuted: so is the extra term).
    """
    import math
    from ... import _capture
    from ...augmented_attention import cuda as attention
    from ...augmented_attention.cuda import apb

    A, B, L, D = single.shape
    DC = cond.shape[-1]  # the conditioning width: 384, or 768 (the pack's cond tables are built for the weights' width)
    if (
        torch.is_grad_enabled()
        or single.dtype not in (torch.bfloat16, torch.float32)
        or pair.dtype != single.dtype
        or (B != 1 and single.dtype is not torch.bfloat16)  # B trunk_seeds: bf16 only (the TF32 core keeps one shared bias)
        or (heads, D) not in ((16, 768), (24, 768), (12, 768), (16, 1024))
        or L % 128
        or cond.shape[-1] not in (384, 768)  # 768: ESMFold2's token DiT, conditioned at its own width
        or tuple(pair.shape[:3]) != (B, L, L)
        or pair.shape[3] not in (128, 256, 512)  # 256: Protenix-v2's diffusion pair; 512: IntelliFold2's
        or not pair.is_contiguous()
        or _capability(single.device) != (9, 0)
    ):
        raise ValueError(
            "Hopper token CUDA candidate requires inference BF16/FP32 bucketed 768 (12 / 16 / 24 heads) or 1024 (16 heads) / cond 384 or 768 / pair 128, 256 or 512"
        )
    if extra is not None and tuple(extra.shape) != (L, L):
        raise ValueError("Hopper token extra pair bias requires an [L, L] bias")
    qs = math.log2(math.e) / math.sqrt(D // heads)
    fp32 = single.dtype is torch.float32
    v = VARIANT_DEFAULT if variant is None else tuple(variant)
    parallel, sig, ada_eps, gated, up, kq_ln, kq_eps = bool(v[0]), bool(v[1]), float(v[2]), bool(v[3]), bool(v[4]), bool(v[5]), float(v[6])
    qk_rna = fp32 and bool(v[7])
    plain = v == VARIANT_DEFAULT
    pack_qk = qk or kq_ln  # the pack leaves q unscaled when a norm of q follows (the norm's weight carries the scale)
    nbase = 25 if pack_qk else 23
    base, tail = weights[:nbase], weights[nbase:]
    asynchronous = _CANDIDATE_INFER_ASYNC if plain else 0
    key = (
        asynchronous,
        qk,
        heads,
        single.dtype,
        v,
        *((w.data_ptr(), w._version) for w in weights),
    )

    def variant_pack():
        """The pieces the canonical pack has no slot for: [W_a; W_b; W_u] in the GEMM dtype (up-gate), the flat q / k LayerNorm affine
        in fp32 with the logit scale folded into q's (kq_ln)."""
        out = {}
        dt = torch.float32 if fp32 else torch.bfloat16
        if up:
            out["wab3"] = torch.cat([base[20], base[21], tail[0]]).to(dt).contiguous()
        if kq_ln:
            wq, bq, wk, bk = (t.float() for t in tail[-4:])
            out["kq"] = ((wq * qs).contiguous(), (bq * qs).contiguous(), wk.contiguous(), bk.contiguous())
        return out

    def prepare():
        if not asynchronous:
            return (
                token_pack(base, heads, pack_qk, qs, single.dtype),
                (weights, None if plain else variant_pack()),
                None,
                None,
                None,
            )
        idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
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
                (weights, None),
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
        return out, (weights, None), cond_ready, matrix_ready, None

    if plain:
        packed, (_, vpack), cond_ready, matrix_ready, late_pack = _capture.lookup(
            _TOKEN_PACKS, key, prepare, limit=_capture.BLOCK_CACHE_LIMIT
        )
    else:
        # a block with model options keeps its weight pack across CUDA-graph captures and replays (per weight pointer / version, as the
        # variants' inference packs always did: the blocks are inference only here, so their weights do not change under a graph; a
        # first call inside a capture records the pack, an in-place weight change after a capture needs a new capture)
        hit = _STATIC_PACKS.get(key)
        made = hit is None
        if hit is None:
            if len(_STATIC_PACKS) >= _capture.BLOCK_CACHE_LIMIT:
                _STATIC_PACKS.pop(next(iter(_STATIC_PACKS)))
            hit = prepare()
            if not gated and not fp32:
                # bf16, no gating query: the gate rows of the q|k|v|g pack are zero, their bias +1e4 (sigmoid = 1): the one GEMM writes
                # the gate columns (no fill per call)
                hit[0][6][3 * D :] = _NO_GATE
            _STATIC_PACKS[key] = hit
        _capture.hold(hit, not made)  # a capture reads the pack: the count limit above must not free it under the graph
        packed, (_, vpack), cond_ready, matrix_ready, late_pack = hit
    w1, b1, w2, b2, wf, wqkvg, bqkvg, wo, wab_fp32, ws, nq, nk, wab = packed
    # B trunk_seeds of one input share the key mask (the caller passes it expanded over B, stride 0): its first row serves them all
    mask = None if mask is None else mask.reshape(-1, L)[0]

    # The augmented-attention module fills fully masked rows with a finite
    # negative value, so its reference softmax is uniform on that edge.
    # AttentionPairBias has a separate zero-probability reference contract.
    def make_bias():
        if fp32:
            dp = pair.shape[-1]
            if dp == 256 and heads == 16:
                # the 256-wide pair (Protenix-v2's diffusion pair), 16 heads: the same one pass, built per width for the augmented
                # attention's TF32 path (``augmented_attention/cuda/pair_rows.cu`` bias32_k: LN + the projections + the key
                # permutation + the mask, Wf in shared memory)
                from miniworld_engine.kernels.augmented_attention.cuda import _rows_ext

                bias = single.new_empty((heads, L, L))
                _rows_ext().pair_bias_rows32(pair.view(L * L, dp), wf, bias, mask, int(L), 1e-5, -1e30)
            elif dp == 128 and _FUSED_PAIR_BIAS_F32 and heads in (12, 16, 24) and pair.is_contiguous() and wf.is_contiguous():
                # LayerNorm + the projections + the key permutation + the mask in ONE pass over the pair (the LN, a cuBLAS GEMM and
                # the mask fill read / wrote it three times: 302 MB each at fp32); masked keys take -1e30 like the fallback below
                from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

                bias = single.new_empty((heads, L, L))
                rows32().pair_bias_tok_cuda(pair.view(L * L, 128), wf, bias, mask, int(L), 1e-5, True, -1e30)
            else:
                # LayerNorm rows written in the TF32 core's key order, one cuBLAS GEMM, the mask (128, 256 or 512 channels)
                zh = single.new_empty((L * L, dp))
                _ext().layernorm128_rows(pair.view(L * L, dp), zh, 1e-5, True)
                bias = torch.mm(wf, zh.t()).view(heads, L, L)
                if mask is not None:  # the masked keys in key_perm order
                    bias.masked_fill_(~mask[key_perm(L, single.device)], -1e30)
            if extra is not None:
                # the TF32 core reads the keys in key_perm order: the extra term is laid out the same way (masked keys stay at -1e30)
                bias += (extra.float() * 1.4426950408889634)[:, key_perm(L, single.device)]
        else:
            dp = pair.shape[-1]
            one = (lambda z: apb.pair_bias(z, wf, mask, L, 1e-5, -1e30)) if dp == 128 else (lambda z: _pair_bias_wide(z, wf, mask, L))
            if B == 1:
                bias = one(pair.view(L * L, dp))
            else:
                # one bias per trunk_seed (its own pair), laid out [H, B, L, L] for the grouped attention (NBG = B); built once per trajectory
                bias = torch.stack([one(pair[b].reshape(L * L, dp)) for b in range(B)], dim=1).contiguous()
            if extra is not None:
                # the bias is in the attention's base-2 domain (``wf`` carries log2 e); a masked key stays at -1e30
                bias += extra.float() * 1.4426950408889634
        # the small key tensors stay referenced (a freed address cannot alias another tensor); the pair -- the input whose storage
        # and bias are big -- only weakly, by its storage (the pair arrives as a fresh alias of the caller's storage at every call):
        # a hit must read that very storage, and the entry goes with it. Holding the pair kept
        # every past trajectory's biases (and, with the static captures pinning them, filled the store so the next trajectory's
        # entries were evicted before its capture read them: the rebuild was recorded into every replay -- +41 % on a 512-token
        # Protenix-v2 sampler from its second call).
        return bias, StorageWeakRef(pair.untyped_storage()), mask, wf, extra

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
        valid=lambda e: not e[1].expired() and e[1].cdata == StorageWeakRef(pair.untyped_storage()).cdata,
        alive=lambda e: not e[1].expired(),
        inputs=(pair, mask, extra),  # make_bias reads these (and the weight pack) only: _capture.refresh_static_inputs remakes it in place
        # the key without the versions: a pair refilled in place (a new version) replaces its old bias instead of keeping both
        lineage=(
            pair.data_ptr(),
            tuple(pair.shape),
            None if mask is None else mask.data_ptr(),
            wf.data_ptr(),
            heads,
            fp32,
            None if extra is None else extra.data_ptr(),
        ),
    )[0]
    shared = cond.shape[0] == 1 or cond.stride(0) == 0
    c = (
        cond[0].reshape(-1, DC) if shared else cond.reshape(-1, DC)
    )  # shared: the [B L] table of the trunk_seeds (rows are [A][B][L]: row % (B L))
    P = c.shape[0]
    # bf16: the conditioning folded into ONE GEMM of the raw c ([W1; W2], fp32 out); the sigmoid AdaLN's tables take the LayerNorm of c
    # in the row kernels that read them (``token_dit_rows.cu`` CondFold), the identity AdaLN (Chai-1, parallel) reads the raw c anyway --
    # the LayerNorm pass and the second GEMM are gone
    fold = not fp32 and (sig or parallel) and c.dtype is torch.bfloat16 and DC <= D
    if not fold:
        cn = single.new_empty((P, DC))
        cc = torch.empty_like(cn)
        cond_rows(c, cn, cc, 1e-5)
    elif c.stride(-1) != 1 or c.stride(0) % 4 or c.data_ptr() % 8:
        c = c.contiguous()
    if late_pack is not None and late_pack["pending"]:
        stream = late_pack["stream"]
        stream.wait_stream(torch.cuda.current_stream(single.device))
        with torch.cuda.stream(stream):
            _ext().token_stage90(weights, packed, heads, qk, qs, fp32, 5, 13)
            matrix_ready.record(stream)
        late_pack["pending"] = False
    if cond_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(cond_ready)
    # the AdaLN tables: the sigmoid form reads LN(c) (its cond-LN weight folded into w1), the identity form the raw c (its + 1 in b1)
    if fold:
        wc, r1, b1f, b2f, fkey = _cond_fold_pack(w1, b1, w2, b2)
        G = _cond_gemm(c, wc, fkey)  # [P, 6 D] fp32: the AdaLN tables (before the LayerNorm of c), then the gates (no b2)
        stats = single.new_empty((P, 2), dtype=torch.float32)
    else:
        g1 = torch.addmm(b1, cn if sig else cc, w1.t()).view(P, 4, D)
        g2 = torch.addmm(b2, cc, w2.t()).view(P, 2, D)
    M = A * B * L
    xin = single.view(M, D)
    # the fp32 residual (the folded passes read xin instead: no fp32 copy of the stream)
    x = single.new_empty((0 if fold else M, D), dtype=torch.float32)
    xa = single.new_empty((M, D))
    # the fused kernel loses to cuBLAS + the SwiGLU rows below ~768 rows (its 128 x 256 tiles leave SMs idle): measured per cell
    fused_expand = fp32 and _FP32_SWIGLU_GEMM and not up and M >= 768 and _swiglu_gemm_ok(xa, wab_fp32)
    if fold:
        xt = torch.empty_like(xa) if parallel else xa
        _ext().adaln_in_rows_cf_cuda(xin, G, r1, b1f, b2f, c, stats, xa, ada_eps, 1e-5, xt if parallel else None, _PDL, sig)
    elif parallel:
        # both parts read the block input: the transition's AdaLN comes out of the same LN pass (TF32-rounded for the fused expand)
        xt = torch.empty_like(xa)
        adaln_in_rows(single.view(M, D), x, g1[:, 0], g1[:, 1], xa, P, ada_eps, sig, second=(g1[:, 2], g1[:, 3], xt), rnd2=fused_expand)
    else:
        xt = xa
        adaln_in_rows(single.view(M, D), x, g1[:, 0], g1[:, 1], xa, P, ada_eps, sig)
    if matrix_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(matrix_ready)
    if gated or not fp32:  # bf16 without a gating query: the pack's gate bias is +1e4 (above)
        qkvg = torch.addmm(bqkvg[: 3 * D], xa, wqkvg[: 3 * D].t()) if fp32 else torch.addmm(bqkvg, xa, wqkvg.t())
    else:
        # fp32, no gating query: the core's gate columns hold +1e4 (sigmoid = 1); the TF32 GEMM writes the q | k columns only
        qkvg = single.new_empty((M, 3 * D))
        qkvg[:, 2 * D :].fill_(_NO_GATE)
        torch.addmm(bqkvg[: 2 * D], xa, wqkvg[: 2 * D].t(), out=qkvg[:, : 2 * D])
    if qk:
        qknorm_rows(qkvg, nq, nk, eq, ek, D)
    elif kq_ln:
        qk_ln_rows(qkvg, *vpack["kq"], kq_eps, D, rnd=qk_rna)  # q / k rounded on the way (qk_rna)
    elif qk_rna:
        rna_rows(qkvg, 2 * D)
    if fp32:
        vt = torch.mm(wqkvg[3 * D :], xa.t())
        if qk_rna:
            rna_rows(vt, M)  # v^T rounded as q / k (the core truncates)
        attention.tf32_gated_inference(qkvg, vt, bias, 0, A, 2 * D, heads, D // heads)
    else:
        attention.gated_inference(qkvg, bias, 0, A * B, heads, D // heads, B)
    ya = torch.mm(qkvg[:, :D], wo.t())
    if not parallel and fold:
        _ext().resgate_adaln_cf_cuda(xin, ya, G, r1, b1f, b2f, stats, xa, ada_eps, _PDL)
    elif not parallel:
        resgate_adaln_rows(x, ya, g2[:, 0], g1[:, 2], g1[:, 3], xa, P, ada_eps, tf32_round=fused_expand, sig=sig)
    h = single.new_empty((M, ws.shape[1]))
    if up and not fp32:
        # silu(a) b u in the SwiGLU GEMM's epilogue (128 x 192 tiles, [Wa 64 | Wb 64 | Wu 64]): [a | b | u] never reaches HBM.
        # H100 graph replay, K 768, H 1536: M = 640 8.3 against 12.1 us for cuBLAS + the row pass, 1920 20.6 / 30.0, 5120 55.6 / 87.8
        op = _token_swiglu(single.device.index)
        op.ext.gemm_swiglu(xt, _bf16_wab64(vpack["wab3"], up=True), h, op.nsm, 64, True, _PDL)
    elif up:
        ab = torch.mm(xt, vpack["wab3"].t())  # [M, 3 n D] a | b | u
        swiglu_rows(ab, h, up=True)
    elif fused_expand:
        # expand + SwiGLU as one TF32 wgmma GEMM: xt is TF32-rounded by its row kernel and the weights once per pack, so the
        # TMA-fed (truncating) operands carry cuBLAS's rounding; [a | b] never reaches HBM
        op = _token_swiglu(single.device.index)
        ht = _swiglu_ht(M, ws.shape[1], op.nsm)
        op.ext.gemm_swiglu(xt, _tf32_wab(wab_fp32, ht), h, op.nsm, ht)
    elif fp32:
        ab = torch.mm(xt, wab_fp32.t())
        swiglu_rows(ab, h)
    else:
        op = _token_swiglu(single.device.index)
        ht = _swiglu_ht(M, ws.shape[1], op.nsm, bf16=True)
        op.ext.gemm_swiglu(xt, wab if ht == 128 else _bf16_wab64(wab), h, op.nsm, ht, False, _PDL)
    y = torch.mm(h, ws.t())
    out = torch.empty_like(xa)
    if fold:  # out = (xin + sigmoid(gA) ya) + sigmoid(gT) y: both schedules' output
        _ext().resgate_out_cf_cuda(xin, ya, y, G, b2f, out, _PDL)
    elif parallel:
        resgate_out_rows(x, ya, g2[:, 0], out, P, y2=y, gl2=g2[:, 1])
    else:
        resgate_out_rows(x, y, g2[:, 1], out, P)
    return out.view(A, B, L, D)


def adaln_rows(x, ms, mb, out, L, eps=1e-5, sig=True):
    """out = LN(x) * f(ms[row % L]) + mb[row % L]; x fp32 [M, D] (D 768 or 1024), ms / mb per-token tables; f = sigmoid (``sig``) or
    the identity (Chai-1's identity AdaLN, its + 1 in the table)."""
    _ext().adaln_rows_cuda(x, ms, mb, out, int(L), float(eps), bool(sig))


def resgate_adaln_rows(x, y, gl, ms, mb, out, L, eps=1e-5, tf32_round=False, sig=True):
    """x += sigmoid(gl[row % L]) * y in place (fp32); then, when ms is given, out = AdaLN(x) for the next half-block (f as in
    ``adaln_rows``)."""
    if tf32_round:
        _ext().resgate_adaln_rows_r_cuda(x, y, gl, ms, mb, out, int(L), float(eps), bool(sig))
    else:
        _ext().resgate_adaln_rows_cuda(x, y, gl, ms, mb, out, int(L), float(eps), bool(sig))


def gate_rows(o, g, out):
    """out = o * sigmoid(g)."""
    _ext().gate_rows_cuda(o, g, out)


def swiglu_rows(ab, out, up=False):
    """out = silu(a) * b for ab = [a | b]; ``up``: out = silu(a) * b * u for ab = [a | b | u] (Boltz-2's up-gate transition)."""
    _ext().swiglu_rows_cuda(ab, out, bool(up))


def adaln_in_rows(xin, x, ms, mb, out, L, eps=1e-5, sig=True, second=None, rnd2=False):
    """x = xin (the step's input rows, fp32 or bf16, into the fp32 residual [M, 768]) and out = LN(x) * f(ms) + mb, one pass
    (adaln_rows after a copy, without the copy). ``second`` = (ms2, mb2, out2): a second AdaLN of the same LN(x) in the same pass (the
    parallel block's transition input); ``rnd2``: out2 rounded to TF32."""
    ms2, mb2, out2 = second if second is not None else (None, None, None)
    _ext().adaln_in_rows_cuda(xin, x, ms, mb, out, int(L), float(eps), bool(sig), ms2, mb2, out2, bool(rnd2))


def resgate_out_rows(x, y, gl, out, L, y2=None, gl2=None):
    """out = x + sigmoid(gl[row % L]) * y [+ sigmoid(gl2[row % L]) * y2] in out's dtype: the block's last residual, straight into the
    step's output (``y2``: the parallel block's second update)."""
    _ext().resgate_out_rows_cuda(x, y, gl, out, int(L), y2, gl2)


def qk_ln_rows(qk, wq, bq, wk, bk, eps, d=768, rnd=False):
    """The flat q / k LayerNorm with affine in place: q = qk[:, 0:d] <- LN(q) wq + bq, k = qk[:, d:2d] <- LN(k) wk + bk (fp32 [d] affine,
    any logit scale folded into q's): RoseTTAFold3's kq-norm. ``rnd``: the results rounded to TF32 (fp32 rows)."""
    _ext().qk_ln_rows_cuda(qk, wq, bq, wk, bk, float(eps), int(d), bool(rnd))


def rna_rows(t, cols):
    """The first ``cols`` columns of every fp32 row of ``t`` rounded to TF32 to nearest in place (q | k of the qkvg rows, v^T whole): the
    operands of the TF32 attention core, which truncates."""
    _ext().rna_rows_cuda(t, int(cols))


def layernorm_rows(z, out, eps=1e-5):
    """out = LN(z) over 128, 256, 384 or 512 columns, no affine (the conditioning rows, the DiT variants' pair rows), z / out fp32
    or bf16."""
    _ext().layernorm_rows_cuda(z, out, float(eps))


def cond_rows(c, cn, cc, eps=1e-5):
    """cn = LN(c), cc = c, both in the GEMM dtype, in one warp-per-row pass over the 384 or 768 conditioning columns."""
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
            rows32.pair_bias_tok_cuda(
                z2d, wf[16 * b : 16 * b + 16], o2[16 * b : 16 * b + 16], m0, int(L), float(eps), bool(perm), float("-inf")
            )
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
    "qk_ln_rows",
    "rna_rows",
    "qknorm_rows",
    "resgate_adaln_rows",
    "resgate_out_rows",
    "swiglu_rows",
]


#: Prefixes of the requirements_kernel.csv names held by the layout-specialized training rows extension (token_dit_train_rows.cu).
_TOKEN_ROWS_KERNELS = ("cond_transition_token_train_", "cond_transition_token_transpose_", "cond_transition_caller_grad_")


def prebuild(target, rows):
    """Every build of the ConditionedTransition / token-DiT row family for ``target`` (kernels/_prebuild.py), from the requirements
    ``rows``: a build is planned when a row needs one of the kernels it holds (the rows are already restricted to the modes each
    kernel's role runs in). The training row kernels are specialized per model layout: the H100 training path builds rpb 2
    (``integrations.token_dit_train._native_rpb``) for the layout (d_single, n_head, d_cond, d_pair, n) of every token-DiT row
    that trains, when the kernels support it; rpb 8 is the default the pair-bias callers take. The 5-sample uniform-mask
    variant (``empty_uniform``) follows the sample count of a row, which ``rows`` does not carry: both are planned."""
    if target.arch == "sm80":  # A100: the token DiT transition of integrations/conditioned_transition_sm80 (its AdaLN rows + tail)
        yield from _prebuild_sm80(rows)
        return
    if target.arch != "sm90":
        return
    import itertools

    from miniworld_engine.kernels._prebuild import _kv

    from . import train
    from .gemm_swiglu import GemmSwigluSm90

    needed = set().union(*(r["kernels"] for r in rows)) if rows else set()

    def holds(*prefixes):
        return any(k.startswith(prefixes) for k in needed)

    train_rows = holds(*_TOKEN_ROWS_KERNELS)
    if any(k.startswith("cond_transition_token_") and "_train_" not in k and "transpose" not in k for k in needed):
        yield functools.partial(_arch_ext, "90a")
    if holds("cond_transition_gemm_swiglu"):
        yield functools.partial(GemmSwigluSm90, 0)
    if train_rows:
        yield train.owned_gemm_ext
        yield functools.partial(train.rows_ext, hopper=True)
    layouts = set()
    for r in rows:
        dims = _kv(r["dims"])
        if (
            r["mode"] != "train"
            or not any(k.startswith(_TOKEN_ROWS_KERNELS) for k in r["kernels"])
            or not {"d_single", "n_head", "d_cond", "d_pair"} <= dims.keys()
        ):
            continue
        d, h = int(dims["d_single"]), int(dims["n_head"])
        dc, dp, n = int(dims["d_cond"]), int(dims["d_pair"]), int(dims.get("n", 2))
        if (
            (d, d // h) in ((768, 48), (768, 32), (768, 64), (1024, 64))
            and dc in (128, 256, 384, 768)
            and n in (2, 4)
            and dp in (128, 256, 512)
        ):
            layouts.add((d, d // h, dc, n, dp))
    for (d, hd, dc, n, dp), empty in itertools.product(sorted(layouts), (False, True)):
        yield functools.partial(train.rows_ext, d, hd, 2, empty_uniform=empty, dc=dc, n=n, dp=dp, hopper=True)


#: classes whose A100 "cuda" backend runs integrations/conditioned_transition_sm80 (TokenDiTProtenix._forward_a100)
_SM80_TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld"})


def _prebuild_sm80(rows):
    """The A100 builds the token DiT transition reaches: the tail extension (``sm80.py``) and the AdaLN rows it composes with
    (``kernels/adaln/cuda/sm80.py``). Width-independent: one build each when a row of those classes exists."""
    if not any(r["module"] in _SM80_TOKEN for r in rows):
        return
    from miniworld_engine.kernels.adaln.cuda import sm80 as adaln_sm80

    from . import sm80

    yield sm80._ext
    yield adaln_sm80._ext
    if any(r["module"] in _SM80_TOKEN and r["mode"] == "train" for r in rows):  # the training path's parameter casts (transition family)
        from miniworld_engine.kernels.transition.cuda import parameter_cast_pack_sm80, parameter_cast_sm80

        yield parameter_cast_sm80._ext
        yield parameter_cast_pack_sm80._ext
