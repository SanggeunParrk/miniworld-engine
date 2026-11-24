"""CUDA row kernels of the fused token DiT (``token_dit_rows.cu``): the passes between the GEMMs, one block per row.

Same entry points and semantics as the Triton module ``kernels/conditioned_transition/triton/token_dit_kernels.py``
(``adaln_rows``, ``resgate_adaln_rows``, ``gate_rows``, ``swiglu_rows``, ``pair_bias_all``; ``qknorm_rows`` has no
Triton twin: QK-norm on the fused step needs CUDA rows on H100 or B200), so the runner can take
either. ``pair_bias_all`` is a CUDA LayerNorm of the pair rows and one cuBLAS GEMM that writes every block's bias
head-major (a fused WMMA LayerNorm + projection kernel measured 3.4-3.6x slower on B200 and was dropped). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch
from miniworld_engine.kernels._streams import record_stream as _record_stream
from torch.multiprocessing.reductions import StorageWeakRef

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._key_len import kernel_mask

from . import token_dit_fp8 as _fp8

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


def _f8_wab(wab, M: int, H: int, device_index):
    """The bf16 SwiGLU operand in the tiling the e4m3 expand GEMM takes for M rows (the bf16 kernel's ``_swiglu_ht`` rule) and that ht. The
    64-row tiling as a callable: ``token_dit_fp8.weight_pack`` makes it only when it makes the pack (its lookup is host time at every call)."""
    ht = _swiglu_ht(M, H, _fp8_nsm(device_index), bf16=True)
    return (wab if ht == 128 else functools.partial(_bf16_wab64, wab)), ht


def _fp8_nsm(device_index) -> int:
    from miniworld_engine.kernels._device import num_sms

    return num_sms(device_index if device_index is not None else torch.cuda.current_device())


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


#: The hoisted pair bias of a 256 / 512-wide pair on the AttentionPairBias kernel (``make_bias``); False: LN rows + cuBLAS (the tests' and the benchmarks' A / B switch)
WIDE_APB = True


def _pair_bias_wide(z2d, wf, mask, L):
    """The pair bias of a pair wider than 128 channels at a head count ``apb.pair_bias`` is not built for (it serves 256 with 8 / 16 heads and
    512 with 16, the registry's rows): the CUDA LayerNorm rows, one cuBLAS GEMM, the masked keys at -1e30. Built once per trajectory (the
    hoisted bias), so the two passes and the mask fill cost nothing a solver step."""
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
#: Boltz-2's block (``kernels/integrations/token_dit.options``): the default plus the up-gate transition (rna set by ``up``); the one other variant the e4m3 expand serves
VARIANT_UP = (0.0, 1.0, 1e-5, 1.0, 1.0, 0.0, 1e-5, 1.0)
_COND_FOLD: dict = _capture.weight_cache("conditioned_transition.cond_fold")  # [W1; W2], W1 1, b1, b2 per weight pack (``_cond_fold_pack``)


def _cond_fold_pack(w1, b1, w2, b2):
    """The folded conditioning's operands of a bf16 weight pack, once per pack: ``[W1; W2]`` (bf16 [6 D, d_cond], the ONE conditioning
    GEMM's weight), ``W1 1`` (fp32 row sums of the packed W1: the LayerNorm of c applied after the GEMM), b1 and b2 in fp32. The entry
    holds the pack's tensors: a freed pack's address cannot alias another while it lives. Last: whether THIS call made the operands (their
    kernels were issued on the caller's stream just now: ``_cond_gemm``'s side stream must not read ``[W1; W2]`` before them)."""
    from ... import _capture

    made = []

    def build():
        made.append(True)
        return (torch.cat([w1, w2]).contiguous(), w1.float().sum(1), b1.float().contiguous(), b2.float().contiguous(), (w1, b1, w2, b2))

    key = tuple((t.data_ptr(), t._version, tuple(t.shape), t.dtype) for t in (w1, b1, w2, b2))
    return (*_capture.lookup(_COND_FOLD, key, build, limit=_capture.BLOCK_CACHE_LIMIT)[:4], key, bool(made))


#: The folded conditioning GEMMs of a stack, grouped and off the blocks' critical path. A stack's blocks read the SAME conditioning
#: tensor (one sampler step: FoldForge's and the engine's TokenDiTStack pass one object to every block), and a block's conditioning
#: GEMM reads nothing else. So:
#:   - the blocks that read one conditioning are learned in call order (per first block: ``_COND_GROUPS``, the last pass's sequence);
#:   - a block whose successors are known issues ONE GEMM for the next chunk of them ([P, d_cond] x [d_cond, n 6 D]: the stack-level
#:     conditioning hoist of the variant stacks, at most ``_COND_CHUNK_BYTES`` of output a chunk) on a side stream, from its own start --
#:     it runs under this block's kernels, which leave SMs idle at a sampler step's few rows -- and each of those blocks joins it before
#:     its first row pass, reading its [P, 6 D] column slice;
#:   - a block not covered by a chunk (the first of a pass, a stack seen once, a chunk weight not made in the capture's weight scope) runs
#:     its own GEMM on the side stream from the start of the previous block that read this conditioning.
#: Per (capture, stream, conditioning tensor) entry (``_COND_PIPE``): the pass's sequence, the chunks issued for later positions, the
#: last block's start event. Capture-scoped by the capture's own id, whatever is declared static: a sampler step's conditioning changes
#: between replays (FoldForge makes it inside the captured denoiser step), so nothing made from it outlives a capture or a replay -- only
#: the ORDER and the grouping of the GEMMs change, never what a block computes (a chunk's columns are the block's own [W1; W2] rows, a
#: block's key is checked before it is served).
#:
#: **What a side-stream GEMM depends on.** A captured graph replays its kernels in the order of the recorded dependencies only (and an
#: eager side stream runs beside the caller's stream, whose kernels may still be queued), so every GEMM forks from an event of the
#: caller's stream that is issued AFTER the kernels that make its operands:
#:   - the conditioning: made before the pass's first block (the pass's start event);
#:   - a block's ``[W1; W2]``: made in an EARLIER call (a cache hit) it is older than the previous block's start, where the block's own
#:     GEMM forks; made in THIS call (``made``: the first call after the weights' version changed, and every pack of a capture outside
#:     ``static_weights()``, whose packs are the capture's own and remade by every replay) it is issued after that start, and the GEMM
#:     forks from an event recorded now. (Forking it from the previous block's start read the pack before its kernels ran: the first
#:     replay of such a capture, and an eager call right after an optimizer step, computed the tables from unwritten memory.)
#:   - a chunk weight: the concatenation of its blocks' ``[W1; W2]``, made on the caller's stream before the fork.
#: **Weight scope.** The groups and the chunk weights are scoped like every weight pack (``_capture.capture_id()``): the eager scope
#: (eager calls, and captures under ``static_weights()``, which read the chunk the warm-up made and never make one), or one capture
#: outside ``static_weights()`` -- whose packs are remade by every replay, so its chunks are concatenated from ITS packs inside the capture
#: (learned in the capture's first pass over the stack, used from its second: a captured rollout of several steps) and a replay after an
#: optimizer step reads the new weights. (An eager chunk served to such a capture kept the weights of the capture's time.)
#: **A capture ends with nothing pending.** In a capture the block that issues a chunk joins it at its end (``token_inference``): the
#: caller's stream then depends on every GEMM issued so far, so a pass that stops before the chunk's blocks (a shorter stack than the
#: learned group) leaves no unjoined work in the graph, and whatever the side stream read (the conditioning, a pack, a chunk weight of the
#: graph's pool) may be freed and its memory reused by the caller's stream after the block.
#: A capture that reads an eager chunk holds it (``_capture.hold``: the LRU of 4 below evicts the chunks of other shapes, and a graph
#: kept per shape still reads its own). Off for big tables (P rows over ``_COND_PIPE_ROWS``).
_COND_PIPE: dict = {}
_COND_PIPE_ROWS = 1024
_COND_CHUNK_BYTES = 32 << 20
_COND_GROUPS: dict = _capture.weight_cache("conditioned_transition.cond_groups")  # (weight scope, first block's key) -> the pass
_COND_CHUNKS: dict = _capture.weight_cache("conditioned_transition.cond_chunks")  # (weight scope, the blocks' keys) -> [n 6 D, d_cond]
_COND_STREAMS: dict = {}
#: bf16 token steps: the CUDA row passes and the SwiGLU GEMM launched as programmatic dependent launches (each waits on
#: griddepcontrol.wait before its first read, so its launch overlaps the end of the kernel before it)
_PDL = True


def _cond_gemm(c, wc, fkey, made=False):
    """G = c [W1; W2]^T in fp32 ([P, 6 D]) of the block whose fold pack key is ``fkey``: grouped and pipelined when possible
    (``_COND_PIPE``). ``made``: ``wc`` was made in this call (``_cond_fold_pack``). Returns G and, in a capture, the event of the
    look-ahead GEMM this call issued (None: none) -- the caller's stream waits for it at the end of the block."""
    P = c.shape[0]
    cid = _capture._graph_capture_id()
    if P > _COND_PIPE_ROWS or cid == 0:
        return torch.mm(c, wc.t(), out_dtype=torch.float32), None
    ws = _capture.capture_id()  # the weight scope: None for eager calls and captures under static_weights(), else this capture
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
            _capture.prune(_COND_GROUPS, ws)  # other captures' groups and chunks (their packs are those captures' own)
            _capture.prune(_COND_CHUNKS, ws)
        start = torch.cuda.Event()
        start.record(cur)  # the conditioning and this block's pack are ready here
        made = False
        E = _COND_PIPE[key] = {"storage": storage, "seq": [], "pending": {}, "last": start, "group": _COND_GROUPS.get((ws, fkey))}
    pos = len(E["seq"])
    E["seq"].append((fkey, wc))
    _COND_GROUPS[(ws, E["seq"][0][0])] = tuple(E["seq"])  # the next pass's group (the longest prefix seen so far in this one)
    hit = E["pending"].pop(pos, None)
    if hit is not None and hit[0] == fkey:
        _, G, done, whole = hit
        cur.wait_event(done)
        _record_stream(whole, cur)  # allocated on the side stream, read here: not reused before this stream's readers ran
    else:
        if made:
            # the pack's kernels were issued after the previous block's start: fork behind them
            ready = torch.cuda.Event()
            ready.record(cur)
            side.wait_event(ready)
        else:
            side.wait_event(E["last"])
        with torch.cuda.stream(side):
            G = torch.mm(c, wc.t(), out_dtype=torch.float32)
            done = torch.cuda.Event()
            done.record(side)
        cur.wait_event(done)
        _record_stream(G, cur)
    # look ahead: one GEMM for the next chunk of this pass's learned group, from this block's start
    join = None
    group, nxt = E["group"], pos + 1
    if group is not None and nxt < len(group) and nxt not in E["pending"] and group[pos][0] == fkey:
        n = max(1, min(len(group) - nxt, _COND_CHUNK_BYTES // (P * wc.shape[0] * 4)))
        keys = tuple(k for k, _ in group[nxt : nxt + n])
        W = _COND_CHUNKS.get((ws, keys))
        if W is None and ws == cid:  # an eager call, or a capture of its own weight scope: the concatenation is recorded into it
            eager = [k for k in _COND_CHUNKS if k[0] is None]
            if ws is None and len(eager) >= 4:  # (a capture's chunks are its graph's memory: they go with the capture's scope)
                for k in eager[: len(eager) - 3]:
                    del _COND_CHUNKS[k]
            W = _COND_CHUNKS[(ws, keys)] = torch.cat([w for _, w in group[nxt : nxt + n]])
        if W is not None:
            # a capture reads the chunk: another shape's chunks (an LRU of 4) must not free it under the graph
            _capture.hold(W, ws is None)
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
            if cid is not None:
                join = ready
    mine = torch.cuda.Event()
    mine.record(cur)  # this block's start: an uncovered next block's GEMM forks here
    E["last"] = mine
    return G, join


_STATIC_PACKS: dict = _capture.weight_cache(
    "conditioned_transition.static_packs"
)  # the weight packs of the blocks with model options (``token_inference``)
_NO_GATE = 1.0e4  # gate logits of a block without a gating query: sigmoid(1e4) == 1.0 in fp32
#: B200 bf16 inference with B trunk_seeds: every fold in one core launch (0: one launch per fold)
_B200_FOLDS = os.environ.get("MINIWORLD_TOKEN_B200_FOLDS", "1") != "0"
#: B200 bf16 inference: the pair pass on a side stream too, issued after the folded conditioning GEMM (part 3: 1-8 % faster at L 128 -
#: 512; part 2 had measured the fork 1-10 % slower with the pair pass issued first, where it held the SMs the chain needed).
#: OFF by default since 2026-11-24 (MINIWORLD_TOKEN_B200_BF16_FORK=1 turns it on): with several shapes' sampler graphs alive in one process
#: (FoldForge's per-shape sampler-graph cache, F4) the bf16 fork gave non-finite coordinates (AF3 12 FoldBench targets in one process: 1 of 12,
#: 27 targets: 9 of 27); with the fork off 12 / 12 finite. Recording the side-stream outputs on the caller's stream did not fix it: cause open.
_B200_BF16_FORK = os.environ.get("MINIWORLD_TOKEN_B200_BF16_FORK", "0") == "1"
#: B200 fp32 inference inside a capture, one sample: the v^T GEMM on the chain (0: beside it)
_B200_VT_MAIN = os.environ.get("MINIWORLD_TOKEN_B200_VT_MAIN", "1") != "0"
#: B200 fp32 inference inside a capture, several samples (round 3 part 6, measured per row): v^T on the chain at L 128 when the
#: conditioning is shared by the samples or there is no gating query (the chain to the core is short there: the side stream's pair pass
#: + v^T were the longer path; Chai-1 r110 / r111 96.1 -> 94.1 us, shared-cond rows 2-11 % faster; per-sample rows with a gate 2-9 %
#: slower, left beside); on a stream of its own at L 256 with a shared conditioning (beside the q | k rows and the pair pass:
#: Boltz-2 r183 132.8 -> 126.9 us). 0: v^T beside the chain as before
_B200_VT_RULE = os.environ.get("MINIWORLD_TOKEN_B200_VT_RULE", "1") != "0"
#: B200 fp32 inference without a gating query (Chai-1): the gate columns from the q | k | g GEMM (zero rows, bias +1e4; 0: fill + a q | k GEMM)
_B200_NOGATE_GEMM = os.environ.get("MINIWORLD_TOKEN_B200_NOGATE_GEMM", "1") != "0"
#: B200 fp32: the pair -> bias pass and the v^T GEMM of a captured inference step as a parallel branch of the graph, joined before the
#: core (``bias_only_dit.cuda.side``; MINIWORLD_TOKEN_B200_STREAMS=0: one chain). The pair pass is the largest stage of a step whose
#: bias is rebuilt per replay (36 of 213 us at L 768 bf16, 115 of 372 us fp32) and reads nothing of the conditioning / single chain.
_B200_STREAMS = os.environ.get("MINIWORLD_TOKEN_B200_STREAMS", "1") != "0"
#: B200 fp32 inference, several samples: one q | k | v | g GEMM, v read MN-major by the core (``-DVMN``; 0: the v^T GEMM)
_B200_VMN = os.environ.get("MINIWORLD_TOKEN_B200_VMN", "1") != "0"
#: B200 parallel blocks inside a capture: the transition on its own stream beside the attention (0: on the chain after it)
_B200_PAR = os.environ.get("MINIWORLD_TOKEN_B200_PAR", "1") != "0"
#: B200 fp32 inference, several samples (part 9): the CTA-pair TF32 GEMMs (``pair32``) for q | k | v | g with the TF32 rounding of q | k | v in
#: the epilogue, the out projection and the up-gate expand with the SwiGLU epilogue (0: cuBLAS + the row passes)
_B200_PAIR32 = int(os.environ.get("MINIWORLD_TOKEN_B200_PAIR32", "5"))  # bits: 1 q | k | v | g, 2 out projection, 4 expand


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

    b200 = _capability(single.device) == (10, 0)  # B200 (sm_100a): this composition with the B200 cores (``b200`` branches below)
    from ...bias_only_dit.cuda import side as _side

    # (fp32 only: bf16 inference steps measured 1-10 % slower with the fork at L 128 / 384 -- the bf16 pair pass is short, the fork
    # and join edges cost more than the overlap wins; fp32 5-9 % faster at L 128, 2026-11-17)
    sfork = (
        _side.fork(single.device) if b200 and _B200_STREAMS and (single.dtype is torch.float32 or _B200_BF16_FORK) else None
    )  # (None outside a capture)
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
        or _capability(single.device) not in ((9, 0), (10, 0))
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
    asynchronous = _CANDIDATE_INFER_ASYNC if plain and not b200 else 0
    # B200 bf16, default options, the folded conditioning (``fold`` below): the one-launch pack (``_b200_pack``)
    b200_pack = b200 and plain and not fp32 and cond.dtype is torch.bfloat16 and DC <= D
    made_now: list = []
    key = (
        b200_pack,
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
        elif b200 and not fp32:
            # B200 bf16: [W_a; W_b] untiled for cuBLAS + the SwiGLU rows (the pack's bf16 wab is the sm_90a GEMM's tiling)
            out["wab2"] = torch.cat([base[20], base[21]]).to(dt).contiguous()
        if kq_ln:
            wq, bq, wk, bk = (t.float() for t in tail[-4:])
            out["kq"] = ((wq * qs).contiguous(), (bq * qs).contiguous(), wk.contiguous(), bk.contiguous())
        return out

    def prepare():
        made_now.append(True)
        if b200_pack:
            packed_b, vpack_b = _b200_pack(base, int(heads), bool(pack_qk), float(qs))
            return packed_b, (weights, vpack_b), None, None, None
        if not asynchronous:
            return (
                (
                    _ext().token_pack_vector(base, int(heads), bool(pack_qk), float(qs), fp32)
                    if b200
                    else token_pack(base, heads, pack_qk, qs, single.dtype)
                ),
                (weights, None if plain and not b200 else variant_pack()),
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

    if plain and not b200:
        packed, (_, vpack), cond_ready, matrix_ready, late_pack = _capture.lookup(
            _TOKEN_PACKS, key, prepare, limit=_capture.BLOCK_CACHE_LIMIT
        )
    else:
        # a block with model options keeps its weight pack across CUDA-graph captures and replays (per weight pointer / version, as the
        # variants' inference packs always did: the blocks are inference only here, so their weights do not change under a graph; a
        # first call inside a capture records the pack, an in-place weight change after a capture needs a new capture)
        # B200 (2026-11-17): the default-option blocks too -- token_inference is inference only (grad off), and a replay that repacked
        # spent 17 of 82 us (bf16, bo_pack_sm100) / 13 of 104 us (fp32, token_pack_vector) at L 128 on weights that had not changed
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
            elif not gated and b200 and _B200_NOGATE_GEMM:
                # B200 fp32, no gating query (2026-11-17, part 4): the q | k | g pack's gate rows are zero (to_gate.weight = 0), their bias
                # +1e4: ONE contiguous GEMM with the bias in its epilogue writes the gate columns too -- the per-call strided fill and the
                # strided bias copy of an addmm into a q | k view were 5.0 + 6.0 us at Chai-1 r110 L 128
                hit[0][6][2 * D : 3 * D] = _NO_GATE
            if b200 and not fp32 and "fold" not in hit[1][1]:
                # B200: the folded conditioning's operands ([W1; W2], W1 1, b1 / b2 in fp32) belong to this static pack: made once
                # with it (``_cond_fold_pack`` is scoped to the capture: a replay re-ran its cat, row sum and two casts, ~23 us of
                # 88 at Boltz-2 master L 128 -- 2026-11-17)
                w1_, b1_, w2_, b2_ = hit[0][:4]
                hit[1][1]["fold"] = (
                    torch.cat([w1_, w2_]).contiguous(),
                    w1_.float().sum(1),
                    b1_.float().contiguous(),
                    b2_.float().contiguous(),
                )
            if b200 and fp32 and _B200_VMN and "qkvg4" not in hit[1][1]:
                # B200 fp32, several samples (part 8): the q | k | v | g weight / bias (the pack's order is q | k | g | v) for ONE GEMM whose
                # v columns the core reads MN-major; made after the no-gate bias above
                wp, bp = hit[0][5], hit[0][6]
                hit[1][1]["qkvg4"] = tuple(torch.cat([t[: 2 * D], t[3 * D : 4 * D], t[2 * D : 3 * D]]).contiguous() for t in (wp, bp))
            if b200 and fp32 and _B200_PAIR32 and "p32" not in hit[1][1] and "qkvg4" in hit[1][1]:
                # B200 fp32, several samples (part 9): the TF32-rounded weights of the CTA-pair GEMMs (``pair32``): q | k | v | g, the
                # out projection, the up-gate expand regrouped for the SwiGLU epilogue
                from . import pair32 as _p32

                wab3 = hit[1][1].get("wab3")
                hit[1][1]["p32"] = (
                    _p32.round_tf32(hit[1][1]["qkvg4"][0]),
                    _p32.round_tf32(hit[0][7].float()) if _B200_PAIR32 & 2 else None,
                    None if wab3 is None else _p32.pack_swiglu(wab3.float(), _p32.CFG["swiglu_up"][1], up=True),
                )
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
        if sfork is not None:  # after the weight pack (a first call inside a capture packs on the current stream after the fork)
            sfork.wait_stream(torch.cuda.current_stream(single.device))
        with _side.on(sfork):  # B200 inside a capture: beside the conditioning / single chain
            return make_bias_here()

    def make_bias_here():
        if fp32:
            dp = pair.shape[-1]
            if b200:
                # B200: the sm_100a TF32 core reads its bias in natural key order. ONE pass over the pair (the training block's
                # ``pair_bias``: LayerNorm + the projection on TF32 tensor cores + the key mask as -inf; wf carries log2 e): the LN rows,
                # a cuBLAS GEMM and the mask fill read / wrote the fp32 pair twice more (2026-11-17: r99 L2048 1.40 of 2.06 ms)
                from .train import ext as _train_ext

                bias = single.new_empty((heads, L, L))
                mk = None if mask is None else mask.reshape(L).to(torch.bool).contiguous()
                if extra is None and L >= 256 and heads in apb.PAIR_WIDTHS.get(dp, ()) and pair.data_ptr() % 16 == 0:
                    # from L 256: AttentionPairBias's TMA-fed pass (``pair_bias32_tc``, natural key order) -- 86 against 131 us at L 768,
                    # 286 against 485 at L 1536 (pair 128, 16 heads; pair 512: 285 against 422 at L 768); at L 128 it is the slower one
                    wfc = wf if wf.is_contiguous() else wf.contiguous()
                    apb.ext().pair_bias32_tc(pair.view(L * L, dp), wfc, mk, bias, None, L, 1e-5, -1e30, False)
                    return bias, StorageWeakRef(pair.untyped_storage()), mask, wf, extra
                _train_ext(D, D // heads, rpb=2, dc=DC, dp=dp).pair_bias(  # (rpb 2: the build the B200 training block loads)
                    pair.view(L * L, dp),
                    wf if wf.is_contiguous() else wf.contiguous(),
                    None if mask is None else mask.reshape(L).to(torch.bool).contiguous(),
                    bias,
                    single.new_empty((L * L, 2)),
                    L,
                    1e-5,
                    None if extra is None else extra.reshape(-1).float().contiguous(),
                    1.4426950408889634,
                    -1e30,  # masked keys finite, as every inference bias here (a fully masked row stays a uniform softmax)
                )
                return bias, StorageWeakRef(pair.untyped_storage()), mask, wf, extra
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
            # one pass over the pair for every width / head count the AttentionPairBias kernel is built for (128; 256 with 8 / 16 heads;
            # 512 with 16: Protenix-v2, ESMFold2, Chai-1, IntelliFold2); the LN rows + GEMM pair serves the rest
            # (B200, 2026-11-17: a wide pair at a head count ``apb.pair_bias`` is built for takes its one pass too: the LayerNorm rows + GEMM +
            # mask fill read / wrote the pair three times, 0.35 against 0.20 ms at L 768 for pair 256 / 128; on B200 with no ``L % 16`` gate and
            # no ``WIDE_APB`` switch, elsewhere the 3.0.3 rule)
            fast = dp == 128 or (heads in apb.PAIR_WIDTHS.get(dp, ()) and (b200 or (WIDE_APB and L % 16 == 0)))
            one = (lambda z: apb.pair_bias(z, wf, mask, L, 1e-5, -1e30)) if fast else (lambda z: _pair_bias_wide(z, wf, mask, L))
            if B == 1:
                bias = one(pair.view(L * L, dp))
            else:
                # one bias per trunk_seed (its own pair), laid out [H, B, L, L] for the grouped attention (NBG = B); built once per trajectory
                # (B200: [B, H, L, L], each trunk_seed's planes contiguous for its core launch)
                if b200 and _B200_FOLDS and dp == 128 and pair.is_contiguous():
                    # B200 (2026-11-17, part 3): every trunk_seed's bias in ONE pass, [H, B, L, L] (the fold core launch reads that layout;
                    # one pass per trunk_seed + a stack were 5 x 4.4 + 1.9 us at L 128)
                    mb = None if mask is None else mask.reshape(1, L).expand(B, L).contiguous()
                    bias = apb.pair_bias_batch(pair.reshape(B, L, L, dp), wf, mb, 1e-5, -1e30)
                else:
                    bias = torch.stack([one(pair[b].reshape(L * L, dp)) for b in range(B)], dim=0 if b200 else 1).contiguous()
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
    def bias_lookup():
        return _capture.lookup_inputs(
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
            inputs=(
                pair,
                mask,
                extra,
            ),  # make_bias reads these (and the weight pack) only: _capture.refresh_static_inputs remakes it in place
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

    # B200 fp32 inside a capture (2026-11-17, part 3): the pair pass (a persistent grid on every SM) is issued after the conditioning
    # GEMMs below -- issued first, it held the SMs while the conditioning rows and their GEMMs queued behind it (~11 us at L 128)
    late_bias = sfork is not None
    bias = None if late_bias else bias_lookup()
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
    if fold and vpack is not None and "fold" in vpack:
        wc, r1, b1f, b2f = vpack["fold"]  # made by the one-launch pack (``_b200_pack``)
        fkey, made = key, bool(made_now)
        G, cond_join = _cond_gemm(c, wc, fkey, made)
        stats = single.new_empty((P, 2), dtype=torch.float32)
    elif fold:
        wc, r1, b1f, b2f, fkey, made = _cond_fold_pack(w1, b1, w2, b2)
        # [P, 6 D] fp32: the AdaLN tables (before the LayerNorm of c), then the gates (no b2)
        G, cond_join = _cond_gemm(c, wc, fkey, made)
        stats = single.new_empty((P, 2), dtype=torch.float32)
    else:
        g1 = torch.addmm(b1, cn if sig else cc, w1.t()).view(P, 4, D)
        if late_bias:  # the gates' GEMM beside the chain (read after the join before the core), then the pair pass
            assert sfork is not None
            sfork.wait_stream(torch.cuda.current_stream(single.device))
            with _side.on(sfork):
                g2 = torch.addmm(b2, cc, w2.t()).view(P, 2, D)
            bias = bias_lookup()
        else:
            g2 = torch.addmm(b2, cc, w2.t()).view(P, 2, D)
    if bias is None:  # (bf16 with the fork: the pair pass after the folded conditioning GEMM)
        bias = bias_lookup()
    M = A * B * L
    xin = single.view(M, D)
    # the fp32 residual (the folded passes read xin instead: no fp32 copy of the stream)
    x = single.new_empty((0 if fold else M, D), dtype=torch.float32)
    xa = single.new_empty((M, D))
    # the fused kernel loses to cuBLAS + the SwiGLU rows below ~768 rows (its 128 x 256 tiles leave SMs idle): measured per cell
    fused_expand = fp32 and _FP32_SWIGLU_GEMM and not up and not b200 and M >= 768 and _swiglu_gemm_ok(xa, wab_fp32)
    # EXPERIMENTAL e4m3 placements (opt-in, default off: ``token_dit_fp8.PLACES`` empty -> f8 empty, the bf16 route below unchanged)
    # (``identity``: Chai-1's form, the parallel block with the identity AdaLN, no up-gate / k-q LayerNorm / per-head q-k norm: expand, out, squeeze only,
    # the transition's AdaLN comes out of the attention's LN pass as e4m3; any other parallel block (RoseTTAFold3 ...): expand only, one quantising row pass)
    identity = parallel and not sig and not up and not kq_ln and not qk
    # ``up``: the default block plus Boltz-2's up-gate transition (its expand reads [Wa 64 | Wb 64 | Wu 64] tiles, ht 64)
    f8 = (
        _fp8.active(
            M, D, fp32, (plain or v == VARIANT_UP) and fold, identity and fold, parallel and fold and not up and not identity, up, b200
        )
        if _fp8.PLACES  # (B200: the tcgen05 e4m3 expand of token_dit_fp8_b200, expand only unless MINIWORLD_TOKEN_DIT_FP8_B200 says more)
        else frozenset()
    )
    if _fp8.PLACES:  # observability only (fp8.route_stats()): plain int bumps at call time
        if f8:
            _fp8.STATS["calls"] += 1
            for _p in f8:
                _fp8.STATS[_p] += 1
        else:
            _fp8.STATS["declined"] += 1
    if f8:
        f8w = _fp8.weight_pack(
            key,
            wqkvg,
            wo,
            ws,
            (_bf16_wab64(vpack["wab3"], up=True), 64)
            if up
            else (vpack["wab2"], 128)
            if b200
            else _f8_wab(wab, M, ws.shape[1], single.device.index),
            bqkvg,
        )
    if f8 and "qkvg" in f8:
        xt = xa
        xa8 = torch.empty((M, D), device=single.device, dtype=_fp8.E4M3)
        sxa = torch.empty((M, 1), device=single.device, dtype=torch.float32)
        _fp8.ext().adaln_in_rows_cf_q8(xin, G, r1, b1f, b2f, c, stats, xa8, sxa, ada_eps, 1e-5, _PDL)
    elif f8 and "expand" in f8 and identity:
        # the parallel block (Chai-1): the transition's AdaLN comes out of the attention's LN pass as e4m3 + a row scale (xt8 / sxt)
        xt = None
        xt8 = torch.empty((M, D), device=single.device, dtype=_fp8.E4M3)
        sxt = torch.empty((M, 1), device=single.device, dtype=torch.float32)
        _fp8.ext().adaln_in_rows_cf_par_q8(xin, G, r1, b1f, b2f, c, stats, xa, xt8, sxt, ada_eps, 1e-5, _PDL, sig)
    elif fold:
        xt = torch.empty_like(xa) if parallel else xa
        _ext().adaln_in_rows_cf_cuda(xin, G, r1, b1f, b2f, c, stats, xa, ada_eps, 1e-5, xt if parallel else None, _PDL, sig)
    elif parallel:
        # both parts read the block input: the transition's AdaLN comes out of the same LN pass (TF32-rounded for the fused expand)
        xt = torch.empty_like(xa)
        adaln_in_rows(single.view(M, D), x, g1[:, 0], g1[:, 1], xa, P, ada_eps, sig, second=(g1[:, 2], g1[:, 3], xt), rnd2=fused_expand)
    else:
        xt = xa
        adaln_in_rows(single.view(M, D), x, g1[:, 0], g1[:, 1], xa, P, ada_eps, sig)
    if f8 and "expand" in f8 and parallel and not identity:
        # the parallel schedule's transition input is the block input's AdaLN rows (one quantising row pass; before any fork: the B200 side-stream
        # schedule runs the transition beside the attention)
        xt8, sxt = _fp8.rows_q8(xt)
    keep: list = []

    def _trans():
        h = single.new_empty((M, ws.shape[1]))
        if f8 and "expand" in f8:
            a8, sa8, ht = f8w["expand"]
            # (B200 parallel schedule: the transition runs beside the attention core on a side stream; a persistent expand grid over every SM would
            # wait for the core's CTAs (Chai-1 L512 A5: 1.044 with 148 CTAs, 0.964 with 96): ``token_dit_fp8.B200_PAR_CTAS``)
            nsm_f8 = _fp8.B200_PAR_CTAS if b200 and xpar else _fp8_nsm(single.device.index)
            _fp8.ext().gemm_swiglu_f8(xt8, a8, h, sxt, sa8, nsm_f8, ht, _PDL, up)
        elif up and not fp32 and not b200:
            # silu(a) b u in the SwiGLU GEMM's epilogue (128 x 192 tiles, [Wa 64 | Wb 64 | Wu 64]): [a | b | u] never reaches HBM.
            # H100 graph replay, K 768, H 1536: M = 640 8.3 against 12.1 us for cuBLAS + the row pass, 1920 20.6 / 30.0, 5120 55.6 / 87.8
            op = _token_swiglu(single.device.index)
            op.ext.gemm_swiglu(xt, _bf16_wab64(vpack["wab3"], up=True), h, op.nsm, 64, True, _PDL)
        elif up and b200 and p32 is not None and _B200_PAIR32 & 4 and p32[2] is not None and ws.shape[1] % 32 == 0:
            # B200 fp32, several samples (part 9): the CTA-pair expand with silu(a) b u in its epilogue ([a | b | u] never reaches HBM)
            from . import pair32 as _p32

            _p32.swiglu(xt, p32[2], h, up=True)
        elif up:
            ab = torch.mm(xt, vpack["wab3"].t())  # [M, 3 n D] a | b | u
            swiglu_rows(ab, h, up=True)
            keep.append(ab)
        elif fused_expand:
            # expand + SwiGLU as one TF32 wgmma GEMM: xt is TF32-rounded by its row kernel and the weights once per pack, so the
            # TMA-fed (truncating) operands carry cuBLAS's rounding; [a | b] never reaches HBM
            op = _token_swiglu(single.device.index)
            ht = _swiglu_ht(M, ws.shape[1], op.nsm)
            op.ext.gemm_swiglu(xt, _tf32_wab(wab_fp32, ht), h, op.nsm, ht)
        elif fp32:
            ab = torch.mm(xt, wab_fp32.t())
            swiglu_rows(ab, h)
            keep.append(ab)
        elif b200:
            # B200 bf16: from gemm_swiglu.MIN_ROWS rows the sm_100a expand GEMM with the SwiGLU epilogue (the composed runner's: [a | b] never
            # reaches HBM), below it cuBLAS + the SwiGLU row pass
            gsw = _b200_swiglu(xt, vpack["wab2"])
            if gsw is not None:
                gsw(xt, vpack["wab2"], h)
            else:
                ab = torch.mm(xt, vpack["wab2"].t())
                swiglu_rows(ab, h)
                keep.append(ab)
        else:
            op = _token_swiglu(single.device.index)
            ht = _swiglu_ht(M, ws.shape[1], op.nsm, bf16=True)
            op.ext.gemm_swiglu(xt, wab if ht == 128 else _bf16_wab64(wab), h, op.nsm, ht, False, _PDL)
        if f8 and "squeeze" in f8:
            y = _fp8.mm(*_fp8.rows_q8(h), f8w["squeeze"])
        else:
            y = torch.mm(h, ws.t())
        keep.append(h)  # (side stream: referenced until the join)
        return y

    # B200 inside a capture, the parallel block (RoseTTAFold3, Chai-1; 2026-11-18, part 8): the transition reads only xt, so it runs on a
    # stream of its own beside the attention, forked after the q | k rows (beside the core, whose 48 work items at five samples L 128 leave
    # most SMs idle) and joined before the output pass. The same kernels, the same bits.
    spar = ypar = None
    xpar = b200 and parallel and sfork is not None and _B200_PAR
    if matrix_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(matrix_ready)
    vt = None
    # (one sample: the side stream's pair pass outlasts the chain to the core -- v^T on the chain, after the q | k rows; five samples keep
    # it beside the chain: measured 4-6 % slower on the chain at L 256 - 512, 2026-11-17 part 3)
    several = fp32 and sfork is not None and A * B > 1 and _B200_VT_RULE
    # B200 fp32, several samples (2026-11-18, part 8): ONE q | k | v | g GEMM and the core reading v MN-major from its columns
    # (``attn_inf_tf32 -DVMN``): no v^T GEMM, q | k | v rounded in one pass
    # (measured per row, VMN on / off: L 128 2-10 us faster except the parallel block with a gate and a conditioning per sample --
    # RoseTTAFold3 r109 80.7 -> 83.7 us, its v^T beside the chain overlaps better; from L 256 the core's two stages lose: L 512
    # 7-11 % slower)
    vmn = (
        fp32
        and b200
        and _B200_VMN
        and A * B > 1
        and L <= 128
        and not (parallel and gated and not shared)
        and vpack is not None
        and "qkvg4" in vpack
        and (gated or _B200_NOGATE_GEMM)
    )
    vt_own = several and shared and 128 < L <= 256 and not vmn
    svt = None
    vt_main = fp32 and sfork is not None and (A * B == 1 or (several and L <= 128 and (shared or not gated))) and _B200_VT_MAIN and not vmn
    # (part 9) the pair GEMMs: five-sample L 128 rows (M a multiple of 128), the TF32 rounding of q | k | v fused only where no q / k norm
    # follows (cuBLAS is the faster GEMM alone)
    p32 = vpack.get("p32") if vmn and M % 128 == 0 and D == 768 else None
    if vmn and p32 is not None and _B200_PAIR32 & 1 and gated and not shared and qk_rna and not qk and not kq_ln:
        from . import pair32 as _p32

        qkvg = _p32.mm(xa, p32[0], vpack["qkvg4"][1], rnac=3 * D)  # [M, 4 D] q | k | v | g, q | k | v rounded to TF32
    elif vmn:
        wq4, bq4 = vpack["qkvg4"]
        qkvg = torch.addmm(bq4, xa, wq4.t())  # [M, 4 D] q | k | v | g
        if qk:
            qknorm_rows(qkvg, nq, nk, eq, ek, D)
            if qk_rna:
                rna_rows(qkvg[:, 2 * D :], D)
        elif kq_ln:
            qk_ln_rows(qkvg, *vpack["kq"], kq_eps, D, rnd=qk_rna)  # q / k rounded on the way (qk_rna)
            if qk_rna:
                rna_rows(qkvg[:, 2 * D :], D)
        elif qk_rna:
            rna_rows(qkvg, 3 * D)
    elif vt_own:  # v^T on a stream of its own (``_B200_VT_RULE``)
        svt = _side.fork(single.device)
        with _side.on(svt):
            vt = torch.mm(wqkvg[3 * D :], xa.t())
            if qk_rna:
                rna_rows(vt, M)
    elif (
        fp32 and sfork is not None and not vt_main
    ):  # B200 fp32 inside a capture: v^T (xa is in) beside the q | k | g GEMM and the q / k rows
        sfork.wait_stream(torch.cuda.current_stream(single.device))
        with _side.on(sfork):
            vt = torch.mm(wqkvg[3 * D :], xa.t())
            if qk_rna:
                rna_rows(vt, M)
    if f8 and "qkvg" in f8:
        qkvg = _fp8.mm(xa8, sxa, f8w["qkvg"], bias=f8w["bqkvg"])
    elif vmn:
        pass
    elif gated or not fp32 or (b200 and _B200_NOGATE_GEMM):  # no gating query: the static pack's gate bias is +1e4 (above)
        qkvg = torch.addmm(bqkvg[: 3 * D], xa, wqkvg[: 3 * D].t()) if fp32 else torch.addmm(bqkvg, xa, wqkvg.t())
    else:
        # fp32, no gating query: the core's gate columns hold +1e4 (sigmoid = 1); the TF32 GEMM writes the q | k columns only
        qkvg = single.new_empty((M, 3 * D))
        qkvg[:, 2 * D :].fill_(_NO_GATE)
        torch.addmm(bqkvg[: 2 * D], xa, wqkvg[: 2 * D].t(), out=qkvg[:, : 2 * D])
    if vmn:
        pass
    elif qk:
        qknorm_rows(qkvg, nq, nk, eq, ek, D)
    elif kq_ln:
        qk_ln_rows(qkvg, *vpack["kq"], kq_eps, D, rnd=qk_rna)  # q / k rounded on the way (qk_rna)
    elif qk_rna:
        rna_rows(qkvg, 2 * D)
    if xpar:  # the transition beside the core, from after the q | k rows
        spar = _side.fork(single.device)
        with _side.on(spar):
            ypar = _trans()
    if vt_main:
        vt = torch.mm(wqkvg[3 * D :], xa.t())
        if qk_rna:
            rna_rows(vt, M)
    # the core skips the key tiles past the last attended key of a padded mask, reading the mask itself (``kernels._key_len``; the same bits)
    _side.join(single.device, sfork)  # the bias (and v^T) are in
    _side.join(single.device, svt)
    kmask = kernel_mask(mask, L)
    if fp32:
        if vmn:
            _b200_core(single.device, torch.float32, heads, D // heads)(qkvg, bias, 0, A, vmn=True)
        elif vt is None:
            vt = torch.mm(wqkvg[3 * D :], xa.t())
            if qk_rna:
                rna_rows(vt, M)  # v^T rounded as q / k (the core truncates)
        if vmn:
            pass
        elif b200:
            _b200_core(single.device, torch.float32, heads, D // heads)(qkvg, bias, 0, A, vt=vt)
        else:
            attention.tf32_gated_inference(qkvg, vt, bias, 0, A, 2 * D, heads, D // heads, kmask=kmask)
    else:
        if b200:  # the masked keys are -1e30 in the bias (no key-tile skip on B200)
            core = _b200_core(single.device, torch.bfloat16, heads, D // heads)
            if B == 1:
                core(qkvg, bias, 0, A)
            elif _B200_FOLDS and pair.shape[-1] == 128 and pair.is_contiguous():  # (the bias is [H, B, L, L]: built for this above)
                # B trunk_seeds (rows [A][B][L]) in ONE launch: a work item pairs two samples of one fold and reads that fold's bias
                # planes (2026-11-17, part 3; the per-fold launches below were 5 x 7.8 us at L 128)
                core.folds(qkvg, bias, A, B, L)
            else:  # B trunk_seeds (rows [A][B][L]): one launch per trunk_seed, its samples B L rows apart, its own bias planes
                for b in range(B):
                    core(qkvg[b * L :], bias[b], 0, A, L=L, srow=B * L)
        else:
            attention.gated_inference(qkvg, bias, 0, A * B, heads, D // heads, B, kmask=kmask)
    if p32 is not None and _B200_PAIR32 & 2:
        from . import pair32 as _p32

        ya = _p32.mm(qkvg[:, :D], p32[1], cfg=_p32.CFG["wo"])
    else:
        if f8 and "out" in f8:
            ya = _fp8.mm(*_fp8.rows_q8(qkvg[:, :D]), f8w["out"])
        else:
            ya = torch.mm(qkvg[:, :D], wo.t())
    if f8 and "expand" in f8 and parallel:
        pass  # xt8 / sxt were made above (identity: by the attention's LN pass; otherwise one quantising row pass)
    elif f8 and "expand" in f8:
        xt8 = torch.empty((M, D), device=single.device, dtype=_fp8.E4M3)
        sxt = torch.empty((M, 1), device=single.device, dtype=torch.float32)
        _fp8.ext().resgate_adaln_cf_q8(xin, ya, G, r1, b1f, b2f, stats, xt8, sxt, ada_eps, _PDL)
    elif not parallel and fold:
        _ext().resgate_adaln_cf_cuda(xin, ya, G, r1, b1f, b2f, stats, xa, ada_eps, _PDL)
    elif not parallel:
        # (B200 part 9: the pair expand reads the TF32 bits of xt -- rounded here, as cuBLAS rounds its operands)
        rnd_xt = fused_expand or (up and p32 is not None and _B200_PAIR32 & 4 and p32[2] is not None)
        resgate_adaln_rows(x, ya, g2[:, 0], g1[:, 2], g1[:, 3], xa, P, ada_eps, tf32_round=rnd_xt, sig=sig)
    y = ypar if ypar is not None else _trans()
    _side.join(single.device, spar)
    out = torch.empty_like(xa)
    if fold:  # out = (xin + sigmoid(gA) ya) + sigmoid(gT) y: both schedules' output
        _ext().resgate_out_cf_cuda(xin, ya, y, G, b2f, out, _PDL)
        if cond_join is not None:
            # a capture: the look-ahead GEMM this block issued joins here (it ran under this block's kernels; the next block reads it first)
            torch.cuda.current_stream(single.device).wait_event(cond_join)
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
    if target.arch == "sm120":  # RTX 5090: the same two extensions built for sm_120a (kernels/_sm120_builds.py)
        yield from _prebuild_sm120(rows)
        return
    if target.arch == "sm100":  # B200: the token DiT rows / training rows / SwiGLU GEMM builds (_prebuild_sm100)
        yield from _prebuild_sm100(rows)
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
    # the opt-in FP8 route (docs/fp8.md, token_dit_fp8.active: bf16 inference at width 768): its e4m3 rows + expand / SwiGLU build
    if any(
        r["mode"] == "eval"
        and r["dtype"] != "float32"
        and _kv(r["dims"]).get("d_single") == "768"
        and any(k.startswith("cond_transition_token_") for k in r["kernels"])
        for r in rows
    ):
        yield _fp8.ext
    if train_rows:
        yield train.owned_gemm_ext
        yield functools.partial(train.rows_ext, hopper=True)
    if any(
        r["mode"] == "train"
        and r["dtype"] != "float32"
        and _kv(r["dims"]).get("d_single") == "768"
        and any(k.startswith(_TOKEN_ROWS_KERNELS) for k in r["kernels"])
        for r in rows
    ):  # width-768 bf16 training blocks: the default byte reductions (nox / hsave) and the opt-in FP8 training placements run these row kernels
        from . import token_dit_fp8_train as _f8train

        yield _f8train.ext
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


#: B200: the token DiT classes of integrations/token_dit (inference) / token_dit_train (training); the bias-only class reads the token row
#: extension in its pair hoist only (bias_only_dit/cuda/runner.hoist); the classes whose inference is the composed runner with its SwiGLU
#: GEMM (default options); the class whose transition has an up gate (no SwiGLU GEMM in training)
_SM100_TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld", "TokenDiTBoltz2", "TokenDiTRoseTTAFold3", "TokenDiTChai1"})
_SM100_COMPOSED = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld"})
_SM100_UP = frozenset({"TokenDiTBoltz2"})


def _prebuild_sm100(rows):
    """The B200 builds: the token DiT rows extension for sm_100 (``_arch_ext("100")``: the composed runner, ``token_inference``, the
    training block's rows and the bias-only hoist); the training rows (``train.rows_ext_sm100``: token_dit_train_rows_sm100.cu, sm_100a only, rpb 2 as the B200 training
    block loads them since 2026-11-17 -- the empty-uniform variants are H100's) per layout of the token DiT rows that train or run fp32
    inference (its one-pass pair bias); the sm_100a SwiGLU GEMM (``gemm_swiglu.GemmSwiglu``)
    where a row's M = samples x length reaches ``gemm_swiglu.MIN_ROWS``: bf16 inference of the composed runner (h only) and bf16 training
    of a block without the up gate (``-DSAVE_AB``; the training block pads the samples to an even count)."""
    from miniworld_engine.kernels._prebuild import _kv

    from . import gemm_swiglu, train

    token = [r for r in rows if r["module"] in _SM100_TOKEN or r["module"] == "TokenDiTMiniWorldBiasOnly"]
    if not token:
        return
    yield functools.partial(_arch_ext, "100")
    layouts, swiglu = set(), set()
    for r in token:
        if r["module"] not in _SM100_TOKEN:
            continue
        dims = _kv(r["dims"])
        d, h = int(dims["d_single"]), int(dims["n_head"])
        dc, dp, n = int(dims["d_cond"]), int(dims["d_pair"]), int(dims.get("n", 2))
        bf16 = r["dtype"] != "float32"
        if r["mode"] == "train" or not bf16:  # (fp32 inference: the one-pass pair bias of the training rows, ``token_inference``)
            if (
                (d, d // h) in ((768, 48), (768, 32), (768, 64), (1024, 64))
                and dc in (128, 256, 384, 768)
                and n in (2, 4)
                and dp in (128, 256, 512)
            ):
                layouts.add((d, d // h, dc, n, dp))
        if r["mode"] == "train":
            rows_m = max((s + s % 2) * (-(-length // 128) * 128) for s in r["samples"] for length in r["lengths"])
            if bf16 and r["module"] not in _SM100_UP and rows_m >= gemm_swiglu.MIN_ROWS:
                swiglu.add((d, n * d, True))
        elif bf16 and r["module"] not in _SM100_UP:  # token_inference / the composed runner: every block without the up gate
            rows_m = max(s * (-(-length // 8) * 8) for s in r["samples"] for length in r["lengths"])
            if rows_m >= gemm_swiglu.MIN_ROWS:
                swiglu.add((d, n * d, False))
    for d, hd, dc, n, dp in sorted(layouts):
        yield functools.partial(train.rows_ext_sm100, d, hd, 2, empty_uniform=False, dc=dc, n=n, dp=dp)
    # fp32 inference with several samples at d 768: the CTA-pair TF32 GEMMs (``pair32``, part 9; the up-gate expand for Boltz-2)
    p32 = {
        r["module"] in _SM100_UP
        for r in token
        if r["module"] in _SM100_TOKEN and r["dtype"] == "float32" and r["mode"] == "eval" and int(_kv(r["dims"])["d_single"]) == 768
    }
    if p32:
        from . import pair32

        seen = set()
        for up in sorted(p32):
            for call in pair32.prebuild_calls(768, up):
                if call.args[1] not in seen:
                    seen.add(call.args[1])
                    yield call
    for k, hidden, save_ab in sorted(swiglu):
        if k % 64 == 0 and hidden % 128 == 0 and hidden % k == 0:  # gemm_swiglu.supported
            yield _swiglu_sm100(k, hidden, save_ab)
    # the opt-in e4m3 expand route (docs/b200/fp8.md, token_dit_fp8.active on a B200): its rows module and the width-768 expand cubin
    f8_rows = [
        r
        for r in token
        if r["mode"] == "eval" and r["dtype"] != "float32" and r["module"] in _SM100_TOKEN and _kv(r["dims"]).get("d_single") == "768"
    ]
    if f8_rows:
        from . import token_dit_fp8_b200

        # (agent E: Boltz-2's up-gate expand has its own cubin)
        yield from token_dit_fp8_b200.prebuild_items(
            plain=any(r["module"] not in _SM100_UP for r in f8_rows), up=any(r["module"] in _SM100_UP for r in f8_rows)
        )


def _swiglu_sm100(k: int, hidden: int, save_ab: bool):
    """The cubin ``gemm_swiglu.GemmSwiglu(idx, K=k, H=hidden, save_ab)`` loads (its defines)."""
    from ...augmented_attention.cuda import sm100
    from . import gemm_swiglu

    defs = (f"DIM={k}", f"HMUL={hidden // k}", "ITEM_SCHED", *(("SAVE_AB",) if save_ab else ()))
    return functools.partial(sm100.cubin, "gemm_swiglu2_sm100", defs, src_dir=gemm_swiglu._dir)


#: classes whose A100 "cuda" backend runs integrations/conditioned_transition_sm80 (TokenDiTProtenix._forward_a100)
_SM80_TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld"})


def _prebuild_sm80(rows):
    """The A100 builds the token DiT transition reaches: the tail extension (``sm80.py``) and the AdaLN rows it composes with
    (``kernels/adaln/cuda/sm80.py``). Width-independent: one build each when a row of those classes exists."""
    if not any(r["module"] in _SM80_TOKEN for r in rows):
        return
    from miniworld_engine.kernels.adaln.cuda import sm80 as adaln_sm80

    from . import sm80

    yield sm80._ext_sm80
    yield adaln_sm80._ext_sm80
    if any(r["module"] in _SM80_TOKEN and r["mode"] == "train" for r in rows):  # the training path's parameter casts (transition family)
        from miniworld_engine.kernels.transition.cuda import parameter_cast_pack_sm80, parameter_cast_sm80

        yield parameter_cast_sm80._ext_sm80
        yield parameter_cast_pack_sm80._ext


#: classes whose RTX 5090 "cuda" backend reaches integrations/conditioned_transition_sm80 or integrations/adaln_sm80 (sm_120a builds)
_SM120_TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld", "TokenDiTMiniWorldBiasOnly", "TokenDiTRoseTTAFold3"})


#: token DiT classes the RTX 5090 e4m3 route (``token_dit_fp8_sm120``, opt-in) serves at width 768 bf16 inference
_SM120_F8_TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld", "TokenDiTBoltz2", "TokenDiTRoseTTAFold3", "TokenDiTChai1"})


def _prebuild_sm120(rows):
    """The RTX 5090 builds of the AdaLN rows and the ConditionedTransition tail (the sm_80 sources for sm_120a), and the opt-in token DiT
    e4m3 row passes (``token_dit_fp8_sm120``: bf16 inference rows at width 768)."""
    from miniworld_engine.kernels._prebuild import _kv

    if any(
        r["module"] in _SM120_F8_TOKEN and r["mode"] == "eval" and r["dtype"] != "float32" and _kv(r["dims"]).get("d_single") == "768"
        for r in rows
    ):
        from . import token_dit_fp8_sm120

        yield token_dit_fp8_sm120.ext
    if not any(r["module"] in _SM120_TOKEN for r in rows):
        return
    from miniworld_engine.kernels.adaln.cuda import sm80 as adaln_sm80

    from . import sm80

    yield sm80._ext_sm120
    yield adaln_sm80._ext_sm120


_B200_CORES: dict = {}


def _b200_core(device: torch.device, dtype: torch.dtype, heads: int, dh: int):
    """B200 (sm_100a): the token DiT's gated inference core (``augmented_attention/cuda/sm100.GatedInferenceCore``: bf16 q | k | v | g,
    fp32 q | k | g + v^T on TF32 tensor cores; logits in exp2 units, sigmoid(g) o written over q), one per device / dtype / layout."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, dtype, heads, dh)
    core = _B200_CORES.get(key)
    if core is None:
        from ...augmented_attention.cuda import sm100

        core = _B200_CORES[key] = sm100.GatedInferenceCore(idx, dtype, heads, dh)
    return core


_B200_SWIGLU: dict = {}


def _b200_swiglu(xt, wab):
    """The sm_100a SwiGLU GEMM (``gemm_swiglu.GemmSwiglu``, h only) for this expand, or None where it does not apply (rows below
    ``gemm_swiglu.MIN_ROWS``, a layout outside its tiles)."""
    from . import gemm_swiglu

    if not (xt.is_contiguous() and wab.is_contiguous() and gemm_swiglu.supported(xt, wab)):
        return None
    idx = xt.device.index if xt.device.index is not None else torch.cuda.current_device()
    key = (idx, wab.shape[1], wab.shape[0] // 2)
    op = _B200_SWIGLU.get(key)
    if op is None:
        op = _B200_SWIGLU[key] = gemm_swiglu.GemmSwiglu(idx, K=key[1], H=key[2])
    return op


def _b200_pack(base, heads: int, qk: bool, qs: float):
    """B200 bf16, default options, the folded conditioning: the block's weight pack in ONE launch (the bias-only runner's segment pack,
    ``kernels/bias_only_dit/cuda/pack_sm100.py``) and one GEMV. It replaces ``token_pack_vector`` plus the [Wa; Wb] cast / cat and the
    cond-fold operands (``_cond_fold_pack``: [W1; W2] cat, W1 1, b1 / b2 in fp32): ~9 small kernels a CUDA-graph replay re-ran (the pack
    is scoped to the capture), ~35 of 86 us at r144 L128. Returns the canonical 13-slot pack (unused slots None) and the fold operands
    (wc [6 D, d_cond] bf16 whose rows are W1 then W2, W1 1 fp32, b1 / b2 fp32). b1 / b2 are the fp32 biases themselves (the canonical
    pack rounded them through bf16 first)."""
    from ...bias_only_dit.cuda.pack_sm100 import Pack

    w_q, b_q, w_k, w_v, w_g, w_o, ln1, s1w, s1b, h1w, g1w, g1b, lnp, wb, ln2, s2w, s2b, h2w, g2w, g2b, wa, wbx, wsq = base[:23]
    D, DC, HH, DP = w_q.shape[0], ln1.numel(), wa.shape[0], lnp.numel()
    dev, bf, f32 = w_q.device, torch.bfloat16, torch.float32
    p = Pack(bf)
    wc = torch.empty(6 * D, DC, device=dev, dtype=bf)
    for i, (w, ln) in enumerate(((s1w, ln1), (h1w, ln1), (s2w, ln2), (h2w, ln2))):
        p.copy(w, wc[i * D : (i + 1) * D], scale=ln)
    p.copy(g1w, wc[4 * D : 5 * D])
    p.copy(g2w, wc[5 * D :])
    b1f = torch.empty(4 * D, device=dev, dtype=f32)
    p.copy(s1b, b1f[:D])
    p.zero(b1f[D : 2 * D])
    p.copy(s2b, b1f[2 * D : 3 * D])
    p.zero(b1f[3 * D :])
    b2f = torch.empty(2 * D, device=dev, dtype=f32)
    p.copy(g1b, b2f[:D])
    p.copy(g2b, b2f[D:])
    wf = p.copy(wb, torch.empty(heads, DP, device=dev, dtype=bf), scale=lnp, cs=1.4426950408889634)
    wqkvg = torch.empty(4 * D, D, device=dev, dtype=bf)
    p.copy(w_q, wqkvg[:D], cs=1.0 if qk else qs)
    p.copy(w_k, wqkvg[D : 2 * D])
    p.copy(w_v, wqkvg[2 * D : 3 * D])
    p.copy(w_g, wqkvg[3 * D :])
    bqkvg = torch.empty(4 * D, device=dev, dtype=bf)
    p.copy(b_q, bqkvg[:D], cs=1.0 if qk else qs)
    p.zero(bqkvg[D:])
    wo = w_o if w_o.dtype is bf else p.copy(w_o, torch.empty(D, D, device=dev, dtype=bf))
    ws = wsq if wsq.dtype is bf else p.copy(wsq, torch.empty(D, HH, device=dev, dtype=bf))
    wab = torch.empty(2 * HH, D, device=dev, dtype=bf)
    p.copy(wa, wab[:HH])
    p.copy(wbx, wab[HH:])
    nq = nk = None
    if qk:
        nq = p.copy(base[23], torch.empty(base[23].numel(), device=dev, dtype=f32), cs=qs)
        nk = p.copy(base[24], torch.empty(base[24].numel(), device=dev, dtype=f32))
    p.launch(dev)
    r1 = torch.mm(wc[: 4 * D], torch.ones(DC, 1, device=dev, dtype=bf), out_dtype=f32).view(4 * D)
    packed = (wc[: 4 * D], None, wc[4 * D :], None, wf, wqkvg, bqkvg, wo, None, ws, nq, nk, None)
    return packed, {"wab2": wab, "fold": (wc, r1, b1f, b2f), "keep": p}
