"""Automatic dispatch to the packaged, selected bidirectional H100 training path. The hidden width equals the pair width, or is
twice it in one direction at D64 / D128 (``_hidden_twice``: AF3 / Protenix template blocks, pair 64, hidden 128)."""

from __future__ import annotations

import functools

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType

#: Widths with a packaged bidirectional CUDA training path (D64, D128 and the wide port).
TRAINING_WIDTHS = (64, 128, 256, 384, 512)
#: Widths with a single-direction CUDA training path (D64: port of the bidirectional D64 kernels; D128: B1/B7;
#: D256/384: the wide sources at hidden D).$
SINGLE_TRAINING_WIDTHS = (64, 128, 256, 384, 512)

#: Samples ONE LAUNCH takes at each pair width (tokens b-major, one launch per stage; MiniWorld folds templates, structures or
#: augments into the batch of one pair stack). The MiniWorld-facing name is ``max_batch(width)``; ``MAX_BATCH`` keeps its D64 meaning
#: (the same as ``trimul_b200.MAX_BATCH``). A call with more samples, or more than fit in memory, is split into launches (``_plan``):
#: any B is served. The bound is what was validated (B up to 48 against B one-sample calls and the FP32 statement); the kernels' only
#: index bound is the int32 token index B L^2 < 2^31 (``_index_ok``). D64 / D128: the native K1 / K3 (inference) and B1 / B2 / B7
#: (training) paths. D256 / D384 / D512: the B200-style port (planes [c, B L, L], its K3 / backward token-flat).
MAX_BATCH_BY_WIDTH = {64: 48, 128: 48, 256: 48, 384: 48, 512: 48}
MAX_BATCH = MAX_BATCH_BY_WIDTH[64]


def max_batch(width: int) -> int:
    """Largest number of samples one H100 TriMul launch takes at pair width ``width`` (1 where the width has no batched path); a call
    with more samples runs in several launches (``_plan``)."""
    return MAX_BATCH_BY_WIDTH.get(int(width), 1)


#: Chunking. A call with more samples than one launch takes (``max_batch``), or whose working set would not fit, is split into
#: launches of at most ``_plan(...)[0]`` samples run one after another (results concatenated): it never runs out of memory because of
#: its batch and never falls back to the PyTorch statements because of size. The chunk is a deterministic function of the call's
#: shapes, width, direction and mode and of the device's TOTAL memory (never of live free memory), so torch.compile and CUDA-graph
#: capture see the same split every time. Working set per pair token = coefficient x D bytes, the coefficients being the measured
#: peaks of one batched call (inputs, outputs, saved tensors and workspace; H100 80GB, 2026-10-19) plus 5 %, which must fit
#: CHUNK_MEMORY_FRACTION of the device. Training that does not fit in one launch is chunked in one of two ways: when the whole batch's
#: saved tensors (held after the forward) fit next to one chunk's backward workspace, the chunks run as plain calls; otherwise each
#: chunk is checkpointed (its forward recomputed in its backward, so only the chunk being differentiated holds saved tensors).
CHUNK_MEMORY_FRACTION = 0.85
_WORKING_SET_PER_CHANNEL = {        # (training, bidirectional, D <= 128) -> bytes per pair token per channel, one launch's peak
    (False, True, True): 18.1, (False, True, False): 18.1,
    (False, False, True): 12.1, (False, False, False): 12.1,
    (True, True, True): 40.3, (True, True, False): 62.2,
    (True, False, True): 32.0, (True, False, False): 42.2,
}
#: Training: bytes per pair token per channel one launch holds from its forward to its backward (saved tensors and y; measured,
#: x / dy excluded), (bidirectional, D <= 128).
_HELD_PER_CHANNEL = {(True, True): 16.1, (True, False): 20.1, (False, True): 10.0, (False, False): 14.0}
#: The caller's own tensors while a chunked training call runs: x, y, dy, dx (2 bytes per channel each). ``_Chunked`` allocates y
#: and dx once and the chunks write into them, so nothing else scales with the batch.
_CALLER_PER_CHANNEL = 8.0
#: Training cells (D, L, bidirectional) -> largest B for which per-sample launches of the preserved single-sample kernels
#: (``_legacy_native_faster``), writing into the batch's y / dx, measured faster than the port (one launch or its chunks): CUDA
#: graph, B 4 / 16 / 48, H100 80GB, 2026-10-19 (docs/gpus/h100/trimul "Batched samples"). Above it, or at a cell not listed, the port.
PER_SAMPLE_LEGACY_TRAINING = {
    (256, 384, True): 48, (256, 384, False): 0, (256, 768, True): 4, (256, 768, False): 4,
    (384, 384, True): 4, (384, 384, False): 4, (384, 768, True): 16, (384, 768, False): 4,
    (512, 384, True): 16, (512, 384, False): 16, (512, 768, True): 16, (512, 768, False): 16,
}
#: Bidirectional hidden 2 D at D64 (``h100_d64_h2``), one sample per launch, bytes per pair token per channel (D = 64), from the planes
#: it allocates (K1 planes 1024 B + contraction 512 B + statistics per token held; backward adds dt / dl / dr / dp / dgl / dxn):
#: held from forward to backward, and the forward + backward peak (x / dy excluded).
_H2_HELD_PER_CHANNEL = 26.0
_H2_PEAK_PER_CHANNEL = 70.0
#: The preserved kernels at B = 1, bytes per pair token per channel (measured, x / dy excluded): held from forward to backward and
#: the forward + backward peak, (D, bidirectional).
_LEGACY_HELD_PER_CHANNEL = {(256, True): 24.1, (384, True): 40.1, (512, True): 40.1,
                            (256, False): 24.1, (384, False): 24.1, (512, False): 24.1}
_LEGACY_PEAK_PER_CHANNEL = {(256, True): 70.2, (384, True): 77.1, (512, True): 78.2,
                            (256, False): 44.2, (384, False): 44.2, (512, False): 44.2}


@functools.lru_cache(None)
def _total_memory(index: int) -> int:
    return torch.cuda.get_device_properties(index).total_memory


def _hidden_twice(module, pair) -> bool:
    """A single-direction module whose hidden width is twice the pair width, at D64 / D128 (AF3 / Protenix's template TriMul: pair
    64, hidden 128). Per token it has exactly the bidirectional block's shapes -- K1 writes 4 D planes, K3 normalises a 2 D-channel
    contraction output and projects it back to D, the same B1 / B2 / B7 backward -- only the cuBLAS contraction between the kernels
    pairs the planes differently (all 2 D channels in the module's one direction instead of D per direction). So the bidirectional
    D64 / D128 paths serve it with ``direction`` 1 / 2 (``h100_d64_b200gate``, ``h100_training``), and inference is the K1 / K3
    unit at hidden 2 D (the bidirectional one) with the one-direction contraction."""
    return hasattr(module, "outgoing") and pair.shape[-1] in (64, 128) and module.d_hidden == 2 * pair.shape[-1]


def _hidden_twice_bidirectional(module, pair) -> bool:
    """The bidirectional module with hidden width 2 D (per direction) at D64: 8 D planes out of K1, LN_out over 4 D = 256 channels.
    Inference: the packaged K1 / K3 unit (c_z 64, 256 channels, ``TILE_TABLE``); training: ``h100_d64_h2`` (the 256 channels run as
    two halves of the D64 backward, ``d64_h2_bwd.cu``)."""
    return not hasattr(module, "outgoing") and pair.shape[-1] == 64 and module.d_hidden == 128


def _hidden_ok(module, pair) -> bool:
    """The hidden widths the D64 / D128 native paths take: the pair width, or twice it in one direction (``_hidden_twice``); at D64 the
    bidirectional module also takes hidden 2 D (``_hidden_twice_bidirectional``)."""
    return (module.d_hidden == pair.shape[-1] or _hidden_twice(module, pair)
            or _hidden_twice_bidirectional(module, pair))


def _plan(module, pair, inference):
    """(chunk, checkpoint): the samples one launch takes and whether training chunks are checkpointed. chunk == B: one call."""
    bsz = pair.shape[0]
    if pair.ndim != 4 or not pair.is_cuda or bsz <= 1:
        return max(bsz, 1), False
    n, d = pair.shape[1], pair.shape[-1]
    granule = TRAINING_GRANULE if d >= 256 and inference else 16
    n = -(-n // granule) * granule                    # the padded length one launch runs (``_padded_inference``)
    bidirectional = not hasattr(module, "outgoing")
    planes4 = bidirectional or _hidden_twice(module, pair)  # hidden 2 D in one direction: the bidirectional working set
    tokens = d * n * n                                 # bytes per sample = coefficient x tokens
    peak = _WORKING_SET_PER_CHANNEL[(not inference, planes4, d <= 128)] * 1.05
    budget = CHUNK_MEMORY_FRACTION * _total_memory(pair.device.index if pair.device.index is not None else 0)
    if _hidden_twice_bidirectional(module, pair):
        if not inference:
            # one sample per launch (``h100_d64_h2``): plain when every sample's saved state fits next to one sample's backward
            # workspace and the caller's x / y / dy / dx, checkpointed otherwise
            held = _H2_HELD_PER_CHANNEL * 1.05
            plain = (bsz * (_CALLER_PER_CHANNEL + held - 2.0) + (_H2_PEAK_PER_CHANNEL * 1.05 - held)) * tokens <= budget
            return 1, not plain
        peak *= 1.8                                    # 8 D planes out of K1 instead of 4 (an over-estimate: only the planes double)
    limit = max(1, min(max_batch(d), (2 ** 31 - 1) // (n * n)))
    if (not inference and d >= 256 and bsz <= PER_SAMPLE_LEGACY_TRAINING.get((d, pair.shape[1], bidirectional), 0)
            and _legacy_native_faster(module, pair, False)):
        # one sample per launch on the preserved kernels: plain when every sample's saved state fits next to one sample's
        # backward workspace and the caller's x / y / dy / dx, checkpointed otherwise
        held = _LEGACY_HELD_PER_CHANNEL[(d, bidirectional)] * 1.05
        peak_legacy = _LEGACY_PEAK_PER_CHANNEL[(d, bidirectional)] * 1.05
        plain = (bsz * (_CALLER_PER_CHANNEL + held - 2.0) + peak_legacy - held) * tokens <= budget
        return 1, not plain
    if bsz * peak * tokens <= budget:
        return min(bsz, limit), False
    if inference:
        # the input and the output stay resident (4 bytes per channel); chunks write into the output
        free = budget - bsz * 4.0 * tokens
        return max(1, min(limit, int(free // (peak * tokens)))), False
    caller = bsz * _CALLER_PER_CHANNEL * tokens
    saved = (_HELD_PER_CHANNEL[(planes4, d <= 128)] * 1.05 - 2.0) * tokens      # y is the caller's
    work = (peak - _CALLER_PER_CHANNEL) * tokens - saved                              # one chunk's backward workspace
    free = budget - caller - bsz * saved
    if free >= work:                                   # every chunk's saved state fits next to one chunk's backward workspace
        return max(1, min(limit, int(free // work))), False
    free = budget - caller
    return max(1, min(limit, int(free // ((peak - _CALLER_PER_CHANNEL) * tokens)))), True


def _split(bsz, chunk):
    """Balanced chunk sizes (they differ by at most one sample)."""
    count = -(-bsz // chunk)
    return [bsz // count + (i < bsz % count) for i in range(count)]


def _one_chunk(module, pair, inference):
    """The first chunk's shape (a view): what ``serves*`` check when the call is split."""
    chunk, _ = _plan(module, pair, inference)
    return pair if chunk >= pair.shape[0] else pair[:chunk]



#: Training lengths: multiples of TRAINING_GRANULE (the B1 / B2 / B7 tiles of 64 tokens must not straddle a pair row and the
#: B200-style output tiles 128 tokens) from 128 up to MAX_TRAINING_LENGTH, B200's bound (80 x 128). The kernels index planes
#: through TMA coordinates or 64-bit offsets; their only index bound is the int32 token index (``_index_ok``). Memory binds
#: first on an 80 GB card: the training step needs ~4.3 kB per pair at D128 bidirectional (L2560: 28 GB, so up to ~L4300) and
#: ~2.1 kB at D64 (L2560: 13.5 GB per sample).
TRAINING_GRANULE = 128
MAX_TRAINING_LENGTH = 10240
#: Inference takes any L >= MIN_INFERENCE_LENGTH: one sample pads to 16 inside K1 / K3 (planes [c, Np, Np], zero pad rows),
#: B samples (whose K1 / K3 stack the samples without a pad) are padded to a multiple of 16 on the host when L is not one.
MIN_INFERENCE_LENGTH = 16


def _index_ok(pair) -> bool:
    """The int32 token index of K1 / K3 / B1 / B2 (B L^2 < 2^31, with L rounded up to the 16-token pad)."""
    n = -(-pair.shape[1] // 16) * 16
    return pair.shape[0] * n * n < 2 ** 31


def _training_length(n) -> bool:
    return n % TRAINING_GRANULE == 0 and TRAINING_GRANULE <= n <= MAX_TRAINING_LENGTH


def _batch_ok(pair, inference=False) -> bool:
    """The native D64 / D128 paths: one sample at every width; B samples (B <= ``max_batch``) at D64 and D128, whose K1 / K3 / B1 /
    B2 / B7 take b-major tokens (rows = B L of the slow token axis, a per-sample token mask and row-dropout scale) and whose
    contraction batches over channels x B. D64 training also needs the native D64 preparation (the per-sample pair mask is built
    there). Inference at L % 16 != 0 pads on the host (``update_inference``). D256+ batches only on the B200-style port
    (``_b200style_serves``)."""
    if pair.ndim != 4 or not _index_ok(pair):
        return False
    bsz, width = pair.shape[0], pair.shape[-1]
    if bsz == 1:
        return True
    if width not in (64, 128) or not 1 <= bsz <= max_batch(width):
        return False
    return inference or (pair.shape[1] % 16 == 0 and (width != 64 or _native_training_enabled()))


#: Cells where the preserved native path (the one served before the port) measured faster than the B200-style port on H100
#: (2026-10-13, bf16, FP32 masters, B1): training L384 / L768 at D256 / D384 / D512, and bidirectional inference at
#: (D384, L768) and (D512, L384). Every other length keeps the B200-style port, which is 1.2-2.1x faster than the
#: general path wherever the preserved path does not exist.
_LEGACY_TRAINING_WIDTHS = (256, 384, 512)
_LEGACY_BIDIR_INFERENCE_CELLS = ((384, 768), (512, 384))


def _legacy_native_faster(module, pair, inference):
    n, d = pair.shape[1], pair.shape[-1]
    if inference:
        if hasattr(module, "outgoing") and n > 1024:
            # single direction D256 / D384 beyond L1024: the packaged tmn K1 / K3 (any L) served these before the port did
            return d in (256, 384)
        if hasattr(module, "outgoing"):
            # single direction D256: the tmn K1 / K3 tiles beat the B200-style port by 0.7-2.5 % at every length but 384
            return d == 256 and n != 384
        return (d, n) in _LEGACY_BIDIR_INFERENCE_CELLS
    return n in (384, 768) and d in _LEGACY_TRAINING_WIDTHS


def _b200style_serves(module, pair, inference=False, padded=False):
    """The Hopper port of the B200 algorithm, except the cells where the preserved native path measured faster. B samples
    (B <= ``max_batch``) always take the port: the preserved paths run one sample. padded=True: B > 1 inference at an L the port
    does not take (L % 128 != 0), which ``update_inference`` pads to the next multiple of 128 (``_padded_inference``)."""
    if pair.ndim == 4 and pair.shape[0] == 1 and _legacy_native_faster(module, pair, inference):
        return False
    if padded and (not inference or pair.ndim != 4 or pair.shape[0] == 1 or pair.shape[1] % TRAINING_GRANULE == 0):
        return False
    length = -(-pair.shape[1] // TRAINING_GRANULE) * TRAINING_GRANULE if padded else pair.shape[1]
    if (not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4
            or not 1 <= pair.shape[0] <= max_batch(pair.shape[-1]) or pair.shape[0] * length * length >= 2 ** 31
            or pair.shape[1] != pair.shape[2]
            or pair.shape[-1] not in (256, 384, 512)
            or not _training_length(length)
            or not pair.is_contiguous() or module.d_hidden != pair.shape[-1]
            or module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5
            or torch.cuda.get_device_capability(pair.device) != (9, 0)):
        return False
    return all(p.dtype == torch.float32 and p.is_contiguous() and p.device == pair.device
               for p in (module.ln_pair.weight, module.ln_pair.bias,
                         module.ln_out.weight, module.ln_out.bias))


def _b200style_update(module, pair, mask, dropscale, bidirectional, out=None):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
        training_unit_scale,
    )
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_b200style import trimul
    n, d = pair.shape[1], pair.shape[-1]
    scale = ((training_unit_scale(pair) if torch.is_grad_enabled() else None)
             if dropscale is None else dropscale.reshape(pair.shape[0] * n, d).contiguous())
    token = torch.empty(0, device=pair.device, dtype=torch.bool) if mask is None else mask.contiguous()
    leaves = [pair, module.to_left.weight, module.to_left_gate.weight,
              module.to_right.weight, module.to_right_gate.weight,
              module.to_gate.weight, module.to_out.weight,
              module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias]
    direction = 0 if bidirectional else 1 if module.outgoing else 2
    if out is None:
        return trimul(leaves, token, scale, direction)
    return trimul(leaves, token, scale, direction, out=out)


def _native_buckets_enabled(pair=None):
    """True = explicit native buckets. Unset (None): D64 / D128 take them at every length 128..1024 (2026-10-13: 1.2-2.4x
    over the Triton fallback they otherwise get outside L384 / L768); D256+ keep the qualified cells only (the explicit
    buckets measured 2.6x slower there at L384 / L768)."""
    policy = settings.current().trimul_h100_native_buckets
    if policy is None and pair is not None:
        return pair.shape[-1] in (64, 128)
    return policy is True


def _selected_uni_d256(module, pair):
    """Qualified outgoing D256/D384 L384 and D512 L384,768 native defaults."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import uni_training
    return (hasattr(module, "outgoing") and uni_training(pair, bool(module.outgoing)))


def _selected_d64_training(module, pair):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import d64_training
    return d64_training(pair, getattr(module, "outgoing", False),
                        not hasattr(module, "outgoing"))


def _selected_wide_training(module, pair):
    """The qualified D256+ native cells (the D128 default stays on the framework RNG so the module compiles fullgraph)."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import bidi_training
    return _selected_uni_d256(module, pair) or (not hasattr(module, "outgoing") and bidi_training(pair))


def _selected_native_training(module, pair):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        bidi_training,
        d128_training,
    )
    return (_selected_uni_d256(module, pair)
            or (not hasattr(module, "outgoing") and bidi_training(pair))
            or d128_training(pair, getattr(module, "outgoing", False),
                             not hasattr(module, "outgoing")))

def serves(module, pair: torch.Tensor) -> bool:
    """Check the mathematical and resource contract before touching CUDA builders. B samples: the contract of one launch's chunk
    (``_plan``); any B is served by chunks."""
    return _serves_one(module, _one_chunk(module, pair, False))


def _serves_one(module, pair: torch.Tensor) -> bool:
    policy = settings.current()
    if (
        module.implementation != ImplementationType.MINIWORLD
        or policy.force_portable_kernels
    ):
        return False
    if _b200style_serves(module, pair):
        return pair.shape[-1] in policy.trimul_h100_training_widths
    if (
        pair.shape[-1] not in policy.trimul_h100_training_widths
        or pair.shape[-1] not in TRAINING_WIDTHS
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or not _batch_ok(pair)
        or pair.shape[1] != pair.shape[2]
        or not (_training_length(pair.shape[1])
                if _native_buckets_enabled(pair) or (pair.shape[-1] == 64 and _native_training_enabled())
                else pair.shape[1] in (384, 768))
        or (pair.shape[-1] > 128 and pair.shape[1] > 1024)   # D256+ beyond L1024: the B200-style port only (above)
    ):
        return False
    if (
        not _hidden_ok(module, pair)
        or module.ln_pair.eps != 1e-5
        or module.ln_out.eps != 1e-5
    ):
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    # Cooperative B1/B7 grids were qualified on full 132-SM H100s, not MIG slices.
    if (
        pair.shape[-1] == 128
        and torch.cuda.get_device_properties(pair.device).multi_processor_count != 132
    ):
        return False
    return all(
        p.dtype == torch.float32
        for p in (
            module.ln_pair.weight,
            module.ln_pair.bias,
            module.ln_out.weight,
            module.ln_out.bias,
        )
    )


def update(
    module,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    dropscale: torch.Tensor | None,
    *, bidirectional: bool = True,
) -> torch.Tensor:
    chunk, checkpoint = _plan(module, pair, False)
    if chunk >= pair.shape[0]:
        return _update_one(module, pair, mask, dropscale, bidirectional=bidirectional)
    return _update_chunked(module, pair, mask, dropscale, chunk, checkpoint, bidirectional)


def _update_chunked(module, pair, mask, dropscale, chunk, checkpoint, bidirectional):
    """B samples in launches of ``chunk`` (``_plan``), one after another (``_Chunked``)."""
    params = list(module.parameters())
    return _Chunked.apply((module, chunk, checkpoint, bidirectional), pair, mask, dropscale, *params)


def _chunk_one(module, x, mask, dropscale, bidirectional):
    """One chunk through the path ``update`` would take for it (no caller hooks: the chunks are differentiated by ``_Chunked``)."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        preserved_d64_context,
    )
    with preserved_d64_context(_selected_d64_training(module, x)):
        return _update(module, x, mask, dropscale, bidirectional=bidirectional)


class _Chunked(torch.autograd.Function):
    """A chunked training call as one autograd node. The batch's y and dx are allocated once; each chunk's launch writes its rows of
    them in place (``_h100_runtime.output_buffers``: the path's own y / dx allocation takes the rows), so nothing is copied or
    concatenated and the batch holds only x, y, dy and dx besides the chunk being run. Plain chunks keep their own autograd graph
    from forward to backward; checkpointed chunks run their forward without saving anything and again, with grad, in the backward.
    Parameter gradients are the chunks' gradients summed in chunk order (deterministic)."""

    @staticmethod
    def forward(ctx, setup, pair, mask, dropscale, *params):
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
            output_buffers,
        )
        module, chunk, checkpoint, bidirectional = setup
        ctx.setup, ctx.n_params = setup, len(params)
        sizes = _split(pair.shape[0], chunk)
        ctx.bounds = [(sum(sizes[:i]), sum(sizes[:i + 1])) for i in range(len(sizes))]
        y = torch.empty_like(pair)
        ctx.graphs = []
        for start, stop in ctx.bounds:
            x = pair[start:stop].detach()
            m = None if mask is None else mask[start:stop]
            d = None if dropscale is None else dropscale[start:stop]
            rows = y[start:stop]
            if checkpoint:
                with torch.no_grad(), output_buffers(y=rows):
                    out = _chunk_one(module, x, m, d, bidirectional)
                ctx.graphs.append(None)
            else:
                x.requires_grad_()
                with torch.enable_grad(), output_buffers(y=rows):
                    out = _chunk_one(module, x, m, d, bidirectional)
                ctx.graphs.append((x, out))
            if out.data_ptr() != rows.data_ptr():
                rows.copy_(out)       # a path that allocated its own y (traced, or not wired to the scope)
        ctx.save_for_backward(pair, mask, dropscale)
        return y

    @staticmethod
    def backward(ctx, dy):
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
            output_buffers,
        )
        module, _, checkpoint, bidirectional = ctx.setup
        pair, mask, dropscale = ctx.saved_tensors
        params = list(module.parameters())
        dx = torch.empty_like(pair)
        total = [None] * len(params)
        for i, (start, stop) in enumerate(ctx.bounds):
            if checkpoint:
                x = pair[start:stop].detach().requires_grad_()
                m = None if mask is None else mask[start:stop]
                d = None if dropscale is None else dropscale[start:stop]
                with torch.enable_grad():
                    out = _chunk_one(module, x, m, d, bidirectional)
            else:
                x, out = ctx.graphs[i]
                ctx.graphs[i] = None
            rows = dx[start:stop]
            with output_buffers(dx=rows):
                grads = torch.autograd.grad(out, [x, *params], dy[start:stop].contiguous(), allow_unused=True)
            del out
            if grads[0].data_ptr() != rows.data_ptr():
                rows.copy_(grads[0])
            for k, g in enumerate(grads[1:]):
                if g is not None:
                    total[k] = g if total[k] is None else total[k].add_(g)
        return (None, dx, None, None, *total)


def _update_one(module, pair, mask, dropscale, *, bidirectional=True):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        preserved_d64_context,
    )
    with preserved_d64_context(_selected_d64_training(module, pair)):
        out = _update(module, pair, mask, dropscale, bidirectional=bidirectional)
    if _b200style_serves(module, pair) or _native_buckets_enabled(pair) or _selected_native_training(module, pair) or _selected_d64_training(module, pair) or (pair.shape[-1] == 64 and _native_training_enabled()):
        from miniworld_engine.integrations.token_dit_train import _native_caller_output
        out = _native_caller_output(out, pair, *module.parameters(), force=True)
    return out


def _update(module, pair, mask, dropscale, *, bidirectional=True):
    """Use native FP32 master gradients where qualified; preserve cast-based fallbacks."""
    if _b200style_serves(module, pair):
        return _b200style_update(module, pair, mask, dropscale, bidirectional)
    if _hidden_twice_bidirectional(module, pair):
        return _update_hidden2(module, pair, mask, dropscale)
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_training import (
        bidirectional_trimul,
    )

    n, d = pair.shape[1], pair.shape[-1]
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        preserve_training,
    )
    # hidden 2 D in one direction: the bidirectional D64 / D128 kernels (``_hidden_twice``), whatever the bucket policy
    planes4 = bidirectional or _hidden_twice(module, pair)
    preserve = preserve_training(d) or _selected_native_training(module, pair)
    # (B samples take the preserved D128 kernels below: the general buckets run one sample)
    if (_native_buckets_enabled(pair) and d != 64 and not preserve and pair.shape[0] == 1
            and (bidirectional or not planes4)):
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
            training_unit_scale,
        )
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_training import (
            native_buckets,
            single_trimul,
        )
        empty_mask = torch.empty(0,device=pair.device,dtype=torch.bool) if mask is None else mask
        scale = training_unit_scale(pair) if dropscale is None else dropscale.reshape(n,d)
        args = (pair, module.to_left.weight, module.to_left_gate.weight,
                module.to_right.weight, module.to_right_gate.weight,
                module.to_gate.weight, module.to_out.weight,
                module.ln_pair.weight, module.ln_pair.bias,
                module.ln_out.weight, module.ln_out.bias, empty_mask, scale)
        if not bidirectional:
            return single_trimul(module.outgoing, *args)
        return native_buckets(True, True, *args)
    # Bidirectional D128 (and hidden 2 D in one direction, its kernels) takes the token mask [n]: its front K1 and B7 form m[i] & m[j]
    # themselves. The other paths read a bf16 pair mask in forward and backward, so it is built once here.
    token_path = planes4 and d == 128 and (mask is None or mask.ndim == 2)
    if preserve and not token_path:
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
            owned_training_mask,
        )
        with torch.no_grad():
            pair_mask = owned_training_mask(pair,mask)
    elif d == 64 and _native_training_enabled():
        # Keep token masks compact until the saved forward boundary constructs
        # its owned FP32 pair mask. No intermediate BF16 square allocation.
        pair_mask = (torch.empty(0, device=pair.device, dtype=torch.bool)
                     if mask is None else mask)
    elif mask is None:
        rows = pair.shape[0] * n   # batched samples: a token mask [B n] / a pair mask [B n, n]
        pair_mask = (torch.ones(rows, device=pair.device, dtype=torch.bool) if token_path
                     else torch.ones((rows, n), device=pair.device, dtype=torch.bfloat16))
    elif token_path:
        pair_mask = mask.reshape(-1)   # [B n], b-major
        if pair_mask.dtype != torch.bool:
            pair_mask = pair_mask != 0
    else:
        pair_mask = (
            (mask.unsqueeze(-1) & mask.unsqueeze(-2)) if mask.ndim == 2 else mask
        )
        pair_mask = pair_mask.to(torch.bfloat16).contiguous()
    if dropscale is None and (preserve or (d == 64 and _native_training_enabled())):
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
            training_unit_scale,
        )
        scale = training_unit_scale(pair)
    else:
        scale = (pair.new_ones((pair.shape[0] * n, d)) if dropscale is None
                 else dropscale.reshape(pair.shape[0] * n, d).contiguous())
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    if planes4:
        if d == 64:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_d64_b200gate import (
                bidirectional_trimul as apply,
            )
        else:
            apply = bidirectional_trimul
        if not bidirectional:   # hidden 2 D in one direction: the contraction pairs all 2 D channels in it
            apply = functools.partial(apply, direction=1 if module.outgoing else 2)
    elif d in (64, 256, 384, 512):
        if d == 64:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_d64_b200gate import (
                single_trimul,
            )
        else:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_training import (
                single_trimul,
            )
        def apply(*args):
            return single_trimul(module.outgoing, *args)
    else:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_single import (
            single_trimul,
        )
        def apply(*args):
            return single_trimul(module.outgoing, *args)
    return apply(
        pair,
        *(w if w.dtype in (torch.bfloat16, torch.float32) else w.to(pair.dtype).contiguous()
          for w in weights),
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
        pair_mask,
        scale,
    )


def _update_hidden2(module, pair, mask, dropscale):
    """Bidirectional hidden 2 D at D64 (``_hidden_twice_bidirectional``): the dedicated hidden-256 training path."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import (
        training_unit_scale,
    )
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_d64_h2 import (
        bidirectional_trimul,
    )

    rows = pair.shape[0] * pair.shape[1]
    token = torch.empty(0, device=pair.device, dtype=torch.bool) if mask is None else mask.contiguous()
    scale = training_unit_scale(pair) if dropscale is None else dropscale.reshape(rows, pair.shape[-1]).contiguous()
    weights = (module.to_left.weight, module.to_left_gate.weight, module.to_right.weight, module.to_right_gate.weight,
               module.to_gate.weight, module.to_out.weight)
    return bidirectional_trimul(
        pair,
        *(w if w.dtype in (torch.bfloat16, torch.float32) else w.to(pair.dtype).contiguous() for w in weights),
        module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias, token, scale)


def _native_training_enabled():
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import (
        cuda_training_aux_enabled,
    )
    return cuda_training_aux_enabled()


def native_training_dropout(module, pair, probability):
    """Share the hand-CUDA Philox primitive; preserve PyTorch RNG progression. (B samples: the selection of one launch's chunk.)"""
    one = _one_chunk(module, pair, False)
    if not (_b200style_serves(module, one) or settings.current().trimul_h100_native_buckets is True or _selected_wide_training(module, one) or _selected_d64_training(module, one) or (one.shape[-1] == 64 and _native_training_enabled())) or not _serves_one(module, one):
        return None
    from miniworld_engine.kernels.triangle_attention.cuda.h100_triattn import extension
    return extension("pack").drop_scale(pair, probability).reshape(
        pair.shape[0], 1, pair.shape[1], pair.shape[-1])


def _native_residual_bucket(module, pair):
    """Measured outgoing D384/L512 inference bucket (contract checked by serves)."""
    return (pair.ndim == 4 and pair.shape[1] == 512 and pair.shape[-1] == 384
            and module.d_hidden == 384 and getattr(module, "outgoing", False))


def serves_inference(
    module, pair: torch.Tensor, *, bidirectional: bool, dropscale=None
) -> bool:
    """The inference contract of one launch's chunk (``_plan``); any B is served by chunks."""
    return _serves_inference_one(module, _one_chunk(module, pair, True), bidirectional=bidirectional, dropscale=dropscale)


def _serves_inference_one(module, pair, *, bidirectional, dropscale=None):
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().force_portable_kernels
    ):
        return False
    if torch.is_grad_enabled() or dropscale is not None:
        return False
    if _b200style_serves(module, pair, inference=True) or _b200style_serves(module, pair, inference=True, padded=True):
        return True
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or not _batch_ok(pair, inference=True)
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] <= 0
    ):
        return False
    if module.ln_pair.eps != module.ln_out.eps:
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    if ((pair.shape[-1] in (64, 128) or (not bidirectional and _native_residual_bucket(module, pair)))
            and (pair.shape[1] < MIN_INFERENCE_LENGTH
                 or not _hidden_ok(module, pair) or module.ln_pair.eps != 1e-5
                 or any(p.dtype != torch.float32 for p in (
                     module.ln_pair.weight, module.ln_pair.bias,
                     module.ln_out.weight, module.ln_out.bias)))):
        return False
    if bidirectional and _wide_inference_ok(module, pair):
        return True
    if not bidirectional and _uni_wide_inference_ok(module, pair):
        return True
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as table

    hidden = module.d_hidden * (2 if bidirectional else 1)
    entry = table.TILE_TABLE.get(("sm_90a", pair.shape[-1], hidden, "b"))
    return bool(entry and entry.get("k1") and entry.get("k3"))


def update_inference(module, pair, mask, *, bidirectional):
    chunk, _ = _plan(module, pair, True)
    if chunk >= pair.shape[0]:
        return _update_inference_one(module, pair, mask, bidirectional=bidirectional)
    # B samples in launches of ``chunk``, one after another, each written into the output as it finishes (no list of chunk outputs)
    out = torch.empty_like(pair)
    start = 0
    for size in _split(pair.shape[0], chunk):
        stop = start + size
        _update_inference_one(module, pair[start:stop], None if mask is None else mask[start:stop], bidirectional=bidirectional,
                              out=out[start:stop])
        start = stop
    return out


def _update_inference_one(module, pair, mask, *, bidirectional, out=None):
    """One launch. out (chunked batches): the caller's rows of the output; the port and the packaged K1 / K3 write them in place,
    the other paths are copied in."""
    y = _update_inference_body(module, pair, mask, bidirectional, out)
    if out is not None and y is not out:
        out.copy_(y)
        return out
    return y


def _update_inference_body(module, pair, mask, bidirectional, out):
    if _b200style_serves(module, pair, inference=True):
        if out is None:   # (the one-launch call keeps the five-argument form tests patch)
            return _b200style_update(module, pair, mask, None, bidirectional)
        return _b200style_update(module, pair, mask, None, bidirectional, out=out)
    if _b200style_serves(module, pair, inference=True, padded=True):
        return _padded_inference(module, pair, mask, bidirectional, TRAINING_GRANULE)
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import inference

    bsz, n = pair.shape[0], pair.shape[1]
    if bsz > 1 and n % 16:
        return _padded_inference(module, pair, mask, bidirectional)
    native_residual = not bidirectional and _native_residual_bucket(module, pair)
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as runtime
    native_aux = runtime.cuda_aux_enabled()
    if (native_residual or native_aux) and mask is not None and (
        mask.device != pair.device or mask.dtype != torch.bool
        or tuple(mask.shape) != (bsz, n) or not mask.is_contiguous()
    ):
        raise ValueError("H100 native TriMul inference requires a contiguous boolean [B,L] mask")
    # K1 forms the pair mask m[i] & m[j] from the token mask itself: no [n, n] tensor per call (B samples: [B n], b-major).
    token = None if mask is None else mask.reshape(bsz * n)
    if token is not None and token.dtype != torch.bool:
        token = token != 0
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    # K1 packs the first four matrices with live stride-aware CUDA operations.
    # Preserve native column-major parameters instead of copying them first.
    if pair.shape[-1] in (64, 128) or native_residual or native_aux:
        from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import cast_weights
        matrices = cast_weights(weights)
    else:
        matrices = [w.to(pair.dtype) if i < 4 else w.to(pair.dtype).contiguous()
                    for i, w in enumerate(weights)]
    weights = [
        *matrices,
        module.ln_pair.weight.float().contiguous(),
        module.ln_pair.bias.float().contiguous(),
        module.ln_out.weight.float().contiguous(),
        module.ln_out.bias.float().contiguous(),
    ]
    if bidirectional and _wide_inference_ok(module, pair):
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_wide_inference import (
            wide_inference,
        )
        return wide_inference(pair, weights, token)
    if not bidirectional and _uni_wide_inference_ok(module, pair):
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_inference import (
            uni_wide_inference,
        )

        if native_aux:
            from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import pair_mask
            pm = pair_mask(token, n, pair.device)
        else:
            pm = (pair.new_ones((n, n)) if mask is None
                  else (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float().contiguous())
        return uni_wide_inference(pair, weights, pm, module.outgoing)
    direction = 0 if bidirectional else 1 if module.outgoing else 2
    if out is not None:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import (
            inference_into,
        )
        inference_into(pair, weights, token, direction, module.ln_pair.eps, out)
        return out
    return inference(pair, weights, token, direction, module.ln_pair.eps)


def _padded_inference(module, pair, mask, bidirectional, granule=16):
    """B samples at L % 16 != 0. K1 / K3 stack the B samples along the slow token axis without a pad (one sample pads inside
    them), so each sample is padded to the next multiple of 16 here: zero pair entries on masked pad tokens. A pad token's
    left / right rows and columns are zero (K1's pair mask m[i] & m[j]), so every real pair (i, j) contracts the same terms as
    without the pad; the pad rows of the output are dropped. granule=128: B samples at D256+ (the B200-style port's 128-token
    tiles), same argument."""
    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    npad = -(-n // granule) * granule
    x = pair.new_zeros((bsz, npad, npad, d))
    x[:, :n, :n] = pair
    token = torch.zeros((bsz, npad), device=pair.device, dtype=torch.bool)
    if mask is None:
        token[:, :n] = True
    else:
        token[:, :n] = mask.reshape(bsz, n) != 0
    return _update_inference_one(module, x, token, bidirectional=bidirectional)[:, :n, :n].contiguous()


def _wide_inference_ok(module, pair: torch.Tensor) -> bool:
    """Bidirectional D256/384/512 at L384/768: fused wide K1/K3 (hidden = D per direction)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_wide_inference import supports

    return (
        supports(pair.shape[-1], pair.shape[1])
        and module.d_hidden == pair.shape[-1]
        and module.ln_pair.eps == 1e-5
        and module.ln_out.eps == 1e-5
    )


def serves_single(module, pair: torch.Tensor) -> bool:
    """Qualified single-direction native training contract; inference is separate. B samples: one launch's chunk (``_plan``)."""
    one = _one_chunk(module, pair, False)
    return (one.shape[-1] in SINGLE_TRAINING_WIDTHS
            and (one.shape[-1] != 512 or _native_buckets_enabled(one)
                 or _selected_native_training(module, one) or _b200style_serves(module, one)) and _serves_one(module, one))


def _uni_wide_inference_ok(module, pair: torch.Tensor) -> bool:
    """Single-direction D512 at L384/768: LN + K1 + contraction + gate GEMM + folded K3."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_uni_wide_inference import (
        serves,
    )

    return serves(module, pair)
