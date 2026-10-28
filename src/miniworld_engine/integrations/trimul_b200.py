"""Automatic dispatch to the B200 (sm_100a) TriMul: inference (``b200_infer``) and training (``b200_train``), D64-D512,
both modules; the hidden width equals the pair width, or is twice it in one direction at D64 / D128 (``b200_infer.hidden_ok``:
AF3 / Protenix template blocks, pair 64, hidden 128).

Batched samples at every width: x [B, L, L, D] runs as one launch per stage (tokens b-major, M = B L L; the token mask and the
row-dropout scale are read per sample, the contraction batches over channels x B), up to ``max_batch(D)`` samples per launch.
A call with more samples, or with more than fit in memory, runs in several launches (``_plan``): any B is served."""

from __future__ import annotations

import functools

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.modules.exceptions import ImplementationType

#: Samples ONE LAUNCH takes at each pair width (MiniWorld folds templates, structures or augments into the batch of one pair stack;
#: it asks ``max_batch(width)``). ``MAX_BATCH`` keeps its D64 meaning (= ``max_batch(64)``, the same name as on H100). The bound is
#: what was validated (B up to 48 against B one-sample calls, bitwise y / dx); the kernels' only index bound is the int32 token
#: index B L^2 < 2^31 per launch (``_plan`` applies it). A call with more samples runs in several launches (``_plan``).
MAX_BATCH_BY_WIDTH = {64: 48, 128: 48, 256: 48, 384: 48, 512: 48}
MAX_BATCH = MAX_BATCH_BY_WIDTH[64]


def max_batch(width: int) -> int:
    """Largest number of samples one B200 TriMul launch takes at pair width ``width`` (1 where the width has no batched path); a call
    with more samples runs in several launches (``_plan``)."""
    return MAX_BATCH_BY_WIDTH.get(int(width), 1)


#: Chunking. A call with more samples than one launch takes (``max_batch``), or whose working set would not fit, is split into
#: launches of at most ``_plan(...)[0]`` samples run one after another: it never runs out of memory because of its batch and never
#: falls back to the PyTorch statements because of size. The chunk is a deterministic function of the call's shapes, width, plane
#: count and mode and of the device's TOTAL memory (never of live free memory), so torch.compile and CUDA-graph capture see the same
#: split every time. One launch's working set per pair token = coefficient x D bytes: the measured peak of one batched call (inputs,
#: outputs, saved tensors and workspace; B200, 2026-10-20) plus 5 %, which must fit CHUNK_MEMORY_FRACTION of the device. Training
#: that does not fit one launch is chunked in one of two ways: plain chunks when every chunk's saved tensors (held from its forward
#: to the backward) fit next to one chunk's backward workspace and the caller's x / y / dy / dx, checkpointed chunks otherwise (each
#: chunk's forward runs again in the backward). Chunks write their rows of y (and dx) in place.
CHUNK_MEMORY_FRACTION = 0.85
#: (training, 4 D planes, D <= 128) -> bytes per pair token per channel D, one launch's peak. "4 D planes": bidirectional, or one
#: direction at hidden 2 D (the same per-token kernels).
_WORKING_SET_PER_CHANNEL = {
    (False, True, True): 18.1, (False, True, False): 16.1,
    (False, False, True): 10.1, (False, False, False): 10.1,
    (True, True, True): 36.2, (True, True, False): 53.3,
    (True, False, True): 21.7, (True, False, False): 33.3,
}
#: Training: bytes per pair token per channel one launch holds from its forward to its backward (saved tensors and y; x / dy
#: excluded), (4 D planes, D <= 128). Measured with the coefficients above (B200, L128 B16 / L384 B4 / L768 B2-4, max).
_HELD_PER_CHANNEL = {(True, True): 16.2, (True, False): 17.2, (False, True): 9.3, (False, False): 11.2}
#: The caller's own tensors while a chunked training call runs: x, y, dy, dx (2 bytes per channel each).
_CALLER_PER_CHANNEL = 8.0
#: Pair tokens (B L^2) one launch takes at most, for speed: the channel-major kernels slow down as their channel stride -- the
#: launch's token count -- grows (k1w's plane stores: D64 0.13 ns per pair token at 0.59 M tokens, 0.24 ns at 2.4 M, also at one
#: sample of L1536; k3w's loads of t: D512 1.5 -> 2.6 ns), so past this size a batch runs faster as several launches. Measured on
#: B200 (2026-10-20, CUDA graph, launch size forced over B 4-48 x L128-768): inference is fastest at 0.44-0.59 M tokens per launch
#: at every width (13-21 % over one launch at L256-768, within 2 % at L128); training at D128+ gains 2-5 % at L256-768 with up to
#: 0.79 M (48 samples of L128); D64 training is fastest in one launch (key None: no cap).
_LAUNCH_TOKENS = {(False, False): 589824, (False, True): 589824, (True, False): 786432, (True, True): None}   # (training, D64)


@functools.lru_cache(None)
def _total_memory(index: int) -> int:
    return torch.cuda.get_device_properties(index).total_memory


def _four_planes(module, bidirectional: bool) -> bool:
    return bidirectional or module.d_hidden == 2 * module.ln_pair.weight.shape[0]


def _plan(module, pair: torch.Tensor, inference: bool, bidirectional: bool) -> tuple[int, bool]:
    """(chunk, checkpoint): the samples one launch takes (at most ``max_batch``, at most ``_LAUNCH_TOKENS`` pair tokens, and what fits
    in memory) and whether training chunks are checkpointed. chunk >= B: one launch."""
    bsz = pair.shape[0]
    if pair.ndim != 4 or bsz <= 1:
        return max(bsz, 1), False
    n, d = pair.shape[1], pair.shape[-1]
    tokens = d * n * n                                 # bytes per sample = coefficient x tokens
    four = _four_planes(module, bidirectional)
    peak = _WORKING_SET_PER_CHANNEL[(not inference, four, d <= 128)] * 1.05
    index = pair.device.index if pair.device.index is not None else 0
    budget = CHUNK_MEMORY_FRACTION * (_total_memory(index) if pair.is_cuda else 0)
    limit = max(1, min(max_batch(d), (2 ** 31 - 1) // (n * n)))
    cap = _LAUNCH_TOKENS[(not inference, d == 64)]
    if cap is not None:
        limit = min(limit, max(1, cap // (n * n)))
    if bsz * peak * tokens <= budget:
        return min(bsz, limit), False
    if inference:
        # the input and the output stay resident (4 bytes per channel); chunks write into the output
        free = budget - bsz * 4.0 * tokens
        return max(1, min(limit, int(free // (peak * tokens)))), False
    caller = bsz * _CALLER_PER_CHANNEL * tokens
    saved = (_HELD_PER_CHANNEL[(four, d <= 128)] * 1.05 - 2.0) * tokens               # y is the caller's
    work = (peak - _CALLER_PER_CHANNEL) * tokens - saved                              # one chunk's backward workspace
    free = budget - caller - bsz * saved
    if free >= work:                                   # every chunk's saved state fits next to one chunk's backward workspace
        return max(1, min(limit, int(free // work))), False
    free = budget - caller
    return max(1, min(limit, int(free // ((peak - _CALLER_PER_CHANNEL) * tokens)))), True


def _split(bsz: int, chunk: int) -> list[int]:
    """Balanced chunk sizes (they differ by at most one sample)."""
    count = -(-bsz // chunk)
    return [bsz // count + (i < bsz % count) for i in range(count)]


def _one_chunk(module, pair, inference, bidirectional):
    """The first launch's shape (a view): what ``serves*`` check when the call is split."""
    chunk, _ = _plan(module, pair, inference, bidirectional)
    return pair if chunk >= pair.shape[0] else pair[:chunk]


def _common_ok(module, pair: torch.Tensor) -> bool:
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().force_portable_kernels
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if pair.ndim != 4 or pair.shape[1] != pair.shape[2] or pair.shape[0] < 1:
        return False
    return module.ln_pair.eps == 1e-5 and module.ln_out.eps == 1e-5


#: Inference pads L to this granule on the host (``_padded_inference``): the kernels take L % 16 == 0.
_GRANULE = 16


def _padded_length(n: int) -> int:
    return -(-n // _GRANULE) * _GRANULE


def serves_inference(module, pair: torch.Tensor, *, bidirectional: bool, dropscale: torch.Tensor | None = None) -> bool:
    """Inference (no autograd) at every width D64-D512, either direction, any B (``b200_infer``; B > ``max_batch`` or more than fit
    in memory: several launches), any L >= 1 (L % 16 != 0: padded on the host, ``_padded_inference``; not with a row-dropout scale)."""
    if torch.is_grad_enabled() or not _common_ok(module, pair):
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import supports

    npad = _padded_length(pair.shape[1])
    if npad != pair.shape[1] and dropscale is not None:
        return False
    one = _one_chunk(module, pair, True, bidirectional)
    if one.shape[0] * npad * npad >= 2 ** 31:
        return False
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if not supports(pair.shape[-1], npad, dropscale is not None, module.d_hidden, direction):
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def _padded_inference(module, pair: torch.Tensor, mask: torch.Tensor | None, *, bidirectional: bool) -> torch.Tensor:
    """L % 16 != 0 (H100's 62acfe721, the same argument): each sample is padded to the next multiple of 16 with zero pair entries on
    masked pad tokens. k1w forms the pair mask m[i] & m[j] from the token mask, so a pad token's left / right rows and columns are zero
    and every real pair (i, j) contracts the same terms as without the pad; the pad rows of the output are dropped."""
    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    npad = _padded_length(n)
    x = pair.new_zeros((bsz, npad, npad, d))
    x[:, :n, :n] = pair
    token = torch.zeros((bsz, npad), device=pair.device, dtype=torch.bool)
    if mask is None:
        token[:, :n] = True
    else:
        token[:, :n] = mask.reshape(bsz, n) != 0
    return update_inference(module, x, token, None, bidirectional=bidirectional)[:, :n, :n].contiguous()


def _inference_params(module, bf):
    params = (module.to_left.weight, module.to_left_gate.weight, module.to_right.weight, module.to_right_gate.weight,
              module.to_gate.weight, module.to_out.weight, module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight,
              module.ln_out.bias)

    def forms():
        w = params
        return (w[0].to(bf), w[1].to(bf), w[2].to(bf), w[3].to(bf), w[4].to(bf).contiguous(), w[5].to(bf).contiguous(),
                w[6].float().contiguous(), w[7].float().contiguous(), w[8].float().contiguous(), w[9].float().contiguous())

    if torch.compiler.is_compiling():
        # traced: the casts are graph ops (the cache's capture-id query is a graph break, and a compiled graph needs no cache)
        return list(forms())
    return list(_capture.param_forms(f"trimul_b200_{bf}", params, forms))


def update_inference(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None,
                     *, bidirectional: bool) -> torch.Tensor:
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import inference, inference_into

    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    if n % _GRANULE:
        return _padded_inference(module, pair, mask, bidirectional=bidirectional)
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
    params = _inference_params(module, pair.dtype)
    scale = None if dropscale is None else dropscale.reshape(bsz * n, d).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    chunk, _ = _plan(module, pair, True, bidirectional)
    if chunk >= bsz:
        return inference([pair, *params], token_mask, scale, direction)
    # B samples in launches of ``chunk``, one after another, each writing its rows of the output in place
    out = torch.empty_like(pair)
    start = 0
    for size in _split(bsz, chunk):
        stop = start + size
        inference_into([pair[start:stop], *params], None if token_mask is None else token_mask[start * n:stop * n],
                       None if scale is None else scale[start * n:stop * n], direction, out[start:stop])
        start = stop
    return out


def serves_train(module, pair: torch.Tensor, *, bidirectional: bool) -> bool:
    """Training (autograd, ``b200_train``) at every width D64-D512, either direction, any B (B > ``max_batch`` or more than fit in
    memory: chunks of one autograd node)."""
    if not torch.is_grad_enabled() or not _common_ok(module, pair):
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import supports

    one = _one_chunk(module, pair, False, bidirectional)
    if one.shape[0] * one.shape[1] * one.shape[1] >= 2 ** 31:
        return False
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if not supports(pair.shape[-1], pair.shape[1], direction, module.d_hidden):
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def update_train(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None,
                 *, bidirectional: bool) -> torch.Tensor:
    """The module's parameters are the leaves (any float dtype; see ``b200_train.trimul_train``)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import trimul_train, trimul_train_chunked

    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
    bf = pair.dtype
    # the parameters themselves: the training op casts them for the kernels outside autograd, so an fp32 master gets fp32 gradients
    leaves = [pair, module.to_left.weight, module.to_left_gate.weight, module.to_right.weight, module.to_right_gate.weight,
              module.to_gate.weight, module.to_out.weight, module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight,
              module.ln_out.bias]
    scale = None if dropscale is None else dropscale.reshape(bsz * n, d).to(bf).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    chunk, checkpoint = _plan(module, pair, False, bidirectional)
    if chunk >= bsz:
        return trimul_train(leaves, token_mask, scale, direction)
    return trimul_train_chunked(leaves, token_mask, scale, direction, _split(bsz, chunk), checkpoint)
