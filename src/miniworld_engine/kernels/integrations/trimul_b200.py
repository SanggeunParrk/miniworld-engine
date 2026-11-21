"""Automatic dispatch to the B200 (sm_100a) TriMul: inference (``b200_infer``) and training (``b200_train``), D64-D512,
both modules; the hidden width equals the pair width, or is twice it in one direction at D64 / D128 (``b200_infer.hidden_ok``:
AF3 / Protenix template blocks, pair 64, hidden 128) or bidirectionally at D64 (8 D planes).

Batched samples at every width: x [B, L, L, D] runs as one launch per stage (tokens b-major, M = B L L; the token mask and the
row-dropout scale are read per sample, the contraction batches over channels x B), up to ``max_batch(D)`` samples per launch.
A call with more samples, or with more than fit in memory, runs in several launches (``_plan``): any B is served."""

from __future__ import annotations

import functools
import os

import torch

from miniworld_engine.kernels import _capture

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
    (False, True, True): 18.1,
    (False, True, False): 16.1,
    (False, False, True): 10.1,
    (False, False, False): 10.1,
    (True, True, True): 36.2,
    (True, True, False): 53.3,
    (True, False, True): 21.7,
    (True, False, False): 33.3,
}
#: The D64 bidirectional hidden-128 block (8 D planes, ``b200_train._eight_planes``): (training) -> the same two coefficients. Estimated
#: from its buffers (inference: x, y, 8 D planes, 4 D of t; training: k1wb's moment -- the saved planes, t, x_n, the plane gradient and
#: [dpre | dg] next to x, dy, dx), +15 %.
_WORKING_SET_EIGHT = {False: 34.0, True: 98.0}
_HELD_EIGHT = 30.0
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
_LAUNCH_TOKENS = {(False, False): 589824, (False, True): 589824, (True, False): 786432, (True, True): None}  # (training, D64)


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
    tokens = d * n * n  # bytes per sample = coefficient x tokens
    four = _four_planes(module, bidirectional)
    eight = bidirectional and module.d_hidden == 2 * d
    peak = (_WORKING_SET_EIGHT[not inference] if eight else _WORKING_SET_PER_CHANNEL[(not inference, four, d <= 128)]) * 1.05
    index = pair.device.index if pair.device.index is not None else 0
    budget = CHUNK_MEMORY_FRACTION * (_total_memory(index) if pair.is_cuda else 0)
    limit = max(1, min(max_batch(d), (2**31 - 1) // (n * n)))
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
    saved = ((_HELD_EIGHT if eight else _HELD_PER_CHANNEL[(four, d <= 128)]) * 1.05 - 2.0) * tokens  # y is the caller's
    work = (peak - _CALLER_PER_CHANNEL) * tokens - saved  # one chunk's backward workspace
    free = budget - caller - bsz * saved
    if free >= work:  # every chunk's saved state fits next to one chunk's backward workspace
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


def _common_refusal(module, pair: torch.Tensor) -> str | None:
    if not pair.is_cuda or torch.cuda.get_device_capability(pair.device) != (10, 0):
        return "the B200 TriMul kernels run on a capability (10, 0) device only"
    if pair.dtype != torch.bfloat16:
        return f"dtype {pair.dtype}: the B200 TriMul kernels are bf16 only (no fp32 TriMul, by policy)"
    if not pair.is_contiguous():
        return "pair must be contiguous"
    if pair.ndim != 4 or pair.shape[1] != pair.shape[2] or pair.shape[0] < 1:
        return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return "the B200 TriMul kernels take LayerNorm eps 1e-5 (LN_in and LN_out)"
    return None


#: Inference pads L to this granule on the host (``_padded_inference``): the kernels take L % 16 == 0.
_GRANULE = 16


def _padded_length(n: int) -> int:
    return -(-n // _GRANULE) * _GRANULE


def inference_refusal(module, pair: torch.Tensor, *, bidirectional: bool, dropout: bool = False) -> str | None:
    """Inference (no autograd) at every width D64-D512, either direction, any B (``b200_infer``; B > ``max_batch`` or more than fit
    in memory: several launches), any L >= 1 (L % 16 != 0: padded on the host, ``_padded_inference``; not with a row-dropout scale).
    ``None`` when served, else the reason."""
    why = _common_refusal(module, pair)
    if why is not None:
        return why
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import supports

    n, d = pair.shape[1], pair.shape[-1]
    npad = _padded_length(n)
    if npad != n and dropout:
        return f"L {n}: a row-dropout scale needs L % 16 == 0 (no host pad with dropout)"
    one = _one_chunk(module, pair, True, bidirectional)
    if one.shape[0] * npad * npad >= 2**31:
        return f"L {n}: one launch past the int32 token index (B L^2 >= 2**31)"
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if not supports(d, npad, dropout, module.d_hidden, direction):
        return (
            f"d_pair {d}, d_hidden {module.d_hidden}, L {n}, dropout {dropout}: outside the B200 inference kernels (b200_infer.supports: "
            "D 64 / 128 / 256 / 384 / 512, hidden = D or 2 D at D64 / D128 one direction or D64 bidirectional, a dropout scale at D <= 128 "
            "needs L % 128 == 0)"
        )
    return None


def _padded_inference(module, pair: torch.Tensor, mask: torch.Tensor | None, *, bidirectional: bool, rf3: bool = False) -> torch.Tensor:
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
    return update_inference(module, x, token, None, bidirectional=bidirectional, rf3=rf3)[:, :n, :n].contiguous()


def _inference_params(module, bf, rf3: tuple | None = None, *, rf3_inline: bool = False):
    """The inference forms; ``rf3`` = (token mask or None, L): RoseTTAFold3's length factor folded into W_l (D <= 128, per call)."""
    params = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
    )
    if bf == torch.bfloat16 and module.ln_pair.weight.shape[0] <= 128:
        # D64 / D128: the same cached forms as training (one trimul_forms launch per block, shared with the training forward of a step)
        from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import kernel_forms

        fl = kernel_forms(params, eval_reads_eager=not module.training, rf3=rf3)
        if module.ln_pair.weight.shape[0] == 128 and module.to_out.weight.shape[1] in (128, 256):
            # D128 inference output on k3w (b200_infer.K3W_D128_CELLS): its LayerNorm-folded Wp' / Wg' / vectors (fold_op), cached apart
            # from the shared D64 / D128 forms (inference only; the training kernels never read them)
            from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import fold_op, front_fold

            # + the LN_in-folded packed front and its correction rows (b200_infer.front_fold, k1w FOLD); RoseTTAFold3 (its 1 / n
            # factor is in W_l: per call in the forms launch, or as torch ops through modules.triangle._LeftScaled -- both must keep
            # the same bits) gets two empty placeholders and keeps the plain front
            plain = rf3 is not None or rf3_inline or type(module).__name__ == "_LeftScaled"

            def fold128():
                wg, wp, gi, bi, go, bo = (t.detach().float().contiguous() for t in params[4:10])
                ff = (wp.new_empty(0), wp.new_empty(0)) if plain else front_fold(params[0].detach(), params[1].detach(), params[2].detach(), params[3].detach(), gi, bi)
                return [*fold_op([wp, go, bo, wg, gi, bi]), *ff]

            if torch.compiler.is_compiling():
                fl = [*fl, *fold128()]  # traced: graph ops
            else:
                tag = f"trimul_b200_k3w128_{bf}_{plain}"
                fl = [*fl, *_capture.param_forms(tag, params, fold128, capture_reads_eager=not module.training)]
        elif module.ln_pair.weight.shape[0] == 64 and module.to_out.weight.shape[1] in (64, 128):
            # D64 inference output on k3hf (b200_infer.K3H_FOLD: both LayerNorms folded, M = 64 tiles): Wp' / Wg' / vectors (fold_op),
            # cached apart from the shared forms (inference only); + the LN_in-folded packed front and its correction rows (k1w FOLD at
            # D64, b200_infer.K1W_FOLD64_CELLS) -- RoseTTAFold3 (its 1 / n factor in W_l) gets two empty placeholders, as at D128
            from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import fold_op, front_fold

            plain = rf3 is not None or rf3_inline or type(module).__name__ == "_LeftScaled"

            def fold64():
                wg, wp, gi, bi, go, bo = (t.detach().float().contiguous() for t in params[4:10])
                ff = (wp.new_empty(0), wp.new_empty(0)) if plain else front_fold(params[0].detach(), params[1].detach(), params[2].detach(), params[3].detach(), gi, bi)
                return [*fold_op([wp, go, bo, wg, gi, bi]), *ff]

            if torch.compiler.is_compiling():
                fl = [*fl, *fold64()]
            else:
                tag = f"trimul_b200_k3hf64_{bf}_{plain}"
                fl = [*fl, *_capture.param_forms(tag, params, fold64, capture_reads_eager=not module.training)]
        return fl

    def forms():
        from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import fold_op, front_fold, pack_front

        w = params
        # Wg / Wp fp32 as stored: k3w's LayerNorm-folded forms (fold_op, made here once per parameter set instead of per call) round
        # bf16(W g) once from the master (2026-11-17)
        wg, wp, gi, bi, go, bo = (t.float().contiguous() for t in w[4:10])
        wb = tuple(t.to(bf) for t in w[:4])
        # + k1w's packed front (b200_infer.pack_front: one TMA box per weight slot), + at D256 / D384 the LN_in-folded packed front and
        # its correction rows (b200_infer.front_fold, k1w FOLD)
        ff = front_fold(*w[:4], gi, bi) if gi.shape[0] in (256, 384) else ()
        return (*wb, wg, wp, gi, bi, go, bo, *fold_op([wp, go, bo, wg, gi, bi]), pack_front(*wb), *ff)

    # traced (torch.compile), param_forms returns forms(): the casts are graph ops
    return list(_capture.param_forms(f"trimul_b200_{bf}", params, forms, capture_reads_eager=not module.training))


def update_inference(
    module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None, *, bidirectional: bool, rf3: bool = False
) -> torch.Tensor:
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import inference, inference_into

    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    if n % _GRANULE:
        return _padded_inference(module, pair, mask, bidirectional=bidirectional, rf3=rf3)
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
    assert not rf3 or (d <= 128 and bsz == 1), "the RoseTTAFold3 factor in the forms: D64 / D128, one sample"
    # RoseTTAFold3 (D64 / D128, one sample): the length factor is applied inside the front kernels on the left projection (S90 part 3: no per-call
    # forms launch; the forms are the plain cached ones). env TRIMUL_B200_RF3 = 0: the factor folded into W_l by the forms launch (A/B)
    rf3_inline = bool(rf3) and os.environ.get("TRIMUL_B200_RF3", "1") == "1"
    params = _inference_params(module, pair.dtype, (token_mask, n) if rf3 and not rf3_inline else None, rf3_inline=rf3_inline)
    if not rf3_inline:
        rf3_arg = -1.0
    elif token_mask is not None:
        rf3_arg = 0.0
    else:
        rf3_arg = float(torch.tensor(1.0 / max(n, 1), dtype=torch.float32).to(torch.bfloat16).float())
    scale = None if dropscale is None else dropscale.reshape(bsz * n, d).to(pair.dtype).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    chunk, _ = _plan(module, pair, True, bidirectional)
    if chunk >= bsz:
        return inference([pair, *params], token_mask, scale, direction, rf3_arg)
    # B samples in launches of ``chunk``, one after another, each writing its rows of the output in place
    out = torch.empty_like(pair)
    start = 0
    for size in _split(bsz, chunk):
        stop = start + size
        inference_into(
            [pair[start:stop], *params],
            None if token_mask is None else token_mask[start * n : stop * n],
            None if scale is None else scale[start * n : stop * n],
            direction,
            out[start:stop],
        )
        start = stop
    return out


def train_refusal(module, pair: torch.Tensor, *, bidirectional: bool) -> str | None:
    """Training (autograd, ``b200_train``) at every width D64-D512, either direction, any B (B > ``max_batch`` or more than fit in
    memory: chunks of one autograd node). ``None`` when served, else the reason."""
    why = _common_refusal(module, pair)
    if why is not None:
        return why
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import supports

    n, d = pair.shape[1], pair.shape[-1]
    one = _one_chunk(module, pair, False, bidirectional)
    if one.shape[0] * n * n >= 2**31:
        return f"L {n}: one launch past the int32 token index (B L^2 >= 2**31)"
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if not supports(d, n, direction, module.d_hidden):
        return (
            f"d_pair {d}, d_hidden {module.d_hidden}, L {n}: outside the B200 training kernels (b200_train.supports: D64 / D128 L % 128 == 0 "
            "and L <= 10240, D256-512 L % 16 == 0; hidden = D, or 2 D at D64 / D128 one direction or D64 bidirectional)"
        )
    return None


def route(module, pair: torch.Tensor, mask: torch.Tensor | None, dropout_p: float | None, *, bidirectional: bool):
    """``("b200_infer" | "b200_train", None)``, else ``(None, reason)``: grad off -> the inference kernels (a live row dropout through
    their dropout-scale epilogue), grad on -> the training kernels."""
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return None, "mask must be a bool [B, L] residue mask"
    dropout_p = module.p_drop if dropout_p is None else dropout_p
    live_drop = bool(dropout_p) and dropout_p > 0.0 and module.training
    if not torch.is_grad_enabled():
        why = inference_refusal(module, pair, bidirectional=bidirectional, dropout=live_drop)
        return ("b200_infer", None) if why is None else (None, why)
    why = train_refusal(module, pair, bidirectional=bidirectional)
    return ("b200_train", None) if why is None else (None, why)


def forward(
    module,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    dropout_p: float | None,
    route_name: str,
    *,
    bidirectional: bool,
    rf3: bool = False,
):
    """The whole update ``pair + drop_row(trimul(pair))`` on the route ``route`` chose; the row-dropout scale is the reference's draw
    (``modules.functional.dropout_scale(pair, p, 1)``: the same bits as the PyTorch statements under one generator state)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import dropout_scale

    dropout_p = module.p_drop if dropout_p is None else dropout_p
    live = bool(dropout_p) and dropout_p > 0.0 and module.training
    if route_name == "b200_infer":
        if not live:  # opt-in FP8 planes (docs/b200/fp8.md; POLICY None = off: the bf16 route below, bit for bit)
            from miniworld_engine.kernels.trimul_inproj.cuda import b200_fp8 as _f8

            y = _f8.module_route(module, pair, mask, bidirectional, rf3) if _f8.POLICY is not None else None
            if y is not None:
                return y
        return update_inference(module, pair, mask, dropout_scale(pair, dropout_p) if live else None, bidirectional=bidirectional, rf3=rf3)
    # training: the draw alone here, its scale made by the training call's forms launch (``b200_train.kernel_forms`` at D <= 128, the wide forms
    # launch at D >= 256: S90)
    draw = b200_train.dropout_draw(pair, dropout_p) if live and b200_train.DROP_IN_FORMS else None
    if draw is not None:
        return update_train(module, pair, mask, None, bidirectional=bidirectional, rf3=rf3, drop=(draw, float(dropout_p)))
    return update_train(module, pair, mask, dropout_scale(pair, dropout_p) if live else None, bidirectional=bidirectional, rf3=rf3)


def update_train(
    module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None, *, bidirectional: bool, rf3: bool = False,
    drop: tuple | None = None,
) -> torch.Tensor:
    """The module's parameters are the leaves (any float dtype; see ``b200_train.trimul_train``). ``drop`` = (the reference's draw
    [B, 1, L, D], p) in place of ``dropscale`` (D64 / D128: the scale made inside the training call's forms launch)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import drop_scale_op
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import trimul_train, trimul_train_chunked

    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
    bf = pair.dtype
    # the parameters themselves: the training op casts them for the kernels outside autograd, so an fp32 master gets fp32 gradients
    leaves = [
        pair,
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
    ]
    scale = None if dropscale is None else dropscale.reshape(bsz * n, d).to(bf).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    chunk, checkpoint = _plan(module, pair, False, bidirectional)
    if chunk >= bsz:
        if drop is not None:
            r = b200_train.draw_rows(drop[0], bsz * n, d)
            return trimul_train(leaves, token_mask, None, direction, n if rf3 else 0, drop=(r, drop[1]))
        return trimul_train(leaves, token_mask, scale, direction, n if rf3 else 0)
    if drop is not None:
        scale = drop_scale_op(b200_train.realize_draw(drop[0]), drop[1]).reshape(bsz * n, d)
    assert not rf3, "the RoseTTAFold3 factor in the forms: one sample"
    return trimul_train_chunked(leaves, token_mask, scale, direction, _split(bsz, chunk), checkpoint)
