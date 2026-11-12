"""Automatic dispatch of the A100 (sm_80) hand-CUDA triangle attention (``kernels.triangle_attention.cuda``): the whole module in inference (``serves_module`` / ``forward``) and in
training (``serves_train`` / ``forward_train``: the same forward with what the backward needs, the module's broadcast dropout, and an autograd function).
The ``cuda`` backend of the triangle-attention classes (``modules/triangle.py``) states ``refusal`` on an A100.

Two whole-module paths, picked by the module's geometry:

  * ``fused``: d_pair 128, 4 heads x 32 (``sm80.py``): the front, the core and the back are three persistent kernels with the weights resident in shared memory;
  * ``wide``: every other width the model code uses (``sm80_wide.py``): d_pair 64 .. 512 (a multiple of 64), 1 .. 16 heads of 16 or 32 channels: hand-CUDA row kernels around cuBLAS GEMMs
    and the same core.
  * ``variant``: what neither takes (``sm80_wide.variant``): AlphaFold 3's ending-node bias orientation (AF3 / RF3 / Chai-1), RoseTTAFold3's gate / out-projection biases,
    Chai-1's untransposed / dual outputs, heads of 64 channels -- the rows engine on the generalised core, the variants as the H100 width-generic path has them.
"""

from __future__ import annotations

from typing import cast

import torch

from miniworld_engine.kernels import _capture


def _weights(module) -> tuple[torch.Tensor, ...]:
    return (
        module.to_query.weight,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_bias.weight,
    )


def _detached_casts(leaves) -> tuple[torch.Tensor, ...]:
    """bf16 copies of ``leaves`` with no autograd history: a cached form built under grad (the training gate checks the inference forms) held a
    ``ToCopyBackward`` -> ``AccumulateGrad`` chain of the warm-up, and a CUDA-graph capture of the training step then failed on its stale stream."""
    with torch.no_grad():
        return tuple(t.to(torch.bfloat16).contiguous() for t in leaves)


def _inference_forms(module) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """The five projection weights and the output weight in the kernels' dtype for an inference call. fp32 master parameters (the harness's
    ``bf16-mixed``: fp32 parameters over bf16 activations, as the trainer feeds the model) are cast once and cached while the parameters are
    unmodified (``_capture.param_forms``: an inference graph under ``static_weights`` replays no cast kernel); bf16 parameters are used as they are."""
    leaves = (*_weights(module), module.to_out.weight)
    bf = torch.bfloat16
    if all(t.dtype == bf for t in leaves) or any(t.dtype not in (bf, torch.float32) for t in leaves):
        return leaves[:5], leaves[5]
    # grad off (inference): a CUDA-graph capture is served the casts its eager warm-up made, so a replay runs no cast kernel (six per call)
    forms = _capture.param_forms(
        "triattn_sm80",
        list(leaves),
        lambda: _detached_casts(leaves),
        capture_reads_eager=not torch.is_grad_enabled(),
    )
    return tuple(forms[:5]), forms[5]


def _train_forms(module) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """As ``_inference_forms`` with autograd: an fp32 master parameter is cast by a differentiable ``to(bf16)``, so the kernels' bf16 weight gradient
    reaches the parameter as the fp32 cast of it (what autocast gives the PyTorch module)."""
    leaves = (*_weights(module), module.to_out.weight)
    bf = torch.bfloat16
    if any(t.dtype not in (bf, torch.float32) for t in leaves):
        return leaves[:5], leaves[5]
    forms = tuple(t if t.dtype == bf else t.to(bf) for t in leaves)
    return forms[:5], forms[5]


def _kind(module, pair: torch.Tensor, mask: torch.Tensor | None):
    """``("fused", None)`` for d_pair 128 / 4 x 32, ``("wide", cfg)`` for the other geometries ``sm80_wide`` serves, else None (before the kernels' own gates)."""
    if not module.use_self_attention or module.use_qk_norm:
        return None
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4:
        return None
    if mask is not None and not (mask.dtype == torch.bool and mask.shape == pair.shape[:2]):
        return None
    d_hidden, d_pair = module.to_query.weight.shape
    if pair.shape[-1] != d_pair:
        return None
    if (d_pair, d_hidden, module.n_head) == (128, 128, 4):
        return ("fused", None)
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    cfg = sm80_wide.cfg_for(d_pair, d_hidden, module.n_head)
    return None if cfg is None else ("wide", cfg)


def _serves_kind(module, pair: torch.Tensor, mask: torch.Tensor | None, *, train: bool) -> bool:
    kind = _kind(module, pair, mask)
    if kind is None:
        return False
    from miniworld_engine.kernels.triangle_attention.cuda import sm80, sm80_wide

    ln = module.ln_pair
    weights, wo = _inference_forms(module)  # the dtypes the call will run with; training casts differentiably in forward_train
    if kind[0] == "fused":
        if train:
            return sm80.supports_train(pair, weights, ln.weight, ln.bias, wo, mask) and sm80._built(pair)
        return sm80.available_module(pair, weights, ln.weight, ln.bias, wo, mask)
    return sm80_wide.supports(pair, kind[1], weights, wo, ln.weight, ln.bias, mask, train=train) and sm80_wide.loads(kind[1])


def _features(module, *, train: bool = True) -> bool:
    """The module has a variant feature the fused / wide paths lack: the ending node's AF3 bias orientation (training only: the inference fronts write
    the flipped bias themselves), projection biases, Chai-1's output layouts."""
    return (
        (module._bias_untransposed and train)
        or module.to_gate.bias is not None
        or module.to_out.bias is not None
        or module.dual_output
        or (module.untransposed_output and not module.starting)
    )


def _is_variant(module, pair: torch.Tensor, *, train: bool = True) -> bool:
    """The call goes to the variant path: a variant feature, or a geometry only it takes (heads of 64 channels). Plain modules the fused / wide paths
    serve stay there."""
    if _features(module, train=train):
        return True
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    d_hidden, d_pair = module.to_query.weight.shape
    if (d_pair, d_hidden, module.n_head) == (128, 128, 4) or sm80_wide.cfg_for(d_pair, d_hidden, module.n_head) is not None:
        return False
    return sm80_wide.variant_cfg(d_pair, d_hidden, module.n_head) is not None


def _out_mode(module) -> int:
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    if module.dual_output:
        return sm80_wide.DUAL
    return sm80_wide.UNTRANSPOSED if (module.untransposed_output and not module.starting) else sm80_wide.STANDARD


def _variant_params(module, train: bool):
    """(q, k, v, g, bias weights), wo, gate bias, out bias, second out-projection in the kernels' dtype (bf16; fp32 masters cast as ``_inference_forms`` /
    ``_train_forms`` do)."""
    weights, wo = _train_forms(module) if train else _inference_forms(module)
    extra = [module.to_gate.bias, module.to_out.bias, module.to_out_t.weight if module.dual_output else None]
    bf = torch.bfloat16
    if train:
        extra = [t if t is None or t.dtype == bf else t.to(bf) for t in extra]
    else:
        present = [t for t in extra if t is not None]
        if any(t.dtype != bf for t in present):
            forms = iter(
                _capture.param_forms(
                    "triattn_sm80_variant", present, lambda: _detached_casts(present), capture_reads_eager=not torch.is_grad_enabled()
                )
            )
            extra = [None if t is None else next(forms) for t in extra]
    return weights, wo, *extra


def _variant_refusal(module, pair: torch.Tensor, mask: torch.Tensor | None, *, train: bool) -> str | None:
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    d_hidden, d_pair = module.to_query.weight.shape
    vc = sm80_wide.variant_cfg(d_pair, d_hidden, module.n_head)
    if vc is None:
        return (
            f"d_pair {d_pair}, {module.n_head} heads x {d_hidden // module.n_head}: outside the A100 kernels (d_pair 64 .. 512 a multiple of 64, "
            "heads of 16 / 32 channels up to 16 or of 64 channels up to 8)"
        )
    if module.dual_output and (module.to_out_t.bias is not None or module.to_out.bias is not None):
        return "a biased second out-projection: no model has one, not part of the A100 kernels"
    weights, wo, gb, bo, wt = _variant_params(module, train=False)
    ln = module.ln_pair
    if not sm80_wide.variant_supports(pair, vc, weights, wo, ln.weight, ln.bias, mask, gate_bias=gb, out_bias=bo, wo_t=wt, train=train):
        return (
            f"L {pair.shape[1]}, d_pair {d_pair}, {module.n_head} heads, {'training' if train else 'inference'}: refused by the A100 variant path's gates "
            "(L % 128 == 0, contiguous [B, L, L, d_pair], a bool [B, L] mask, the training bias-gradient budget) or its extensions failed to build"
        )
    return None


def _variant_forward(module, pair: torch.Tensor, mask: torch.Tensor | None, *, train: bool) -> torch.Tensor:
    """``pair + drop(update(pair))`` on ``sm80_wide.variant`` (with autograd when ``train``)."""
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    d_hidden, d_pair = module.to_query.weight.shape
    vc = cast("sm80_wide.VCfg", sm80_wide.variant_cfg(d_pair, d_hidden, module.n_head))  # (not None: serves() accepted this layout)
    weights, wo, gb, bo, wt = _variant_params(module, train=train)
    ln = module.ln_pair
    ds = None
    if train and module.training and module.p_drop > 0.0:
        b, length = pair.shape[:2]
        ds = module._make_drop_scale(pair, module.p_drop).reshape(b, length, pair.shape[-1])
    return sm80_wide.variant(
        pair,
        vc,
        weights,
        wo,
        ln.weight,
        ln.bias,
        ln.eps,
        mask,
        transposed=not module.starting,
        flip=module._bias_untransposed,
        out_mode=_out_mode(module),
        gate_bias=gb,
        out_bias=bo,
        wo_t=wt,
        ds=ds,
    )


def refusal(module, pair: torch.Tensor, mask: torch.Tensor | None, *, train: bool) -> str | None:
    """Why the A100 path cannot take this call of ``module`` (a ``_TriangleAttentionBase``), or ``None``. ``train``: the autograd path
    (``forward_train``: gradients or the live dropout), else the inference path (``forward``). Shapes / dtypes / flags, then the kernels'
    own gates (which build or load the extensions)."""
    if not module.use_self_attention:
        return "bias-only attention (use_self_attention=False) is not part of the A100 kernels"
    if module.use_qk_norm:
        return "q / k RMSNorm is not part of the A100 kernels"
    if pair.dtype != torch.bfloat16:
        return f"dtype {pair.dtype}: the A100 TriangleAttention kernels are bf16 only (no fp32 kernel)"
    d_hidden, d_pair = module.to_query.weight.shape
    if _is_variant(module, pair, train=train):
        return _variant_refusal(module, pair, mask, train=train)
    kind = _kind(module, pair, mask)
    if kind is None:
        return (
            f"d_pair {d_pair}, {module.n_head} heads x {d_hidden // module.n_head}: outside the A100 kernels (d_pair 128 / 4 x 32 fused; "
            "d_pair 64 .. 512 a multiple of 64 with 1 .. 16 heads of 16 / 32 channels)"
        )
    if not _serves_kind(module, pair, mask, train=train):
        return (
            f"L {pair.shape[1]}, d_pair {d_pair}, {module.n_head} heads, {'training' if train else 'inference'}: refused by the A100 kernels' "
            "gates (L % 128 == 0 for training and the fused d_pair 64 / 128 geometries, L > 128 for inference of the others, contiguous "
            "[B, L, L, d_pair], the training bias-gradient budget) or their extensions failed to build"
        )
    return None


def serves_module(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The whole module in inference, ``pair + TriangleAttention(pair)``: no autograd, dropout off, then ``refusal``."""
    if torch.is_grad_enabled() or (module.training and module.p_drop > 0.0):
        return False
    return refusal(module, pair, mask, train=False) is None


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + TriangleAttention(pair)``.  ``fused``: front (LayerNorm, q | k | v | g, bias), attention (with device-side key compaction on large masked calls), back (gate, output projection,
    residual).  ``wide``: ``sm80_wide.forward`` (row kernels around two GEMMs and the core).  The ending node reads and writes its transposed positions inside the kernels
    (no transposing copies)."""
    if _is_variant(module, pair, train=False):
        return _variant_forward(module, pair, mask, train=False)
    kind = _kind(module, pair, mask)
    ln = module.ln_pair
    ending = not module.starting
    weights, wo = _inference_forms(module)
    if kind is not None and kind[0] == "wide":
        from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

        # L not a multiple of 128, or a whole-pair front output past sm80_wide.ROWS_BLOCKED_BYTES (FoldForge OpenDDE's refiner: 2976 / 3968
        # tokens at d_pair 384): blocks of the attended frame's rows, at L padded to a multiple of 128
        call = sm80_wide.forward_rows if sm80_wide.rows_blocked(pair, kind[1]) else sm80_wide.forward
        return call(
            pair,
            kind[1],
            weights,
            wo,
            ln.weight,
            ln.bias,
            ln.eps,
            mask,
            transposed=ending,
            flip=module._bias_untransposed,
        )
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    qkvg, bias, _ = sm80.front(pair, weights, ln.weight, ln.bias, ln.eps, mask, transposed=ending, flip=module._bias_untransposed)
    q, k, v, _ = sm80.qkv_views(qkvg)
    out, _ = sm80.attention(q, k, v, bias, compact_key_mask=mask)
    b, length = pair.shape[:2]
    o = out.permute(0, 2, 3, 1, 4).reshape(b, length, length, 128)  # "B H L L2 D -> B L L2 (H D)": a view of the core's token-major result
    return sm80.back(o, qkvg, wo, pair, transposed=ending)


def serves_train(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The whole module with autograd (training or an eval-mode module under grad): gradients wanted by the pair tensor or a parameter, then
    ``refusal``."""
    if not torch.is_grad_enabled():
        return False
    wo = module.to_out.weight
    if not (pair.requires_grad or wo.requires_grad or any(w.requires_grad for w in _weights(module))):
        return False
    return refusal(module, pair, mask, train=True) is None


def forward_train(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` with autograd.  The dropout scale is the module's own draw on the original layout (``[B, 1, L, C]`` starting /
    ``[B, L, 1, C]`` ending); in the kernels' starting frame both index the token's second index, so either is ``[B, L, C]`` as it is."""
    if _is_variant(module, pair):
        return _variant_forward(module, pair, mask, train=True)
    kind = _kind(module, pair, mask)
    ln = module.ln_pair
    weights, wo = _train_forms(module)
    ds = None
    if module.training and module.p_drop > 0.0:
        b, length = pair.shape[:2]
        ds = module._make_drop_scale(pair, module.p_drop).reshape(b, length, pair.shape[-1])
    if kind is not None and kind[0] == "wide":
        from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

        return sm80_wide.trainable(
            pair,
            kind[1],
            weights,
            wo,
            ln.weight,
            ln.bias,
            ln.eps,
            mask,
            transposed=not module.starting,
            ds=ds,
        )
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    return sm80.trainable(
        pair,
        weights,
        ln.weight,
        ln.bias,
        wo,
        ln.eps,
        mask,
        transposed=not module.starting,
        ds=ds,
    )
