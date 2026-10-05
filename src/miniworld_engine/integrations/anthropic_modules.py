"""Module compositions using the pinned Anthropic kernels and ordinary torch GEMMs.

Explicit Anthropic requests enter here before any MiniWorld fast path. These
forward-only compositions never silently borrow a MiniWorld kernel or backward.
"""
from math import prod
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from miniworld_engine.integrations import anthropic as upstream
from miniworld_engine.integrations import anthropic_swa_ops as swa_ops


def _require(x):
    upstream._inference()
    from miniworld_engine import settings
    if settings.current().engine_backend == "triton":
        raise ValueError("anthropic conflicts with engine_backend=triton")
    if not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Anthropic module kernels require CUDA floating-point inputs")


def _linear(x, layer):
    return F.linear(x, layer.weight.to(x.dtype), None if layer.bias is None else layer.bias.to(x.dtype))


def local_atom_attention(module, single, cond, pair, mask=None):
    """AF3 32-query/128-key attention via the unchanged upstream atom entry point.

    The surrounding AdaLN/transition modules use their own Anthropic dispatch.
    Batch elements have independent pair biases; augmentations share each bias.
    """
    from miniworld_engine.modules.local_dit.module import (
        KEYS,
        QUERIES,
        _key_windows,
        windows,
    )

    _require(single)
    att = module.attention
    x = att.ada_ln_in(single, cond)
    samples, batch, length, width = x.shape
    heads, nwin = att.n_head, windows(length)
    if tuple(pair.shape[:4]) != (batch, nwin, QUERIES, KEYS):
        raise ValueError("AF3 atom pair must have shape [B, ceil(N/32), 32, 128, d_pair]")
    z, ln_selection = upstream.layer_norm(
        pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
    bias = _linear(z, att.to_bias).permute(0, 4, 1, 2, 3).contiguous()
    xkv = att.ada_ln_kv(x, cond) if module.cross_attention else x
    q, k, v = (_linear(t, layer).reshape(samples, batch, length, heads, width // heads)
               for t, layer in ((x, att.to_query), (xkv, att.to_key), (xkv, att.to_value)))
    keep = torch.ones(batch, length, device=x.device, dtype=torch.bool) if mask is None else mask
    valid = _key_windows(keep[..., None], length)[..., 0].bool()
    bias = bias.masked_fill(~valid[:, None, :, None, :], float("-inf"))
    # The upstream row has no empty-window guard. Give empty windows a finite
    # temporary bias, then zero their outputs to match the module's reference.
    live = valid.any(-1)
    bias = torch.where(live[:, None, :, None, None], bias, torch.zeros_like(bias))
    outputs, selections = [], []
    for b in range(batch):
        out, selection = upstream.atom_attention(q[:, b], k[:, b], v[:, b], bias[b], row="fpf_atom")
        outputs.append(out)
        selections.append(str(selection))
    out = torch.stack(outputs, dim=1).reshape(samples, batch, length, width)
    live_queries = live.repeat_interleave(QUERIES, dim=-1)[:, :length]
    out = out.masked_fill(~live_queries[None, :, :, None], 0)
    out = _linear(torch.sigmoid(_linear(x, att.to_gate)) * out, att.to_out)
    module.anthropic_selection = {"attention": selections, "pair_norm": str(ln_selection),
                                  "window": "32x128", "cross_attention": module.cross_attention}
    return torch.sigmoid(_linear(cond, att.to_scale)) * out


def _periodic_rows(tensor, shape):
    """Hand the upstream DTK a shared leading block, not an expanded copy.

    Broadcasts inside the row geometry (e.g. [A,1,L,C] -> [A,B,L,C])
    are not periodic and must be materialized in the normal broadcast order.
    """
    source = (1,) * (len(shape) - tensor.ndim) + tuple(tensor.shape)
    first = next((i for i, size in enumerate(source[:-1]) if size != 1), len(source) - 1)
    if source[first:-1] == tuple(shape[first:-1]):
        return tensor.reshape(-1, shape[-1]).contiguous(), prod(source[:-1])
    expanded = tensor.expand(shape).reshape(-1, shape[-1]).contiguous()
    return expanded, expanded.shape[0]


def rms_norm(module, x):
    _require(x)
    eps = module.eps if module.eps is not None else torch.finfo(torch.float32).eps
    module.anthropic_selection = {"row": "dtk_kernels.ln_modulate", "rms": True}
    return upstream.carried_kernel("dtk_kernels").ln_modulate(
        x.contiguous(), weight=module.weight, eps=eps, rms=True)


def adaptive_layer_norm(module, x, cond):
    _require(x)
    rows = upstream.carried_kernel("dtk_kernels")
    cn = rows.ln_modulate(cond.contiguous(), weight=module.ln_cond.weight, eps=module.ln_cond.eps)
    scale, shift = _linear(cn, module.to_scale), _linear(cn, module.to_bias)
    shape = torch.broadcast_shapes(x.shape, scale.shape)
    xx = x.expand(shape).contiguous()
    scale, period = _periodic_rows(scale, shape)
    shift, _ = _periodic_rows(shift, shape)
    module.anthropic_selection = {"row": "dtk_kernels.ln_modulate", "conditioning": "upstream LN + cuBLAS",
                                  "mod_period": period}
    return rows.ln_modulate(xx, scale, shift, eps=module.ln_in.eps, mod_period=period)


def conditioned_transition(module, x, cond, *, residual):
    _require(x)
    rows = upstream.carried_kernel("dtk_kernels")
    norm = adaptive_layer_norm(module.ada_ln_in, x, cond)
    ab = torch.cat((_linear(norm, module.expand_a), _linear(norm, module.expand_b)), -1)
    hidden = rows.swiglu(ab.reshape(-1, ab.shape[-1])).reshape(*ab.shape[:-1], -1)
    out = _linear(hidden, module.squeeze)
    gate, period = _periodic_rows(_linear(cond, module.to_scale), out.shape)
    module.anthropic_selection = {"row": "dtk_kernels.ln_modulate+swiglu+gate_residual", "gemm": "cuBLAS",
                                  "gate_period": period}
    return rows.gate_residual(out.reshape(-1, out.shape[-1]), gate=gate, gate_period=period,
                              res=x.expand_as(out).contiguous().reshape(-1, out.shape[-1]) if residual else None,
                              out_dtype=x.dtype).reshape(out.shape)


def attention_pair_bias(module, single, pair, mask):
    _require(single)
    norm = module.ln_single(single)
    q, k, v, gate = (_linear(norm, layer).unflatten(-1, (module.n_head, -1))
                      for layer in (module.to_query, module.to_key, module.to_value, module.to_gate))
    if module.use_qk_norm:
        q, k = module.norm_query(q), module.norm_key(k)
    bias = _linear(module.ln_pair(pair), module.to_bias).permute(0, 3, 1, 2).float().contiguous()
    if mask is not None:
        # The module's finite-min bias makes an entirely masked batch uniform.
        # Explicit zero logits preserve that contract across online-softmax tiles.
        empty = ~mask.bool().any(-1)
        q, k = (torch.where(empty[:, None, None, None], 0, t) for t in (q, k))
        bias = torch.where(empty[:, None, None, None], 0, bias)
        mask = mask | empty[:, None]
    out, selected = upstream.module_pair_bias_attention(q, k, v, bias, mask, gate)
    module.anthropic_selection = selected
    return single + _linear(out.flatten(-2), module.to_out).to(single.dtype)


def augmented_attention(module, single, cond, pair, mask, compute_dtype=None, *, bias_only=False):
    _require(single)
    norm = adaptive_layer_norm(module.ada_ln_in, single, cond)
    v, gate = (_linear(norm, layer).unflatten(-1, (module.n_head, -1))
               for layer in (module.to_value, module.to_gate))
    if bias_only:
        q = k = torch.zeros_like(v)
    else:
        q, k = (_linear(norm, layer).unflatten(-1, (module.n_head, -1))
                for layer in (module.to_query, module.to_key))
    if not bias_only and module.use_qk_norm:
        q, k = module.norm_query(q), module.norm_key(k)
    bias = _linear(module.ln_pair(pair), module.to_bias).permute(0, 3, 1, 2).float().contiguous()
    outputs, selections = [], []
    for b in range(single.shape[1]):
        key_mask = None if mask is None else (mask[b:b+1] if mask.ndim == 2 else mask[:, b])
        operands = [t[:, b].to(compute_dtype or t.dtype).contiguous() for t in (q, k, v, gate)]
        out, selected = upstream.module_pair_bias_attention(operands[0], operands[1], operands[2],
                                                            bias[b:b+1], key_mask, operands[3])
        outputs.append(out)
        selections.append(selected._asdict())
    out = _linear(torch.stack(outputs, 1).flatten(-2).to(single.dtype), module.to_out)
    scale, period = _periodic_rows(_linear(cond, module.to_scale), out.shape)
    module.anthropic_selection = {"batches": selections, "zero_qk": bias_only, "gate_period": period,
                                  "surround": "dtk LN/modulation/gate + upstream LN + cuBLAS"}
    return upstream.carried_kernel("dtk_kernels").gate_residual(
        out.reshape(-1, out.shape[-1]), gate=scale, gate_period=period,
        out_dtype=single.dtype).reshape(out.shape)


def triangle_attention_composition(module, pair, mask, *, bidirectional=False):
    """Projected APB composition for bias-only and bidirectional triangle attention."""
    _require(pair)
    if module.training and getattr(module, "p_drop", 0):
        raise RuntimeError("Anthropic triangle attention requires eval() with dropout")
    norm = module.ln_pair(pair)
    values = _linear(norm, module.to_value)
    biases = _linear(norm, module.to_bias)
    gates = _linear(norm, module.to_gate)
    if module.use_self_attention:
        queries, keys = (_linear(norm, layer) for layer in (module.to_query, module.to_key))
    else:
        queries = keys = torch.zeros_like(values)
    chunks = 2 if bidirectional else 1
    updates, selections = [], []
    for i, (q, k, v, bias) in enumerate(zip(queries.chunk(chunks, -1), keys.chunk(chunks, -1),
                                          values.chunk(chunks, -1), biases.chunk(chunks, -1), strict=True)):
        ending = i == 1 if bidirectional else not module.starting
        if ending:
            q, k, v, bias = (t.transpose(1, 2) for t in (q, k, v, bias))
        q, k, v = (t.unflatten(-1, (module.n_head, -1)).contiguous() for t in (q, k, v))
        bias = bias.permute(0, 3, 1, 2).float().contiguous()
        batches = []
        for b in range(pair.shape[0]):
            out, sel = upstream.module_pair_bias_attention(q[b], k[b], v[b], bias[b:b+1],
                                                           None if mask is None else mask[b:b+1])
            batches.append(out.flatten(-2))
            selections.append(sel._asdict())
        out = torch.stack(batches)
        updates.append(out.transpose(1, 2) if ending else out)
    out = torch.cat(updates, -1).contiguous()
    out = upstream.carried_kernel("dtk_kernels").gate_residual(
        out.reshape(-1, out.shape[-1]), gate=gates.reshape(-1, gates.shape[-1])).reshape(out.shape)
    module.anthropic_selection = {"composition": "projected APB + DTK gate + cuBLAS", "batches": selections,
                                  "zero_qk": not module.use_self_attention, "bidirectional": bidirectional}
    return pair + _linear(out, module.to_out)


class _StockNorm:
    """The upstream MSA recipes take a stock fp32-affine LayerNorm object."""
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __call__(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def outer_product_mean(module, msa, mask, token_asym_id, residual):
    _require(msa)
    op = upstream.operation("msa_opm")
    view = SimpleNamespace(norm=_StockNorm(module.ln_msa), proj_a=module.to_left,
                           proj_b=module.to_right, proj_o=module.to_out)
    if mask is None:
        mask = torch.ones(msa.shape[:3], device=msa.device, dtype=torch.bfloat16)
    if module.normalize_before_proj:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            update = op.forward_mask_norm(view, msa, mask.to(torch.bfloat16), chunk_size=None).to(msa.dtype)
        selection = {"row": "msa_opm.forward_mask_norm", "config": op.cfg_for("mask_norm", update.shape[-1], msa.device)}
    else:
        # ESMFold2 divides the projected outer sum, including output bias, by
        # each pair's actual valid MSA count. The scalar core with divisor one
        # computes that numerator; a tensor division preserves arbitrary masks.
        norm, _ = upstream.layer_norm(msa, msa.shape[-1], module.ln_msa.weight,
                                      module.ln_msa.bias, module.ln_msa.eps, out_dtype=msa.dtype)
        a = (_linear(norm, module.to_left) * mask[..., None]).to(torch.bfloat16)
        b = (_linear(norm, module.to_right) * mask[..., None]).to(torch.bfloat16)
        bias = module.to_out.bias
        cache = {"CH": a.shape[-1], "CZ": module.to_out.weight.shape[0],
                 "wout_t": module.to_out.weight.to(torch.bfloat16).t().contiguous(),
                 "bias32": None if bias is None else torch.cat((bias.to(torch.bfloat16).float(), bias.float()))}
        cfg = op.CFG_VARIANTS["a1"]
        numerator = torch.stack([op.opm_core(aa.contiguous(), bb.contiguous(), cache, "scalar_norm",
                                             norm_scalar=1, cfg=cfg) for aa, bb in zip(a, b, strict=True)])
        counts = torch.einsum("bsi,bsj->bij", mask.float(), mask.float()).clamp_min(1)
        update = (numerator.float() / counts[..., None]).to(msa.dtype)
        selection = {"row": "msa_opm.opm_core", "config": "a1", "normalization": "after projection, masked counts"}
    if module.mask_interchain and token_asym_id is not None:
        update = update * (token_asym_id[:, :, None] == token_asym_id[:, None, :]).unsqueeze(-1)
    module.anthropic_selection = selection
    return update if residual is None else residual + update


def pair_weighted_averaging(module, msa, pair, mask):
    _require(msa)
    if module.training and module.drop_msa.p_drop:
        raise RuntimeError("Anthropic PWA requires eval() when dropout is enabled")
    view = SimpleNamespace(norm_m=_StockNorm(module.ln_msa), proj_m=module.to_value, proj_g=module.to_gate,
                           norm_z=_StockNorm(module.ln_pair), proj_z=module.to_bias, proj_o=module.to_out,
                           inf=1e6, num_heads=module.n_head, c_h=module.to_value.weight.shape[0] // module.n_head)
    n = msa.shape[2]
    if mask is None:
        mask = torch.ones(msa.shape[0], n, device=msa.device, dtype=torch.bool)
    pair_mask = mask[:, None, :].expand(-1, n, -1).to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        update = upstream.operation("msa_pwa").forward_masked(view, msa, pair, pair_mask, chunk_heads=False)
    module.anthropic_selection = {"row": "msa_pwa.forward_masked", "norm": "stock upstream recipe"}
    return msa + update.to(msa.dtype)


def swiglu_ffn(module, x):
    _require(x)
    ab = module.w_up(x)
    hidden = swa_ops.swiglu(ab.reshape(-1, ab.shape[-1]))
    module.anthropic_selection = {"row": "dtk_kernels.swiglu", "gemm": "cuBLAS"}
    return module.w_down(hidden.reshape(*ab.shape[:-1], -1))


def swa_attention(module, x, attention_params):
    """Gather attention with the module's exact sliding window, including padding.

    K=129 is a candidate in upstream's cell table, locally qualified on A100.
    This is a composition, not the release's different 32-query/128-key atom op.
    """
    _require(x)
    from miniworld_engine.modules.swa_atom_attention.module import apply_rotary_emb_3d

    n, s, c = x.shape
    cos, sin, _, _, _, valid = attention_params
    q, k, v = module.Wqkv(x).view(n, s, 3, module.n_heads, module.head_dim).unbind(2)
    eps = torch.finfo(torch.float32).eps
    q, k = (apply_rotary_emb_3d(swa_ops.rms(t.contiguous(), None, None, eps), cos, sin)
            for t in (q, k))
    # Rank-space windows also preserve the reference semantics for interior gaps.
    ranks = valid.to(torch.int64).cumsum(-1) - 1
    positions = torch.arange(s, device=x.device).expand(n, -1)
    packed = torch.where(valid, positions, s).sort(-1).values
    offsets = torch.arange(-module.half_window, module.half_window + 1, device=x.device)
    key_rank = ranks[:, :, None] + offsets
    keep = (key_rank >= 0) & (key_rank < valid.sum(-1)[:, None, None]) & valid[:, :, None]
    indices = packed.gather(1, key_rank.clamp(0, s-1).flatten(1)).view_as(key_rank)
    indices = torch.where(keep, indices, -1)
    # A padded query gets a harmless self key, then its output is zeroed.
    indices[:, :, 0] = torch.where(valid, indices[:, :, 0], positions)
    indices = indices.sort(-1).values.to(torch.int32).contiguous()
    bias = torch.zeros((), device=x.device, dtype=torch.float32).expand(1, s, s, module.n_heads)
    out = swa_ops.gather(
        q.flatten(-2).contiguous(), k.flatten(-2).contiguous(), v.flatten(-2).contiguous(),
        bias, indices, module.n_heads,
        scale=module.scale)
    out = swa_ops.gate(out.reshape(-1, c), module.gate_proj(x).reshape(-1, c), None, True).reshape(n, s, c)
    module.anthropic_selection = {"row": "gather_attn", "qualification": "local A100 candidate",
                                  "window_keys": 2 * module.half_window + 1,
                                  "surround": "dtk RMSNorm + torch RoPE + cuBLAS"}
    return module.out_proj(out.reshape(n, s, c) * valid[:, :, None])


def swa_dit(module, x, cond, attention_params):
    _require(x)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = module.adaln_modulation(cond).chunk(6, -1)
    eps = torch.finfo(torch.float32).eps
    def norm(t, scale, shift):
        return swa_ops.rms(t.contiguous(), (1 + scale).reshape(-1, x.shape[-1]).contiguous(),
                               shift.reshape(-1, x.shape[-1]).contiguous(),
                               eps)
    def residual(t, gate, res):
        return swa_ops.gate(t.reshape(-1, x.shape[-1]), gate.reshape(-1, x.shape[-1]).contiguous(),
                            res.reshape(-1, x.shape[-1]), False).reshape(x.shape)
    x = residual(module.attn(norm(x, scale_a, shift_a), attention_params), gate_a, x)
    module.anthropic_selection = {"row": "dtk_kernels + gather_attn", "gemm": "cuBLAS"}
    return residual(module.ffn(norm(x, scale_f, shift_f)), gate_f, x)
