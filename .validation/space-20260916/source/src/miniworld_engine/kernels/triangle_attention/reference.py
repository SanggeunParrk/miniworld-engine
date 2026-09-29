"""PyTorch reference for the triangle (gated self-)attention kernels.

Two entry points are mirrored, both in the exact tensor layout and argument order the
kernels use.

1. :func:`triangle_attention_pair_bias_pytorch` mirrors the primitive attention kernel
   ``triton_triangle_attention_pair_bias(q, k, v, bias)``
   (``kernels/triangle_attention/triton/main.py``). ``q``/``k``/``v`` are
   ``(B, H, L, L, D)`` and ``bias`` is ``(B, H, L, L)``. Axis 2 (``i``) is a
   BATCH-LIKE row axis — the kernel maps it to ``program_id(2)`` — and axis 3 is the
   ATTENTION axis: ``j`` for the query, ``k`` for the key/value. The pair bias carries
   no row axis at all; it is indexed ``bias[b, h, j, k]`` and broadcast over ``i``::

       logits[b,h,i,j,k] = (sum_d q[b,h,i,j,d] * k[b,h,i,k,d]) * D**-0.5 + bias[b,h,j,k]
       p[b,h,i,j,:]      = softmax_k(logits[b,h,i,j,:])
       out[b,h,i,j,d]    = sum_k p[b,h,i,j,k] * v[b,h,i,k,d]

   This is AF3 Algorithm 14 with ``b_jk`` shared across the triangle apex ``i``.

2. :func:`triangle_attention_pytorch` mirrors the whole-op
   ``kernels/triangle_attention/whole_op.py::triangle_attention``: pair-shaped
   ``(B, L, L, d_pair)`` in and out, weights-as-args, with the LayerNorm, the
   q/k/v/bias/gate projections, the optional qk-RMSNorm, the sigmoid gate and the
   out-projection composed around the attention above.

Masking. ``mask`` is a ``(B, L)`` residue mask and enters ONLY through the bias, on the
KEY axis: ``bias = bias.masked_fill(~mask[:, None, None, :], finfo(bias.dtype).min)``.
A row whose keys are all masked produces an exactly-zero output, matching the kernel's
``l_i`` guard: the kernel floors the running row max at ``-1e38``, so every masked
logit underflows to ``exp2(-inf) == 0``, the denominator is 0, and the guard replaces it
with 1 instead of dividing 0/0. The softmax here floors the max the same way.

Dtypes. Data stays in the caller's dtype (bf16 in production); the QK product, the
softmax and the PV accumulation are done in fp32, and the probabilities are rounded to
the value dtype before the PV contraction because the kernel feeds ``p.to(v.dtype)`` to
its second ``tl.dot`` while summing the fp32 ``p`` for the denominator.

:class:`TriangleAttentionPairBiasReference` is the module form of (1); it holds NO
parameters, because the primitive kernel owns no weights — its q/k/v/bias inputs are
already-projected activations. :class:`TriangleAttentionReference` is the module form of
(2) and owns every projection as an ``nn.Parameter``, so a test can compare both the
forward output and the backward gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _acc_dtype(t: torch.Tensor) -> torch.dtype:
    """fp32 accumulation for bf16/fp16/fp32 data; fp64 inputs (gradcheck) keep fp64."""
    return torch.promote_types(t.dtype, torch.float32)


def triangle_attention_pair_bias_pytorch(
    q: torch.Tensor,  # (B, H, L, L, D) — axis 2 = row i, axis 3 = query token j
    k: torch.Tensor,  # (B, H, L, L, D) — axis 3 = key token k
    v: torch.Tensor,  # (B, H, L, L, D)
    bias: torch.Tensor,  # (B, H, L, L) — [b, h, j, k], broadcast over the row axis i
) -> torch.Tensor:
    """Pair-biased attention over axis 3, batched over ``(B, H)`` and the row axis.

    Returns ``(B, H, L, L, D)``. Differentiable in all four inputs.
    """
    d_head = q.shape[-1]
    scale = d_head**-0.5
    acc = _acc_dtype(q)

    logits = torch.einsum("bhijd,bhikd->bhijk", q.to(acc), k.to(acc)) * scale
    logits = logits + bias.to(acc)[:, :, None, :, :]

    # Flooring the row max at -1e38 is what turns a fully-masked row into zeros rather
    # than a uniform average: the finfo.min sentinel then underflows to exp(-inf) = 0.
    row_max = logits.amax(dim=-1, keepdim=True).clamp_min(-1e38)
    p = torch.exp(logits - row_max)
    denom = p.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    out = torch.einsum("bhijk,bhikd->bhijd", p.to(v.dtype).to(acc), v.to(acc))
    return (out / denom).to(v.dtype)


def triangle_attention_pytorch(
    pair: torch.Tensor,  # (B, L, L, d_pair)
    mask: torch.Tensor | None = None,  # (B, L) residue mask
    *,
    n_head: int,
    ln_pair_weight: torch.Tensor,  # (d_pair,)
    ln_pair_bias: torch.Tensor,  # (d_pair,)
    to_value_weight: torch.Tensor,  # (d_hidden, d_pair)
    to_bias_weight: torch.Tensor,  # (n_head, d_pair)
    to_gate_weight: torch.Tensor,  # (d_hidden, d_pair)
    to_out_weight: torch.Tensor,  # (d_pair, d_hidden)
    to_query_weight: torch.Tensor | None = None,  # (d_hidden, d_pair) — enables self-attn
    to_key_weight: torch.Tensor | None = None,  # (d_hidden, d_pair)
    norm_query_weight: torch.Tensor | None = None,  # (d_head,) — enables qk-norm
    norm_key_weight: torch.Tensor | None = None,  # (d_head,)
    starting: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Whole-op triangle attention in pure PyTorch. Returns ``(B, L, L, d_pair)``.

    Weight convention is ``nn.Linear``'s (``(out, in)``, applied as ``x @ weight.T``).
    ``use_self_attention`` is inferred from ``to_query_weight``/``to_key_weight`` and
    ``use_qk_norm`` from ``norm_query_weight``/``norm_key_weight``, as in the whole-op.
    """
    # Both weights, not just the query: the branch below uses to_key_weight too, so deriving the
    # flag from to_query_weight alone let `to_query_weight=W, to_key_weight=None` reach
    # F.linear(x, None). Testing the pair also narrows both to Tensor for a checker.
    use_self_attention = to_query_weight is not None and to_key_weight is not None
    if (to_query_weight is None) != (to_key_weight is None):
        msg = "to_query_weight and to_key_weight must be given together (self-attention) or both omitted"
        raise TypeError(msg)
    use_qk_norm = norm_query_weight is not None

    if not starting:
        pair = pair.transpose(1, 2).contiguous()

    x = F.layer_norm(pair, (pair.shape[-1],), ln_pair_weight, ln_pair_bias, eps)
    b, length, length2, _ = x.shape

    def to_heads(proj: torch.Tensor) -> torch.Tensor:
        """(B, L, L2, H*D) -> (B, H, L, L2, D)."""
        return proj.view(b, length, length2, n_head, -1).permute(0, 3, 1, 2, 4)

    value = to_heads(F.linear(x, to_value_weight))
    bias = F.linear(x, to_bias_weight).permute(0, 3, 1, 2)  # (B, L, L2, H) -> (B, H, L, L2)
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)

    if use_self_attention:
        query = to_heads(F.linear(x, to_query_weight))
        key = to_heads(F.linear(x, to_key_weight))
        if use_qk_norm:
            d_head = query.shape[-1]
            query = F.rms_norm(query, (d_head,), norm_query_weight, eps)
            key = F.rms_norm(key, (d_head,), norm_key_weight, eps)
        out = triangle_attention_pair_bias_pytorch(query, key, value, bias)
    else:
        # Bias-only path: softmax(bias) has no row axis, so it contracts straight
        # against the value's key axis.
        attention = F.softmax(bias, dim=-1, dtype=_acc_dtype(bias)).to(value.dtype)
        out = torch.einsum("bhjk,bhikd->bhijd", attention, value)

    out = out.permute(0, 2, 3, 1, 4).reshape(b, length, length2, -1)
    out = torch.sigmoid(F.linear(x, to_gate_weight)) * out
    out = F.linear(out, to_out_weight)

    if not starting:
        out = out.transpose(1, 2)
    return out


class TriangleAttentionPairBiasReference(nn.Module):
    """nn.Module form of the primitive attention kernel. Holds NO parameters.

    The kernel entry point ``triton_triangle_attention_pair_bias`` owns no weights: it
    consumes already-projected ``q``/``k``/``v`` and a pre-built pair ``bias``. The
    module exists only so the primitive can be driven through the same
    ``module(*inputs)`` interface as the weighted references; gradients flow to the
    four input tensors.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, H, L, L, D) x3`` + ``(B, H, L, L)`` -> ``(B, H, L, L, D)``."""
        return triangle_attention_pair_bias_pytorch(q, k, v, bias)


class TriangleAttentionReference(nn.Module):
    """nn.Module reference for the whole-op (forward + backward ground truth).

    Owns every weight the whole-op takes as an ``nn.Parameter``, named exactly like the
    whole-op's keyword arguments, so a test can feed ``ref.<name>`` straight into the
    kernel and compare both outputs and grads, e.g.::

        ref = TriangleAttentionReference(64, 4).cuda().to(torch.bfloat16)
        y = ref(pair, mask)
        y_k = triangle_attention(pair, mask, n_head=4, ln_pair_weight=ref.ln_pair_weight, ...)
    """

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        d_hidden = d_pair if d_hidden is None else d_hidden
        if d_hidden % n_head != 0:
            msg = f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        self.eps = eps

        def proj(out_features: int, in_features: int) -> nn.Parameter:
            w = torch.empty(out_features, in_features, device=device, dtype=dtype)
            nn.init.xavier_uniform_(w)
            return nn.Parameter(w)

        def ones(n: int) -> nn.Parameter:
            return nn.Parameter(torch.ones(n, device=device, dtype=dtype))

        self.ln_pair_weight = ones(d_pair)
        self.ln_pair_bias = nn.Parameter(torch.zeros(d_pair, device=device, dtype=dtype))
        self.to_value_weight = proj(d_hidden, d_pair)
        self.to_bias_weight = proj(n_head, d_pair)
        self.to_gate_weight = proj(d_hidden, d_pair)
        self.to_out_weight = proj(d_pair, d_hidden)

        self.to_query_weight = proj(d_hidden, d_pair) if use_self_attention else None
        self.to_key_weight = proj(d_hidden, d_pair) if use_self_attention else None
        qk_norm = use_self_attention and use_qk_norm
        self.norm_query_weight = ones(d_hidden // n_head) if qk_norm else None
        self.norm_key_weight = ones(d_hidden // n_head) if qk_norm else None

    def forward(
        self,
        pair: torch.Tensor,  # (B, L, L, d_pair)
        mask: torch.Tensor | None = None,  # (B, L) bool
    ) -> torch.Tensor:
        """Raw triangle-attention op (no residual, no dropout). -> ``(B, L, L, d_pair)``."""
        return triangle_attention_pytorch(
            pair,
            mask,
            n_head=self.n_head,
            ln_pair_weight=self.ln_pair_weight,
            ln_pair_bias=self.ln_pair_bias,
            to_value_weight=self.to_value_weight,
            to_bias_weight=self.to_bias_weight,
            to_gate_weight=self.to_gate_weight,
            to_out_weight=self.to_out_weight,
            to_query_weight=self.to_query_weight,
            to_key_weight=self.to_key_weight,
            norm_query_weight=self.norm_query_weight,
            norm_key_weight=self.norm_key_weight,
            starting=self.starting,
            eps=self.eps,
        )
