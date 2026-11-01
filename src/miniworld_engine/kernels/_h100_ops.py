"""H100 (sm_90) CUDA wiring of the functional ``ops.*`` facades (docs/records/h100_ops_gate_perf_20261101.md).

H100 / B200 launch no Triton through ``ops.*`` (``modules.dispatch.restricted_card``). A facade that has weights-as-args cannot call the
module integrations directly, so each function here is a thin wrapper: it builds what the module path builds (the packed weights, a
module-shaped view of the weights, the head-major layout of the core) and calls the hand-written CUDA kernels that already serve the op.
Every function answers ``None`` (or ``False``) when no CUDA kernel serves the call -- another card (B200 keeps the PyTorch statements),
a dtype / width / layout outside the kernel's contract, a CPU tensor, a ``torch.compile`` trace -- and the caller then runs the PyTorch
form, chunked deterministically where the statements would not fit in memory.

Nothing here launches Triton and nothing here is a new kernel.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F


def on_h100(*tensors: torch.Tensor) -> bool:
    """Real CUDA tensors on an sm_90 device, outside ``torch.compile`` tracing and fake-tensor mode (the raw extension calls below are
    not traceable): the precondition of every CUDA wrapper in this file."""
    t = tensors[0]
    if t.device.type != "cuda" or torch.compiler.is_compiling():
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if any(isinstance(x, FakeTensor) for x in tensors):
        return False
    index = t.device.index if t.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index) == (9, 0)


# ---------------------------------------------------------------------------------------------------------------- triangle attention
class _TriangleCore(torch.autograd.Function):
    """The sm_90 triangle-attention core (``h100_sm90``: forward, delta, key-side and query-side backward) as an autograd function of the
    projected activations: q, k, v [B, N, S, H, 32] bf16 (token-major: the layout the projections' GEMMs leave), bias [B, H, S, S] (any
    float dtype, masked keys at the dtype's minimum). No residual, gate or projection: those stay the facade's statements."""

    @staticmethod
    def forward(ctx, q, k, v, bias):  # noqa: D102
        from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as core

        scale = core.D**-0.5
        sbias = core.scaled_bias(bias, scale)
        need = any(ctx.needs_input_grad)
        o, lse, _, _ = core._core().triattn_fwd(q, k, v, sbias, scale, need)
        o = o.view(q.shape)
        if need:
            ctx.save_for_backward(q, k, v, sbias, o, lse)
            ctx.bias_dtype = bias.dtype
        return o

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as core

        q, k, v, sbias, o, lse = ctx.saved_tensors
        scale = core.D**-0.5
        bw = core._bwd()
        do = do.contiguous()
        delta = bw.triattn_delta(do, o)
        dk, dv = bw.triattn_bwd_kv(q, k, v, sbias, do, lse, delta, scale)
        B, _, S, H, _ = q.shape
        dbias = torch.zeros(B, H, S, S, device=q.device, dtype=torch.float32)  # the query-side kernel adds its partials in
        dq = core.bwd_q(bw, q, k, v, sbias, do, lse, delta, scale, dbias)
        return dq, dk, dv, dbias.to(ctx.bias_dtype)


def _triangle_core_cuda_ok(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> bool:
    """bf16, heads of 32 channels (at most 16 heads, as the module path), a square pair, sm_90."""
    B, H, L, L2, D = q.shape
    return (
        D == 32 and H <= 16 and L == L2 and bias.shape == (B, H, L, L)
        and all(t.dtype is torch.bfloat16 for t in (q, k, v)) and on_h100(q, k, v, bias)
    )


#: Bytes of fp32 attention scores one chunk of the PyTorch core may hold (one tensor; the softmax holds a few of them).
_SCORE_BUDGET = 4 << 30


def triangle_core_pytorch(q, k, v, bias):
    """The PyTorch core ``(B, H, L, L2, D)`` -> same, chunked over the row axis ``i`` so no chunk's score tensor passes ``_SCORE_BUDGET`` bytes
    (the one-shot form builds ``B H L L2 L2`` scores: 16 GiB at L = 1024, 4 heads). In training each chunk is checkpointed (its scores are
    recomputed in the backward, never kept), so memory stays one chunk's. The split is a function of the shapes alone: the result is
    bit-reproducible and independent of the free memory."""
    from miniworld_engine.kernels.triangle_attention.reference import (
        triangle_attention_pair_bias_pytorch as core,
    )

    B, H, L, L2, _ = q.shape
    acc = 8 if q.dtype is torch.float64 else 4
    rows = max(1, _SCORE_BUDGET // (B * H * L2 * L2 * acc))
    if rows >= L:
        return core(q, k, v, bias)
    from torch.utils.checkpoint import checkpoint

    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, bias))
    outs = []
    for start in range(0, L, rows):
        sl = slice(start, min(L, start + rows))
        if grad:
            outs.append(checkpoint(core, q[:, :, sl], k[:, :, sl], v[:, :, sl], bias, use_reentrant=False))
        else:
            outs.append(core(q[:, :, sl], k[:, :, sl], v[:, :, sl], bias))
    return torch.cat(outs, dim=2)


def triangle_core(q, k, v, bias):
    """Pair-biased attention over axis 3 of ``(B, H, L, L2, D)`` q / k / v (bias ``(B, H, L, L2)``, shared by the rows): the sm_90 CUDA core
    where it serves the call (bf16, 32-channel heads, square pair; L padded to 128 with masked keys), else the chunked PyTorch core.
    Returns ``(B, H, L, L2, D)`` (for the CUDA core a permuted view of the token-major result: ``rearrange`` to channels is free)."""
    if not _triangle_core_cuda_ok(q, k, v, bias):
        return triangle_core_pytorch(q, k, v, bias)
    S = q.shape[3]
    pad = -(-S // 128) * 128 - S
    q5, k5, v5 = (t.permute(0, 2, 3, 1, 4) for t in (q, k, v))  # [B, N, S, H, 32]
    q5, k5, v5 = (t if t.is_contiguous() else t.contiguous() for t in (q5, k5, v5))
    if pad:
        q5, k5, v5 = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q5, k5, v5))
        bias = F.pad(F.pad(bias, (0, pad), value=torch.finfo(bias.dtype).min), (0, 0, 0, pad))
    o = _TriangleCore.apply(q5, k5, v5, bias)
    if pad:
        o = o[:, :, :S]
    return o.permute(0, 3, 1, 2, 4)


# ---------------------------------------------------------------------------------------------------------------- TriMul
def _trimul_shim(x, bidirectional, outgoing, d_hidden, eps, ln_in_w, ln_in_b, wl, wlg, wr, wrg, ln_out_w, ln_out_b, w_out, w_gate):
    """A module-shaped view of the facade's weights (the attributes ``integrations.trimul_h100`` reads): the LayerNorm affine as fp32 (an exact
    cast of the usual fp32 parameters), the six projections as they are, ``outgoing`` only for the single-direction update (the integration
    tells the two modules apart by ``hasattr(module, "outgoing")``)."""
    from miniworld_engine.modules.exceptions import ImplementationType

    def lin(w):
        return types.SimpleNamespace(weight=w, bias=None)

    def norm(w, b):
        return types.SimpleNamespace(weight=w.float(), bias=b.float(), eps=eps)

    shim = types.SimpleNamespace(
        implementation=ImplementationType.MINIWORLD, d_hidden=d_hidden,
        ln_pair=norm(ln_in_w, ln_in_b), ln_out=norm(ln_out_w, ln_out_b),
        to_left=lin(wl), to_left_gate=lin(wlg), to_right=lin(wr), to_right_gate=lin(wrg), to_gate=lin(w_gate), to_out=lin(w_out),
    )
    if not bidirectional:
        shim.outgoing = outgoing
    shim.parameters = lambda: [
        wl, wlg, wr, wrg, w_gate, w_out, shim.ln_pair.weight, shim.ln_pair.bias, shim.ln_out.weight, shim.ln_out.bias
    ]
    return shim


def trimul(x, mask, bidirectional, outgoing, d_hidden, eps, ln_in_w, ln_in_b, wl, wlg, wr, wrg, ln_out_w, ln_out_b, w_out, w_gate):
    """``x + update(x)`` through the hand-CUDA TriMul of ``modules.TriangleMultiplication`` / ``BidirectionalTriangleMultiplication``
    (``integrations.trimul_h100``: K1 / K3 inference, B1..B7 training, chunked batches), bf16 only; None when the integration does not serve
    the call (the caller runs the PyTorch statements). ``mask``: None or a (B, L) bool residue mask."""
    if x.dtype is not torch.bfloat16 or x.ndim != 4 or not on_h100(x, wl, w_out):
        return None
    if any(t is None for t in (ln_in_w, ln_in_b, ln_out_w, ln_out_b)):
        return None
    if mask is not None and (mask.dtype is not torch.bool or mask.shape != x.shape[:2]):
        return None
    from miniworld_engine.integrations import trimul_h100 as h100

    x = x if x.is_contiguous() else x.contiguous()
    shim = _trimul_shim(x, bidirectional, outgoing, d_hidden, eps, ln_in_w, ln_in_b, wl, wlg, wr, wrg, ln_out_w, ln_out_b, w_out, w_gate)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (x, wl, wlg, wr, wrg, w_out, w_gate, ln_in_w, ln_in_b, ln_out_w, ln_out_b))
    if not grad:
        if h100.serves_inference(shim, x, bidirectional=bidirectional):
            return h100.update_inference(shim, x, mask, bidirectional=bidirectional)
        return None
    if (h100.serves(shim, x) if bidirectional else h100.serves_single(shim, x)):
        return h100.update(shim, x, mask, None, bidirectional=bidirectional)
    return None


# ---------------------------------------------------------------------------------------------------------------- SwiGLU / conditioned transition
def _tf32_or_bf16(x: torch.Tensor) -> bool:
    return x.dtype is torch.bfloat16 or (x.dtype is torch.float32 and torch.backends.cuda.matmul.allow_tf32)


class _SwiGLUFFN(torch.autograd.Function):
    """``Ws (silu(Wa x) * (Wb x))``: one cuBLAS GEMM over [Wa; Wb], the CUDA SwiGLU row kernel (``ct_rows``), one GEMM; backward the
    kernels of the conditioned-transition tail without its gate."""

    @staticmethod
    def forward(ctx, x, wa, wb, ws):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.ct_train import swiglu_fwd

        dt = x.dtype
        wab = torch.cat([wa, wb]).to(dt)
        wsb = ws.to(dt)
        ab = torch.mm(x, wab.t())
        h = swiglu_fwd(ab)
        out = torch.mm(h, wsb.t())
        ctx.save_for_backward(x, ab, h, wab, wsb)
        ctx.dtypes = (wa.dtype, wb.dtype, ws.dtype)
        return out

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.ct_train import swiglu_bwd

        x, ab, h, wab, wsb = ctx.saved_tensors
        dy = dy.contiguous()
        dws = torch.mm(dy.t(), h)
        dab = swiglu_bwd(torch.mm(dy, wsb), ab)
        dwab = torch.mm(dab.t(), x)
        dx = torch.mm(dab, wab)
        nd = ab.shape[1] // 2
        da, db, ds = ctx.dtypes
        return dx, dwab[:nd].to(da), dwab[nd:].to(db), dws.to(ds)


_GATE_ONE: dict = {}


def _gate_of_one(x: torch.Tensor):
    """(Wsc, bias) of a gate that is exactly one for any cond: zero weights and bias 30 (sigmoid(30) rounds to 1.0f). Cached per (device, dtype,
    width) so the weight-pack cache of ``ct_inference`` keeps one entry instead of one per call."""
    key = (x.device, x.dtype, x.shape[-1])
    got = _GATE_ONE.get(key)
    if got is None:
        d = x.shape[-1]
        got = _GATE_ONE[key] = (x.new_zeros(d, d), x.new_full((d,), 30.0))
    return got


def swiglu_ffn(x, wa, wb, ws):
    """``ops.swiglu_ffn`` on the CUDA kernels of the conditioned-transition tail, or None. Inference (bf16, or fp32 under TF32): the fused
    expand + SwiGLU GEMM and the squeeze GEMM of ``ct_inference`` (the hidden activation is the only HBM round trip) with a gate that is exactly
    one (``cond = x``, see ``_gate_of_one``). Otherwise (training, fp32 at the default matmul precision): ``_SwiGLUFFN``."""
    if not on_h100(x, wa, wb, ws) or x.dtype not in (torch.bfloat16, torch.float32) or x.shape[-1] % 8 or wa.shape[0] % 8:
        return None
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, wa, wb, ws)):
        return _SwiGLUFFN.apply(x2, wa, wb, ws).reshape(*shape[:-1], ws.shape[0])
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_inference as ct

    d = shape[-1]
    if _tf32_or_bf16(x) and ct.supported(x2, x2, wa.shape[0], d):
        wsc, bsc = _gate_of_one(x2)
        return ct.ct_inference(x2, x2, wa, wb, ws, wsc, bsc).reshape(*shape[:-1], ws.shape[0])
    return _SwiGLUFFN.apply(x2, wa, wb, ws).reshape(*shape[:-1], ws.shape[0])


def conditioned_transition(x, cond, wa, wb, ws, wsc, bsc):
    """``sigmoid(cond Wsc^T + b) * (SwiGLU tail of x)`` on the hand-CUDA conditioned-transition tail (``ct_inference`` / ``ct_tail_train``), or None."""
    if not on_h100(x, cond, wa, wb, ws, wsc) or not _tf32_or_bf16(x) or x.dtype is not cond.dtype:
        return None
    d, dc = x.shape[-1], cond.shape[-1]
    x2, c2 = x.reshape(-1, d), cond.reshape(-1, dc)
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_inference as ci
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_train as ct

    tensors = (x, cond, wa, wb, ws, wsc, bsc)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        if not ct.supported(x2, c2, wa.shape[0]):
            return None
        return ct.ct_tail_train(x2, c2, wa, wb, ws, wsc, bsc).reshape(*x.shape[:-1], ws.shape[0])
    if not ci.supported(x2, c2, wa.shape[0], d):
        return None
    return ci.ct_inference(x2, c2, wa, wb, ws, wsc, bsc).reshape(*x.shape[:-1], ws.shape[0])


# ---------------------------------------------------------------------------------------------------------------- gates
class _GatedLinear(torch.autograd.Function):
    """``Linear(sigmoid(gate) * value)``: the CUDA sigmoid-gate row kernels of the conditioned-transition tail around one GEMM."""

    @staticmethod
    def forward(ctx, gate, value, weight, bias):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.ct_train import sigmul_fwd

        u = sigmul_fwd(value, gate, None)
        out = F.linear(u, weight, bias)
        ctx.save_for_backward(gate, value, weight, u)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.ct_train import sigmul_bwd

        gate, value, weight, u = ctx.saved_tensors
        dy = dy.contiguous()
        du = torch.mm(dy, weight)
        dvalue, dgate, _ = sigmul_bwd(du, value, gate)
        dw = torch.mm(dy.t(), u)
        return dgate, dvalue, dw, (dy.sum(0) if ctx.has_bias else None)


def gated_linear(gate, value, weight, bias):
    """``F.linear(sigmoid(gate) * value, weight, bias)`` for bf16 rows through the CUDA gate kernels, or None."""
    if not on_h100(gate, value, weight) or value.dtype is not torch.bfloat16 or value.shape[-1] % 8 or not value.is_contiguous() or not gate.is_contiguous():
        return None
    shape = value.shape
    out = _GatedLinear.apply(gate.reshape(-1, shape[-1]), value.reshape(-1, shape[-1]), weight, bias)
    return out.reshape(*shape[:-1], weight.shape[0])


class _GatedResidual(torch.autograd.Function):
    """``x + gate * branch``: one ``addcmul`` kernel forward; backward the two products (autograd's ``addcmul`` backward runs four kernels)."""

    @staticmethod
    def forward(ctx, x, gate, branch):  # noqa: D102
        ctx.save_for_backward(gate, branch)
        return torch.addcmul(x, gate, branch)

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        gate, branch = ctx.saved_tensors
        return g, (g * branch if ctx.needs_input_grad[1] else None), (g * gate if ctx.needs_input_grad[2] else None)


def gated_residual(x, gate, branch):
    """``x + gate * branch`` as one fused elementwise kernel (``addcmul``: one launch, one rounding) instead of a multiply and an add."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gate, branch)):
        return _GatedResidual.apply(x, gate, branch)
    return torch.addcmul(x, gate, branch)


# ---------------------------------------------------------------------------------------------------------------- layer_norm_linear
class _LayerNormProject(torch.autograd.Function):
    """``Linear(LayerNorm(x) * w)`` for d_pair 128 -> 16 heads on the token DiT's CUDA pair-bias kernels (``token_dit_train_rows.cu``): one
    pass reads the pair rows (LayerNorm statistics, the projection on TF32 tensor cores, the head-major result written once, LN(x) never
    stored); the backward is one pass too (the LayerNorm backward and the weight-gradient partials), summed in a fixed order."""

    @staticmethod
    def forward(ctx, x2, ln_weight, proj_weight, eps, side):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

        R, d = x2.shape
        h = proj_weight.shape[0]
        wf = (proj_weight.float() * ln_weight.float()[None, :]).contiguous()
        bias = torch.empty(h, side, side, device=x2.device, dtype=x2.dtype)
        pst = torch.empty(R, 2, device=x2.device, dtype=torch.float32)
        ext().pair_bias(x2, wf, None, bias, pst, side, float(eps), None, 1.0)
        ctx.save_for_backward(x2, pst, wf, ln_weight, proj_weight)
        ctx.side = side
        return bias.view(h, R).t()

    @staticmethod
    def backward(ctx, dout):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

        x2, pst, wf, ln_weight, proj_weight = ctx.saved_tensors
        T = ext()
        R, d = x2.shape
        h = proj_weight.shape[0]
        db = dout.t().contiguous()  # [h, R]
        dz = torch.empty_like(x2)
        part = torch.empty(T.pair_grid(R), h * d, device=x2.device, dtype=torch.float32)
        T.pair_bias_bwd(db, x2, pst, wf, dz, part)
        dwf = part.sum(0).view(h, d)
        dproj = dwf * ln_weight.float()[None, :]
        dln = (dwf * proj_weight.float()).sum(0)
        return dz, dln.to(ln_weight.dtype), dproj.to(proj_weight.dtype), None, None


def layer_norm_linear_serves(x, ln_weight, proj_weight) -> bool:
    """d_pair 128 -> 16 heads, bf16 / fp32, rows forming a square (one pair of side * side rows), sm_90."""
    if (
        x.shape[-1] != 128 or proj_weight.shape != (16, 128) or ln_weight.shape != (128,)
        or x.dtype not in (torch.bfloat16, torch.float32) or not on_h100(x, ln_weight, proj_weight)
    ):
        return False
    rows = x.numel() // 128
    return math.isqrt(rows) ** 2 == rows and x.stride(-1) == 1


def layer_norm_linear(x, ln_weight, proj_weight, eps):
    """``Linear(LayerNorm(x))`` (d_pair 128 -> 16, no biases, a pair of side * side rows) on the CUDA pair-bias kernels, or None."""
    if not layer_norm_linear_serves(x, ln_weight, proj_weight):
        return None
    rows = x.numel() // 128
    side = math.isqrt(rows)
    x2 = x.reshape(rows, 128)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (x, ln_weight, proj_weight))
    if not grad and x.dtype is torch.bfloat16 and side % 16 == 0:
        # no gradient, bf16: the pair-bias kernel of the AttentionPairBias stack (``apb_rows.cu``: LayerNorm + the bf16 projection with the LayerNorm
        # weight folded into the projection weight, head-major result)
        from miniworld_engine.kernels.augmented_attention.cuda import apb

        wf = (proj_weight.float() * ln_weight.float()[None, :]).to(torch.bfloat16).contiguous()
        bias = apb.pair_bias(x2 if x2.is_contiguous() else x2.contiguous(), wf, None, side, float(eps), 0.0)
        return bias.view(16, rows).t().view(*x.shape[:-1], 16)
    out = _LayerNormProject.apply(x2, ln_weight, proj_weight, eps, side)
    return out.view(*x.shape[:-1], 16)


# ---------------------------------------------------------------------------------------------------------------- rms_norm_modulation
_ONE_ZERO: dict = {}


def _one_plus_bias(d: int, like: torch.Tensor) -> torch.Tensor:
    """The GEMM bias [1 .. 1 | 0 .. 0 | 0 .. 0] (3 d) of ``rms_norm_modulation``: the projection leaves ``1 + scale``. Cached per (device, dtype, d)."""
    key = (like.device, like.dtype, d)
    got = _ONE_ZERO.get(key)
    if got is None:
        got = _ONE_ZERO[key] = torch.cat([like.new_ones(d), like.new_zeros(2 * d)])
    return got


def rms_norm_modulation(q, c, w_scale, w_shift, w_gate, weight, eps):
    """``(rmsnorm(q) * (1 + c Ws^T) + c Wsh^T, c Wg^T)`` in the activation's dtype with as few launches as the statements allow: one RMSNorm, one
    GEMM over [Ws; Wsh; Wg] whose bias epilogue adds the 1 of ``1 + scale`` (fp32 accumulation), one fused ``addcmul``. Replaces the fp32 statements
    of ``rmsnorm_adamod_reference`` (a dozen launches and fp32 temporaries of the whole activation). The gate comes back as a view of the GEMM result."""
    dt = q.dtype
    d = q.shape[-1]
    w_all = torch.cat([w_scale, w_shift, w_gate]).to(dt)
    proj = F.linear(c.to(dt), w_all, _one_plus_bias(d, q))
    scale1, shift, gate = proj[..., :d], proj[..., d : 2 * d], proj[..., 2 * d :]
    normed = F.rms_norm(q, (d,), None if weight is None else weight.to(dt), eps)
    return torch.addcmul(shift, normed, scale1), gate


# ---------------------------------------------------------------------------------------------------------------- augmented attention
def augmented_attention_sdpa(query, key, value, bias, mask):
    """Pair-biased attention for the calls the sm_90 core does not take (any dtype, head width and length): ``(A, B, H, L, D)`` q / k / v,
    bias ``(B, H, L, L)``, key mask ``(A, B, L)``, through ``F.scaled_dot_product_attention`` (memory-efficient kernel with an additive
    bias; fp32 accumulation, the scores are never materialised). Masked keys carry the dtype's minimum (a fully masked row averages uniformly
    where the reference returns zeros: no row of the model's masks is fully masked)."""
    A, B, H, L, D = query.shape
    add = bias.unsqueeze(0).expand(A, B, H, L, L)
    if mask is not None:
        add = torch.where(mask[:, :, None, None, :], add, torch.finfo(add.dtype).min)
    q, k, v = (t.reshape(A * B, H, L, D) for t in (query, key, value))
    add = add.reshape(A * B, H, L, L).to(query.dtype)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value, bias))
    if not grad and query.dtype is torch.bfloat16 and on_h100(query):
        # no gradient: the cuDNN attention (1.3 - 1.6x the memory-efficient kernel at bf16; it has no gradient for an additive bias), the
        # memory-efficient kernel when cuDNN refuses the call
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=add).reshape(A, B, H, L, D)
        except RuntimeError:
            pass
    return F.scaled_dot_product_attention(q, k, v, attn_mask=add).reshape(A, B, H, L, D)


def rms_norm_rows(x, weight, eps):
    """RMSNorm over the last axis of a contiguous activation on the general CUDA RMSNorm (``kernels/rmsnorm/cuda``: any width, bf16 / fp16 / fp32,
    fp32 statistics, forward and backward with a bit-reproducible weight gradient), or None. The triangle-attention QK norm (rows of 32 channels)."""
    if not on_h100(x) or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float32):
        return None
    from miniworld_engine.kernels.layernorm.cuda.general import supported
    from miniworld_engine.kernels.rmsnorm.cuda import rmsnorm_cuda

    if not supported(x):
        return None
    return rmsnorm_cuda(x, weight, eps)
