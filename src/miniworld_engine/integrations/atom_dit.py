"""The AF3-style atom DiT block on B200 (sm_100a): inference and training through ``kernels/augmented_attention/cuda/sm100_atom``.

One ``DiTBlock`` call at atom widths runs, in place of the module path,

    forward   cond_fwd -> pre_fwd -> pair bias (both layouts) -> attn_fwd -> post_fwd
    backward  tr_bwd_gate -> post_bwd -> attn_dkv / attn_dq / attn_dbias -> pair-bias backward -> pre_bwd -> cond_bwd,
              then the weight gradients as cuBLAS GEMMs on the activations those kernels write.

Both residuals are inside post_fwd (the block returns the stream, as DiTBlock does). Inference skips the saved activations.

``serves()`` is the whole gate: B200, the engine's kernel backend (implementation MINIWORLD or TRITON), bf16 single / cond /
pair (``compute_dtype`` None or bf16), the atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair = 16, transition n = 2,
no QK-norm), B == 1, any N, a [B, N] bool key mask or none, LayerNorm eps 1e-5, eager (not under torch.compile or fake
tensors). N that is not a multiple of 128 is padded here (zero single / cond rows, the padded keys masked; the pair tensor is
read in place) and the first N rows come back; the key mask and the padding are folded into the pair bias (masked keys
-1e4), so the attention kernels run unmasked. Anything else keeps the module path. ``MINIWORLD_ATOM_DIT_SM100=0`` turns it off. A build or
load failure warns once and keeps the module path.

Parameters may be bf16 or fp32 (the kernels read bf16 copies, rebuilt when a parameter changes); their gradients come back
in their own dtype. Numerics and timings: ``docs/gpus/b200/atom_dit/atom_dit.md``.
"""

from __future__ import annotations

import os
import warnings
import weakref

import torch

from miniworld_engine import settings

DS_, DC_, DP_, NH_, DH_ = 128, 128, 16, 4, 32
EPS = 1e-5
BF = torch.bfloat16
_LOADED: set[int] = set()
_FAILED = False


def serves(module, single, cond, pair, mask, compute_dtype=None) -> bool:
    global _FAILED
    if _FAILED or os.environ.get("MINIWORLD_ATOM_DIT_SM100", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or a.use_qk_norm:
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF) or compute_dtype not in (None, BF):
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != DS_ or single.shape[2] == 0:
        return False
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (1, single.shape[2]) or mask.device != single.device):
        return False
    if tuple(cond.shape) != (*single.shape[:3], DC_) or tuple(pair.shape) != (1, single.shape[2], single.shape[2], DP_):
        return False
    if (a.n_head, a.to_query.weight.shape[0], a.to_bias.weight.shape[1], tr.expand_a.weight.shape[0]) != (NH_, DS_, DP_, 2 * DS_):
        return False
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, tr.ada_ln_in.ln_in, tr.ada_ln_in.ln_cond)
    if any(n.eps != EPS for n in norms):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling() or any(isinstance(t, FakeTensor) for t in (single, cond, pair)):
        return False
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    if index not in _LOADED:
        try:
            sm100.load_all(index)
        except Exception as exc:  # a toolchain or driver problem keeps the module path
            _FAILED = True
            warnings.warn(f"sm_100a atom DiT kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
            return False
        _LOADED.add(index)
    return True


_PACKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _weights(blk):
    """The kernels' stacked bf16 weights and the backward's transposed A operands, rebuilt when a parameter changes."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    key = tuple((p.data_ptr(), p._version) for p in blk.parameters())
    ent = _PACKS.get(blk)
    if ent is None or ent[0] != key:
        with torch.no_grad():
            wc, wp, wo = sm100.cond_params(blk), sm100.pre_params(blk), sm100.post_params(blk)
            def tr(t):
                return t.t().contiguous()
            wb = {"WsT": tr(wo[2]), "WuT": tr(wo[1]), "WoT": tr(wo[0]), "WpT": tr(wp[0]), "WmodT": tr(wc[0])}
        ent = (key, wc, wp, wo, wb)
        _PACKS[blk] = ent
    return ent[1:]


def _padded(t, n):
    """[A, 1, N, d] -> [A, 1, n, d], zero rows appended (the tensor itself when n == N)."""
    return t if t.shape[2] == n else torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[2]))


def _keys(mask):
    """The valid-key bytes [N] the pair-bias kernels read: the bool mask itself (None: every key valid; the kernels treat
    keys past N, the padding, as masked)."""
    return None if mask is None else mask[0].contiguous()


def _unpadded(t, shape):
    """[A, 1, n, d] (a view of the kernels' rows) -> the caller's [A, 1, N, d]."""
    return t if t.shape == shape else t[:, :, : shape[2]].contiguous()


def _infer(blk, single, cond, pair, mask):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    a = blk.attention
    A, _, N, _ = single.shape
    n = (N + 127) // 128 * 128
    kv = _keys(mask)
    M = A * n
    s2, c2 = _padded(single, n).reshape(M, DS_).contiguous(), _padded(cond, n).reshape(M, DC_).contiguous()
    wc, wp, wo, _ = _weights(blk)
    mod = sm100.cond_fwd(c2, *wc)
    q, k, v, sg, _ = sm100.pre_fwd(s2, mod, *wp)
    bias, _ = sm100.pair_bias_fwd(pair, a.ln_pair.weight, a.to_bias.weight, trans=False, n=n, kv=kv)
    O, _ = sm100.attn_fwd(q.view(A, n, DS_), k.view(A, n, DS_), v.view(A, n, DS_), bias)
    return _unpadded(sm100.post_fwd(s2, O, sg, mod, *wo)[0].view(A, 1, n, DS_), single.shape)


class _AtomBlock(torch.autograd.Function):
    """The whole block forward and backward on the sm_100a kernels; gradients for single, cond, pair and every parameter
    (``named_parameters`` order)."""

    @staticmethod
    def forward(ctx, blk, mask, single, cond, pair, *params):
        from miniworld_engine.kernels.augmented_attention.cuda import (
            sm100_atom as sm100,
        )

        a = blk.attention
        A, _, N0, _ = single.shape
        N = (N0 + 127) // 128 * 128                                    # the kernels' length (padded atoms are masked keys)
        kv = _keys(mask)
        M = A * N
        wc, wp, wo, _ = _weights(blk)
        s2, c2 = _padded(single, N).reshape(M, DS_).contiguous(), _padded(cond, N).reshape(M, DC_).contiguous()
        mod = sm100.cond_fwd(c2, *wc)
        q, k, v, sg, x1 = sm100.pre_fwd(s2, mod, *wp, save=True)
        bias, bias_t = sm100.pair_bias_fwd(pair, a.ln_pair.weight, a.to_bias.weight, trans=True, n=N, kv=kv)
        O, LSE = sm100.attn_fwd(q.view(A, N, DS_), k.view(A, N, DS_), v.view(A, N, DS_), bias)
        a3, u, a2, x2, t = sm100.post_fwd(s2, O, sg, mod, *wo, save=True)
        ctx.save_for_backward(s2, c2, pair, mod, q, k, v, sg, x1, bias, bias_t, O, LSE, u, a2, x2, t)
        ctx.blk, ctx.dims, ctx.shapes, ctx.kv = blk, (A, N), (single.shape, cond.shape), kv
        return _unpadded(a3.view(A, 1, N, DS_), single.shape)

    @staticmethod
    def backward(ctx, dy):
        from miniworld_engine.kernels.augmented_attention.cuda import (
            sm100_atom as sm100,
        )

        s2, c2, z, mod, q, k, v, sg, x1, bias, bias_t, O, LSE, u, a2, x2, t = ctx.saved_tensors
        blk = ctx.blk
        a = blk.attention
        A, N = ctx.dims
        M = A * N
        wc, _, wo, wb = _weights(blk)
        dev = s2.device
        dy = _padded(dy, N).reshape(M, DS_).to(BF).contiguous()       # zero rows: the padded atoms contribute nothing
        dmod = torch.empty(M, 6 * DS_, device=dev, dtype=BF)          # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]
        dP = torch.empty(M, 4 * DS_, device=dev, dtype=BF)            # [dq | dk | dv | dg]
        DBIAS = torch.zeros(6 * DS_, device=dev)
        DBQ = torch.zeros(DS_, device=dev)
        DG = torch.zeros(2 * DS_, device=dev)
        Dd = torch.empty(A, NH_, N, device=dev)
        DT, HH, DAB = sm100.tr_bwd_gate(dy, t, mod, x2, wo[1], wb["WsT"], dmod, DBIAS[640:])
        DA, DU, GATED, DO = sm100.post_bwd(DAB, dy, a2, mod, u, sg, O, wb["WuT"], wb["WoT"], dmod, dP, Dd, DBIAS, N)
        DB = sm100.attn_bwd(q.view(A, N, DS_), k.view(A, N, DS_), v.view(A, N, DS_), DO.view(A, N, DS_), bias, bias_t, LSE, Dd, dP)
        dz, dgam, dwb = sm100.pair_bias_bwd(z, a.ln_pair.weight, a.to_bias.weight, DB, kv=ctx.kv)
        DS = sm100.pre_bwd(dP, s2, mod, DA, wb["WpT"], dmod, DBIAS, DBQ)
        dc, cn1, cn2 = sm100.cond_bwd(dmod, c2, wb["WmodT"], wc[2], wc[3], DG)
        # weight gradients (cuBLAS)
        dWs = DT.t() @ HH
        dWu = DAB.t() @ x2
        dWo = DU.t() @ GATED
        dWp = dP.t() @ x1
        dW1, dW2, dW3 = dmod[:, 0:256].t() @ cn1, dmod[:, 256:512].t() @ cn2, dmod[:, 512:768].t() @ c2
        g = {"attention.ada_ln_in.ln_cond.weight": DG[:128], "attention.ada_ln_in.to_scale.weight": dW1[:128],
             "attention.ada_ln_in.to_scale.bias": DBIAS[0:128], "attention.ada_ln_in.to_bias.weight": dW1[128:],
             "attention.to_query.weight": dWp[0:128], "attention.to_query.bias": DBQ, "attention.to_key.weight": dWp[128:256],
             "attention.to_value.weight": dWp[256:384], "attention.ln_pair.weight": dgam, "attention.to_bias.weight": dwb,
             "attention.to_gate.weight": dWp[384:512], "attention.to_out.weight": dWo, "attention.to_scale.weight": dW3[:128],
             "attention.to_scale.bias": DBIAS[512:640], "transition.ada_ln_in.ln_cond.weight": DG[128:],
             "transition.ada_ln_in.to_scale.weight": dW2[:128], "transition.ada_ln_in.to_scale.bias": DBIAS[256:384],
             "transition.ada_ln_in.to_bias.weight": dW2[128:], "transition.expand_a.weight": dWu[:256],
             "transition.expand_b.weight": dWu[256:], "transition.squeeze.weight": dWs, "transition.to_scale.weight": dW3[128:],
             "transition.to_scale.bias": DBIAS[640:768]}
        pg = [g[n].to(p.dtype).view(p.shape) for n, p in blk.named_parameters()]
        sshape, cshape = ctx.shapes
        return (None, None, _unpadded(DS.view(A, 1, N, DS_), sshape), _unpadded(dc.view(A, 1, N, DC_), cshape), dz.view(z.shape), *pg)


def block(module, single, cond, pair, mask=None):
    """The block's output stream (both residuals included), on the sm_100a kernels; ``serves()`` must have accepted the call."""
    params = tuple(module.parameters())
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *params)):
        return _AtomBlock.apply(module, mask, single, cond, pair, *params)
    return _infer(module, single, cond, pair, mask)
