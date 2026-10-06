"""The AF3-style atom DiT block on B200 (sm_100a) and qualified H100 shapes.

H100 inference and training use the CUDA block for A in {1, 5}, N in {1024, 2048, 4096},
with contiguous bf16 activations and bf16 or fp32 parameters. Unsupported H100 atom
contracts raise instead of falling through to the composed module path. Training
writes attention gradients directly into the packed projection buffer; A1 fuses
dBias with dQ, while A5 retains each bias tile across augmentations with Hopper TMA.

One ``DiTBlock`` call at atom widths runs, in place of the module path,

    forward   cond_fwd -> pre_fwd -> pair bias (both layouts) -> attn_fwd -> post_fwd
    backward  tr_bwd_gate -> post_bwd -> attn_dkv / attn_dq / attn_dbias -> pair-bias backward -> pre_bwd -> cond_bwd,
              then the weight gradients as cuBLAS GEMMs on the activations those kernels write.

Both residuals are inside post_fwd (the block returns the stream, as DiTBlock does). Inference skips the saved activations.

``serves()`` is the whole gate: B200, the engine's kernel backend (implementation MINIWORLD or TRITON), bf16 single / cond /
pair (``compute_dtype`` None or bf16), the atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair = 16, transition n = 2,
no QK-norm), B == 1, any N, a [B, N] bool key mask or none, LayerNorm eps 1e-5. N that is not a multiple of 128 is padded
here (zero single / cond rows, the padded keys masked; the pair tensor is read in place) and the first N rows come back; the
key mask and the padding are folded into the pair bias (masked keys -1e4), so the attention kernels run unmasked. Anything
else keeps the module path. ``MINIWORLD_ATOM_DIT_SM100=0`` turns it off. A build or load failure warns once and keeps the
module path.

The block is two opaque ops, ``atom_dit_block_fwd`` and ``atom_dit_block_bwd`` (``kernels._compile.opaque``), so a
``torch.compile``d model keeps them in its graph and the call is served inside compiled training and inference as well as
eagerly. The ops take the 23 parameters as a list (``WEIGHTS``), not the module: packing them for the kernels (stacked
bf16 copies, the backward's transposed operands) happens inside the op and is reused while every weight's tensor object and
version are unchanged, except while a CUDA graph is captured (a hit would record no pack kernel, and every replay would keep
reading the weights of the capture-time step).

Parameters may be bf16 or fp32 (the kernels read bf16 copies); their gradients come back in their own dtype. Numerics and
timings: ``docs/gpus/b200/atom_dit/atom_dit.md``.
"""

from __future__ import annotations

import os
import warnings
import weakref
from types import SimpleNamespace

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque

DS_, DC_, DP_, NH_, DH_ = 128, 128, 16, 4, 32
EPS = 1e-5
BF = torch.bfloat16
_LOADED: set[int] = set()
_FAILED = False

#: The block's parameters, in the order the ops take and return them (every ``DiTBlock.named_parameters`` entry at atom widths).
WEIGHTS = (
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_scale.bias",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_query.weight",
    "attention.to_query.bias",
    "attention.to_key.weight",
    "attention.to_value.weight",
    "attention.ln_pair.weight",
    "attention.to_bias.weight",
    "attention.to_gate.weight",
    "attention.to_out.weight",
    "attention.to_scale.weight",
    "attention.to_scale.bias",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.to_bias.weight",
    "transition.expand_a.weight",
    "transition.expand_b.weight",
    "transition.squeeze.weight",
    "transition.to_scale.weight",
    "transition.to_scale.bias",
)
_GAMMA, _WBIAS = WEIGHTS.index("attention.ln_pair.weight"), WEIGHTS.index("attention.to_bias.weight")


@device_constant
def _kernels_ready(index: int) -> bool:
    """Build and load the sm_100a kernels on ``index`` once; False (after one warning) when a toolchain or driver problem keeps
    the module path. A constant to ``torch.compile``: evaluated while tracing, never inside the graph."""
    global _FAILED
    if _FAILED:
        return False
    if index in _LOADED:
        return True
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    try:
        sm100.load_all(index)
    except Exception as exc:  # a toolchain or driver problem keeps the module path
        if sm100._hopper(index):
            raise RuntimeError("Hopper atom DiT kernel build failed") from exc
        _FAILED = True
        warnings.warn(f"sm_100a atom DiT kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    _LOADED.add(index)
    return True


def serves(module, single, cond, pair, mask, compute_dtype=None) -> bool:
    qualified = _serves_hopper(module, single, cond, pair, mask, compute_dtype)
    if qualified is not None:
        return qualified
    if _FAILED or os.environ.get("MINIWORLD_ATOM_DIT_SM100", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or a.use_qk_norm:
        return False
    cap = torch.cuda.get_device_capability(single.device) if single.is_cuda else None
    hopper = cap == (9, 0) and os.environ.get("MINIWORLD_ATOM_DIT_H100", "0") == "1"
    if not (cap == (10, 0) or hopper):
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
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    return _kernels_ready(index)


def _serves_hopper(module, single, cond, pair, mask, compute_dtype):
    """The measured six-shape H100 contract for both modes; None leaves other families alone."""
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or not single.is_cuda:
        return None
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return None
    # Identify the atom family from its parameters, not potentially malformed inputs.
    if (a.to_query.weight.shape[1], a.to_bias.weight.shape[1]) != (DS_, DP_):
        return None
    weights = tuple(module.get_parameter(name) for name in WEIGHTS)

    def require(ok, detail):
        if not ok:
            raise NotImplementedError(f"H100 atom DiT requires {detail}; no composed fallback")

    require(settings.current().engine_backend != "triton", "the CUDA engine backend")
    require(not a.use_qk_norm, "use_qk_norm=False")
    require((a.n_head, a.to_query.weight.shape[0], a.ada_ln_in.to_scale.weight.shape[1],
             tr.expand_a.weight.shape[0]) == (NH_, DS_, DC_, 2 * DS_), "128/128/16 widths, four heads, transition n=2")
    lengths = (1024, 2048, 4096)
    if module.training and torch.is_grad_enabled() and os.environ.get("MINIWORLD_ATOM_DIT_H100_TRAIN_N8192") == "1":
        # Shape-extension candidate; inference retains its measured contract.
        lengths = (*lengths, 8192)
    require(single.ndim == 4 and single.shape[0] in (1, 5) and single.shape[1] == 1
            and single.shape[2] in lengths and single.shape[3] == DS_,
            f"single [A,1,N,128] with A in {{1,5}}, N in {lengths}")
    n = single.shape[2]
    require(tuple(cond.shape) == (*single.shape[:3], DC_) and tuple(pair.shape) == (1, n, n, DP_),
            "matching contiguous conditioning and dense pair shapes")
    require(compute_dtype in (None, BF), "bf16 compute")
    require(all(t.dtype == BF and t.device == single.device and t.is_contiguous() for t in (single, cond, pair)),
            "contiguous bf16 activations on one CUDA device")
    require(all(w.dtype in (BF, torch.float32) and w.device == single.device and w.is_contiguous() for w in weights),
            "contiguous bf16 or fp32 parameters on the input device")
    require(mask is None or (mask.dtype == torch.bool and tuple(mask.shape) == (1, n)
            and mask.device == single.device and mask.is_contiguous()), "a contiguous [1,N] bool mask or None")
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, tr.ada_ln_in.ln_in, tr.ada_ln_in.ln_cond)
    require(all(norm.eps == EPS for norm in norms), "LayerNorm eps=1e-5")
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    if not _kernels_ready(index):
        raise RuntimeError("H100 atom DiT CUDA kernels unavailable; no composed fallback")
    return True


# ------------------------------------------------------------------------------------------------------------ weight packs
def _namespace(weights) -> SimpleNamespace:
    """The attribute tree ``sm100_atom.cond_params`` / ``pre_params`` / ``post_params`` read (``block.attention.to_query.weight``)."""
    block = SimpleNamespace()
    for path, weight in zip(WEIGHTS, weights, strict=True):
        node = block
        *parents, leaf = path.split(".")
        for name in parents:
            if not hasattr(node, name):
                setattr(node, name, SimpleNamespace())
            node = getattr(node, name)
        setattr(node, leaf, weight)
    return block


def _pack(weights, backward=False):
    """The kernels' stacked bf16 weights and the backward's transposed A operands."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    block = _namespace(weights)
    if sm100._hopper(weights[0].device.index):
        return sm100.pack_hopper(weights, backward=backward)
    with torch.no_grad():
        wc, wp, wo = sm100.cond_params(block), sm100.pre_params(block), sm100.post_params(block)

        def transposed(t):
            return t.t().contiguous()

        wb = {"WsT": transposed(wo[2]), "WuT": transposed(wo[1]), "WoT": transposed(wo[0]), "WpT": transposed(wp[0]),
              "WmodT": transposed(wc[0])}
    return wc, wp, wo, wb


#: Packs keyed by the weights' tensor objects (held weakly), valid while every version is unchanged. An address reused by a new
#: tensor with the same shape and version must not be served the old tensor's pack, so identity -- not (pointer, version) -- decides.
_PACKS: dict = {}


def _weights(weights, backward=False):
    """Scoped to the CUDA-graph capture (``kernels._capture``): a capture packs once and records it."""
    try:
        versions = tuple(w._version for w in weights)
    except RuntimeError:  # inference tensors carry no version counter: nothing could invalidate an entry
        return _pack(weights, backward=backward)
    key = _capture.scoped((bool(backward), *(id(w) for w in weights)))
    if key is None:
        return _pack(weights, backward=backward)
    hit = _PACKS.get(key)
    if hit is not None:
        refs, seen, pack = hit
        if seen == versions and all(r() is w for r, w in zip(refs, weights, strict=True)):
            return pack
    _capture.prune(_PACKS)
    if len(_PACKS) >= 64:  # a model's worth of blocks; drop the oldest
        _PACKS.pop(next(iter(_PACKS)))
    pack = _pack(weights, backward=backward)
    _PACKS[key] = (tuple(weakref.ref(w) for w in weights), versions, pack)
    return pack


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


def _padded_length(n0: int) -> int:
    return (n0 + 127) // 128 * 128


# ------------------------------------------------------------------------------------------------------------ the two ops
def _fwd_fake(single, cond, pair, mask, weights, save):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    rows = a * n

    def tile(width=DS_):
        return single.new_empty((rows, width))

    return [out, tile(6 * DS_), tile(), tile(), tile(), tile(), tile(), single.new_empty((NH_, n, n)),
            single.new_empty((NH_, n, n)), tile(), single.new_empty((a, NH_, n), dtype=torch.float32), tile(), tile(), tile(), tile()]


@opaque(fake=_fwd_fake, name="atom_dit_block_fwd")
def _block_fwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
               weights: list[torch.Tensor], save: bool) -> list[torch.Tensor]:
    """The block output [A, 1, N, 128] bf16 and, when ``save``, the activations the backward reads (mod, q, k, v, sg, x1, bias,
    bias_t, O, LSE, u, a2, x2, t: all fresh tensors; the padded single / cond rows are not among them, the backward pads its own
    inputs again)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    rows = a * n
    keys = _keys(mask)
    with torch.cuda.device(single.device):
        wc, wp, wo, _ = _weights(weights, backward=save)
        s2 = _padded(single, n).reshape(rows, DS_).contiguous()
        c2 = _padded(cond, n).reshape(rows, DC_).contiguous()
        mod = sm100.cond_fwd(c2, *wc)
        q, k, v, sg, x1 = sm100.pre_fwd(s2, mod, *wp, save=save)
        bias, bias_t = sm100.pair_bias_fwd(pair, weights[_GAMMA], weights[_WBIAS], trans=save, n=n, kv=keys)
        o, lse = sm100.attn_fwd(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), bias)
        sm100.empty_uniform_hopper(keys, v, o, n)
        out, u, a2, x2, t = sm100.post_fwd(s2, o, sg, mod, *wo, save=save)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    return [out, mod, q, k, v, sg, x1, bias, bias_t, o, lse, u, a2, x2, t]


def _bwd_fake(dy, single, cond, pair, mask, weights, saved):
    return [torch.empty_like(single), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(w) for w in weights)]


@opaque(fake=_bwd_fake, name="atom_dit_block_bwd")
def _block_bwd(dy: torch.Tensor, single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
               weights: list[torch.Tensor], saved: list[torch.Tensor]) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per weight in ``WEIGHTS`` order, in the weight's dtype (all fresh tensors)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    mod, q, k, v, sg, x1, bias, bias_t, o, lse, u, a2, x2, t = saved
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    rows = a * n
    dev = single.device
    with torch.cuda.device(dev):
        hopper = sm100._hopper(dev.index)
        if hopper and (dy.dtype != BF or dy.device != dev or dy.shape != single.shape or not dy.is_contiguous()):
            raise NotImplementedError("H100 atom DiT backward requires contiguous bf16 upstream gradient matching single")
        wc, _, wo, wb = _weights(weights, backward=True)
        s2 = _padded(single, n).reshape(rows, DS_).contiguous()
        c2 = _padded(cond, n).reshape(rows, DC_).contiguous()
        dy = _padded(dy, n).reshape(rows, DS_).to(BF).contiguous()   # zero rows: the padded atoms contribute nothing
        dmod = torch.empty(rows, 6 * DS_, device=dev, dtype=BF)      # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]
        dp = torch.empty(rows, 4 * DS_, device=dev, dtype=BF)        # [dq | dk | dv | dg]
        allocate = torch.empty if hopper else torch.zeros
        dbias_cols = allocate(6 * DS_, device=dev)
        dbq = allocate(DS_, device=dev)
        dgamma_cond = allocate(2 * DS_, device=dev)
        dgparts = torch.empty((rows + 63) // 64, 2 * DS_, device=dev) if hopper else dgamma_cond
        dd = torch.empty(a, NH_, n, device=dev)
        dt, hh, dab = sm100.tr_bwd_gate(dy, t, mod, x2, wo[1], wb["WsT"], dmod, dbias_cols[640:])
        da, du, gated, do = sm100.post_bwd(dab, dy, a2, mod, u, sg, o, wb["WuT"], wb["WoT"], dmod, dp, dd, dbias_cols, n)
        db = sm100.attn_bwd(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), do.view(a, n, DS_), bias, bias_t, lse, dd, dp,
                            native_hopper=hopper)
        sm100.empty_uniform_hopper(_keys(mask), do, dp, n, value_gradient=True)
        dz, dgamma, dwb = sm100.pair_bias_bwd(pair, weights[_GAMMA], weights[_WBIAS], db, kv=_keys(mask), native_hopper=hopper)
        ds = sm100.pre_bwd(dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq)
        dc, cn1, cn2 = sm100.cond_bwd(dmod, c2, wb["WmodT"], wc[2], wc[3], dgparts)
        # weight gradients (cuBLAS)
        if hopper:
            sm100.finish_columns_hopper(dmod, dp, dgparts, dbias_cols, dbq, dgamma_cond)
            def wgrad(g, x):
                if os.environ.get("MINIWORLD_ATOM_DIT_H100_OWNED_WGRAD", "0") == "1":
                    return sm100.owned_wgrad_hopper(g, x)
                return torch.mm(g.t(), x, out_dtype=torch.float32)
            dws, dwu, dwo, dwp = wgrad(dt, hh), wgrad(dab, x2), wgrad(du, gated), wgrad(dp, x1)
            dw1, dw2, dw3 = wgrad(dmod[:, 0:256], cn1), wgrad(dmod[:, 256:512], cn2), wgrad(dmod[:, 512:768], c2)
        else:
            dws, dwu, dwo, dwp = dt.t() @ hh, dab.t() @ x2, du.t() @ gated, dp.t() @ x1
            dw1, dw2, dw3 = dmod[:, 0:256].t() @ cn1, dmod[:, 256:512].t() @ cn2, dmod[:, 512:768].t() @ c2
    grads = {
        "attention.ada_ln_in.ln_cond.weight": dgamma_cond[:128], "attention.ada_ln_in.to_scale.weight": dw1[:128],
        "attention.ada_ln_in.to_scale.bias": dbias_cols[0:128], "attention.ada_ln_in.to_bias.weight": dw1[128:],
        "attention.to_query.weight": dwp[0:128], "attention.to_query.bias": dbq, "attention.to_key.weight": dwp[128:256],
        "attention.to_value.weight": dwp[256:384], "attention.ln_pair.weight": dgamma, "attention.to_bias.weight": dwb,
        "attention.to_gate.weight": dwp[384:512], "attention.to_out.weight": dwo, "attention.to_scale.weight": dw3[:128],
        "attention.to_scale.bias": dbias_cols[512:640], "transition.ada_ln_in.ln_cond.weight": dgamma_cond[128:],
        "transition.ada_ln_in.to_scale.weight": dw2[:128], "transition.ada_ln_in.to_scale.bias": dbias_cols[256:384],
        "transition.ada_ln_in.to_bias.weight": dw2[128:], "transition.expand_a.weight": dwu[:256],
        "transition.expand_b.weight": dwu[256:], "transition.squeeze.weight": dws, "transition.to_scale.weight": dw3[128:],
        "transition.to_scale.bias": dbias_cols[640:768],
    }
    # an op returns fresh, unaliased tensors: the gradients are slices of shared buffers, so each is copied
    per_weight = (sm100.grad_pack_hopper([grads[name] for name in WEIGHTS], weights) if hopper else
                  [grads[name].to(w.dtype).reshape(w.shape).clone() for name, w in zip(WEIGHTS, weights, strict=True)])
    return [_unpadded(ds.view(a, 1, n, DS_), single.shape), _unpadded(dc.view(a, 1, n, DC_), cond.shape), dz.view(pair.shape),
            *per_weight]


# ------------------------------------------------------------------------------------------------------------ autograd
class _AtomBlock(torch.autograd.Function):
    """The whole block forward and backward on the sm_100a kernels; gradients for single, cond, pair and every weight."""

    @staticmethod
    def forward(ctx, mask, single, cond, pair, *weights):
        out, *saved = _block_fwd(single, cond, pair, mask, list(weights), True)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd(dy, single, cond, pair, mask, list(weights), list(saved))
        return (None, *grads)


def _infer(single, cond, pair, mask, weights):
    return _block_fwd(single, cond, pair, mask, list(weights), False)[0]


def block(module, single, cond, pair, mask=None):
    """The block's output stream (both residuals included), on the sm_100a kernels; ``serves()`` must have accepted the call."""
    weights = [module.get_parameter(name) for name in WEIGHTS]
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
        out = _AtomBlock.apply(mask, single, cond, pair, *weights)
        if torch.cuda.get_device_capability(single.device) == (9, 0):
            from .token_dit_train import _native_caller_output
            out = _native_caller_output(out, single, cond, pair, *weights, force=True)
        return out
    return _infer(single, cond, pair, mask, weights)
