"""Selected H100 bidirectional training kernels, callable from installed modules.

D128 forward saves x_n, left/right, tri, output LN statistics, and packed weights;
its backward uses the selected CUDA B1/B7 bodies and four cuBLAS contractions.
D256/384/512 use ``h100_wide_training`` (its own saved set). Every call owns its
saved tensors; there is no global activation or weight cache.
"""

import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm as _H100_BMM
from torch.autograd.function import once_differentiable
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_training as WIDE


def _backward_transposes(leaves):
    """W^T of (W_l, W_lg, W_r, W_rg, W_g) for B1/B7. Row-major front weights are read in place by B7 (no copies; None);
    the column-major storage makes these transposes views."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_b7 import front_row_major

    from ._h100_pack import dense_native
    copy = dense_native if T.preserve_d128_training() else lambda w:w.contiguous()
    front = [None] * 4 if front_row_major(leaves) else [copy(w.t()) for w in leaves[1:5]]
    return front + [copy(leaves[5].t())]


def _data(leaves, mask, ds, *, packed=None, for_backward=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n = x.shape[1]
    # Batched samples (x [B, n, n, 128], B > 1): tokens b-major, rows = B n of the slow token axis; a token mask [B n], a pair mask
    # [B n, n] and the dropout scale [B n, 128] carry one block per sample (B = 1: the single-sample shapes).
    rows = x.shape[0] * n
    # The same packed weights feed K1 and B7. Save this small, per-forward
    # tensor in autograd instead of rebuilding it during backward. It is never
    # cached across optimizer steps or shared between outstanding forwards.
    w1 = packed
    if w1 is None:
        w1 = x.new_empty((1024, 128))
        T.pack_into(w1, wl, wlg, wr, wrg)
    if T.preserve_d128_training() and mask.ndim != 1 and not for_backward:
        from ._h100_pack import dense_native
        mask = dense_native(mask.reshape(rows,n),torch.float32)
    return dict(
        n=n,
        x=x,
        leaves=leaves,
        # A token mask [n] goes to K1 and B7 as is (they form m[i] & m[j]); a pair mask keeps the old contract.
        mask=mask if mask.ndim == 1 else mask.reshape(rows, n) if for_backward else mask.reshape(rows, n).float(),
        ds=ds.reshape(rows, 128),
        wt=_backward_transposes(leaves) if for_backward else [],
        wp=wp,
        gi=gi,
        bi=bi,
        go=go,
        bo=bo,
        w1=w1,
    )


def _fold(t, n):
    """[c, B n, n] -> [c B, n, n] (a view): the samples join the channel planes in cuBLAS's batch dim (B = 1: the same shape)."""
    return t.view(-1, n, n)


def _contract(ab, n, direction=0):
    """Planes [512, B n, n] -> tri [256, B n, n]: outgoing (left right^T) on the first 128 channels, incoming (left^T right) on the
    rest, every sample in the same two cuBLAS calls. direction 1 / 2 (a single-direction module with hidden 256 = 2 D: per token
    the bidirectional shapes, see ``bidirectional_trimul``): planes [left | right] (256 each), all 256 channels outgoing / incoming
    in one call."""
    tri = ab.new_empty((256, ab.shape[1], n))
    f = lambda t: _fold(t, n)
    if direction == 1:
        _H100_BMM.bmm(f(ab[:256]), f(ab[256:]).transpose(-1, -2), out=f(tri))
    elif direction == 2:
        _H100_BMM.bmm(f(ab[:256]).transpose(-1, -2), f(ab[256:]), out=f(tri))
    else:
        _H100_BMM.bmm(f(ab[:128]), f(ab[256:384]).transpose(-1, -2), out=f(tri[:128]))
        _H100_BMM.bmm(f(ab[128:256]).transpose(-1, -2), f(ab[384:]), out=f(tri[128:]))
    return tri


def _contract_grads(ab, dt, n, direction=0):
    """dl / dr (the gradients of the left / right planes) from dt, pairing the planes as ``_contract``."""
    dl = torch.empty_like(dt)
    dr = torch.empty_like(dt)
    f = lambda t: _fold(t, n)   # batched samples: every sample in the same cuBLAS calls
    if direction == 1:
        _H100_BMM.bmm(f(dt), f(ab[256:]), out=f(dl))
        _H100_BMM.bmm(f(dt).transpose(-1, -2), f(ab[:256]), out=f(dr))
    elif direction == 2:
        _H100_BMM.bmm(f(ab[256:]), f(dt).transpose(-1, -2), out=f(dl))
        _H100_BMM.bmm(f(ab[:256]), f(dt), out=f(dr))
    else:
        _H100_BMM.bmm(f(dt[:128]), f(ab[256:384]), out=f(dl[:128]))
        _H100_BMM.bmm(f(dt[:128]).transpose(-1, -2), f(ab[:128]), out=f(dr[:128]))
        _H100_BMM.bmm(f(ab[384:]), f(dt[128:]).transpose(-1, -2), out=f(dl[128:]))
        _H100_BMM.bmm(f(ab[128:256]), f(dt[128:]), out=f(dr[128:]))
    return dl, dr


USE_INFERENCE_K1 = True  # D128 front: the tuned inference K1 (same planes, accuracy unchanged, 94.6 -> 81 us at L384)


def _front(d):
    """Input LayerNorm + gated projections -> planes [512, n, n]; the inference K1 build when it is selected, else the training front."""
    if USE_INFERENCE_K1 and d["x"].shape[-1] == 128:
        from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as K
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_inference as INF

        n = d["n"]
        x = d["x"]
        bsz = x.shape[0]
        rows = bsz * n
        # batched samples: one K1 over the B samples stacked along the slow token axis (planes [512, B n, n]), as at D64
        ab = x.new_empty((512, rows, n))
        cfg = (2, 64, 4, 2) if n == 128 else K.TILE_TABLE[("sm_90a", 128, 256, "b")]["k1"]
        w = {"w1": d["w1"], "ln_in_w": d["gi"], "ln_in_b": d["bi"], "ln_out_w": d["go"], "ln_out_b": d["bo"], "wo": d["wp"],
             "wg": d["leaves"][5]}
        mask = d["mask"]
        INF._kernels().k1(x[0] if bsz == 1 else x.view(rows, n, 128), mask if mask.ndim == 1 else mask.reshape(rows, n).float(),
                          w, ab, N=n, Np=n, cz=128, ch=256, cfg=cfg, lnm=2, eps=1e-5, cache={}, token_mask=mask.ndim == 1,
                          rows=None if bsz == 1 else rows)
        return ab
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_output as O

    return O.front(d)[0]


def _forward_fake(leaves, mask, ds, direction=0):
    """Return output, activations, FP32 LN statistics and per-forward weight pack."""
    x = leaves[0]
    n = x.shape[1]
    d = x.shape[-1]
    if WIDE.supports(d, n):
        return [torch.empty_like(x), *WIDE.saved_like(x, selected=T.bidi_training(x))]
    rows = x.shape[0] * n   # batched samples: planes [c, B n, n], statistics [2 B n n]
    return [
        torch.empty_like(x),
        x.new_empty((4 * d, rows, n)),
        x.new_empty((2 * d, rows, n)),
        torch.empty_like(x),
        x.new_empty((2 * rows * n,), dtype=torch.float32),
        x.new_empty((8 * d, d)),
        x.new_empty((0,), dtype=torch.float32),
    ]


# Op names carry the saved-tensor contract: the wide port replaced the old wide saves,
# so previously compiled artifacts must not bind to these operators.
@opaque(fake=_forward_fake, name="trimul_h100_train_fwd_wide_port")
def forward(
    leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int = 0
) -> list[torch.Tensor]:
    """Run training forward with independently owned saved activations (direction: ``_contract``)."""
    x = leaves[0]
    n = x.shape[1]
    D = x.shape[-1]
    with T.native_context(x.device), T.preserved_d128_context(T.d128_training(x, bidirectional=True)), T.preserved_bidi_context(x):
        if WIDE.supports(D, n):
            return WIDE.forward(leaves, mask, ds)
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_output as O

        d = _data(leaves, mask, ds)
        # Same K1 body/schedule as the selected training reference.
        ab = _front(d)
        tri = _contract(ab, n, direction)
        y, s = O.output(d, tri, ln=3, stats=2, method=-1)
        return [y, ab, tri, s["xn"].reshape_as(x), s["ro"], d["w1"],
                x.new_empty((0,), dtype=torch.float32)]


def _forward_nograd_fake(leaves, mask, ds, direction=0):
    """y like x."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_h100_dropout_nograd_wide_port")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int = 0) -> torch.Tensor:
    """Preserve training dropout/residual without backward-only D128 saves."""
    x = leaves[0]
    n, width = x.shape[1], x.shape[-1]
    with T.native_context(x.device), T.preserved_d128_context(T.d128_training(x, bidirectional=True)), T.preserved_bidi_context(x):
        if WIDE.supports(width, n):
            return WIDE.forward(leaves, mask, ds)[0]
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_output as O
        d = _data(leaves, mask, ds)
        ab = _front(d)
        tri = _contract(ab, n, direction)
        return O.output(d, tri, ln=0, stats=0, method=-1)[0]


def _backward_fake(leaves, mask, ds, saved, dy, fp32_dw=False, native=False, direction=0):
    """Match each differentiable input gradient shape and dtype."""
    grads = [torch.empty_like(t, dtype=torch.float32 if fp32_dw and 1 <= i <= 6 else t.dtype) for i, t in enumerate(leaves)]
    if leaves[0].shape[-1] == 128:
        # Return the common four-weight allocation once: custom ops prohibit
        # aliasing between outputs, even for disjoint views. Unbind outside the
        # custom op so autograd sees the views without any copies.
        front = leaves[0].new_empty((4, 128, 256), dtype=torch.float32 if fp32_dw else leaves[0].dtype)
        return [grads[0], front, *grads[5:]]
    return grads


@opaque(fake=_backward_fake, name="trimul_h100_train_bwd_wide_port")
def backward(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    saved: list[torch.Tensor],
    dy: torch.Tensor, fp32_dw: bool = False, native: bool = False, direction: int = 0,
) -> list[torch.Tensor]:
    """Run CUDA B1/B7 and contraction gradients using saved values."""
    x = leaves[0]
    n = x.shape[1]
    D = x.shape[-1]
    if WIDE.supports(D, n):
        with T.native_context(x.device), T.preserved_bidi_context(x):
            return WIDE.backward(leaves, mask, ds, saved, dy, fp32_dw)
    ab, tri, xn, stats, packed, _ = saved
    dy = dy.contiguous()
    with T.native_context(x.device), T.preserved_d128_context(native and D == 128):
        from miniworld_engine.kernels.trimul_inproj.cuda import (
            h100_b1 as B1,
            h100_b7 as B7,
        )

        d = _data(leaves, mask, ds, packed=packed, for_backward=True)
        d["master_weights"] = fp32_dw
        configs = T.read_config("b1/configs.json")
        cfg = configs[str(n) if str(n) in configs else "384" if n <= 384 else "768"]
        cfg = dict(cfg)
        cfg["defines"] = dict(cfg.get("defines") or {}, MASTER_FP32=int(fp32_dw))
        b1 = B1.Plan(dict(d, x=xn), dy, tri, stats, **cfg)
        dg, dwg, dt, dgo, dbo, dwp = b1()
        dl, dr = _contract_grads(ab, dt, n, direction)
        b7 = B7.Plan(d, dy, dl, dr, dg, xn=xn)
        dx, dwl, dwlg, dwr, dwrg, dgi, dbi = b7()
        from ._h100_pack import dense_native
        return [
            dx.reshape_as(x),
            b7.dw,
            # Output gate stays row-major for K3. Match its parameter layout
            # here rather than moving the copy into AccumulateGrad/optimizer.
            dense_native(dwg.t()) if T.preserve_d128_training() else dwg.t().contiguous(),
            dwp,
            dgi,
            dbi,
            dgo,
            dbo,
        ]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, *args):
        leaves = list(args[:11])
        mask, ds = args[11:]
        ctx.direction = direction
        weights = leaves[1:7]
        ctx.master = any(w.dtype == torch.float32 for w in weights)
        ctx.native = T.preserve_training(leaves[0].shape[-1]) or T.d128_training(leaves[0], bidirectional=True) or T.bidi_training(leaves[0])
        if ctx.native:
            from ._h100_pack import cast_training_weights
            leaves = [leaves[0],*cast_training_weights(weights),*leaves[7:]]
        elif ctx.master:
            copies = [torch.empty_like(w, dtype=leaves[0].dtype) for w in weights]
            torch._foreach_copy_(copies, weights)
            leaves = [leaves[0], *copies, *leaves[7:]]
        y, *saved = forward(leaves, mask, ds, direction)
        ctx.saved_count = len(saved)
        ctx.save_for_backward(*args, *saved, *leaves[1:7])
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        end = 13 + ctx.saved_count
        leaves = [vals[0], *vals[end:], *vals[7:11]]
        grads = backward(leaves, vals[11], vals[12], list(vals[13:end]), dy, ctx.master, ctx.native, ctx.direction)
        if vals[0].shape[-1] == 128:
            grads = [grads[0], *(w.t() for w in grads[1].unbind()), *grads[2:]]
        if ctx.native:
            from ._h100_pack import native_training_grads
            grads = native_training_grads(grads,list(vals[:11]))
        else:
            grads = [g.to(t.dtype) for g, t in zip(grads, vals[:11], strict=True)]
        return (None, *grads, None, None)


def bidirectional_trimul(*args, direction=0):
    """(x [B, n, n, D], wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, mask, dropscale [B n, D]) -> y. direction 1 / 2: a single-direction
    D128 module with hidden 256 (2 D): the same kernels, the contraction pairs the planes per ``_contract``."""
    if not torch.is_grad_enabled():
        leaves = list(args[:11])
        if T.preserve_training(leaves[0].shape[-1]) or T.d128_training(leaves[0], bidirectional=True) or T.bidi_training(leaves[0]):
            from ._h100_pack import cast_training_weights
            leaves = [leaves[0], *cast_training_weights(leaves[1:7]), *leaves[7:]]
        elif any(w.dtype != leaves[0].dtype for w in leaves[1:7]):
            weights = [torch.empty_like(w, dtype=leaves[0].dtype) for w in leaves[1:7]]
            torch._foreach_copy_(weights, leaves[1:7])
            leaves = [leaves[0], *weights, *leaves[7:]]
        return forward_nograd(leaves, *args[11:], direction)
    return _Training.apply(direction, *args)
