"""Selected H100 bidirectional training kernels, callable from installed modules.

Forward saves x_n, left/right, tri, output LN statistics, and packed weights. Backward
uses the selected CUDA B1/B7 bodies and four cuBLAS contractions. Every call
owns its saved tensors; there is no global activation or weight cache.
"""

import json
import torch
from torch.autograd.function import once_differentiable
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


def _data(leaves, mask, ds, *, packed=None, for_backward=False):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n = x.shape[1]
    # The same packed weights feed K1 and B7. Save this small, per-forward
    # tensor in autograd instead of rebuilding it during backward. It is never
    # cached across optimizer steps or shared between outstanding forwards.
    w1 = packed
    if w1 is None:
        w1 = x.new_empty((1024, 128))
        T.pack_into(w1, wl, wlg, wr, wrg)
    return dict(
        n=n,
        x=x,
        leaves=leaves,
        mask=mask.reshape(n, n) if for_backward else mask.reshape(n, n).float(),
        ds=ds.reshape(n, 128),
        wt=[w.t().contiguous() for w in (wl, wlg, wr, wrg, wg)]
        if for_backward else [],
        wp=wp,
        gi=gi,
        bi=bi,
        go=go,
        bo=bo,
        w1=w1,
    )


def _forward_fake(leaves, mask, ds):
    """Return output, activations, FP32 LN statistics and per-forward weight pack."""
    x = leaves[0]
    n = x.shape[1]
    d = x.shape[-1]
    return [
        torch.empty_like(x),
        x.new_empty((4 * d, n, n)),
        x.new_empty((2 * d, n, n)),
        torch.empty_like(x),
        x.new_empty((2 * n * n if d == 128 else 0,), dtype=torch.float32),
        x.new_empty((8 * d, d)),
        x.new_empty((0,) if d == 128 else (n, n), dtype=torch.float32),
    ]


@opaque(fake=_forward_fake, name="trimul_h100_train_fwd_prepared_allwidths")
def forward(
    leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor
) -> list[torch.Tensor]:
    """Run training forward with independently owned saved activations."""
    x = leaves[0]
    n = x.shape[1]
    D = x.shape[-1]
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        if D != 128:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import Training

            model = Training(*leaves, mask, ds, x.new_empty(x.shape))
            return [
                model.forward(),
                model.front.ab,
                model.tri,
                model.xn.reshape_as(x),
                x.new_empty((0,), dtype=torch.float32),
                model.w1,
                model.mask,
            ]
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_output as O

        d = _data(leaves, mask, ds)
        # Same K1 body/schedule as the selected training reference.
        ab, _ = O.front(d)
        tri = x.new_empty((256, n, n))
        torch.bmm(ab[:128], ab[256:384].transpose(-1, -2), out=tri[:128])
        torch.bmm(ab[128:256].transpose(-1, -2), ab[384:], out=tri[128:])
        y, s = O.output(d, tri, ln=3, stats=2, method=-1)
        return [y, ab, tri, s["xn"].reshape_as(x), s["ro"], d["w1"],
                x.new_empty((0,), dtype=torch.float32)]


def _backward_fake(leaves, mask, ds, saved, dy):
    """Match each differentiable input gradient shape and dtype."""
    grads = [torch.empty_like(t) for t in leaves]
    if leaves[0].shape[-1] == 128:
        # Return the common four-weight allocation once: custom ops prohibit
        # aliasing between outputs, even for disjoint views. Unbind outside the
        # custom op so autograd sees the views without any copies.
        front = leaves[0].new_empty((4, 128, 256))
        return [grads[0], front, *grads[5:]]
    return grads


@opaque(fake=_backward_fake, name="trimul_h100_train_bwd_prepared_allwidths")
def backward(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    """Run CUDA B1/B7 and contraction gradients using saved values."""
    x = leaves[0]
    n = x.shape[1]
    D = x.shape[-1]
    ab, tri, xn, stats, packed, prepared_mask = saved
    dy = dy.contiguous()
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        if D != 128:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import Training

            model = Training(*leaves, mask, ds, dy, saved=(ab, tri, xn),
                             packed=packed, prepared_mask=prepared_mask)
            return list(model.backward(dy))
        from miniworld_engine.kernels.trimul_inproj.cuda import (
            h100_b1 as B1,
            h100_b7 as B7,
        )

        d = _data(leaves, mask, ds, packed=packed, for_backward=True)
        cfg = json.loads((T.SOURCES / "b1/configs.json").read_text())[str(n)]
        b1 = B1.Plan(dict(d, x=xn), dy, tri, stats, **cfg)
        dg, dwg, dt, dgo, dbo, dwp = b1()
        dl = torch.empty_like(tri)
        dr = torch.empty_like(tri)
        torch.bmm(dt[:128], ab[256:384], out=dl[:128])
        torch.bmm(dt[:128].transpose(-1, -2), ab[:128], out=dr[:128])
        torch.bmm(ab[384:], dt[128:].transpose(-1, -2), out=dl[128:])
        torch.bmm(ab[128:256], dt[128:], out=dr[128:])
        b7 = B7.Plan(d, dy, dl, dr, dg, xn=xn)
        dx, dwl, dwlg, dwr, dwrg, dgi, dbi = b7()
        return [
            dx.reshape_as(x),
            b7.dw,
            # Output gate stays row-major for K3. Match its parameter layout
            # here rather than moving the copy into AccumulateGrad/optimizer.
            dwg.t().contiguous(),
            dwp,
            dgi,
            dbi,
            dgo,
            dbo,
        ]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        mask, ds = args[11:]
        y, *saved = forward(leaves, mask, ds)
        ctx.save_for_backward(*args, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = backward(list(vals[:11]), vals[11], vals[12], list(vals[13:]), dy)
        if vals[0].shape[-1] == 128:
            grads = [grads[0], *(w.t() for w in grads[1].unbind()), *grads[2:]]
        return (*grads, None, None)


def bidirectional_trimul(*args):
    return _Training.apply(*args)
