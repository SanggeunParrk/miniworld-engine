"""Selected H100 bidirectional training kernels, callable from installed modules.

Forward saves x_n, left/right, tri, and tiny output LN statistics. Backward
uses the selected CUDA B1/B7 bodies and four cuBLAS contractions. Every call
owns its saved tensors; there is no global activation or weight cache.
"""

import json
import torch
from torch.autograd.function import once_differentiable
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


def _data(leaves, mask, ds):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n = x.shape[1]
    w1 = x.new_empty((1024, 128))
    T.pack_into(w1, wl, wlg, wr, wrg)
    return dict(
        n=n,
        x=x,
        leaves=leaves,
        mask=mask.reshape(n, n).float(),
        ds=ds.reshape(n, 128),
        wt=[w.t().contiguous() for w in (wl, wlg, wr, wrg, wg)],
        wp=wp,
        gi=gi,
        bi=bi,
        go=go,
        bo=bo,
        w1=w1,
    )


def _forward_fake(leaves, mask, ds):
    """Return output, left/right, tri, x_n, and FP32 output-LN statistics."""
    x = leaves[0]
    n = x.shape[1]
    d = x.shape[-1]
    return [
        torch.empty_like(x),
        x.new_empty((4 * d, n, n)),
        x.new_empty((2 * d, n, n)),
        torch.empty_like(x),
        x.new_empty((2 * n * n if d == 128 else 0,), dtype=torch.float32),
    ]


@opaque(fake=_forward_fake, name="trimul_prep_before_fwd")
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
                model.xn,
                x.new_empty((0,), dtype=torch.float32),
            ]
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_output as O

        d = _data(leaves, mask, ds)
        # Same K1 body/schedule as the selected training reference.
        ab, _ = O.front(d)
        tri = x.new_empty((256, n, n))
        torch.bmm(ab[:128], ab[256:384].transpose(-1, -2), out=tri[:128])
        torch.bmm(ab[128:256].transpose(-1, -2), ab[384:], out=tri[128:])
        y, s = O.output(d, tri, ln=3, stats=2, method=-1)
        return [y, ab, tri, s["xn"].reshape_as(x), s["ro"]]


def _backward_fake(leaves, mask, ds, saved, dy):
    """Match each differentiable input gradient shape and dtype."""
    return [torch.empty_like(t) for t in leaves]


@opaque(fake=_backward_fake, name="trimul_prep_before_bwd")
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
    ab, tri, xn, stats = saved
    dy = dy.contiguous()
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        if D != 128:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import Training

            model = Training(*leaves, mask, ds, dy, saved=(ab, tri, xn))
            return list(model.backward(dy))
        from miniworld_engine.kernels.trimul_inproj.cuda import (
            h100_b1 as B1,
            h100_b7 as B7,
        )

        d = _data(leaves, mask, ds)
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
            dwl.t().contiguous(),
            dwlg.t().contiguous(),
            dwr.t().contiguous(),
            dwrg.t().contiguous(),
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
        return (*grads, None, None)


def bidirectional_trimul(*args):
    return _Training.apply(*args)
