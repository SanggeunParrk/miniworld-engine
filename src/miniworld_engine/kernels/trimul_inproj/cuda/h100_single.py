"""Single-direction H100 autograd: native K1/K3, fused B1 and streamed B7.

Save x_n, left/right, tri and packed weights. Projections/gates and output LN
are recomputed on chip by backward. All activations belong to one forward.
"""

import json
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda.h100_native import Front
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_output import (
    Output,
    Plan as Back,
)
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_b7 import Plan as FrontBack


@lru_cache(None)
def selection(n):
    return json.loads((T.SOURCES / "single_output/selection.json").read_text())[str(n)]


class Plan:
    def __init__(
        self,
        x,
        wl,
        wlg,
        wr,
        wrg,
        wg,
        wp,
        gi,
        bi,
        go,
        bo,
        mask,
        ds,
        dy,
        saved=None,
        outgoing=True,
    ):
        self.x = x
        self.n = x.shape[1]
        self.outgoing = outgoing
        self.dy = dy
        n = self.n
        if (
            x.shape != (1, n, n, 128)
            or n not in (384, 768)
            or x.dtype != torch.bfloat16
        ):
            raise ValueError("single H100 training requires BF16 B1/D128/L384 or L768")
        self.config = selection(n)
        self.weights = (wl, wlg, wr, wrg)
        self.mask = mask.reshape(n, n).bfloat16().contiguous()
        self.ds = ds.reshape(n, 128)
        self.w1 = saved[3] if saved else x.new_empty((512, 128))
        self.tri = saved[1] if saved else x.new_empty((128, n, n))
        self.gi, self.bi, self.go, self.bo = gi, bi, go, bo
        self.wg, self.wp = wg, wp
        self.front = None
        if saved is None:
            self.front = Front(
                x[0],
                self.w1,
                self.mask.float(),
                gi,
                bi,
                tuple(self.config["k1"]),
                hidden=128,
            )
            self.ab = self.front.ab
            self.xn = self.front.xn
            self.output = Output(
                x,
                self.tri,
                wp,
                wg,
                gi,
                bi,
                go,
                bo,
                self.ds,
                cfg=tuple(self.config["k3"]),
            )
        else:
            self.ab = saved[0]
            self.xn = saved[2]
        self.back = None
        self.front_back = None

    def forward(self):
        T.pack_into(self.w1, *self.weights)
        self.front()
        left, right = self.ab[:128], self.ab[128:]
        if self.outgoing:
            torch.bmm(left, right.transpose(-1, -2), out=self.tri)
        else:
            torch.bmm(left.transpose(-1, -2), right, out=self.tri)
        return self.output()

    def backward(self, dy=None):
        if dy is not None:
            self.dy = dy.contiguous()
        self.back = Back(
            self.x,
            self.xn,
            self.tri,
            self.wp,
            self.wg,
            self.go,
            self.bo,
            self.ds,
            self.dy,
            **self.config["b1"],
        )
        dg, dwg, dt, dgo, dbo, dwp = self.back.backward()
        dl = torch.empty_like(dt)
        dr = torch.empty_like(dt)
        left, right = self.ab[:128], self.ab[128:]
        if self.outgoing:
            torch.bmm(dt, right, out=dl)
            torch.bmm(dt.transpose(-1, -2), left, out=dr)
        else:
            torch.bmm(right, dt.transpose(-1, -2), out=dl)
            torch.bmm(left, dt, out=dr)
        d = dict(
            n=self.n,
            x=self.x,
            mask=self.mask,
            ds=self.ds,
            w1=self.w1,
            wt=[w.t().contiguous() for w in (*self.weights, self.wg)],
            gi=self.gi,
            bi=self.bi,
        )
        self.front_back = FrontBack(
            d, self.dy, dl, dr, dg, xn=self.xn, **self.config["b7"]
        )
        dx, *_, dgi, dbi = self.front_back()
        self.outputs = (
            dx.reshape_as(self.x),
            self.front_back.dw,
            dwg,
            dwp,
            dgi,
            dbi,
            dgo,
            dbo,
        )
        return self.outputs


def _forward_fake(leaves, mask, ds, outgoing):
    x = leaves[0]
    n, d = x.shape[1], x.shape[-1]
    return [
        torch.empty_like(x),
        x.new_empty((2 * d, n, n)),
        x.new_empty((d, n, n)),
        torch.empty_like(x),
        x.new_empty((4 * d, d)),
    ]


@opaque(fake=_forward_fake, name="trimul_h100_single_stream_train_fwd")
def forward(
    leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, outgoing: bool
) -> list[torch.Tensor]:
    x = leaves[0]
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        plan = Plan(*leaves, mask, ds, x, outgoing=outgoing)
        y = plan.forward()
        return [y, plan.ab, plan.tri, plan.xn.reshape_as(x), plan.w1]


def _backward_fake(leaves, mask, ds, saved, dy, outgoing):
    return [
        torch.empty_like(leaves[0]),
        leaves[0].new_empty((4, 128, 128)),
        *[torch.empty_like(v) for v in leaves[5:]],
    ]


@opaque(fake=_backward_fake, name="trimul_h100_single_stream_train_bwd")
def backward(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    outgoing: bool,
) -> list[torch.Tensor]:
    x = leaves[0]
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        plan = Plan(*leaves, mask, ds, dy.contiguous(), saved=saved, outgoing=outgoing)
        return list(plan.backward())


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outgoing, *args):
        y, *saved = forward(list(args[:11]), args[11], args[12], outgoing)
        ctx.outgoing = outgoing
        ctx.save_for_backward(*args, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        grads = backward(list(v[:11]), v[11], v[12], list(v[13:]), dy, ctx.outgoing)
        grads = [grads[0], *(w.t() for w in grads[1].unbind()), *grads[2:]]
        return (None, *grads, None, None)


def single_trimul(outgoing, *args):
    return _Training.apply(outgoing, *args)
