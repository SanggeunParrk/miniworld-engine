"""Single-direction Hopper training with TMA/WGMMA and one contraction.

Derived from the engine's width-parametric CUDA training extension of Anthropic
primitives. Saves x_n, left/right, tri and the small packed weight tensor;
recomputes projections/gates in backward. Each call owns its activations.
"""

import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width_base as BASE
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda.h100_native import Front
from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import launch, tm, tuning
from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import pack_into


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
        assert (
            x.dtype == torch.bfloat16 and x.shape[0] == 1 and x.shape[1] == x.shape[2]
        )
        n = x.shape[1]
        D = x.shape[-1]
        H = D
        M = n * n
        assert D == 128 and n in (384, 768)
        self.D, self.n, self.M = D, n, M
        self.weights = (wl, wlg, wr, wrg)
        self.x, self.dy, self.ds = x, dy, ds
        self.mask = mask.float().contiguous()
        self.outgoing = outgoing
        self.ks, self.path = BASE.build(D, H)
        self.grid = torch.cuda.get_device_properties(
            x.device
        ).multi_processor_count * min(
            4,
            min(
                int(
                    k.unit.drv._unwrap(
                        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                        k.unit.drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                            k.unit.drv.d.CUfunction(int(k.handle)),
                            (128 * tuning(D)[0]),
                            (tuning(D)[2]),
                        ),
                    )
                )
                for k in self.ks.values()
            ),
        )
        self.w1 = saved[3] if saved else x.new_empty((4 * H, D))
        self.gi, self.bi = gi, bi
        self.front = Front(
            x[0], self.w1, self.mask, gi, bi, (2, 64, 8, 2, 1), saved=saved, hidden=H
        )
        self.xn = self.front.xn
        self.tri = saved[1] if saved else x.new_empty((H, n, n))
        self.y = torch.empty_like(x)
        self.dx = torch.empty_like(x)
        self.dg = x.new_empty((M, D))
        self.dt = torch.empty_like(self.tri)
        self.gp_all = x.new_empty((4, M, H))
        self.gp = list(self.gp_all.unbind())
        self.dw = [torch.empty_like(w) for w in self.weights]
        self.dwp = torch.empty_like(wp)
        self.dwg = torch.empty_like(wg)
        norm = x.new_empty((M, H))
        dp = x.new_empty((M, D))
        dn = x.new_empty((M, H))
        dxn = x.new_empty((M, D))
        self.dl = torch.empty_like(self.tri)
        self.dr = torch.empty_like(self.tri)
        self.tensors = [
            x,
            self.tri,
            dy,
            ds,
            self.y,
            self.xn,
            norm,
            dp,
            self.dg,
            dn,
            dxn,
            self.dx,
            self.dt,
            *self.gp,
            *self.dw,
            self.dwp,
            self.dwg,
            None,
        ]
        self.floats = [
            gi,
            bi,
            go,
            bo,
            self.mask,
            *[torch.empty(M, device=x.device) for _ in range(2)],
            torch.empty((tuning(D)[1], 5 * H * D + D * D), device=x.device),
            *[torch.empty(c, device=x.device) for c in (D, D, H, H)],
            torch.empty(32, device=x.device, dtype=torch.int64),
        ]
        maps = [
            tm(self.xn.reshape(M, D)),
            tm(wp),
            tm(wg),
            tm(norm),
            tm(dp),
            tm(self.dg),
            *[tm(g) for g in self.gp],
            *[tm(w) for w in self.weights],
            tm(dxn),
            tm(dn),
        ]
        L = T._launch_module()
        self.params = L.Struct([*maps, *self.tensors, *self.floats, M, n])
        t7 = self.tensors.copy()
        t7[22:24] = [self.dl, self.dr]
        maps7 = maps.copy()
        maps7[14:16] = [tm(self.dl.reshape(H, M)), tm(self.dr.reshape(H, M))]
        self.params7 = L.Struct([*maps7, *t7, *self.floats, M, n])
        self.maps = maps
        self.maps7 = maps7
        self.outputs = (self.dx, *self.dw, self.dwg, self.dwp, *self.floats[8:12])

    def forward(self):
        pack_into(self.w1, *self.weights)
        ab, _ = self.front()
        d = self.D
        h = d
        left, right = ab[:h], ab[h:]
        if self.outgoing:
            torch.bmm(left, right.transpose(-1, -2), out=self.tri)
        else:
            torch.bmm(left.transpose(-1, -2), right, out=self.tri)
        launch(self.ks["forward"], self.params, self.grid, D=self.D)
        return self.y

    def bind_dy(self, dy):
        self.tensors[2] = dy.contiguous()
        U = T._launch_module()
        self.params = U.Struct(
            [*self.maps, *self.tensors, *self.floats, self.M, self.n]
        )
        t7 = self.tensors.copy()
        t7[22:24] = [self.dl, self.dr]
        self.params7 = U.Struct([*self.maps7, *t7, *self.floats, self.M, self.n])

    def backward(self, dy=None):
        if dy is not None:
            self.bind_dy(dy)
        launch(self.ks["b1"], self.params, self.grid, D=self.D)
        ab = self.front.ab
        d = self.D
        h = d
        left, right = ab[:h], ab[h:]
        if self.outgoing:
            torch.bmm(self.dt, right, out=self.dl)
            torch.bmm(self.dt.transpose(-1, -2), left, out=self.dr)
        else:
            torch.bmm(right, self.dt.transpose(-1, -2), out=self.dl)
            torch.bmm(left, self.dt, out=self.dr)
        launch(self.ks["b7"], self.params7, self.grid, D=self.D)
        return self.outputs

    def __call__(self):
        return self.forward(), self.backward()


