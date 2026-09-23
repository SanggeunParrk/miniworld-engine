import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class GP:
    def __init__(self, m, cfg=None, parameters_only=False):
        n = m.n
        D = m.D
        cfg = tuple(cfg or m.front.cfg)
        bi, bj, slots, sk, mb = cfg
        self.cfg = cfg
        if parameters_only:
            from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import gp_smem

            self.k, self.smem, self.path = None, gp_smem(D, cfg), None
        else:
            raise ValueError("Only the selected fused producer is packaged")
        U = T._launch_module()
        tm = lambda t, box, dims, strides: U.tensor_map(
            t, box, dims=dims, strides_bytes=strides, swizzle="128B", l2="128B"
        )
        mz = tm(m.xn, [64, bj, bi], [D, n, n], [D * 2, n * D * 2])
        mw = tm(m.w1, [64, 64], [D, 8 * D], [D * 2])
        ma = tm(m.front.ab, [64, 1, 32], [n, n, 4 * D], [n * 2, n * n * 2])
        tj = n // bj
        tiles = (n // bi) * tj
        self.params = U.Struct(
            [
                mz,
                mw,
                ma,
                m.mask,
                m.gi,
                m.bi,
                m.front.ab,
                None,
                None,
                n,
                n,
                tj,
                tiles,
                1,
                n,
                1,
                1e-5,
                n * D,
                D,
                0,
                0,
                *([0] * 8),
                tm(m.dl, [64, 1, 32], [n, n, 2 * D], [n * 2, n * n * 2]),
                tm(m.dr, [64, 1, 32], [n, n, 2 * D], [n * 2, n * n * 2]),
                tm(m.gp_all, [64, 1, 32], [n, n, 8 * D], [n * 2, n * n * 2]),
            ]
        )
        self.grid = min(
            tiles,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            * mb,
        )
        self.threads = 128 * (bi * bj // 64 + 1)

    def __call__(self):
        self.k.launch((self.grid, 1, 1), (self.threads, 1, 1), [self.params], self.smem)
