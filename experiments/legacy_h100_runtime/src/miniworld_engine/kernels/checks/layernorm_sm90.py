"""TMA LayerNorm checker consumes the kernel's saved FP32 statistics."""


def layernorm_bwd_split_sm90_cute():
    from miniworld_engine.kernels.drivers.layernorm_sm90 import inputs
    from miniworld_engine.kernels.layernorm.cute.tma_backward import backward_impl

    result = {}
    for layout in ("col", "row"):
        dy, x, w, mean, rs = inputs(layout=layout)
        dx, dw, db = backward_impl(dy, x, w, mean, rs, x.stride())
        xh = (x.float() - mean[:, None]) * rs[:, None]
        wd = dy.float() * w
        ref = (wd - ((wd * xh).mean(1)[:, None] * xh + wd.mean(1)[:, None])) * rs[
            :, None
        ]
        result.update(
            {
                layout + "_dx": (dx, ref),
                layout + "_dw": (dw, (dy.float() * xh).sum(0)),
                layout + "_db": (db, dy.float().sum(0)),
            }
        )
    return result
