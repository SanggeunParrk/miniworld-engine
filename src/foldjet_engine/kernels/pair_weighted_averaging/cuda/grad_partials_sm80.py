"""Unreduced slabs from existing CUDA PWA kernels for E113 opt-in fusion.

The kernel, tile, dropout and input layout stay unchanged. All returned
buffers are fresh; their original deterministic reduction is performed by
grad_reduce_fused_sm80 at the end of backward.
"""

import torch


from . import sm80


def glue(m, dres, o, lnw, lnb, wg, wo, keep, eps, dscale, tuned):
    rows = sm80.ext().bwd_grids(m.shape[0], m.shape[1], 64, 32)[0]
    do = torch.empty_like(o)
    dgp = torch.empty_like(o)
    partial = m.new_empty((rows, 4, 8192), dtype=torch.float32)
    keep_tensor = m.new_empty((0,), dtype=torch.bfloat16) if keep is None else keep
    if tuned:
        from . import glue_tuned_sm80

        glue_tuned_sm80._ext().glue(
            m,
            dres,
            o,
            lnw,
            lnb,
            wg,
            wo,
            keep_tensor,
            do,
            dgp,
            partial,
            eps,
            dscale,
            2 if False else 1,
        )
    else:
        sm80.ext().glue(m, dres, o, lnw, lnb, wg, wo, keep_tensor, do, dgp, partial, eps, dscale)
    return do, dgp, partial.view(rows, -1)


def dgv(m, dres, dgp, dv, lnw, lnb, wg, wv, eps):
    rows = sm80.ext().bwd_grids(m.shape[0], m.shape[1], 64, 32)[1]
    dm = torch.empty_like(m)
    partial = m.new_empty((rows, 16512), dtype=torch.float32)
    sm80.ext().dgv_bwd(m, dres, dgp, dv, lnw, lnb, wg, wv, dm, partial, eps)
    return dm, partial


def pair(z, w, dw, mask, lnw, lnb, wb, eps):
    mask_tensor = z.new_empty((0,), dtype=torch.uint8) if mask is None else mask
    if dw.dim() == 4:
        from . import pair_partial_sm80

        return pair_partial_sm80.pair(z, w, dw, mask_tensor, lnw, lnb, wb, eps)
    dz = torch.empty_like(z)
    pm = z.new_empty((z.shape[0], 1024), dtype=torch.float32)
    ps = z.new_empty((z.shape[0], 8), dtype=torch.float32)
    variant = 0
    sm80.ext().pair_bwd(z, w, dw, mask_tensor, lnw, lnb, wb, dz, pm, ps, eps, variant)
    return dz, pm, ps
