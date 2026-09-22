"""Audited saved-training adapter: live compiled packing, original gate weight.

Does not modify the independent K1/K3 implementation or production dispatch.
All layout conversions still run on every call (including graph replay).
"""
import torch
from types import SimpleNamespace
from compare_cueq_training import C, T, B


def _pack(wl, wlg, wr, wrg, wg):
    transposed = tuple(q.t().contiguous() for q in (wl, wlg, wr, wrg, wg))
    gate, proj = torch.cat((wlg, wrg), 0), torch.cat((wl, wr), 0)
    packed = torch.stack((gate.reshape(-1, 32, 128),
                          proj.reshape(-1, 32, 128)), 1).reshape(1024, 128)
    # A single output allocation lets Inductor combine the six layout writes.
    # These views retain the original contiguous shapes and 128-byte alignment.
    values = (*transposed, packed)
    flat = torch.cat(tuple(t.reshape(-1) for t in values))
    chunks = flat.split(tuple(t.numel() for t in values))
    return tuple(chunk.reshape_as(t) for chunk,t in zip(chunks, values))


pack = torch.compile(_pack, fullgraph=True, dynamic=False,
                     options={'triton.cudagraphs': False})


def forward(d, k1=None, k3=None):
    n, x = d['n'], d['x']
    k1 = k1 or ((3, 64, 2, 2, 1) if n == 384 else (1, 128, 2, 2, 1))
    k3 = k3 or T.default_config(256)
    packed = pack(*d['leaves'][1:6])
    d['wt'], d['w1'] = packed[:5], packed[5]
    xn, mu, rs, ab, pre = C.front(x, d['w1'], d['mask'], d['gi'], d['bi'],
                                True, k1, (1, 1))
    lf, rf = ab[:256], ab[256:]
    tri = B.packed_forward(lf, rf, 128)
    # The public low-level API already accepts nn.Linear's original Wg.
    # Keep Wg.T for backward but do not copy it back for the output kernel.
    y, norm, mo, ro, proj, gate = T.output_training(
        tri, xn.reshape(n, n, 128), d['wp'], d['leaves'][5], d['go'], d['bo'],
        x.reshape(n*n, 128), d['ds'], 1e-5, list(k3))
    ctx = SimpleNamespace(
        saved_tensors=(xn, *d['wt'], d['wp'], d['go'], pre, lf, rf,
                       tri, norm, mo, ro, gate, proj),
        eps=1e-5, h=128, mm=d['mask'], sm90_dual_bwd=False,
        dropscale=d['ds'], seq_len=n)
    return y.reshape_as(x), (ctx, mu, rs)
