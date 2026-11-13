"""``kernels._key_len.kernel_mask``: the key mask as the attention cores read it (bool rows, contiguous), a view when it already is one."""

from __future__ import annotations

import torch

from miniworld_engine.kernels._key_len import kernel_mask


def test_kernel_mask_views_and_copies() -> None:
    L = 300
    m = torch.ones(3, L, dtype=torch.bool)
    m[1, 200:] = False
    out = kernel_mask(m, L)
    assert out is not None and out.data_ptr() == m.data_ptr() and out.shape == (3, L)
    shared = torch.ones(1, L, dtype=torch.bool).expand(4, L)  # stride 0 over the rows: materialised
    out = kernel_mask(shared, L)
    assert out is not None and out.is_contiguous() and out.shape == (4, L) and bool(out.all())
    as_int = kernel_mask(m.to(torch.uint8), L)
    assert as_int is not None and as_int.dtype is torch.bool and torch.equal(as_int, m)
    assert kernel_mask(None, L) is None
