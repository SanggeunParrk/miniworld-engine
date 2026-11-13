"""``kernels._key_len.key_len``: 1 + the last attended key per mask row, the full length for a row that attends none, None without a mask."""

from __future__ import annotations

import torch

from miniworld_engine.kernels._key_len import key_len


def test_key_len_values() -> None:
    L = 300
    m = torch.ones(5, L, dtype=torch.bool)
    m[0, 200:] = False  # prefix of 200
    m[1, :] = False  # none attended: every key kept
    m[2, 1:] = False  # key 0 only
    m[3, 10:20] = False  # a hole: the last key is attended
    m[4, :] = False
    m[4, 150] = True  # one key in the middle
    out = key_len(m, L)
    assert out is not None
    assert out.dtype == torch.int32 and out.is_contiguous()
    assert out.tolist() == [200, L, 1, L, 151]
    assert key_len(None, L) is None


def test_key_len_flattens_leading_dims() -> None:
    m = torch.ones(2, 3, 64, dtype=torch.bool)
    m[1, 2, 40:] = False
    out = key_len(m, 64)
    assert out is not None
    assert out.tolist() == [64, 64, 64, 64, 64, 40]
