"""``miniworld_engine.utils.optimizer.align_optimizer_state_layout_``: optimizer moments saved for a column-major parameter (engine 2.4.x
TriMul D128 front matrices) are re-laid to the row-major parameter of 2.5 on resume, values preserved."""

from __future__ import annotations

import torch

from miniworld_engine.utils.optimizer import align_optimizer_state_layout_


def _column_major(t: torch.Tensor) -> torch.Tensor:
    return t.t().contiguous().t()


def test_column_major_moments_follow_row_major_parameter():
    torch.manual_seed(0)
    old = torch.nn.Parameter(_column_major(torch.randn(6, 4)))  # the 2.4.x layout
    assert old.stride() == (1, 6)
    opt_old = torch.optim.AdamW([old], lr=1e-3)
    old.grad = _column_major(torch.randn(6, 4))
    opt_old.step()
    saved = opt_old.state_dict()
    assert saved["state"][0]["exp_avg"].stride() == (1, 6)

    new = torch.nn.Parameter(old.detach().contiguous())  # the 2.5 layout
    opt_new = torch.optim.AdamW([new], lr=1e-3)
    opt_new.load_state_dict(saved)
    before = {k: v.clone() for k, v in opt_new.state[new].items()}
    assert opt_new.state[new]["exp_avg"].stride() != new.stride()

    assert align_optimizer_state_layout_(opt_new) == 2  # exp_avg and exp_avg_sq; `step` is scalar
    for name, value in opt_new.state[new].items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert value.dtype == before[name].dtype
        if value.shape == new.shape:
            assert value.stride() == new.stride()
    assert align_optimizer_state_layout_(opt_new) == 0  # idempotent


def test_matching_layouts_untouched():
    p = torch.nn.Parameter(torch.randn(3, 5))
    opt = torch.optim.Adam([p])
    p.grad = torch.randn(3, 5)
    opt.step()
    ids = {k: id(v) for k, v in opt.state[p].items()}
    assert align_optimizer_state_layout_(opt) == 0
    assert {k: id(v) for k, v in opt.state[p].items()} == ids
