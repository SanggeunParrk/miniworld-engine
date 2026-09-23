"""Native storage remains a real parameter with the legacy checkpoint contract."""
import pytest
import torch

from miniworld_engine.integrations.optimizer import align_optimizer_state_layout_
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_front_parameter_layout_checkpoint_and_optimizer(dtype):
    model = BidirectionalTriangleMultiplication(128, implementation="miniworld").to(dtype)
    legacy = BidirectionalTriangleMultiplication(128, implementation="pytorch").to(dtype)
    model.load_state_dict(legacy.state_dict())
    assert model.state_dict().keys() == legacy.state_dict().keys()
    names = ("to_left", "to_left_gate", "to_right", "to_right_gate")
    for name in names:
        p = getattr(model, name).weight
        assert p.is_leaf
        assert p.shape == (256, 128)
        assert p.stride() == (1, 256)
        assert p.t().is_contiguous()
        torch.testing.assert_close(p, getattr(legacy, name).weight, rtol=0, atol=0)
    opts = [torch.optim.AdamW(m.parameters(), lr=.001, foreach=False) for m in (model, legacy)]
    ids = tuple(id(p) for p in model.parameters())
    for _ in range(3):
        for name in names:
            grad = torch.randn_like(getattr(legacy, name).weight)
            getattr(model, name).weight.grad = grad.t().contiguous().t()
            getattr(legacy, name).weight.grad = grad.clone()
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for name in names:
            torch.testing.assert_close(getattr(model, name).weight, getattr(legacy, name).weight, rtol=0, atol=0)
    assert tuple(id(p) for p in model.parameters()) == ids
    legacy.load_state_dict(model.state_dict())
    for name in names:
        torch.testing.assert_close(getattr(model, name).weight, getattr(legacy, name).weight, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["foreach", "fused"])
def test_native_layout_adamw_resume_on_cuda(mode):
    model = BidirectionalTriangleMultiplication(128, implementation="miniworld").cuda().bfloat16()
    legacy = BidirectionalTriangleMultiplication(128, implementation="pytorch").cuda().bfloat16()
    model.load_state_dict(legacy.state_dict())
    opts = [torch.optim.AdamW(m.parameters(), lr=.001, **{mode: True}) for m in (model, legacy)]
    for step in range(3):
        for p, q in zip(model.parameters(), legacy.parameters(), strict=True):
            grad = torch.randn_like(q)
            p.grad = torch.empty_like(p).copy_(grad)
            q.grad = grad
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for p, q in zip(model.parameters(), legacy.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        if step == 0:
            # Legacy optimizer state can have row-major moments: resume remains valid.
            import copy
            opts[0].load_state_dict(copy.deepcopy(opts[1].state_dict()))
            assert align_optimizer_state_layout_(opts[0]) == 8
            assert align_optimizer_state_layout_(opts[0]) == 0
