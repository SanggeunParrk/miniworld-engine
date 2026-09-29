"""Real CUDA regression checks for independent reference paths and vendor composition."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention
from miniworld_engine.modules.swa_atom_attention.module import build_attention_params
from miniworld_engine.modules.swa_dit import SWADiTBlock
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

class NoMiniWorldExceptFlash(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        assert not name.startswith("miniworld_engine::") or "flash_window" in name, name
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("block", [False, True])
def test_pytorch_swa_uses_only_torch_and_flash_on_cuda(block):
    torch.manual_seed(23)
    model = (SWADiTBlock(128,128,implementation=ImplementationType.PYTORCH) if block else
             SWA3DRoPEAttention(128,4,implementation=ImplementationType.PYTORCH)).cuda().to(torch.bfloat16).train()
    x = torch.randn(2,128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    cond = torch.randn_like(x,requires_grad=True)
    cos = torch.randn(1,128,16,device="cuda")
    sin = torch.randn_like(cos)
    valid = torch.arange(128,device="cuda")[None,:] < torch.tensor([[117],[121]],device="cuda")
    ap = build_attention_params(cos,sin,valid,num_aug=2)
    with NoMiniWorldExceptFlash():
        y = model(x,cond,ap) if block else model(x,ap)
        y.backward(torch.randn_like(y))
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("length",[128,384])
def test_actual_vendor_bidir_matches_output_and_all_gradients(length):
    torch.manual_seed(37)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref=BidirectionalTriangleMultiplication(128,p_drop=0,implementation=ImplementationType.PYTORCH).cuda().to(torch.bfloat16)
    actual=BidirectionalTriangleMultiplication(128,p_drop=0,implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for name,p in ref.named_parameters():
            if "ln_" not in name: p.normal_(std=128**-.5)
    actual.load_state_dict(ref.state_dict())
    x=torch.randn(1,length,length,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    xr=x.detach().clone().requires_grad_(True)
    mask=torch.rand(1,length,device="cuda")>.125
    y,yr=actual(x,mask),ref(xr,mask)
    dy=torch.randn_like(y);y.backward(dy);yr.backward(dy)
    pairs=[("output",y,yr),("input gradient",x.grad,xr.grad)]
    pairs += [(name,p.grad,rp.grad) for (name,p),(_,rp) in zip(actual.named_parameters(),ref.named_parameters(),strict=True)]
    for name,a,b in pairs:
        assert a is not None, name
        assert b is not None, name
        assert torch.isfinite(a).all(), name
        assert torch.isfinite(b).all(), name
        relative=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)
        assert relative < .03,(name,float(relative))
