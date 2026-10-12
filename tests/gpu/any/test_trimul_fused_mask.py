"""Mask fusion must preserve the residual, unmasked output gate, and every gradient."""
import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

def models(kind):
    cls=BidirectionalTriangleMultiplication if kind=="bidir" else TriangleMultiplication
    options={} if kind=="bidir" else {"outgoing": kind=="outgoing"}
    ref=cls(128,p_drop=0,implementation=ImplementationType.PYTORCH,**options).cuda().float()
    actual=cls(128,p_drop=0,implementation=ImplementationType.MINIWORLD,**options).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for name,p in ref.named_parameters():
            if "ln_" not in name:p.normal_(std=128**-.5)
    actual.load_state_dict(ref.state_dict())
    # Compare the same representable weights, retaining an FP32 reference calculation.
    ref.load_state_dict(actual.state_dict())
    return actual,ref

def check(name,a,b):
    assert a is not None, name
    assert b is not None, name
    assert torch.isfinite(a).all(), name
    assert torch.isfinite(b).all(), name
    err=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)
    assert err<.025,(name,float(err))

@pytest.mark.parametrize("kind",["outgoing","incoming","bidir"])
@pytest.mark.parametrize("length",[128,384])
@pytest.mark.parametrize("mask_kind",["none","all_valid","holes","all_invalid"])
def test_training_fused_mask_output_and_all_gradients(kind,length,mask_kind):
    torch.manual_seed(61);torch.backends.cuda.matmul.allow_tf32=False
    actual,ref=models(kind)
    x=torch.randn(1,length,length,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    xr=x.detach().float().requires_grad_(True)
    mask=None if mask_kind=="none" else torch.ones(1,length,device="cuda",dtype=torch.bool)
    if mask_kind=="holes":
        assert mask is not None
        mask[:,::3]=False
    if mask_kind=="all_invalid":
        assert mask is not None
        mask.zero_()
    y,yr=actual(x,mask),ref(xr,mask)
    dy=torch.randn_like(y);y.backward(dy);yr.backward(dy.float())
    check("output",y,yr);check("input",x.grad,xr.grad)
    for (name,p),(rn,rp) in zip(actual.named_parameters(),ref.named_parameters(),strict=True):
        assert name==rn
        check(name,p.grad,rp.grad)
    if mask_kind=="all_invalid":
        torch.testing.assert_close(y,x,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,dy,rtol=0,atol=0)

@pytest.mark.parametrize("kind",["outgoing","incoming","bidir"])
@pytest.mark.parametrize("length",[128,384])
@pytest.mark.parametrize("mask_kind",["holes","all_invalid"])
def test_inference_fused_mask_preserves_reference(kind,length,mask_kind):
    torch.manual_seed(62);torch.backends.cuda.matmul.allow_tf32=False
    actual,ref=models(kind);actual.eval();ref.eval()
    x=torch.randn(1,length,length,128,device="cuda",dtype=torch.bfloat16)
    mask=torch.ones(1,length,device="cuda",dtype=torch.bool)
    if mask_kind=="holes":
        assert mask is not None
        mask[:,::3]=False
    else:mask.zero_()
    with torch.no_grad():y,yr=actual(x,mask),ref(x.float(),mask)
    check("output",y,yr)
    if mask_kind=="all_invalid":torch.testing.assert_close(y,x,rtol=0,atol=0)


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "bidir"])
@pytest.mark.parametrize("training", [False, True])
def test_hopper_mask_keeps_output_gate_and_custom_eps(kind, training):
    """A nonzero output-LN bias exposes masking the gate input by mistake."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper qualification")
    torch.manual_seed(710)
    actual, ref = models(kind)
    actual.train(training)
    ref.train(training)
    with torch.no_grad():
        for model in (actual, ref):
            model.ln_pair.eps = .03
            model.ln_out.eps = .07
        actual.ln_pair.bias.fill_(.2)
        actual.ln_out.bias.fill_(.4)
    ref.load_state_dict(actual.state_dict())
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16,
                    requires_grad=training)
    xr = x.detach().float().requires_grad_(training)
    mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    parameters_before = dict(actual.named_parameters())
    with torch.set_grad_enabled(training):
        y, yr = actual(x, mask), ref(xr, mask)
    check("masked output", y, yr)
    assert dict(actual.named_parameters()).keys() == parameters_before.keys()
    if training:
        dy = torch.randn_like(y)
        y.backward(dy)
        yr.backward(dy.float())
        check("input", x.grad, xr.grad)
        for name, param in parameters_before.items():
            assert dict(actual.named_parameters())[name] is param
            check(name, param.grad, dict(ref.named_parameters())[name].grad)
