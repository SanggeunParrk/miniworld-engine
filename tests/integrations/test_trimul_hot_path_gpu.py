"""Warm forwards/backwards must not run compiler-cache or source-file checks."""
from pathlib import Path
import pytest
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as H
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]

@pytest.mark.parametrize('d', [64, 128])
def test_warm_native_path_has_no_source_io(d, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('Hopper required')
    torch.manual_seed(117)
    n=384
    x=torch.randn(1,n,n,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=[torch.randn(h,k,device='cuda',dtype=x.dtype)*k**-.5 for h,k in [(2*d,d)]*4+[(d,d),(d,2*d)]]
    if d==128:w[:4]=[v.t().contiguous().t() for v in w[:4]]
    w=[v.requires_grad_() for v in w]
    norms=[(torch.ones(c,device='cuda') if i%2==0 else torch.zeros(c,device='cuda')).requires_grad_() for i,c in enumerate((d,d,2*d,2*d))]
    args=[x,*w,*norms]
    mask=torch.ones(n,n,device='cuda',dtype=x.dtype)
    ds=(torch.rand(n,d,device='cuda')>.25).to(x.dtype)/.75
    dy=torch.randn_like(x)
    y=H.bidirectional_trimul(*args,mask,ds)
    expected=torch.autograd.grad(y,args,dy)
    def forbidden(*a,**kw):raise AssertionError('compiler cache queried on warm path')
    monkeypatch.setattr(T,'compile',forbidden)
    read=Path.read_text
    def guarded(path,*a,**kw):
        if str(path).startswith(str(T.SOURCES)):
            raise AssertionError('packaged config read on warm path')
        return read(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',guarded)
    # Independent outstanding forwards must retain independent saved tensors.
    a=H.bidirectional_trimul(*args,mask,ds)
    b=H.bidirectional_trimul(*args,mask,ds)
    actual=torch.autograd.grad(a,args,dy)
    second=torch.autograd.grad(b,args,dy)
    torch.testing.assert_close(a,y,rtol=0,atol=0)
    for ref,first,last in zip(expected,actual,second):
        for got in (first,last):
            if d == 64 and ref.dtype == torch.float32:
                # The unchanged wide LN reduction uses atomicAdd; its summation
                # order is not bitwise deterministic across launches.
                error=(got-ref).norm()/ref.norm().clamp_min(1e-12)
                assert error < 1e-4, error.item()
            else:
                torch.testing.assert_close(got,ref,rtol=0,atol=0)
