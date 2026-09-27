import sys,runpy
import torch
if sys.argv[1]=='pwa':
    scope=runpy.run_path('tests/integrations/test_pwa_train_gpu.py')
    scope['test_glue_dropout_shared_tiles_match_materialized_gradient'](384)
else:
    from miniworld_engine.modules.outer_product import OuterProductMean
    m=OuterProductMean(64,128,32,implementation='miniworld').cuda().bfloat16()
    with torch.no_grad():m.to_out.weight.normal_(std=.03)
    x=torch.randn(1,256,384,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    residual=torch.randn(1,384,384,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    mask=torch.rand(1,256,384,device='cuda')>.2
    with torch.no_grad():torch.testing.assert_close(m(x,mask,residual=residual),m(x,mask)+residual,rtol=0,atol=0)
    y=m(x,mask,residual=residual);g=torch.randn_like(y)
    grads=torch.autograd.grad(y,(x,residual,*m.parameters()),g)
    torch.testing.assert_close(grads[1],g,rtol=0,atol=0)
    assert all(bool(v.isfinite().all()) for v in grads)
torch.cuda.synchronize()
print('SANITY PASS',sys.argv[1],flush=True)
