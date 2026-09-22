from width_plan import *
ks,path=build(256);U=T._launch_module()
for ta in (0,1):
 for tb in (0,1):
  torch.manual_seed(11);a=torch.randn((128,64) if ta else (64,128),device='cuda',dtype=torch.bfloat16);b=torch.randn((128,128) if tb else (128,128),device='cuda',dtype=torch.bfloat16);out=torch.empty((64,128),device='cuda',dtype=torch.bfloat16)
  maps=[tm(a),tm(b)]+[tm(a)]*14;p=U.Struct([*maps,out,*([None]*23),*([None]*13),64,2*ta+tb]);launch(ks['probe'],p,1,False,D=256);ref=(a.t() if ta else a)@(b if tb else b.t());torch.cuda.synchronize();err=float((out.float()-ref.float()).norm()/ref.float().norm());print('PROBE',ta,tb,err,flush=True);assert err<.001
print('PASS',path,flush=True)
