import torch
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_saved import front,output

torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(910)
def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
for n,h in ((64,128),(72,256)):
 c=128;kw=dict(device='cuda',dtype=torch.bfloat16)
 xn=torch.randn(1,n,n,c,**kw);ws=[torch.randn(c,h,**kw)/c**.5 for _ in range(4)]
 mask=(torch.rand(1,n,n,device='cuda')>.3).bfloat16()
 a,b,pre=front(xn,*ws,pair_mask=mask)
 ps=[(xn.float()@w.float()).reshape(n*n,h).T for w in ws]
 ar=(torch.sigmoid(ps[1])*ps[0]).bfloat16()*mask.reshape(1,n*n)
 br=(torch.sigmoid(ps[3])*ps[2]).bfloat16()*mask.reshape(1,n*n)
 pr=torch.cat((torch.stack((ps[1],ps[0]),1).reshape(2*h,-1),torch.stack((ps[3],ps[2]),1).reshape(2*h,-1))).bfloat16()
 assert max(rel(a.reshape(h,-1),ar),rel(b.reshape(h,-1),br),rel(pre,pr))<.003
 norm=torch.randn(n*n,h,**kw);wp=torch.randn(c,h,**kw)/h**.5;wg=torch.randn(c,c,**kw)/c**.5
 ds=(torch.rand(n,c,device='cuda')>.25).bfloat16()/.75;res=torch.randn(n*n,c,**kw)
 y,p,g=output(norm,xn.reshape(n*n,c),wp,wg,res,ds,n)
 pp=norm@wp.T;gg=torch.sigmoid((xn.reshape(n*n,c)@wg).float())
 yr=(pp.float()*gg*ds.repeat(n,1).float()+res.float()).bfloat16()
 assert max(rel(y,yr),rel(p,pp),rel(g,gg.bfloat16()))<.003
 torch.cuda.synchronize();print('OK',n,h,flush=True)
