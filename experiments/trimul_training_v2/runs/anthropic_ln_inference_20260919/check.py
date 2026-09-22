from core import *
for n in (64,72):
 torch.manual_seed(43);x=torch.randn(1,n,n,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
 w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(n,n,device='cuda')>.2).float()
 xn=ln(x,g,b,('tma',128,0,1));af=front(x,w,mask,g,b,True,(2,64,8,2,-1,232,2));ap=front(xn,w,mask,g,b,False,(2,64,8,2,-1,232,2));assert torch.equal(af,ap)
 tri=torch.randn(256,n,n,device='cuda',dtype=x.dtype)
 yf=output(tri,x,wp,wg,g,b,go,bo,x,True,(2,64,4,1,1,1));yp=output(tri,xn,wp,wg,g,b,go,bo,x,False,(2,64,4,1,1,1));torch.cuda.synchronize()
 print(n,'front',torch.equal(af,ap),'output',torch.equal(yf,yp),'max',(yf.float()-yp.float()).abs().max().item(),flush=True);assert torch.equal(yf,yp)
 # Numerical check against independent Torch formula, not just two matching implementations.
 nr=torch.nn.functional.layer_norm(x.float(),(128,),g,b,1e-5).bfloat16();nt=torch.nn.functional.layer_norm(tri.permute(1,2,0).float(),(256,),go,bo,1e-5).bfloat16()
 yr=(torch.sigmoid(nr.float()@wg.float().T)*(nt.float()@wp.float().T)).bfloat16();yr=(x+yr).bfloat16()
 err=(yf.float()-yr.float()).norm()/yr.float().norm();print('torch rel',err.item(),flush=True);assert err<.005
 yt=triton_ln(x,g,b,(16,128,4,2));err=(yt.float()-xn.float()).norm()/xn.float().norm();assert err<.003
print('PASS',flush=True)
