from core import *
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
rows=json.loads((R/'results.json').read_text());records=[]
for row in rows:
 cf=row['configs'];lc=tuple(cf['ln_anthropic']);tc=tuple(cf['ln_triton']);kc=tuple(cf['k1_split']);oc=tuple(cf['k3_split']);f1=tuple(cf['k1_original']);f3=tuple(cf['k3_original'])
 for n in (64,72):
  torch.manual_seed(976);x=torch.randn(1,n,n,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
  w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(n,n,device='cuda')>.2).float()
  def calc(fused,triton=False):
   xn=x if fused else triton_ln(x,g,b,tc) if triton else ln(x,g,b,lc)
   ab=front(xn,w,mask,g,b,fused,f1 if fused else kc);tri=packed_forward(ab[:256],ab[256:],128)
   return output(tri,xn,wp,wg,g,b,go,bo,x,fused,f3 if fused else oc,tma_residual=row.get("residual_tma",False) and not fused)
  yr=calc(True);ya=calc(False);yt=calc(False,True);torch.cuda.synchronize();assert torch.equal(yr,ya)
  err=((yt.float()-yr.float()).norm()/yr.float().norm()).item();assert err<.005
  for scale,offset in ((1e-4,0.),(1.,64.),(0.,0.)):
   z=(torch.randn_like(x)*scale+offset).contiguous();a=ln(z,g,b,lc);t=triton_ln(z,g,b,tc);r=torch.nn.functional.layer_norm(z.float(),(128,),g,b,1e-5).bfloat16()
   for out in (a,t):assert ((out.float()-r.float()).norm()/r.float().norm().clamp_min(1e-30)).item()<.005
  records.append(dict(selected_for=row['N'],tested_N=n,anthropic_bitwise=True,triton_relative_l2=err))
print(json.dumps(records),flush=True);(R/'validation.json').write_text(json.dumps(records,indent=2));print('PASS',flush=True)
