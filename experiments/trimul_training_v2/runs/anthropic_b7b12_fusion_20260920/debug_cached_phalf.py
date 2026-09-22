from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);ref,dc,dxn=baseline(a,True);p=CachedDxPlan(a,80,'front_cached_phalf',debug=1);p();torch.cuda.synchronize()
 acc=torch.zeros_like(p.dx,dtype=torch.float32)
 for side in range(2):
  g,pw=[a[k] for k in (['wlg','wl'] if side==0 else ['wrg','wr'])]
  acc+=torch.mm(dc[side*512:side*512+256].t().float(),g.t().float())
  for h in range(2):
   lo=side*512+256+h*128;acc+=torch.mm(dc[lo:lo+128].t().float(),pw[:,h*128:(h+1)*128].t().float())
   got=p.partw.flatten()[:256*8].reshape(256,8)[:,side*4+h*2:side*4+h*2+2];re=acc[::64,::64];print('STAGE',side,h,'whole',rel(got,re),'tiles',[(j,got[j].tolist(),re[j].tolist()) for j in [0,79,80,159,160,255]],flush=True)
