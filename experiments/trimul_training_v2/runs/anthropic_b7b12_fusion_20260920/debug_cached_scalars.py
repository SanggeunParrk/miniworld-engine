from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);ref,dc,dxn=baseline(a,True);p=CachedDxPlan(a,80,'front_cached_scalars',debug=1);p();torch.cuda.synchronize()
 acc=torch.zeros_like(p.dx,dtype=torch.float32)
 for i,k in enumerate(['wlg','wl','wrg','wr']):
  acc+=torch.mm(dc[i*256:(i+1)*256].t().float(),a[k].t().float())
  got=p.partw.flatten()[:256*8].reshape(256,8)[:,i*2:i*2+2];re=acc[::64,::64];print('STAGE',i,'whole',rel(got,re),'tiles',[(j,got[j].tolist(),re[j].tolist()) for j in [0,79,80,159,160,255]],flush=True)
