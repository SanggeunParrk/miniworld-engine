from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);ref,dc,dxn=baseline(a,True);p=CachedDxPlan(a,80,'front_cached_stages',debug=1);p();torch.cuda.synchronize()
 acc=torch.zeros_like(p.dx,dtype=torch.float32)
 for i,k in enumerate(['wlg','wl','wrg','wr']):
  acc+=torch.mm(dc[i*256:(i+1)*256].t().float(),a[k].t().float())
  got=p.debugdc[i*128:(i+1)*128].t().float();dif=(got-acc).reshape(-1,64,128)
  print('STAGE',i,'whole',rel(got,acc),'tiles',[(j,float(dif[j].norm())) for j in [0,79,80,159,160,255]],flush=True)
