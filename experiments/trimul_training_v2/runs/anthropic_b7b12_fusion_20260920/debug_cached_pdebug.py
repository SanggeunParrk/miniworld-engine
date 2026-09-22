from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);ref,dc,dxn=baseline(a,True);p=CachedDxPlan(a,80,'front_cached_pdebug',debug=1);p();torch.cuda.synchronize()
 for side in range(2):
  v=p.debugdc[side*256:(side+1)*256].float();ref=dc[side*512+256:side*512+512].float();e=(v-ref).t().reshape(-1,64,256);print('SIDE',side,rel(v,ref),'TILES',[(i,float(e[i].norm())) for i in [0,79,80,159,160,255]],flush=True)
