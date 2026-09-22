from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);ref,dc,dxn=baseline(a,True);p=CachedDxPlan(a,80,'front_cached_debug_loop',debug=1);p();torch.cuda.synchronize();gdx=torch.mm(a['dg'].reshape(-1,128),a['wg'].t());print('DEBUG',dict(dc=rel(p.debugdc,dc),gate=rel(p.debugxn,gdx),dxn=rel(p.partw,dxn),output=rel(p.dx,ref[0])),flush=True)
 print('NORMS',p.debugdc.float().norm().item(),dc.float().norm().item(),p.debugxn.float().norm().item(),gdx.float().norm().item(),p.partw.norm().item(),dxn.float().norm().item(),flush=True)
 torch.save(dict(gotgate=p.debugxn.cpu(),gate=gdx.cpu(),gotdc=p.debugdc.cpu(),dc=dc.cpu(),gotdxn=p.partw.cpu(),dxn=dxn.cpu()),R/'debug-cached-loop-L128.pt')
