from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);p=CachedDxPlan(a,80,'front_cached_dumpw',debug=1);p();torch.cuda.synchronize();v=p.debugdc.flatten().view(torch.int32)[:256*256*96].reshape(256,256,96).cpu();print('WEIGHT_DIFFS',[(i,torch.count_nonzero(v[i]-v[i%80]).item()) for i in [0,79,80,159,160,255]],flush=True)
 torch.save(v,R/'cached-registers-L128.pt')
