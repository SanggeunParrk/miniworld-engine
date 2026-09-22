from cached_dx_plan import *
with torch.no_grad():
 a=setup(128);p=CachedDxPlan(a,80,'front_cached_hash',debug=1);p();torch.cuda.synchronize();v=p.partw.flatten().view(torch.int32)[:256*256].reshape(256,256).cpu();print('DIFFS',[(i,torch.count_nonzero(v[i]-v[i%80]).item()) for i in [0,79,80,159,160,255]],flush=True)
