from dual import *
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from bench_k3 import graph
records=[]
with torch.no_grad():
 for n in ((64,72) if '--small' in sys.argv else (64,72,384,768)):
  for part in (1,2):
   d,dy,s=data(n);p=Unified(d,dy,s,min(132,n*n//64),part);o=p();torch.cuda.synchronize();ref=baseline(d,dy,s);errors=[rel(x,y) for x,y in zip(o,ref)];assert max(errors)<.001,errors
   gg=graph(p)
   for _ in range(20):gg[0].replay()
   torch.cuda.synchronize();assert torch.count_nonzero(p.workspace[-1])==0
   d['ds'].zero_();gg[0].replay();torch.cuda.synchronize();assert all(torch.count_nonzero(x)==0 for x in o)
   d['ds'].fill_(1);dy.normal_();ref=baseline(d,dy,s);gg[0].replay();torch.cuda.synchronize();e2=[rel(x,y) for x,y in zip(o,ref)];assert max(e2)<.001,e2
   rec=dict(L=n,part=part,errors=errors,live_input_errors=e2,zero_mask_exact=True,counters_reset=True);records.append(rec);print('PASS',rec,flush=True)
 if '--small' not in sys.argv:(R/'validation.json').write_text(json.dumps(records,indent=2))
