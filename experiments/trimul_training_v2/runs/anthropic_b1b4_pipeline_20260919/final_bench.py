from dual import *
from integrate import backward_cuda
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'));from bench_k3 import graph,paired
sys.path.insert(0,str(R.parent/'anthropic_ln_equal_saves_20260919'));import core_saved as C
records=[]
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);ref=baseline(d,dy,s);bf=C.backward(d,s,dy);p=Unified(d,dy,s,132,2);o=p();cf=backward_cuda(d,s,dy,p);torch.cuda.synchronize();errors=[rel(x,y) for x,y in zip(o,ref)];ferrors=[rel(x,y) for x,y in zip(cf,bf)];print('ERRORS',n,errors,ferrors,flush=True);assert max(errors+ferrors)<.001
  funcs={'B1-B4 Triton/cuBLAS':lambda:baseline(d,dy,s),'B1-B4 CUDA one kernel':p,'whole backward baseline':lambda:C.backward(d,s,dy),'whole backward new':lambda:backward_cuda(d,s,dy,p)}
  times=paired({k:graph(f) for k,f in funcs.items()},reps=80,rounds=20);rec=dict(L=n,dropout=.25,count=132,mode='cooperative single kernel',region_errors=errors,whole_backward_errors=ferrors,times=times);records.append(rec);print('RESULT',n,{k:round(v['median_us'],2) for k,v in times.items()},flush=True)
  (R/'final-results.json').write_text(json.dumps(records,indent=2))
