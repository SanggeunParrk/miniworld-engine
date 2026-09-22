from width_plan import *
from width_autograd import D128,fixed_policy
from fixture import setup
import argparse,statistics
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--length',type=int,required=True);a=p.parse_args();D,N=a.width,a.length
leaves,dy,mask,ds,*_=setup(D,N)
def snap(o):return tuple(t.clone() for t in (o[0],*o[1]))
with torch.no_grad():
 if D==128:
  m=D128(leaves,mask,ds);m.a['dy']=dy;F=fixed_policy();old=F.Fixed(m.a) if N==384 else F.P.Regression(m.a);new=lambda:(m.forward(),m.backward(dy))
 else:
  m=Training(*leaves,mask,ds,dy);old=baseline_module().Training(*leaves,mask,ds,dy);new=m
 before=snap(old());after=snap(new());torch.cuda.synchronize();errors=[float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)) for a,b in zip(after,before)];assert max(errors)<.001,errors
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream());graphs={}
 for k,fn in [('previous',old),('current',new)]:
  with torch.cuda.stream(st):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=st):fn()
  graphs[k]=g
 ev={k:[] for k in graphs}
 for rep in range(3):
  for g in graphs.values():
   for _ in range(5):g.replay()
  for i in range(30):
   for k in list(graphs) if (i+rep)%2 else list(graphs)[::-1]:
    aa,bb=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);aa.record();graphs[k].replay();bb.record();ev[k].append((aa,bb))
 torch.cuda.synchronize();times={k:dict(median_us=statistics.median(a.elapsed_time(b)*1000 for a,b in v),samples_us=[a.elapsed_time(b)*1000 for a,b in v]) for k,v in ev.items()}
 result=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),times=times,errors=errors,speedup=times['previous']['median_us']/times['current']['median_us'],complete=True);(R/f'paired-D{D}-L{N}.json').write_text(json.dumps(result,indent=2));print('PAIRED',D,N,{k:v['median_us'] for k,v in times.items()},result['speedup'],flush=True)
