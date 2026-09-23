import importlib.util, json, statistics
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single import Plan
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_b7 import Plan as B7
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_output import Plan as B1,Output
from miniworld_engine.kernels.trimul_inproj.cuda.h100_native import Front,configs,k1_smem
spec=importlib.util.spec_from_file_location('cases','tests/integrations/test_trimul_single_h100_gpu.py');cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
T._launch_module()._make_context_current(0)
rows=[]
def timeit(fn):
 for _ in range(3):fn()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=fn()
 for _ in range(10):g.replay()
 times=[]
 for _ in range(3):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(40):g.replay()
  b.record();b.synchronize();times.append(a.elapsed_time(b)*25)
 return statistics.median(times)
for n in (384,768):
 m,x,mask,ds=cases.setup(n,True);dy=torch.randn_like(x)
 w=[m.to_left.weight,m.to_left_gate.weight,m.to_right.weight,m.to_right_gate.weight,m.to_gate.weight,m.to_out.weight,m.ln_pair.weight,m.ln_pair.bias,m.ln_out.weight,m.ln_out.bias]
 with torch.no_grad():
  p=Plan(x,*w,(mask[:,:,None]&mask[:,None,:]).bfloat16(),ds,dy)
  y=p.forward();p.backward();fb=p.front_back
  for c in (4,6,8,10,12):
   cl=264//(8+c)
   for rings in (8,12):
    b=B7(fb.d,dy,*fb.inputs[:3],xn=p.xn,clusters=cl,consumers=c,rings=rings)
    g=b();torch.cuda.synchronize()
    errors=[cases.relative(a,v) for a,v in zip(g,fb.outputs)]
    assert max(errors)<.0025,(c,rings,errors)
    row=dict(n=n,kernel='b7',config=dict(clusters=cl,consumers=c,rings=rings),us=timeit(b),errors=errors)
    rows.append(row);print(json.dumps(row),flush=True)
  for count in (66,132,264,396):
    b=B1(x,p.xn,p.tri,w[5],w[4],w[8],w[9],ds,dy,count=count)
    g=b.backward();torch.cuda.synchronize()
    errors=[cases.relative(a,v) for a,v in zip(g,[p.back.dg,p.back.dwg,p.back.dt,p.back.dgamma,p.back.dbeta,p.back.dwp])]
    assert max(errors)<.0025,errors
    row=dict(n=n,kernel='b1',config=dict(count=count),us=timeit(b.backward),errors=errors)
    rows.append(row);print(json.dumps(row),flush=True)
  for cfg in [(2,64,4,1),(2,64,6,1),(2,64,8,1),(2,64,4,2),(1,64,4,1),(1,128,4,1)]:
    try:
     b=Output(x,p.tri,w[5],w[4],*w[6:],ds,cfg=cfg)
     z=b();torch.cuda.synchronize();assert cases.relative(z,y)<.0001
     row=dict(n=n,kernel='k3',config=cfg,us=timeit(b))
    except (ValueError,RuntimeError) as e:
     row=dict(n=n,kernel='k3',config=cfg,error=str(e)[:300])
    rows.append(row);print(json.dumps(row),flush=True)
  for cfg in [(1,64,4,2,2),(2,64,8,2,1),(1,128,8,2,1),(1,64,4,1,2),(2,64,4,2,1),(1,128,4,2,1)]:
    b=Front(x[0],p.w1,p.mask.float(),w[6],w[7],cfg,hidden=128)
    ab,xn=b();torch.cuda.synchronize();assert cases.relative(ab,p.ab)<.0001
    row=dict(n=n,kernel='k1',config=cfg,us=timeit(b));rows.append(row);print(json.dumps(row),flush=True)
 Path('verdicts/trimul-single-20260923/tune.json').write_text(json.dumps(rows,indent=2))
