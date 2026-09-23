exec(open('verdicts/trimul-single-20260923/probe.py').read().split('for direction in')[0])
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_b7 import Plan as B7
from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import launch
plan=Plan(x,*w,*p,mask,ds,dy,outgoing=True)
with torch.no_grad():plan()
data=dict(n=L,x=x,mask=mask,ds=ds,wt=[v.t().contiguous() for v in w[:5]],wp=w[5],gi=p[0],bi=p[1],go=p[2],bo=p[3],w1=plan.w1)
b7=B7(data,dy,plan.dl,plan.dr,plan.dg,xn=plan.xn)
with torch.no_grad():g=b7()
torch.cuda.synchronize()
print('B7ERR',[float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip(g,[plan.dx.reshape(-1,128),*(v.t() for v in plan.dw),*plan.floats[8:10]])],flush=True)

def bench(fn):
 for _ in range(3):fn()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):fn()
 for _ in range(5):graph.replay()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(100):graph.replay()
 b.record();b.synchronize()
 return a.elapsed_time(b)*10
for name,fn in [('front',plan.front),('output',lambda:launch(plan.ks['forward'],plan.params,plan.grid,D=128)),('b1',lambda:launch(plan.ks['b1'],plan.params,plan.grid,D=128)),('old_b7',lambda:launch(plan.ks['b7'],plan.params7,plan.grid,D=128)),('new_b7',b7)]:
 print('TIMING_US',name,bench(fn),flush=True)
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_output import Plan as Out
op=Out(x,plan.xn,plan.tri,w[5],w[4],p[2],p[3],ds,dy)
with torch.no_grad():
    yo=op.forward();go=op.backward()
torch.cuda.synchronize()
print('OUTPUT_ERR',float((yo.float()-plan.y.float()).norm()/plan.y.float().norm()),flush=True)
print('B1ERR',[float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip(go,[plan.dg,plan.dwg,plan.dt,*plan.floats[10:12],plan.dwp])],flush=True)
print('TIMING_US new_output',bench(op.forward),flush=True)
print('TIMING_US new_b1',bench(op.backward),flush=True)
from miniworld_engine.kernels.trimul_inproj.cuda.h100_single_output import Output
k3=Output(x,plan.tri,w[5],w[4],*p,ds)
with torch.no_grad():yk=k3()
torch.cuda.synchronize()
print('K3_ERR',float((yk.float()-yo.float()).norm()/yo.float().norm()),flush=True)
print('TIMING_US k3',bench(k3),flush=True)
