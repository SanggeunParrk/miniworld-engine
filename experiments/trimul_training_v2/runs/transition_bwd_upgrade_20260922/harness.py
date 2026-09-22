from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,torch,statistics
import drv
R=Path(__file__).resolve().parent
import miniworld_engine.kernels.transition.cuda
s=importlib.util.spec_from_file_location('miniworld_engine.kernels.transition.cuda.transition_upgrade_baseline',R/'fused_sm90a.py');N=importlib.util.module_from_spec(s);sys.modules[s.name]=N;s.loader.exec_module(N)
NAMES=('dx','dgamma','dbeta','dWa','dWb','dWs')

def fixture(L=384,seed=2319):
 torch.manual_seed(seed);m=L*L;dev='cuda';bf=torch.bfloat16
 x=torch.randn(m,128,device=dev,dtype=bf);g=torch.rand(128,device=dev)+.5;b=torch.randn(128,device=dev)*.1
 wa=(torch.randn(512,128,device=dev)/128**.5).to(bf);wb=(torch.randn_like(wa.float())/128**.5).to(bf);ws=(torch.randn(128,512,device=dev)/512**.5).to(bf);dy=torch.randn_like(x)
 with torch.no_grad():y,xn,rs,c1=N._fwd_launch(x,g,b,wa,wb,ws.t().contiguous(),1e-5,True)
 return dict(L=L,M=m,x=x,gamma=g,beta=b,wa=wa,wb=wb,ws=ws,dy=dy,xn=xn,rs=rs,c1=c1,y=y)

def refresh(d):
 with torch.no_grad():y,xn,rs,c1=N._fwd_launch(d['x'],d['gamma'],d['beta'],d['wa'],d['wb'],d['ws'].t().contiguous(),1e-5,True)
 d['xn'].copy_(xn);d['rs'].copy_(rs);d['c1'].copy_(c1)

def build(name,rep=8,ctas=132):
 src=R/(name+'.cu');flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(R/'anthropic_v5'),'-DNCTA='+str(ctas),'-DDW_REPL='+str(rep)]
 key=hashlib.sha256(src.read_bytes()+str(flags).encode()).hexdigest();p=R/'build'/(key+'.cubin');p.parent.mkdir(exist_ok=True)
 if not p.exists():
  q=subprocess.run(['nvcc',*flags,str(src),'-o',str(p)],capture_output=True,text=True);p.with_suffix('.log').write_text(q.stdout+q.stderr)
  if q.returncode:raise RuntimeError(q.stderr)
 return p

class Plan:
 def __init__(self,d,name='baseline',rep=8,ctas=132):
  self.d=d;self.rep=rep;self.ctas=ctas;self.ndw=rep*8;self.ndx=ctas-self.ndw;self.path=build(name,rep,ctas);self.k=drv.Kernel(str(self.path),'transition_bwd_fused',231424);self.kr=drv.Kernel(str(self.path),'reduce_partials',0)
  tm=lambda t,dims,stride,box:drv.TensorMap(t,dims,stride,box)
  self.maps=[tm(d[t],[128,d['M']],256,[64,64]) for t in ('dy','xn','x')]+[tm(d['ws'],[512,128],1024,[64,128])]+[tm(d[t],[128,512],256,[64,64]) for t in ('wa','wb')]
  self.out=(torch.empty_like(d['x']),torch.empty_like(d['gamma']),torch.empty_like(d['gamma']),torch.empty_like(d['wa']),torch.empty_like(d['wb']),torch.empty_like(d['ws']))
  self.pw=torch.empty((self.ndw,3,64,128),device='cuda');self.pl=torch.empty((self.ndx,8,256),device='cuda');self.name=name
 def main(self):
  d=self.d;dx,dg,db,*_=self.out
  self.k((self.ctas,1,1),(256,1,1),*self.maps,d['rs'],d['c1'],d['gamma'],dx,dg,db,self.pw,self.pl,d['M'],d['M']//128)
 def reduce(self):
  dx,dg,db,wa,wb,ws=self.out
  extra=4 if 'parallel' in self.name else 1
  self.kr((768+extra,1,1),(256,1,1),self.pw,wa,wb,ws,self.pl,dg,db)
 def __call__(self):self.main();self.reduce();return self.out

def capture(fn):
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(stream)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):out=fn()
 return g,out

def errors(out,ref,require=True):
 result={}
 for n,x,y in zip(NAMES,out,ref):
  r=float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30));limit=5e-6 if n in ('dgamma','dbeta') else 2e-5 if n=='dx' else 5e-4
  result[n]=dict(relative_l2=r,finite=bool(x.isfinite().all()),bit_exact=torch.equal(x,y),limit=limit)
  if require:assert result[n]['finite'] and r<=limit,(n,result[n])
 return result

def paired(graphs,iters=120,rounds=4):
 events={n:[] for n in graphs};names=list(graphs)
 for r in range(rounds):
  for g in graphs.values():
   for _ in range(50):g.replay()
  torch.cuda.synchronize()
  for i in range(iters):
   for n in names if (i+r)%2==0 else names[::-1]:
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[n].replay();b.record();events[n].append((a,b))
 torch.cuda.synchronize()
 return {n:dict(median_us=statistics.median(a.elapsed_time(b)*1000 for a,b in es),samples_us=[a.elapsed_time(b)*1000 for a,b in es]) for n,es in events.items()}
