from native import *
import sys,statistics,os,torch.nn.functional as F
D=int(sys.argv[1]);H=2*D
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
record=dict(D=D,job=os.environ.get('SLURM_JOB_ID'),results={})
opts=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
@torch.compile(**opts)
def pk(dst,wl,wlg,wr,wrg):dst.copy_(torch.stack((torch.cat((wlg,wrg)).reshape(-1,32,D),torch.cat((wl,wr)).reshape(-1,32,D)),1).reshape(8*D,D))
@torch.compile(**opts)
def ln(dst,x,g,b):dst.copy_(F.layer_norm(x.float(),(D,),g,b,1e-5).to(x.dtype))
@torch.compile(**opts)
def reference(*args):return B.bidirectional_trimul_triton(*args[:-1],1e-5,1e-5,D,mask=args[-1])
def capture(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):o=fn()
 g.replay();torch.cuda.synchronize();return g,o
with torch.no_grad():
 for n in (384,768):
  torch.manual_seed(20260922);x=torch.randn(1,n,n,D,device='cuda',dtype=torch.bfloat16);weights=[(torch.randn(s,device='cuda')/s[-1]**.5).bfloat16() for s in [(H,D)]*4+[(D,D),(D,H)]];wl,wlg,wr,wrg,wg,wp=weights
  gi,bi,go,bo=[torch.ones(c,device='cuda')+.1*torch.randn(c,device='cuda') if i%2==0 else .05*torch.randn(c,device='cuda') for i,c in enumerate([D,D,H,H])];mask=(torch.rand(1,n,n,device='cuda')>.15).float()
  @torch.compile(**opts)
  def output(t,z):return B.trimul_back_triton(t.reshape(1,H,n,n),z.reshape_as(x),wp.t().contiguous(),wg.t().contiguous(),go,bo,1e-5,x)
  def make(separate):
   row=json.loads((R/f'native-D{D}{"-separate" if separate else ""}.json').read_text())['results'][str(n)];w1=torch.empty((8*D,D),device='cuda',dtype=x.dtype);pk(w1,wl,wlg,wr,wrg);xn=torch.empty_like(x) if separate else x;ln(xn,x,gi,bi) if separate else None
   p=Front(xn[0],w1,mask,gi,bi,row['selected_k1'],emit_xn=not separate,normalize=not separate);tri=torch.empty((H,n,n),device='cuda',dtype=x.dtype)
   def run():
    pk(w1,wl,wlg,wr,wrg)
    if separate:ln(xn,x,gi,bi)
    ab,emitted=p();torch.bmm(ab[:D],ab[H:H+D].transpose(-1,-2),out=tri[:D]);torch.bmm(ab[D:H].transpose(-1,-2),ab[H+D:],out=tri[D:]);return output(tri,xn if separate else emitted)
   return run
  fns={'triton':lambda:reference(x,*weights,gi,bi,go,bo,mask),'fused_ln':make(False),'separate_ln':make(True)};gs={};outs={}
  for k,fn in fns.items():gs[k],outs[k]=capture(fn)
  ref=outs['triton'].clone();checks={}
  for k,o in outs.items():
   er=float((o.float()-ref.float()).norm()/ref.float().norm());assert er<.005;checks[k]=er
  x.neg_();wl.neg_()
  for k,fn in fns.items():
   e=fn().clone();gs[k].replay();torch.cuda.synchronize();assert torch.equal(e,outs[k]),k
  x.neg_();wl.neg_();ev={k:[] for k in gs}
  for r in range(5):
   for g in gs.values():
    for _ in range(30):g.replay()
   for i in range(100):
    for k in list(gs) if (i+r)%2==0 else list(gs)[::-1]:
     s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();gs[k].replay();e.record();ev[k].append((s,e))
  torch.cuda.synchronize();times={k:dict(median_us=statistics.median(s.elapsed_time(e)*1000 for s,e in v),samples_us=[s.elapsed_time(e)*1000 for s,e in v]) for k,v in ev.items()};record['results'][str(n)]=dict(checks=checks,mutation_exact=True,times=times);(R/f'ln-compare-D{D}.json').write_text(json.dumps(record,indent=2));print('LN_AB',D,n,{k:v['median_us'] for k,v in times.items()},flush=True)
record['complete']=True;(R/f'ln-compare-D{D}.json').write_text(json.dumps(record,indent=2))
