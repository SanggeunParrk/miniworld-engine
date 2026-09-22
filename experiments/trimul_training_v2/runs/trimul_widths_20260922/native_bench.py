from native import *
import argparse,statistics,gc,os
from miniworld_engine import settings
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--no-save-xn',action='store_true');p.add_argument('--separate-ln',action='store_true');a=p.parse_args();D=a.width;H=2*D
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
record=dict(D=D,job=os.environ.get('SLURM_JOB_ID'),source='Anthropic v5 K1/K3, K1 occupancy widened; shared-memory-limited K3 uses existing Triton',results={},candidates=configs(D))
def save():(R/f'native-D{D}{"-separate" if a.separate_ln else "-nosave" if a.no_save_xn else ""}.json').write_text(json.dumps(record,indent=2))
def graph(fn):
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=st):out=fn()
 return g,out

def timer(g,reps=40):
 for _ in range(15):g.replay()
 ss=[]
 for _ in range(3):
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record()
  for i in range(reps):g.replay()
  e.record();torch.cuda.synchronize();ss.append(s.elapsed_time(e)*1000/reps)
 return statistics.median(ss)
def rel(x,y):return float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-20))
@torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
def pack(wl,wlg,wr,wrg):
 gate,proj=torch.cat((wlg,wrg),0),torch.cat((wl,wr),0)
 return torch.stack((gate.reshape(-1,32,D),proj.reshape(-1,32,D)),1).reshape(8*D,D)
@torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
def pack_into(dst,wl,wlg,wr,wrg):
 gate,proj=torch.cat((wlg,wrg),0),torch.cat((wl,wr),0)
 dst.copy_(torch.stack((gate.reshape(-1,32,D),proj.reshape(-1,32,D)),1).reshape(8*D,D))
 return dst
@torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
def old(*args):return B.bidirectional_trimul_triton(*args[:-1],1e-5,1e-5,D,mask=args[-1],dropscale=None)
with torch.no_grad():
 for n in (384,768):
  torch.manual_seed(20260922);x=torch.randn(1,n,n,D,device='cuda',dtype=torch.bfloat16)
  wl,wlg,wr,wrg,wg,wp=[(torch.randn(s,device='cuda')/s[-1]**.5).bfloat16() for s in [(H,D)]*4+[(D,D),(D,H)]]
  gi,bi,go,bo=[torch.ones(c,device='cuda')+.1*torch.randn(c,device='cuda') if i%2==0 else .05*torch.randn(c,device='cuda') for i,c in enumerate([D,D,H,H])]
  mask=(torch.rand(1,n,n,device='cuda')>.15).float();w1=pack(wl,wlg,wr,wrg);xn_ref=F.layer_norm(x.float(),(D,),gi,bi,1e-5).bfloat16()
  la=(torch.sigmoid(F.linear(xn_ref,wlg))*F.linear(xn_ref,wl))*mask[...,None];ra=(torch.sigmoid(F.linear(xn_ref,wrg))*F.linear(xn_ref,wr))*mask[...,None];abref=torch.cat((la,ra),-1).reshape(n,n,4*D).permute(2,0,1).contiguous();del la,ra
  results=[];best=None;besttime=float('inf')
  for cfg in configs(D):
   try:
    q=Front(xn_ref[0] if a.separate_ln else x[0],w1,mask,gi,bi,cfg,emit_xn=not(a.no_save_xn or a.separate_ln),normalize=not a.separate_ln);out,xn=q();torch.cuda.synchronize();es=dict(ab=rel(out,abref),xn=0. if (a.no_save_xn or a.separate_ln) else rel(xn,xn_ref[0]));assert max(es.values())<.01,es
    g,_=graph(q);us=timer(g);log=Path(q.path).with_suffix('.log').read_text();row=dict(config=cfg,us=us,errors=es,cubin=q.path,ptxas=log);results.append(row);print('K1',D,n,cfg,us,flush=True)
    if us<besttime:best,besttime=q,us
    del g
   except Exception as e:results.append(dict(config=cfg,error=repr(e)));print('K1_FAIL',D,n,cfg,str(e)[-300:],flush=True)
  assert best is not None
  ab,xn=best();xn=xn_ref if a.separate_ln else xn;tri=B.packed_forward(ab[:H],ab[H:],D);outs=[];oq=None;ot=float('inf')
  for cfg in itertools.product((1,2),(64,), (4,6,8),(1,2)):
   try:k3_smem(D,cfg)
   except ValueError:continue
   try:
    q=Output(tri,x[0],wp,wg,gi,bi,go,bo,cfg);g,y=graph(q);us=timer(g);outs.append(dict(config=cfg,us=us,cubin=q.path));print('K3',D,n,cfg,us,flush=True)
    if us<ot:oq,ot=q,us
   except Exception as e:outs.append(dict(config=cfg,error=repr(e)))
  @torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
  def fallback(t,xnorm):return B.trimul_back_triton(t.reshape(1,H,n,n),xnorm.reshape(1,n,n,D),wp.t().contiguous(),wg.t().contiguous(),go,bo,1e-5,x)
  # Keep fixed contraction buffer for the native K3 tensor map; copy is avoided by cuBLAS out= slices.
  def contract():
   torch.bmm(ab[:D],ab[H:H+D].transpose(-1,-2),out=tri[:D]);torch.bmm(ab[D:H].transpose(-1,-2),ab[H+D:],out=tri[D:])
  @torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
  def normalize_into(dst,xx,g,b):
   dst.copy_(F.layer_norm(xx.float(),(D,),g,b,1e-5).to(xx.dtype));return dst
  def native():
   if a.separate_ln:normalize_into(xn_ref,x,gi,bi)
   pack_into(w1,wl,wlg,wr,wrg);best();contract()
   return oq().reshape_as(x) if oq is not None else fallback(tri,xn)
  def baseline():return old(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask)
  gr,yr=graph(baseline);gn,yn=graph(native);gr.replay();gn.replay();torch.cuda.synchronize();es=rel(yn,yr);assert es<.005,es
  # Live x/weights mutation: exact restoration and comparison against same path eager.
  x.neg_();eg=native().clone();gn.replay();torch.cuda.synchronize();mut=rel(yn,eg);assert mut==0,mut;x.neg_()
  times={k:[] for k in ('triton','native')}
  for rr in range(5):
   for k,g in ([('triton',gr),('native',gn)] if rr%2==0 else [('native',gn),('triton',gr)]):times[k].append(timer(g,100))
  row=dict(k1=results,k3=outs,selected_k1=best.cfg,k1_us=besttime,k3_us=None if oq is None else ot,k3_route='Anthropic' if oq is not None else 'Triton fused output',relative_l2=es,mutation_graph_eager=mut,times={k:statistics.median(v) for k,v in times.items()},rounds=times)
  record['results'][str(n)]=row;save();print('FULL',D,n,row['times'],es,flush=True)
  del gr,gn,yr,yn,best,oq,ab,xn,tri,x,abref,xn_ref;gc.collect();torch.cuda.empty_cache()
record['complete']=True;save()
