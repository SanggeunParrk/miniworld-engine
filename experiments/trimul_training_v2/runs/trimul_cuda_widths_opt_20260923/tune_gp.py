from width_plan import *
from native_gp import GP
from native import configs
from fixture import setup
import statistics,gc
D=int(sys.argv[1]);N=int(sys.argv[2]) if len(sys.argv)>2 else 384
leaves,dy,mask,ds,*_=setup(D,N);rows=[]
with torch.no_grad():
 m=Training(*leaves,mask,ds,dy);m();torch.cuda.synchronize();ref=m.gp_all.clone()
 for cfg in configs(D):
  if cfg[0:2]==(1,128):continue
  if cfg[2]>4 and cfg!=tuple(m.front.cfg):continue
  gp=GP(m,cfg);gp();torch.cuda.synchronize();err=float((m.gp_all.float()-ref.float()).norm()/ref.float().norm());assert err<.001,(cfg,err)
  st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(st):gp()
  torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=st):gp()
  for _ in range(5):g.replay()
  ev=[]
  for i in range(40):
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();ev.append((a,b))
  torch.cuda.synchronize();row=dict(cfg=cfg,us=statistics.median(a.elapsed_time(b)*1000 for a,b in ev),error=err,cubin=gp.path);rows.append(row);(R/f'gp-tune-D{D}-L{N}.json').write_text(json.dumps(rows,indent=2));print('GP',D,N,row,flush=True);del gp,g;gc.collect()
