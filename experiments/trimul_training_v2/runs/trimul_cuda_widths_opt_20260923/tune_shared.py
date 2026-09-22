from width_plan import *
from fixture import setup
import statistics,gc
D=int(sys.argv[1]);N=int(sys.argv[2]) if len(sys.argv)>2 else 384
leaves,dy,mask,ds,*_=setup(D,N)
rows=[]
with torch.no_grad():
 for mb in (2,):
  for shared in (1,):
   group=2;splits=32;sk=gp_config(D)[3];slots=1
   os.environ.update(B7_MINB=str(mb),GP_SHARED_A=str(shared));build.cache_clear()
   m=Training(*leaves,mask,ds,dy);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):m()
   torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=stream):out=m()
   for _ in range(5):g.replay()
   events=[]
   for i in range(30):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();events.append((a,b))
   torch.cuda.synchronize();t=statistics.median(a.elapsed_time(b)*1000 for a,b in events);row=dict(shared=shared,minb=mb,group=group,splits=splits,grid=m.grid,grid7=m.grid7,sk=sk,slots=slots,us=t);rows.append(row);print('TUNE',D,N,row,flush=True);(R/f'shared-tune-D{D}-L{N}.json').write_text(json.dumps(rows,indent=2));del m,g,out;gc.collect()
