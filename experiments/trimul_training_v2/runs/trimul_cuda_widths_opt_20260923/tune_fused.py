from width_plan import *
from fixture import setup
import statistics,gc
D=int(sys.argv[1]);N=int(sys.argv[2]) if len(sys.argv)>2 else 384
leaves,dy,mask,ds,*_=setup(D,N)
rows=[]
with torch.no_grad():
 for sk in ([1] if D==64 else [1,2,4] if D in (256,512) else [1,2,3,6]):
  for slots in ([1,2,4,8] if D==64 else [1]):
   group=1 if D==64 else 2;splits=128 if D==64 else 32
   os.environ.update(FUSED_GP='1',GP_SK=str(sk),GP_SLOTS=str(slots));build.cache_clear()
   m=Training(*leaves,mask,ds,dy);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):m()
   torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=stream):out=m()
   for _ in range(5):g.replay()
   events=[]
   for i in range(30):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();events.append((a,b))
   torch.cuda.synchronize();t=statistics.median(a.elapsed_time(b)*1000 for a,b in events);row=dict(group=group,splits=splits,grid=m.grid,grid7=m.grid7,sk=sk,slots=slots,us=t);rows.append(row);print('TUNE',D,N,row,flush=True);(R/f'fused-tune-D{D}-L{N}.json').write_text(json.dumps(rows,indent=2));del m,g,out;gc.collect()
