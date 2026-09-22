from width_plan import *
from width_plan import _compile
from fixture import setup
import statistics
D=int(sys.argv[1]);N=384
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad():
 m=Training(*leaves,mask,ds,dy);before=m();before=tuple(t.clone() for t in (before[0],*before[1]));torch.cuda.synchronize()
 os.environ.update(WIDTH_GROUPS='1',WIDTH_N='128',WIDTH_SOURCE='widths_pack.cu')
 ks,path=_compile(D,1,False);m.ks['forward']=ks['forward'];m.ks['b1']=ks['b1']
 m.grid=132*min(4,min(int(k.unit.drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',k.unit.drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(k.unit.drv.d.CUfunction(int(k.handle)),128,tuning(D)[2]))) for k in (ks['forward'],ks['b1'])))
 out=m();torch.cuda.synchronize();errors=[float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)) for a,b in zip((out[0],*out[1]),before)];assert max(errors)<.001,errors
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=st):m()
 for _ in range(5):g.replay()
 ev=[]
 for i in range(40):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();ev.append((a,b))
 torch.cuda.synchronize();record=dict(D=D,grid=m.grid,errors=errors,us=statistics.median(a.elapsed_time(b)*1000 for a,b in ev),cubin=path);(R/f'pack-D{D}.json').write_text(json.dumps(record,indent=2));print('PACK',record,flush=True)
