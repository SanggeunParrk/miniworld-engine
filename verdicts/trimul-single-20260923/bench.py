import importlib.util,json,statistics,time
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
spec=importlib.util.spec_from_file_location('cases','tests/integrations/test_trimul_single_h100_gpu.py'); cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
settings.configure(engine_backend='auto')
records=[]
for length in (384,768):
 for outgoing in (True,False):
    torch.compiler.reset()
    m,x,mask,ds=cases.setup(length,outgoing)
    ref=TriangleMultiplication(128,outgoing=outgoing,implementation='triton',p_drop=.25).cuda().bfloat16()
    ref.load_state_dict(m.state_dict());ref._make_drop_row_scale=lambda pair,p:ds
    xr=x.detach().clone().requires_grad_(); dy=torch.randn_like(x)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    row={'L':length,'D':128,'outgoing':outgoing,'dropout':.25,'fixed_mask':True,'times_ms':{}}
    with torch.cuda.stream(stream):
     graphs={}; keep=[]
     for label,model,inp in [('cuda',m,x),('triton',ref,xr)]:
        fn=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
        def run():
            y=fn(inp,mask)
            return y,torch.autograd.grad(y,(inp,*model.parameters()),dy)
        for _ in range(3):run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): out=run()
        graphs[label]=graph;keep.append(out)
        fgraph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(fgraph,stream=stream): f=fn(inp,mask)
        graphs[label+'_fwd']=fgraph;keep.append(f)
     for g in graphs.values():
        for _ in range(20):g.replay()
     samples={k:[] for k in graphs}
     for i in range(10):
        for k in (list(graphs) if i%2==0 else list(graphs)[::-1]):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(100):graphs[k].replay()
            b.record();b.synchronize();samples[k].append(a.elapsed_time(b)/100)
     row['times_ms']={k:statistics.median(v) for k,v in samples.items()}
     row['samples_ms']=samples
     row['speedup']=row['times_ms']['triton']/row['times_ms']['cuda']
    torch.cuda.current_stream().wait_stream(stream)
    print('RESULT',json.dumps(row),flush=True);records.append(row)
    Path('verdicts/trimul-single-20260923/bench.json').write_text(json.dumps(records,indent=2))
    del graphs,keep,m,ref,x,xr,out,f,fn
    torch.cuda.synchronize();torch.cuda.empty_cache()
