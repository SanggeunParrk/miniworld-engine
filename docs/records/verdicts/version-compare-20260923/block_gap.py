"""Same-GPU paired diagnostics; prepared_only is NOT a production benchmark."""
import ast,json,sys,statistics,threading,subprocess,collections
from pathlib import Path
R=Path(__file__).resolve().parent
sys.argv=[str(R/'bench.py'),'--module','block','--length','384','--arm','engine2']
tree=ast.parse((R/'bench.py').read_text());prefix=[]
for node in tree.body:
    if isinstance(node,ast.Try):break
    prefix.append(node)
ns={'__name__':'gap_setup','__file__':str(R/'bench.py')}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(R/'bench.py'),'exec'),ns)
torch=ns['torch'];m,inputs=ns['setup']();m.train()
dy=torch.randn_like(inputs[0]);leaves=(inputs[0],*m.parameters())
ds=(torch.rand(1,1,384,128,device='cuda')>.25).bfloat16()*(4/3)
from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as native
graphs={};values={};original=native._data
for name in ('production_rng','fixed_dropout','prepared_only'):
    if name=='fixed_dropout':m.trimul._make_drop_row_scale=lambda pair,p:ds
    if name=='prepared_only':
        cache=[]
        def prepared(*args):
            if not cache:cache.append(original(*args))
            return cache[0]
        native._data=prepared
    torch.compiler.reset()
    fn=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def step(fn=fn):
        y=fn(*inputs)
        return y,torch.autograd.grad(y,leaves,dy)
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):step()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=side):values[name]=step()
    torch.cuda.current_stream().wait_stream(side);graphs[name]=g
native._data=original
for g in graphs.values():g.replay()
torch.cuda.synchronize()
errors=[]
for x,y in zip((values['fixed_dropout'][0],*values['fixed_dropout'][1]),(values['prepared_only'][0],*values['prepared_only'][1])):
    errors.append(float((x.float()-y.float()).norm()/x.float().norm().clamp_min(1e-12)))
assert max(errors)<1e-5,errors
for g in graphs.values():
    for _ in range(200):g.replay()
torch.cuda.synchronize()
uid=str(torch.cuda.get_device_properties(0).uuid);uid=uid if uid.startswith('GPU-') else 'GPU-'+uid
stop=threading.Event();telemetry=[]
def watch():
    while not stop.is_set():
        p=subprocess.run(['nvidia-smi','-i',uid,'--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        telemetry.append(p.stdout.strip());stop.wait(.1)
t=threading.Thread(target=watch);t.start();rounds={k:[] for k in graphs};names=list(graphs)
try:
    for r in range(12):
        for name in names[r%3:]+names[:r%3]:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(80):graphs[name].replay()
            b.record();b.synchronize();rounds[name].append(a.elapsed_time(b)/80)
finally:stop.set();t.join()
record=dict(ms={k:statistics.median(v) for k,v in rounds.items()},rounds_ms=rounds,gpu_uuid=uid,telemetry=telemetry,prepared_vs_fixed_relative_l2=errors,
 note='prepared_only freezes TriMul weight packs/transposes and converted mask outside replay; diagnostic only, no live-weight semantics')
for name,g in graphs.items():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:g.replay();torch.cuda.synchronize()
    out=R/f'gap-{name}-trace.json';p.export_chrome_trace(str(out));events=json.loads(out.read_text())['traceEvents']
    c=collections.defaultdict(lambda:[0,0.])
    for e in events:
        if e.get('cat')=='kernel':c[e['name']][0]+=1;c[e['name']][1]+=e['dur']
    record.setdefault('profile',{})[name]=dict(c)
(R/'block-gap.json').write_text(json.dumps(record,indent=2)+'\n');print('GAP',record['ms'],flush=True)
