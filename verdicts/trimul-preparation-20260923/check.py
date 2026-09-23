import ast, importlib.util, json, sys, statistics, copy, os, types
from pathlib import Path
R=Path(__file__).resolve().parent
repo=R.parents[1]
bp=repo/'verdicts/version-compare-20260923/bench.py'
sys.argv=[str(bp),'--module','block','--length','384','--arm','engine2']
tree=ast.parse(bp.read_text()); prefix=[]
for node in tree.body:
    if isinstance(node,ast.Try): break
    prefix.append(node)
setup_module=types.ModuleType('prep_setup');sys.modules['prep_setup']=setup_module
ns=setup_module.__dict__;ns['__file__']=str(bp)
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(bp),'exec'),ns)
torch=ns['torch']
bench_stream=torch.cuda.Stream();torch.cuda.set_stream(bench_stream)
from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as native
from miniworld_engine.integrations import trimul_h100 as wiring
src=(R/'h100_training.before.py').read_text().replace('name="trimul_h100_train_', 'name="trimul_prep_before_')
path=R/'baseline.py';path.write_text(src)
spec=importlib.util.spec_from_file_location('prep_baseline',path);base=importlib.util.module_from_spec(spec);sys.modules['prep_baseline']=base;spec.loader.exec_module(base)
newfn=native.bidirectional_trimul
import warnings
warnings.filterwarnings('error', message='fused sm90a Transition unavailable.*')
results={}
for module in ('trimul','block'):
 for L in (384,768):
    torch.compiler.reset();ns['args'].module=module;ns['args'].length=L
    m,inputs=ns['setup']();old=copy.deepcopy(m)
    tm=m if module=='trimul' else m.trimul
    ot=old if module=='trimul' else old.trimul
    for proj in (ot.to_left,ot.to_left_gate,ot.to_right,ot.to_right_gate):
        proj.weight=torch.nn.Parameter(proj.weight.detach().contiguous())
    assert all(p.weight.stride()==(1,256) for p in (tm.to_left,tm.to_left_gate,tm.to_right,tm.to_right_gate))
    assert set(old.state_dict())==set(m.state_dict())
    m.load_state_dict(old.state_dict())
    # Real autograd/optimizer compatibility, with the same fixed row dropout.
    ds=(torch.rand(1,1,L,128,device='cuda')>.25).bfloat16()*(4/3)
    ot._make_drop_row_scale=lambda pair,p: ds
    tm._make_drop_row_scale=lambda pair,p: ds
    dy=torch.randn_like(inputs[0]);graphs={};outs={};fns={}
    params={};opts={name:torch.optim.AdamW(model.parameters(),lr=.001,foreach=False) for name,model in [('before',old),('after',m)]}
    for name,model in [('before',old),('after',m)]:
        native.bidirectional_trimul=base.bidirectional_trimul if name=='before' else newfn
        model.train();params[name]=(inputs[0],*model.parameters())
        y=model(*inputs);gs=torch.autograd.grad(y,params[name],dy);outs[name]=(y.detach(),*[g.detach() for g in gs])
        # Real .backward accumulation and AdamW update.
        model(*inputs).backward(dy)
        opts[name].step();opts[name].zero_grad(set_to_none=True);inputs[0].grad=None
    errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(outs['before'],outs['after'])]
    assert max(errors)==0,errors
    for a,b in zip(old.parameters(),m.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # New weights must be used by every subsequent invocation.
    for name,model in [('before',old),('after',m)]:
        native.bidirectional_trimul=base.bidirectional_trimul if name=='before' else newfn
        fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        fns[name]=fn
        def step(fn=fn,par=params[name]):
            y=fn(*inputs);return y,torch.autograd.grad(y,par,dy)
        side=bench_stream
        with torch.cuda.stream(side):
            for _ in range(4):step()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=side):outs[name]=step()
        torch.cuda.current_stream().wait_stream(side);graphs[name]=g
    for g in graphs.values():g.replay()
    torch.cuda.synchronize()
    for a,b in zip((outs['before'][0],*outs['before'][1]),(outs['after'][0],*outs['after'][1])):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Timed production graphs generate their own dropout masks, just as the public module does.
    del ot._make_drop_row_scale
    del tm._make_drop_row_scale
    torch.compiler.reset()
    graphs={};outs={};fns={}
    for name,model in [('before',old),('after',m)]:
        native.bidirectional_trimul=base.bidirectional_trimul if name=='before' else newfn
        fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False});fns[name]=fn
        def step(fn=fn,par=params[name]):
            y=fn(*inputs);return y,torch.autograd.grad(y,par,dy)
        for _ in range(4):step()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=bench_stream):outs[name]=step()
        graphs[name]=g
    rounds={k:[] for k in graphs}
    for r in range(10):
        for name in (('before','after') if r%2==0 else ('after','before')):
            g=graphs[name]
            for _ in range(40):g.replay()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(100):g.replay()
            b.record();b.synchronize();rounds[name].append(a.elapsed_time(b)/100)
    key=f'{module}-{L}';res=dict(ms={k:statistics.median(v) for k,v in rounds.items()},rounds=rounds,fixed_mask_bit_exact=True,adamw_bit_exact=True,dropout_rng_included=True)
    for name,g in graphs.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:g.replay();torch.cuda.synchronize()
        p.export_chrome_trace(str(R/f'{key}-{name}-trace.json'))
    results[key]=res;(R/'results.json').write_text(json.dumps(results,indent=2));print(key,res['ms'],flush=True)
    del graphs,outs,fns,m,old,opts,params
    torch.cuda.empty_cache()
native.bidirectional_trimul=newfn
print('ALL PASS',flush=True)
