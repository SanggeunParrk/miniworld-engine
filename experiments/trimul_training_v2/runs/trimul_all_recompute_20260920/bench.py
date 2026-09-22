"""Save-all vs full rematerialization; real H100 CUDA Graph timings.

Both paths include live weight packing, mask, dropout=.25, residual, all
eleven gradients, and unchanged B1-B12 CUDA/cuBLAS work. Recompute forward
returns only y; its backward recreates all intermediates from current inputs.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import sys
import torch
import adapter as A
import compare_cueq_training as Q

R=Path(__file__).resolve().parent

def errors(out,ref):
    return {k:dict(relative_l2=Q.rel(x,y),bit_exact=bool(torch.equal(x,y)),
                   finite=bool(torch.isfinite(x).all()))
            for k,x,y in zip(('forward',*Q.NAMES),(out[0],*out[1]),(ref[0],*ref[1]))}

def clone(out):return out[0].clone(),tuple(v.clone() for v in out[1])

def trace(g,path):
    from collections import defaultdict
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(5):g.replay()
        torch.cuda.synchronize()
    p.export_chrome_trace(str(path));rows=defaultdict(list)
    for e in json.loads(path.read_text())['traceEvents']:
        if e.get('cat')=='kernel':rows[e['name']].append(e['dur'])
    return [dict(name=k,calls_per_replay=len(v)//5,us=sum(v)/5) for k,v in rows.items()]

def memory(n,k3):
    # Actual PyTorch allocated bytes for forward only, same persistent inputs.
    # Excludes driver-only module memory and allocator reserve, not a model peak.
    d=A.C.setup(n)
    fs={'saved':lambda:A.F.forward(d,k3=k3),'recompute':lambda:A.forward_no_save(d)}
    for f in fs.values():
        for _ in range(3):f()
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    result={}
    for name,f in fs.items():
        torch.cuda.synchronize();base=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out=f();torch.cuda.synchronize()
        result[name]=dict(retained_including_y_bytes=torch.cuda.memory_allocated()-base,
                         forward_peak_increment_bytes=torch.cuda.max_memory_allocated()-base)
        del out;gc.collect();torch.cuda.synchronize()
    result['scope']='Eager forward incremental allocated memory, including output y. Shared inputs and weights pre-exist. Not whole-model or backward peak.'
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,choices=(384,768),required=True)
    ap.add_argument('--iterations',type=int,default=200);ap.add_argument('--blocks',type=int,default=3)
    ap.add_argument('--memory-only',action='store_true');ap.add_argument('--sanitize',action='store_true')
    args=ap.parse_args();n=args.length
    torch.backends.cuda.matmul.allow_tf32=False
    k3=json.loads((A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner']
    if args.memory_only:
        with torch.no_grad():rec=memory(n,k3)
        (R/('memory-L%d.json'%n)).write_text(json.dumps(rec,indent=2));print('MEMORY',rec,flush=True);return
    with torch.no_grad():
        print('SETUP',n,flush=True)
        a=Q.setup(n);d=a['d'];t=A.Training(a,k3)
        y,saved=t.forward_saved();baseline=y,A.C.backward(d,saved,a['dy'])
        saved_out=clone(t.saved())
        checks={'saved_vs_reference':errors(saved_out,baseline)}
        assert checks['saved_vs_reference']['forward']['bit_exact'],checks
        assert all(v['finite'] and v['relative_l2']<=5e-4 for k,v in checks['saved_vs_reference'].items()),checks
        print('CHECK saved',checks,flush=True)
        recompute_out=t.recompute();torch.cuda.synchronize()
        checks['recompute_vs_saved']=errors(recompute_out,saved_out)
        print('CHECK recompute',checks['recompute_vs_saved'],flush=True)
        # Do not silently relax a BF16 rounding mismatch: both contracts match.
        assert all(v['finite'] and v['bit_exact'] for v in checks['recompute_vs_saved'].values()),checks
        if args.sanitize:
            # Bound full path for memcheck; repeated no-save path for racecheck.
            for _ in range(3):A.forward_no_save(d)
            torch.cuda.synchronize();print('SANITIZE_DONE',flush=True);return
        results=dict(L=n,C=128,hidden_per_direction=128,dropout=.25,checks=checks,
            metadata=dict(gpu=torch.cuda.get_device_name(),hostname=platform.node(),torch=torch.__version__,
            mode='Full activation checkpoint; backward materializes fresh intermediates before unchanged B1-B12.',
            includes=['live weight packing','both directions','shared output LN','pair mask','dropout','residual','all eleven gradients'],
            excludes=['optimizer','RNG generation','CPU dispatch','compilation'],
            k3_saved=k3,k3_no_save=[2,64,4,1,1,1],
            retained_forward='original input, weights, supplied pair mask and dropout scale; zero intermediate activations',
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),R/'adapter.py',R/'no_save_k3.cu',A.P/'training_forward_adapter.py')}),
            blocks={},times={},traces={},replay_checks={})
        # Bwd-only saved has existing activations. Recompute bwd has none and
        # executes rematerialization inside its graph, charged to bwd timing.
        scopes={
            'forward':{'saved':t.forward_saved,'recompute':lambda:A.forward_no_save(d)},
            'backward':{'saved':lambda:t.backward(saved),'recompute':t.backward_recompute},
            'forward_backward':{'saved':t.saved,'recompute':t.recompute}}
        for scope,fs in scopes.items():
            graphs,outputs={},{}
            for name,fn in fs.items():
                print('CAPTURE',scope,name,flush=True)
                graphs[name],outputs[name]=Q.capture_outputs(fn)
                graphs[name].replay();torch.cuda.synchronize()
                if scope=='forward_backward':
                    es=errors(outputs[name],saved_out)
                    assert all(v['finite'] and v['bit_exact'] for v in es.values()),es
            blocks=[Q.paired(graphs,iterations=args.iterations) for _ in range(args.blocks)]
            results['blocks'][scope]=blocks;results['times'][scope]=Q.pool(blocks)
            print('RESULT',scope,{k:v['median_us'] for k,v in results['times'][scope].items()},flush=True)
            results['traces'][scope]={k:trace(g,R/('trace-%s-%s-L%d.json'%(scope,k,n))) for k,g in graphs.items()}
            if scope=='forward_backward':
                # Change inputs AND weights AND dy/dropout in place after
                # capture to expose stale saves, layouts or masks.
                originals={k:v.clone() for k,v in {'x':d['x'],'wl':d['leaves'][1],'wg':d['leaves'][5],'dy':a['dy'],'ds':d['ds']}.items()}
                d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91)
                d['ds'].copy_(d['ds'].roll(1,0))
                fresh=clone(t.saved())
                for name,g in graphs.items():
                    g.replay();torch.cuda.synchronize();es=errors(outputs[name],fresh)
                    assert all(v['finite'] and v['bit_exact'] for v in es.values()),es
                    results['replay_checks'][name]=es
                for k,v in {'x':d['x'],'wl':d['leaves'][1],'wg':d['leaves'][5],'dy':a['dy'],'ds':d['ds']}.items():v.copy_(originals[k])
            (R/('results-L%d.json'%n)).write_text(json.dumps(results,indent=2))
            del graphs,outputs;gc.collect()
        print('DONE',n,flush=True)

if __name__=='__main__':main()
