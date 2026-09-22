"""Bounded paired production-KG128 timing; no GPU work at import."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('dual15_' + name, ROOT / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    import triton
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel, dual_shape_key
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['baseline', 'one_wg_no_cta', 'hoist_fences', 'one_wg_hoist_fences'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[384])
    parser.add_argument('--rep', type=int, default=150)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--profile')
    parser.add_argument('--cold', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'ablation.json')
    args = parser.parse_args()
    mods = {name:load(name) for name in args.variants}
    c = dict(BLOCK_M1=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=3)
    results = []
    for length in args.lengths:
        torch.manual_seed(93)
        m,kg,kp,n = length*length,128,1024,128
        kw = dict(device='cuda',dtype=torch.bfloat16)
        g = torch.randn(m,kg,**kw)
        f = torch.randn(kp,m,**kw).t()
        w = torch.randn(n,kg,**kw).t()
        v = torch.randn(kp,n,**kw)
        y = torch.empty(m,n,**kw)
        def tri():
            _input_dual_bwd_kernel.fn[(triton.cdiv(m,c['BLOCK_M1'])*triton.cdiv(n,c['BLOCK_N']),)](
                g,f,w,v,y,m,kg,kp,n,*g.stride(),*f.stride(),*w.stride(),*v.stride(),
                shape_key=dual_shape_key(length,kg,kp,n),**c)
            return y
        funcs = {'triton':tri}
        funcs.update({name:lambda mod=mod:mod.input_dual_bwd_sm90_impl(g,f,w,v,length,c) for name,mod in mods.items()})
        ref = tri()
        errors = {}
        for name,fn in funcs.items():
            z = fn()
            torch.cuda.synchronize()
            errors[name] = dict(relative_l2=((z-ref).float().norm()/ref.float().norm()).item(), different=int((z!=ref).sum().item()))
            assert errors[name]['relative_l2'] <= 1e-4, (length,name,errors[name])
        if args.profile:
            fn=funcs[args.profile]
            for _ in range(10):fn()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            fn()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            return
        cold_funcs = {}
        graph_outputs = []
        if args.cold:
            for name, fn in funcs.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_outputs.append(fn())
                cold_funcs[name] = graph.replay
        times={name:[] for name in funcs}
        cold_times={name:[] for name in funcs}
        for r in range(args.rounds):
            names=list(funcs)
            names=names[r%len(names):]+names[:r%len(names)]
            for name in names:
                times[name].append(triton.testing.do_bench_cudagraph(funcs[name],rep=args.rep))
                if args.cold:
                    cold_times[name].append(triton.testing.do_bench(cold_funcs[name],warmup=10,rep=args.rep,return_mode='median'))
        row=dict(L=length,KG=kg,KP=kp,N=n,config=c,errors=errors,times_ms=times,
                 median_ms={name:statistics.median(t) for name,t in times.items()})
        if args.cold:
            row['cold_times_ms']=cold_times
            row['cold_median_ms']={name:statistics.median(t) for name,t in cold_times.items()}
        print(json.dumps(row),flush=True)
        results.append(row)
        args.output.write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
