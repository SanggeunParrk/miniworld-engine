"""Compare the user's selective activation policy with saved/full checkpoint.

Keep ab and tri. Recreate only xn/pre/norm/proj/gate and LN statistics.
No repeated triangular GEMM. Dropout25%, mask, all eleven gradients.
"""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,platform,sys,torch
import adapter_selective as S
import compare_cueq_training as Q
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('checkpoint_bench_helpers',R.parent/'trimul_all_recompute_20260920/bench.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def check_saved(actual,reference):
    # Check every intermediate, including the operands/results kept in forward.
    names=['xn','wl','wlg','wr','wrg','wg','wp','go','pre','lf','rf','tri','norm','mo','ro','gate','proj','mu','rs']
    aa=(*actual[0].saved_tensors,*actual[1:]);bb=(*reference[0].saved_tensors,*reference[1:])
    errors={k:dict(relative_l2=Q.rel(a,b),bit_exact=bool(torch.equal(a,b))) for k,a,b in zip(names,aa,bb)}
    assert all(v['bit_exact'] for v in errors.values()),errors
    return errors

def memory(n,k3):
    d=S.C.setup(n)
    fs={'saved':lambda:S.F.forward(d,k3=k3),'selective':lambda:S.forward(d),'full_checkpoint':lambda:S.A.forward_no_save(d)}
    for f in fs.values():
        for _ in range(3):f()
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    result={}
    for name,f in fs.items():
        base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        out=f();torch.cuda.synchronize()
        result[name]=dict(retained_including_y_bytes=torch.cuda.memory_allocated()-base,
                         forward_peak_increment_bytes=torch.cuda.max_memory_allocated()-base)
        del out;gc.collect();torch.cuda.synchronize()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,choices=(384,768),required=True)
    ap.add_argument('--memory-only',action='store_true');ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--iterations',type=int,default=200);ap.add_argument('--blocks',type=int,default=3)
    args=ap.parse_args();n=args.length
    torch.backends.cuda.matmul.allow_tf32=False
    k3=json.loads((S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner']
    with torch.no_grad():
        if args.memory_only:
            result=memory(n,k3);(R/('memory-L%d.json'%n)).write_text(json.dumps(result,indent=2));print('MEMORY',result,flush=True);return
        print('SETUP',n,flush=True)
        a=Q.setup(n);d=a['d'];t=S.Training(a,k3)
        ref=H.clone(t.saved())
        y,original=t.forward_saved();_,kept=S.forward(d)
        regenerated=S.rematerialize(d,kept,t.rk1,t.rk3)
        checks={'saved_tensors':check_saved(regenerated,original)}
        for name,fn in [('selective',t.selective),('full_checkpoint',t.recompute)]:
            out=fn();torch.cuda.synchronize();es=H.errors(out,ref)
            print('CHECK',name,es,flush=True)
            assert all(v['finite'] and v['bit_exact'] for v in es.values()),es
            checks[name]=es
        if args.check_only:
            for mode in ('normal','zero_drop','changed'):
                if mode=='zero_drop':d['ds'].zero_()
                if mode=='changed':
                    d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03)
                    d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1));d['ds'].fill_(1)
                yy,ss=t.forward_saved();yy2,kk=S.forward(d)
                assert torch.equal(yy,yy2)
                for _ in range(3):check_saved(S.rematerialize(d,kk,t.rk1,t.rk3),ss)
                print('PASS',mode,flush=True)
            print('CHECK_ONLY_DONE',flush=True);return
        result=dict(L=n,C=128,hidden_per_direction=128,dropout=.25,checks=checks,times={},blocks={},traces={},replay_checks={},
            metadata=dict(hostname=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            policy='Retain lf/rf/tri; recreate xn, input preactivations, output norm/proj/gate and LN stats with two CUDA kernels. No repeated contraction.',
            scope='Recomputed tensors are materialized transiently in HBM; unchanged B1-B12 reads them. Not register-local backward fusion.',
            includes=['live weight packing','mask','dropout25','residual','all eleven gradients'],
            excludes=['optimizer','RNG generation','CPU dispatch','compilation'],
            rk1=t.rk1,rk3=t.rk3,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),R/'adapter_selective.py',R/'recompute_front.cu',R/'recompute_output.cu')}))
        scopes={
            'forward':{'saved':t.forward_saved,'selective':lambda:S.forward(d),'full_checkpoint':lambda:S.A.forward_no_save(d)},
            'recompute_only':{'front':lambda:S.front(d,kept[0],t.rk1),
                              'output':lambda:S.output(d,kept[1],regenerated[0].saved_tensors[0],t.rk3),
                              'combined':lambda:S.rematerialize(d,kept,t.rk1,t.rk3)},
            'backward':{'saved':lambda:t.backward(original),'selective':lambda:t.backward_selective(kept),'full_checkpoint':t.backward_recompute},
            'forward_backward':{'saved':t.saved,'selective':t.selective,'full_checkpoint':t.recompute}}
        for scope,fs in scopes.items():
            # Standalone recompute graphs must capture the strongly retained
            # packing in kept, not weights from a just-destroyed forward graph.
            # rematerialize rebinds d during capture; do this before *all* three
            # recompute graphs so their inputs share the same stable lifetime.
            if scope=='recompute_only':
                d['wt'],d['w1']=kept[2][:5],kept[2][5]
            graphs,outs={},{}
            for name,fn in fs.items():
                print('CAPTURE',scope,name,flush=True)
                graphs[name],outs[name]=Q.capture_outputs(fn)
                graphs[name].replay();torch.cuda.synchronize()
                if scope=='forward_backward':
                    es=H.errors(outs[name],ref);assert all(v['finite'] and v['bit_exact'] for v in es.values()),es
            blocks=[Q.paired(graphs,iterations=args.iterations) for _ in range(args.blocks)]
            result['blocks'][scope]=blocks;result['times'][scope]=Q.pool(blocks)
            print('RESULT',scope,{k:v['median_us'] for k,v in result['times'][scope].items()},flush=True)
            result['traces'][scope]={k:H.trace(g,R/('trace-%s-%s-L%d.json'%(scope,k,n))) for k,g in graphs.items()}
            if scope=='forward_backward':
                tensors={'x':d['x'],'wl':d['leaves'][1],'wg':d['leaves'][5],'dy':a['dy'],'ds':d['ds'],'mask':d['mask']}
                originals={k:v.clone() for k,v in tensors.items()}
                d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91)
                d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
                fresh=H.clone(t.saved())
                for name,g in graphs.items():
                    g.replay();torch.cuda.synchronize();es=H.errors(outs[name],fresh)
                    assert all(v['finite'] and v['bit_exact'] for v in es.values()),es
                    result['replay_checks'][name]=es
                for k,v in tensors.items():v.copy_(originals[k])
                a['mask'].copy_(d['mask'].reshape(-1))
                # Count real cuBLAS calls: checkpoint must repeat two forward
                # contractions; selective must have the same count as saved.
                counts={name:sum(x['calls_per_replay'] for x in trace if 'nvjet_' in x['name'] or 'cublas' in x['name'].lower())
                        for name,trace in result['traces'][scope].items()}
                assert counts['saved']==counts['selective']==6 and counts['full_checkpoint']==8,counts
                result['contraction_launch_counts']=counts
            (R/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2))
            del graphs,outs;gc.collect()
        print('DONE',n,flush=True)

if __name__=='__main__':main()
