"""End-to-end comparison: inference forward + both on-chip recompute regions."""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,platform,torch
import plans as P
import compare_cueq_training as Q
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('report_helpers',R.parent/'trimul_all_recompute_20260920/bench.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

class Training:
    def __init__(self,a,k3,s1,s7):
        self.a=a;self.d=a['d'];self.n=self.d['n']
        self.p1=P.B1(self.d,a['dy'],a['s'][0].saved_tensors[11],splits=s1)
        self.p7=P.B7(self.d,a['dy'],a['dl'],a['dr'],a['dg'],splits=s7)
        self.p7.mask=a['mask']
    def forward(self):return P.S.forward(self.d)
    def backward(self,kept):
        a,d,n=self.a,self.d,self.n;ab,tri,packed=kept;d['wt'],d['w1']=packed[:5],packed[5]
        self.p1.bind(a['dy'],tri)
        dg,dwg,dt,dgo,dbo,dwp=self.p1()
        dl,dr=P.B.packed_backward(dt,ab[:256],ab[256:],128)
        self.p7.bind(dl,dr,dg,a['dy'])
        dx,dwl,dwlg,dwr,dwrg,dgi,dbi=self.p7()
        return (dx.reshape_as(d['x']),dwl.t(),dwlg.t(),dwr.t(),dwrg.t(),dwg.t(),dwp,dgi,dbi,dgo,dbo)
    def __call__(self):
        y,kept=self.forward();return y,self.backward(kept)

def assert_full(out,ref):
    es=H.errors(out,ref)
    assert es['forward']['bit_exact'],es
    assert all(v['finite'] and v['relative_l2']<=5e-4 for v in es.values()),es
    return es

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,choices=(384,768),required=True)
    ap.add_argument('--iterations',type=int,default=200);ap.add_argument('--blocks',type=int,default=3)
    ap.add_argument('--profile',action='store_true');ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();n=args.length
    torch.backends.cuda.matmul.allow_tf32=False
    k3=json.loads((P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner']
    cfg={k:json.loads((R/('tune-%s-L%d.json'%(k,n))).read_text())['winner']['splits'] for k in ('b1','b7')}
    with torch.no_grad():
        a=Q.setup(n);d=a['d'];old=P.S.A.Training(a,k3);new=Training(a,k3,cfg['b1'],cfg['b7'])
        y,ss=old.forward_saved();_,kept=new.forward()
        ref=(y,P.S.C.backward(d,ss,a['dy']))
        out=new();torch.cuda.synchronize();checks={'new':assert_full(out,ref),'old':assert_full(old.saved(),ref)}
        print('CHECK',checks,flush=True)
        if args.check_only:
            for _ in range(3):out=new()
            torch.cuda.synchronize();assert_full(out,ref);print('CHECK_ONLY_DONE',flush=True);return
        if args.profile:
            for _ in range(10):new()
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();new();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();return
        result=dict(L=n,C=128,dropout=.25,checks=checks,configs=cfg,blocks={},times={},traces={},replay_checks={},
            metadata=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),hostname=platform.node(),
            forward='Anthropic inference K1/K3 with matched training dropout/residual and BF16 rounding; retain left/right/tri',
            backward='B1-B4 and B7-B12 reconstruct LN/projection/gate on chip; no global activation restoration; unchanged cuBLAS B5-B6',
            includes=['live weight packing','both directions','mask','dropout25','residual','all eleven gradients'],
            excludes=['optimizer','RNG generation','CPU dispatch','compilation'],
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),R/'plans.py',R/'b1_fused.cu',R/'b7_fused.cu',R/'common_recompute.cuh',R/'b1_saved_math.inc')}))
        scopes={'forward':{'saved':old.forward_saved,'fused_recompute':new.forward},
                'backward':{'saved':lambda:old.backward(ss),'fused_recompute':lambda:new.backward(kept)},
                'forward_backward':{'saved':old.saved,'fused_recompute':new}}
        for scope,fs in scopes.items():
            gs,outs={},{}
            for name,f in fs.items():
                print('CAPTURE',scope,name,flush=True);gs[name],outs[name]=Q.capture_outputs(f);gs[name].replay();torch.cuda.synchronize()
                if scope=='forward_backward':assert_full(outs[name],ref)
            blocks=[Q.paired(gs,iterations=args.iterations) for _ in range(args.blocks)]
            result['blocks'][scope]=blocks;result['times'][scope]=Q.pool(blocks)
            print('RESULT',scope,{k:v['median_us'] for k,v in result['times'][scope].items()},flush=True)
            result['traces'][scope]={k:H.trace(g,R/('trace-%s-%s-L%d.json'%(scope,k,n))) for k,g in gs.items()}
            if scope=='forward_backward':
                tensors={'x':d['x'],'wl':d['leaves'][1],'wg':d['leaves'][5],'dy':a['dy'],'ds':d['ds'],'mask':d['mask']}
                original={k:v.clone() for k,v in tensors.items()}
                d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91)
                d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
                yy,ss2=old.forward_saved();fresh=(yy,P.S.C.backward(d,ss2,a['dy']))
                # Isolate replay correctness from differences already present in
                # the saved CUDA implementation vs the independent reference.
                saved_now=old.saved();saved_now=(saved_now[0].clone(),tuple(t.clone() for t in saved_now[1]))
                eager_now=new();eager_now=(eager_now[0].clone(),tuple(t.clone() for t in eager_now[1]))
                result['mutated_new_vs_saved']=assert_full(eager_now,saved_now)
                for name,g in gs.items():
                    g.replay();torch.cuda.synchronize()
                    print('REPLAY_CHECK',name,H.errors(outs[name],fresh),flush=True)
                    # Keep the fixed acceptance bound for the new kernel. The
                    # unchanged saved baseline can itself miss it after mutation;
                    # record that explicitly instead of masking new-path checks.
                    result['replay_checks'][name]=H.errors(outs[name],fresh)
                    result[name+'_mutated_reference_pass']=all(v['finite'] and v['relative_l2']<=5e-4 for v in result['replay_checks'][name].values())
                    if name=='fused_recompute':
                        result['graph_vs_fresh_eager']=H.errors(outs[name],eager_now)
                        assert all(v['finite'] and v['relative_l2']<=1e-6 for v in result['graph_vs_fresh_eager'].values()),result['graph_vs_fresh_eager']
                for k,v in tensors.items():v.copy_(original[k])
                a['mask'].copy_(d['mask'].reshape(-1))
                trace=result['traces'][scope]['fused_recompute']
                names=[x['name'] for x in trace]
                assert not any('mw_saved_front' in s or 'mw_k3_train' in s or 'mw_recompute_' in s for s in names),names
                assert sum(t['calls_per_replay'] for t in trace if 'nvjet_' in t['name'])==6,trace
                result['activation_restore_kernels']=0;result['contraction_calls']=6
            (R/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2))
            del gs,outs;gc.collect()
        print('DONE',n,flush=True)

if __name__=='__main__':main()
