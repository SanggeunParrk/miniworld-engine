"""Re-measure forward and trace the actual CUDA graph kernels of both routes."""
from compare_cueq_training import *
from miniworld_engine import settings
from miniworld_engine.autotune import trimul_sm90_config as sm90_cfg
import collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, choices=(384,768), required=True)
    args = ap.parse_args()
    n = args.length
    with torch.no_grad():
        d = C.setup(n)
        settings.configure(autotune_miss_cap=24)
    manifest = R.parent / ('trimul_sm90_round2_20260917/module/measured-configs-L%d.json'%n)
    configs = json.loads(manifest.read_text())
    original_resolve = sm90_cfg.resolve
    actual_configs = []
    def resolve(op, tensors, **kw):
        if op not in configs:
            return original_resolve(op,tensors,**kw)
        c = configs[op]
        assert kw['feasibility'](c) is None
        actual_configs.append(dict(op=op,config=c))
        return c
    sm90_cfg.resolve = resolve

    def repack():
        _,wl,wlg,wr,wrg,wg,*_ = d['leaves']
        d['wt'] = [q.t().contiguous() for q in (wl,wlg,wr,wrg,wg)]
        gate,proj = torch.cat((wlg,wrg),0),torch.cat((wl,wr),0)
        d['w1'] = torch.stack((gate.reshape(-1,32,128),proj.reshape(-1,32,128)),1).reshape(1024,128)

    def current(cfg=(3,64,2,2,1), pack=True):
        with torch.no_grad():
            if pack:
                repack()
            return C.forward(d,True,cfg,(1,1))

    def old(*inputs):
        return B.bidirectional_trimul_triton(*inputs[:-2],1e-5,1e-5,128,
                    mask=inputs[-2],dropscale=inputs[-1],output_backend='triton')
    compiled_old = torch.compile(old,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    funcs = {'current':current, 'current_no_pack':lambda:current(pack=False)}
    streams = {}
    for label,native in [('old_triton',()),('old_h100',('front','f567','dual_bwd'))]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            leaves = tuple(v.detach().clone().requires_grad_(True) for v in d['leaves'])
            mask = d['mask'].reshape(1,n,n).to(torch.bfloat16)
            ds = d['ds'].reshape(1,1,n,128)
        def run(leaves=leaves,mask=mask,ds=ds,native=native):
            settings.configure(trimul_sm90_kernels=native,autotune_miss_cap=24)
            with torch.enable_grad():
                return compiled_old(*leaves,mask,ds)
        funcs[label] = run
        streams[label] = stream
    if n == 768:
        funcs['current_recorded_K1_winner'] = lambda:current((1,128,2,2,1))

    graphs,outputs = {},{}
    for name,fn in funcs.items():
        print('CAPTURE',name,flush=True)
        graphs[name],outputs[name] = capture_outputs(fn,streams.get(name))
        graphs[name].replay()
        torch.cuda.synchronize()
    ref = outputs['current'][0]
    errs = {k:rel((v[0] if isinstance(v,tuple) else v),ref) for k,v in outputs.items()}
    assert max(errs.values())<.001,errs
    blocks = [paired(graphs) for _ in range(3)]
    times = pool(blocks)
    print('EVENT_TIMES',{k:v['median_us'] for k,v in times.items()},flush=True)
    traces = {}
    for name in ('current','old_triton','old_h100'):
        for _ in range(10): graphs[name].replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5): graphs[name].replay()
            torch.cuda.synchronize()
        path = R/('forward-trace-%s-L%d.json'%(name,n))
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        kernels = collections.defaultdict(list)
        for ev in trace['traceEvents']:
            if ev.get('cat') == 'kernel': kernels[ev['name']].append(ev['dur'])
        assert kernels,'Profiler did not record graph kernels'
        traces[name] = [dict(name=k,count=len(v),mean_us=sum(v)/len(v),total_us_per_replay=sum(v)/5)
                        for k,v in kernels.items()]
        print('TRACE',name,traces[name],flush=True)
    report = dict(L=n,dropout=.25,forward_saves=True,weight_pack_included=True,
        current_K1=(3,64,2,2,1),current_K3=T.default_config(256),old_h100_configs=configs,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                      (Path(__file__),Path(C.__file__),Path(T.__file__),Path(B.__file__))},
        forward_errors=errs,times=times,blocks=blocks,traces=traces,
        profiler_note='Five graph replays/path; per-kernel durations are diagnostic, use paired CUDA events for module latency')
    (R/('forward-audit-L%d.json'%n)).write_text(json.dumps(report,indent=2))
    print('DONE',n,flush=True)


if __name__=='__main__':main()
