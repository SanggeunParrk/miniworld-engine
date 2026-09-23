"""Production-module version comparison. Each backend runs in a fresh process."""
import argparse
import collections
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import traceback

# Dependency location only; do not change historical kernel dispatch or algorithms.
os.environ.setdefault('MINIWORLD_MATHDX_HOME','/home/psk6950/mathdx_dl/extracted/nvidia/mathdx')

import torch
from torch import nn
import miniworld_engine
from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.transition import Transition

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--arm', required=True)
parser.add_argument('--module', required=True)
parser.add_argument('--length', type=int, required=True)
parser.add_argument('--msa-depth', type=int, default=1024)
args = parser.parse_args()
dest = ROOT / f'{args.module}-{args.length}-{args.arm}.json'
record = dict(arm=args.arm, module=args.module, L=args.length, source=miniworld_engine.__file__, harness_revision=2,
              gpu=torch.cuda.get_device_name(), uuid=str(torch.cuda.get_device_properties(0).uuid),
              job=os.getenv('SLURM_JOB_ID'), modes={}, dtype='bf16', compile=True,
              graph=True, mask=True, dropout_rng_included=True,
              versions={k:importlib.metadata.version(k) for k in ('torch','triton','cuequivariance-ops-torch-cu12')})
record['engine_backend']=getattr(settings.current(),'engine_backend','historical architecture default')
record['msa_depth']=args.msa_depth if args.module in ('opm','pwa','msa') else None
def save():
    dest.write_text(json.dumps(record, indent=2))

class Block(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.trimul = BidirectionalTriangleMultiplication(128, implementation=impl, p_drop=.25)
        self.transition = Transition(128, n=4, implementation=I.PYTORCH if impl == I.CUEQUIVARIANCE else impl)
    def forward(self,x,mask):
        return self.transition(self.trimul(x,mask))

class OPMResidual(nn.Module):
    def __init__(self,impl):
        super().__init__()
        from miniworld_engine.modules.outer_product import OuterProductMean
        self.opm=OuterProductMean(64,128,32,implementation=impl)
    def forward(self,msa,mask,pair):
        return self.opm(msa,mask,residual=pair)

def setup():
    impl = {'pytorch':I.PYTORCH,'cuequiv':I.CUEQUIVARIANCE}.get(args.arm,I.MINIWORLD)
    L = args.length
    torch.manual_seed(90323)
    kw = dict(device='cuda',dtype=torch.bfloat16)
    def rand(*shape): return torch.randn(*shape,**kw).requires_grad_()
    mask = torch.rand(1,L,device='cuda')>.1
    name = args.module
    if name in ('trimul','single','transition','block'):
        m = {'trimul':lambda:BidirectionalTriangleMultiplication(128,implementation=impl,p_drop=.25),
             'single':lambda:TriangleMultiplication(128,implementation=impl,p_drop=.25),
             'transition':lambda:Transition(128,n=4,implementation=impl),
             'block':lambda:Block(impl)}[name]()
        x=rand(1,L,L,128)
        inputs=(x,) if name=='transition' else (x,mask)
    elif name=='opm':
        m=OPMResidual(impl)
        inputs=(rand(1,args.msa_depth,L,64),torch.rand(1,args.msa_depth,L,device='cuda')>.1,rand(1,L,L,128))
    elif name=='pwa':
        from miniworld_engine.modules.msa_pair_weighted_averaging import MSAPairWeightedAveraging
        m=MSAPairWeightedAveraging(64,128,8,32,implementation=impl,p_drop=.15)
        inputs=(rand(1,args.msa_depth,L,64),rand(1,L,L,128),mask)
    elif name=='msa':
        if args.arm=='engine1':
            # Load the same block body without current team-gm's unrelated SWA imports,
            # which require symbols that did not exist in the v1.0.0 engine.
            import ast
            import __future__
            from miniworld_engine.modules import OuterProductMean, MSAPairWeightedAveraging
            source=ROOT.parents[2]/'src/miniworld/modules/mini_msa_module.py'
            tree=ast.parse(source.read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMSAModuleBlock')
            namespace=dict(__name__=__name__,nn=nn,torch=torch,ImplementationType=I,
                _to_engine_impl=lambda impl:impl,typecheck=lambda fn:fn,
                OuterProductMean=OuterProductMean,MSAPairWeightedAveraging=MSAPairWeightedAveraging,
                BidirectionalTriangleMultiplication=BidirectionalTriangleMultiplication,Transition=Transition)
            exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec',
                         flags=__future__.annotations.compiler_flag),namespace)
            m=namespace['MiniMSAModuleBlock'](last_block=False,implementation=I.MINIWORLD)
            record['block_import']='unmodified class body; direct historical engine imports; typecheck disabled'
        else:
            from miniworld.modules.mini_msa_module import MiniMSAModuleBlock
            from team_gm.modules.exceptions import ImplementationType as ModelImpl
            m=MiniMSAModuleBlock(last_block=False,implementation=(
                ModelImpl.PYTORCH if args.arm=='pytorch' else ModelImpl.MINIWORLD_ENGINE))
        inputs=(rand(1,args.msa_depth,L,64),torch.rand(1,args.msa_depth,device='cuda')>.1,
                rand(1,L,L,128),mask)
        record['scope']='MiniMSAModuleBlock(last_block=False), both MSA and pair output gradients'
    else:
        from miniworld_engine.modules.dit import DiTBlock
        m=DiTBlock(implementation=impl)
        inputs=(rand(1,1,L,768),rand(1,1,L,384),rand(1,L,L,128),mask)
    m=m.cuda().bfloat16()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if p.ndim>=2:p.normal_(std=p.shape[-1]**-.5)
            elif 'weight' in n:p.copy_(1+.1*torch.randn_like(p))
            else:p.normal_(std=.05)
    record['resolved_backends']={n:str(v._backend) for n,v in m.named_modules() if hasattr(v,'_backend')}
    record['parameter_dtypes']=sorted({str(p.dtype) for p in m.parameters()})
    return m,inputs

try:
    if args.arm=='cuequiv':
        import cuequivariance_ops_torch
        cuequivariance_ops_torch.init_triton_cache()
    m,inputs=setup()
    for mode in ('inference','training'):
        result={};record['modes'][mode]=result;save()
        try:
            torch.compiler.reset()
            train=mode=='training';m.train(train)
            compiled=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.set_grad_enabled(train), torch.cuda.stream(side):
                if args.arm=='engine1':
                    # The historical release initializes imports and patches in its first forward.
                    warm=m(*inputs)
                    if train:
                        warm_tuple=warm if isinstance(warm,tuple) else (warm,)
                        warm_grads=torch.autograd.grad(warm_tuple,tuple(x for x in inputs if x.requires_grad)+tuple(m.parameters()),tuple(torch.ones_like(v) for v in warm_tuple))
                        del warm_grads
                    del warm
                y=compiled(*inputs)
                dy=tuple(torch.randn_like(v) for v in y) if isinstance(y,tuple) else torch.randn_like(y)
                del y
                leaves=tuple(x for x in inputs if x.requires_grad)+tuple(m.parameters())
                def step(compiled=compiled,leaves=leaves,dy=dy,train=train):
                    out=compiled(*inputs)
                    return (out,torch.autograd.grad(out,leaves,dy)) if train else (out,())
                for _ in range(6):step()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=side):out,grads=step()
            torch.cuda.current_stream().wait_stream(side)
            graph.replay();torch.cuda.synchronize()
            outputs=out if isinstance(out,tuple) else (out,)
            result['finite']=all(bool(v.isfinite().all()) for v in outputs) and all(bool(g.isfinite().all()) for g in grads)
            assert result['finite']
            probe_start=torch.cuda.Event(enable_timing=True);probe_end=torch.cuda.Event(enable_timing=True)
            probe_start.record()
            for _ in range(10):graph.replay()
            probe_end.record();probe_end.synchronize()
            estimated_ms=probe_start.elapsed_time(probe_end)/10
            warm_replays=min(10000,max(30,int(300/max(estimated_ms,.001))))
            for _ in range(warm_replays):graph.replay()
            result['warm_replays']=warm_replays
            torch.cuda.synchronize()
            rounds=[]
            for _ in range(7):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):graph.replay()
                end.record();end.synchronize();rounds.append(start.elapsed_time(end)/50)
            result.update(ms=statistics.median(rounds),rounds_ms=rounds)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                graph.replay();torch.cuda.synchronize()
            trace=dest.with_name(dest.stem+'-'+mode+'-trace.json');prof.export_chrome_trace(str(trace))
            events=json.loads(trace.read_text())['traceEvents']
            result['kernels']=dict(collections.Counter(e['name'] for e in events if e.get('cat')=='kernel'))
            result['profile_scope']='graph replay'
            if not result['kernels']:
                with torch.set_grad_enabled(train), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step();torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace))
                events=json.loads(trace.read_text())['traceEvents']
                result['kernels']=dict(collections.Counter(e['name'] for e in events if e.get('cat')=='kernel'))
                result['profile_scope']='compiled step (replay trace empty); timing remains graph replay'
            print('RESULT',args.module,args.length,args.arm,mode,result['ms'],flush=True)
            del graph,compiled,out,grads,dy,step
        except Exception:
            result['error']=traceback.format_exc();print(result['error'],flush=True)
        save()
except Exception:
    record['error']=traceback.format_exc();print(record['error'],flush=True);save()
