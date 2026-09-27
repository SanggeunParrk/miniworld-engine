"""H100 inference qualification: engine adapters, independent reference, CUDA graphs.

One candidate per process isolates incompatible binary loaders. NCU mode brackets
one warmed launch with cudaProfilerStart/Stop; setup is excluded from counters.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from miniworld_engine.integrations import anthropic as A
from miniworld_engine import settings

p = argparse.ArgumentParser()
p.add_argument('--family', required=True, choices=['trimul','triattn','transition','ln','apb','atom_window','gather','template','dtk','adaln','swiglu','msa_ln','opm','ln_linear','opm_core','pwa'])
p.add_argument('--length', type=int, default=384)
p.add_argument('--width', type=int, default=128)
p.add_argument('--direction', default='outgoing')
p.add_argument('--row', required=True)
p.add_argument('--output', required=True)
p.add_argument('--profile', action='store_true')
p.add_argument('--module', action='store_true')
a = p.parse_args()
torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
L,C=a.length,a.width
if a.row=='engine_triton':
    settings.configure(engine_backend='triton')
dev='cuda'; dt=torch.bfloat16
record=vars(a).copy()
record.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
              upstream_revision=A.REVISION, precision='bf16 operands; fp32 LN affine; TF32 disabled',
              measurement='warm one-call CUDA graph, 5 rounds x 50 replays; candidates in separate processes',
              engine_backend=settings.current().engine_backend,
              dropout=0, training=False)


def rand(*shape, scale=1., dtype=dt):
    return torch.randn(*shape,device=dev,dtype=dtype)*scale


def init(m):
    m=m.to(dev).eval()
    for name,t in m.named_parameters():
        if t.ndim>=2:
            t.data=t.data.to(dt)
            t.normal_(std=1/math.sqrt(t.shape[-1]))
        else:
            t.data=t.data.float()
            if name.endswith('weight'): t.normal_(1,.05)
            else: t.normal_(0,.05)
    return m


def error(y,r):
    d=y.float()-r.float()
    return {'rel_rms':float(d.square().mean().sqrt()/(r.float().square().mean().sqrt()+1e-12)),
            'max_abs':float(d.abs().max()),'finite':bool(torch.isfinite(y).all())}


def selection(sel):
    if isinstance(sel,dict):return sel
    if hasattr(sel,'as_dict'): return sel.as_dict()
    if hasattr(sel,'_asdict'): return sel._asdict()
    return str(sel)


def fixture():
    residual=None
    if a.family in {'trimul','transition'} or (a.family=='triattn' and a.module):
        from miniworld_engine.modules import TriangleMultiplication, Transition, TriangleAttention
        if a.family=='trimul':
            m=init(TriangleMultiplication(C, outgoing=a.direction=='outgoing', implementation='pytorch'))
        elif a.family=='transition':
            m=init(Transition(C, implementation='pytorch'))
        else:
            m=init(TriangleAttention(C,n_head=4,d_hidden=C,implementation='pytorch'))
        x=rand(1,L,L,C)
        mask=torch.ones(1,L,device=dev,dtype=torch.bool); mask[:,::7]=False
        mr=copy.deepcopy(m).float()
        ref=mr(x.float()) if a.family=='transition' else mr(x.float(),mask)
        residual=x
        def call(mod): return mod(x) if a.family=='transition' else mod(x,mask)
        if a.row in {'pytorch','pytorch_compile','engine_triton'}:
            if a.row=='engine_triton':
                from miniworld_engine.modules.dispatch import KernelBackend
                m._backend=KernelBackend.TRITON
            if a.row=='pytorch_compile': m=torch.compile(m, dynamic=False, fullgraph=True)
            return lambda:call(m),ref,residual
        if a.family=='trimul':
            target=init(TriangleMultiplication(C,outgoing=a.direction=='outgoing',implementation='anthropic',anthropic_row=a.row))
        elif a.family=='transition':
            target=init(Transition(C,implementation='anthropic',anthropic_row=a.row))
        else:
            target=init(TriangleAttention(C,n_head=4,d_hidden=C,implementation='anthropic',anthropic_row=a.row))
        target.load_state_dict(m.state_dict())
        def fn():
            y=call(target)
            if hasattr(target,'anthropic_selection'):
                record['selection']=selection(target.anthropic_selection)
            return y
        return fn,ref,residual
    if a.family=='triattn':
        H,D=4,C//4
        q,k,v=(rand(1,L,H,L,D) for _ in range(3))
        b=rand(1,1,H,L,L,dtype=torch.float32,scale=.5)
        mask=torch.ones(1,L,1,1,L,device=dev,dtype=torch.bool);mask[...,::7]=False
        # Chunk only the independent outer rows, with the full attention length.
        pieces=[]
        for start in range(0,L,16):
            s=(q[:,start:start+16].float() @ k[:,start:start+16].float().transpose(-1,-2))/math.sqrt(D)
            s=(s+b).masked_fill(~mask[:,start:start+16],float('-inf'))
            pieces.append(s.softmax(-1) @ v[:,start:start+16].float())
        ref=torch.cat(pieces,1); del pieces
        if a.row=='engine_triton':
            from miniworld_engine.kernels.triangle_attention.triton.main import triton_triangle_attention_pair_bias
            qe,ke,ve=(t.permute(0,2,1,3,4).contiguous() for t in (q,k,v))
            be=b[:,0].to(dt).masked_fill(~mask[:,0,0,0,None,None,:],torch.finfo(dt).min).contiguous()
            return lambda:triton_triangle_attention_pair_bias(qe,ke,ve,be).permute(0,2,1,3,4),ref,None
        if a.row=='cueq':
            from cuequivariance_torch import triangle_attention
            return lambda:triangle_attention(q,k,v,b,mask,D**-.5),ref,None
        T=A.provider('triattn')
        stack=T._tensor_facts(q,k)
        from opt_core.kernels.triattn import cuda_sm90a
        sel=T.select(*stack,word=a.row,stack=cuda_sm90a.stack_key())
        record['selection']=selection(sel)
        return lambda:A.triangle_attention(q,k,v,b,mask,row=a.row,selection=sel),ref,None
    if a.family=='ln':
        x=rand(1,L,L,C);w=rand(C,dtype=torch.float32,scale=.05)+1;b=rand(C,dtype=torch.float32,scale=.05)
        ref=torch.nn.functional.layer_norm(x.float(),(C,),w,b,1e-5)
        if a.row=='engine_triton':
            from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
            return lambda:triton_layernorm(x,w,b,1e-5),ref,None
        if a.row=='pytorch': return lambda:torch.nn.functional.layer_norm(x.float(),(C,),w,b,1e-5).to(dt),ref,None
        def fn():
            y,s=A.layer_norm(x,C,w,b,row=a.row,out_dtype=dt,n_tokens=L)
            record['selection']=selection(s)
            return y
        return fn,ref,None
    if a.family=='apb':
        H,D=16,24;S=2
        q,k,v=(rand(S,L,H,D) for _ in range(3))
        b=rand(1,H,L,L,dtype=torch.float32,scale=.5)
        mask=torch.ones(1,L,device=dev,dtype=torch.bool);mask[:,::7]=False
        scores=q.float().permute(0,2,1,3) @ k.float().permute(0,2,3,1)/math.sqrt(D)
        ref=((scores+b).masked_fill(~mask[:,None,None,:],float('-inf')).softmax(-1) @ v.float().permute(0,2,1,3)).permute(0,2,1,3)
        def fn():
            y,s=A.pair_bias_attention(q,k,v,b,mask,row=a.row,cell='pf_h16d24')
            record['selection']=selection(s)
            return y
        return fn,ref,None
    if a.family in {'atom_window','gather','template'}:
        import importlib.util
        filename={'atom_window':'test_atom_window_gpu','gather':'test_gather_attn_gpu','template':'test_templ_embed_gpu'}[a.family]
        path=A.configure()/'common/opt_core/tests/gpu'/f'{filename}.py'
        spec=importlib.util.spec_from_file_location('_upstream_fixture',path)
        T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
        record['fixture_source']=str(path)
        if a.family=='atom_window':
            K=A.carried_kernel('atom_window')
            precision='tf32rn' if a.row=='tf32rn' else 'ieee'
            x,lin,cond,bias,amask,ks,nreal=T._problem(2,L*8,L*8-7,128,4,123)
            ref=T._reference(x,lin,cond,bias,amask,ks,nreal,4,dtype=torch.float32)
            def fn():
                qkvg=K.ln_qkvg(x,cond['gq'],cond['lsq'],cond['gk'],cond['lsk'],lin['q'],lin['k'],lin['v'],lin['g'],1e-5,32**-.5,precision=precision)
                return K.window_attn(qkvg,x,bias,ks,nreal,amask,cond['gate'],lin['o'].weight,lin['o'].bias,4,32,128,1e9,precision=precision)
            record['precision']='upstream fp32 atom-window interface; explicit dot precision '+precision
            return fn,ref,x
        if a.family=='gather':
            K=A.carried_kernel('gather_attn')
            q,k,v,bias,idx,H,gate=T._case(B=2,LQ=L,LK=L,H=4,dh=32,k=128,shared_bias=True,shared_idx=True,qk16=True)
            idx=torch.sort(idx,dim=-1).values.contiguous()
            record['index_preparation']='sorted once outside replay; ensure_sorted=False avoids upstream host sync'
            ref=K.reference(q,k,v,bias,idx,H,gate=gate,round_qk=True)
            return lambda:K.gather_attn(q,k,v,bias,idx,H,gate=gate,allow_candidate=False,ensure_sorted=False),ref,None
        K=A.carried_kernel('templ_embed')
        dg,uv,pbm,bbm,asym,restype,zp,W=T._problem(2,L,seed=123)
        P=K.pack_weights(**{k:v for k,v in W.items() if k not in ('w_aa1','w_aa2')})
        zp=zp.to(dt)
        ref=K.reference_embed(dg,uv,pbm,bbm,asym,restype,zp.float(),**{k:v for k,v in W.items() if k!='w_t'},dtype=torch.float32)
        wa,wb=W['w_aa1'].to(dt),W['w_aa2'].to(dt)
        def fn():
            ri=restype.to(dt) @ wa.T;rj=restype.to(dt) @ wb.T
            return K.embed(dg,uv,pbm,bbm,asym,ri.contiguous(),rj.contiguous(),zp,P)
        return fn,ref,None
    if a.family in {'adaln','swiglu'}:
        K=A.carried_kernel('dtk_kernels')
        if a.family=='swiglu':
            ab=rand(5*L,3072);aa,bb=ab.float().chunk(2,-1)
            return lambda:K.swiglu(ab),torch.nn.functional.silu(aa)*bb,None
        x=rand(5*L,768);scale=rand(L,768);shift=rand(L,768)
        w=rand(768,dtype=torch.float32,scale=.05)+1;b=rand(768,dtype=torch.float32,scale=.05)
        ref=torch.nn.functional.layer_norm(x.float(),(768,),w,b,1e-5)*scale.float().sigmoid().repeat(5,1)+shift.float().repeat(5,1)
        return lambda:K.ln_modulate(x,scale,shift,w,b,mod_period=L),ref,None
    if a.family in {'msa_ln','ln_linear'}:
        x=rand(64,L,128);w=rand(128,dtype=torch.float32,scale=.05)+1;b=rand(128,dtype=torch.float32,scale=.05)
        ws=[rand(64,128,scale=128**-.5),rand(64,128,scale=128**-.5)]
        norm=torch.nn.functional.layer_norm(x.float(),(128,),w,b,1e-5)
        if a.family=='msa_ln':
            K=A.operation('msa_fused.msa_triton')
            ref=torch.cat([norm.to(dt).float() @ t.float().T for t in ws],-1)
            return lambda:torch.cat(K.ln_linear(x,w,b,ws),-1),ref,None
        K=A.carried_kernel('ln_proj');wt=torch.cat(ws,0)
        packed=K.pack_ln_linear_weights(w,b,wt,None,1e-5,x.device)
        return lambda:K.ln_linear(x.reshape(-1,128),packed).reshape_as(x),norm @ wt.float().T,None
    if a.family=='opm_core':
        K=A.operation('msa_opm')
        aa=rand(64,L,32);bb=rand(64,L,32);wt=rand(1024,C,scale=1024**-.5);bias=rand(C,dtype=torch.float32,scale=.05)
        cache=dict(CH=32,CZ=C,wout_t=wt,bias32=bias)
        outer=torch.einsum('sic,sjd->ijcd',aa.float(),bb.float()).to(dt)
        ref=(outer.float().reshape(L,L,1024) @ wt.float()+bias).to(dt).float()/64
        return lambda:K.opm_core(aa,bb,cache,'scalar_norm',norm_scalar=64),ref,None
    if a.family=='pwa':
        K=A.operation('msa_pwa2')
        from types import SimpleNamespace
        m=rand(1,64,L,64,dtype=torch.float32);z=rand(1,L,L,128,dtype=torch.float32)
        mod=SimpleNamespace(norm_m=torch.nn.LayerNorm(64).cuda(),norm_z=torch.nn.LayerNorm(128).cuda(),
            proj_m=torch.nn.Linear(64,256,bias=False).cuda(),proj_g=torch.nn.Linear(64,256,bias=False).cuda(),
            proj_z=torch.nn.Linear(128,8,bias=False).cuda(),proj_o=torch.nn.Linear(256,64,bias=False).cuda(),
            training=False,inf=1e9,num_heads=8,c_h=32)
        mask=torch.ones(1,L,L,device=dev);mask[...,::7]=0
        mn=mod.norm_m(m);zn=mod.norm_z(z)
        v=(mn @ mod.proj_m.weight.T).view(1,64,L,8,32)
        g=(mn @ mod.proj_g.weight.T).sigmoid().view_as(v)
        b=(zn @ mod.proj_z.weight.T).permute(0,3,1,2)+(1-mask[:,None])*-mod.inf
        out=torch.einsum('bhij,bsjhd->bsihd',b.softmax(-1),v)*g
        ref=out.reshape(1,64,L,256) @ mod.proj_o.weight.T
        def fn():
            with torch.autocast('cuda',dtype=dt):return K.pwa_forward(mod,m,z,mask,chunk_heads=False)
        return fn,ref,None
    if a.family=='opm':
        K=A.operation('msa_fused.msa_triton')
        outer=rand(L,32,L,32).permute(0,2,1,3)
        w=rand(1024,128,scale=1024**-.5);b=rand(128);norm=torch.rand(L,L,device=dev)+1
        out=torch.empty(L,L,128,device=dev,dtype=dt)
        ref=((outer.float().reshape(L,L,1024) @ w.float()+b.float()).to(dt).float()/norm[...,None])
        return lambda:K.opm_out(outer,w,b,norm,out),ref,None
    if a.family=='dtk':
        K=A.carried_kernel('dtk_kernels')
        x=rand(5*L,768);g=rand(L,768);res=rand(5*L,768,dtype=torch.float32)
        mask=torch.rand(L,device=dev)>.2
        ref=res+mask.repeat(5)[:,None]*torch.sigmoid(g.float()).repeat(5,1)*x.float()
        return lambda:K.gate_residual(x,gate=g,gate_period=L,rowmask=mask,mask_period=L,res=res,out_dtype=torch.float32),ref,res
    raise NotImplementedError(a.family)


try:
    with torch.no_grad():
        fn,ref,residual=fixture()
        t=time.time();y=fn();torch.cuda.synchronize()
        record['first_call_s']=time.time()-t
        record['error']=error(y,ref)
        if residual is not None: record['update_error']=error(y.float()-residual.float(),ref.float()-residual.float())
        e=record.get('update_error',record['error'])
        if not e['finite'] or e['rel_rms']>.03: raise AssertionError(f'Numerical check failed: {e}')
        for _ in range(3): fn()
        torch.cuda.synchronize()
        if a.profile:
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f'{a.family}/{a.row}/L{L}/C{C}')
            fn()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                fn();s.synchronize()
                with torch.cuda.graph(graph,stream=s): out=fn()
            torch.cuda.current_stream().wait_stream(s)
            graph.replay();torch.cuda.synchronize()
            record['graph_error']=error(out,ref)
            if not record['graph_error']['finite'] or record['graph_error']['rel_rms']>.03:
                raise AssertionError(f"CUDA graph numerical check failed: {record['graph_error']}")
            times=[]
            for _ in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):graph.replay()
                end.record();end.synchronize()
                times.append(start.elapsed_time(end)/50)
            record['ms']=statistics.median(times);record['samples_ms']=times
        record['status']='passed'
except (Exception, SystemExit) as e:
    record.update(status='failed',error_type=type(e).__name__,reason=str(e))
    traceback.print_exc()
finally:
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(record,indent=2,default=str)+'\n')
    print(json.dumps(record,default=str),flush=True)
