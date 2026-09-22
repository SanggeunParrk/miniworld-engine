from core import *
row=json.loads((R/'results.json').read_text())[-1];cf=row['configs'];n=row['N'];torch.manual_seed(677)
x=torch.randn(1,n,n,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(n,n,device='cuda')>.2).float();tri=torch.randn(256,n,n,device='cuda',dtype=x.dtype)
lnc=tuple(cf['ln_anthropic']);tc=tuple(cf['ln_triton']);xn=ln(x,g,b,lnc)
ops={'ln-anthropic':lambda:ln(x,g,b,lnc),'ln-triton':lambda:triton_ln(x,g,b,tc)}
for kind in ('k1','k3'):
 for mode in ('original','split'):
  cfg=tuple(cf[kind+'_'+mode]);fused=mode=='original';operand=x if fused else xn
  if kind=='k1':ops[kind+'-'+mode]=lambda cfg=cfg,fused=fused,operand=operand:front(operand,w,mask,g,b,fused,cfg)
  else:ops[kind+'-'+mode]=lambda cfg=cfg,fused=fused,operand=operand:output(tri,operand,wp,wg,g,b,go,bo,x,fused,cfg,tma_residual=row.get("residual_tma",False) and not fused)
for name,fn in ops.items():
 torch.cuda.nvtx.range_push(name);y=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop();print(name,flush=True)
records=[]
for kind in ('k1','k3'):
 for mode in ('original','split'):
  cfg=tuple(cf[kind+'_'+mode]);impl='k3_tma' if kind=='k3' and mode=='split' and row.get('residual_tma',False) else kind;k,sm=kernel(impl,mode=='original',cfg);p=build(impl,mode=='original',cfg)
  records.append(dict(kind=kind,mode=mode,config=cfg,smem=sm,attrs=k.attrs(),cubin=str(p)))
(R/'binary-selected.json').write_text(json.dumps(records,indent=2))
