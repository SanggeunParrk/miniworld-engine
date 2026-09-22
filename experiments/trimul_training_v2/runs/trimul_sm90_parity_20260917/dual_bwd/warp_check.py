import json,torch,triton
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cute.parity_dual_bwd import input_dual_bwd_sm90_impl,DEFAULT_CONFIG
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel,dual_shape_key
from miniworld_engine.kernels._tiles import tile_grid

torch.manual_seed(73)
results=[]
configs=[DEFAULT_CONFIG,{**DEFAULT_CONFIG,'BLOCK_M1':128,'num_warps':8,'GROUP_M':4},
 {**DEFAULT_CONFIG,'BLOCK_K':32,'BLOCK_N':32,'num_stages':3,'GROUP_M':8},
 {**DEFAULT_CONFIG,'BLOCK_K':128,'BLOCK_N':64,'num_stages':3,'GROUP_M':2}, {**DEFAULT_CONFIG,'BLOCK_K':32,'num_stages':4}]
configs=[{**DEFAULT_CONFIG,'num_warps':8}, {**DEFAULT_CONFIG,'BLOCK_M1':128,'num_warps':4}]
for shape in [(512,256,1024,128),(523,40,72,48)]:
 m,kg,kp,n=shape
 g=torch.randn(m,kg+8,device='cuda',dtype=torch.bfloat16)[:,:kg]*.1
 # preserve pitched rowmajor after scaling
 g=torch.randn(m,kg+8,device='cuda',dtype=torch.bfloat16)[:,:kg]
 f=torch.randn(kp,((m+7)//8)*8+8,device='cuda',dtype=torch.bfloat16)[:,:m].t()
 w=torch.randn(n,kg+8,device='cuda',dtype=torch.bfloat16)[:,:kg].t()
 v=torch.randn(kp,n+8,device='cuda',dtype=torch.bfloat16)[:,:n]
 r=(f.float()@v.float()+(g.float()@w.float()).bfloat16().float()).bfloat16()
 for c in configs:
  print("launch",shape,c,flush=True)
  y=input_dual_bwd_sm90_impl(g,f,w,v,128,c)
  torch.cuda.synchronize();print("cute returned",flush=True)
  z=torch.empty_like(y)
  tc={k:val for k,val in c.items() if k not in ('num_warps','num_stages')}
  _input_dual_bwd_kernel.fn[lambda _: (triton.cdiv(m,c['BLOCK_M1'])*triton.cdiv(n,c['BLOCK_N']),)](g,f,w,v,z,m,kg,kp,n,*g.stride(),*f.stride(),*w.stride(),*v.stride(),**tc,shape_key=dual_shape_key(128,kg,kp,n),num_warps=c['num_warps'],num_stages=c['num_stages'])
  torch.cuda.synchronize()
  rel=((y-r).float().norm()/r.float().norm()).item();tr=((y-z).float().norm()/z.float().norm()).item()
  item=dict(shape=shape,config=c,reference_rel_l2=rel,triton_rel_l2=tr,maxerr=(y-z).abs().max().item())
  print(json.dumps(item),flush=True);results.append(item)
  assert rel<.0001 and tr<.0001,item
Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/dual_bwd/warp-check.json').write_text(json.dumps(results,indent=2))
