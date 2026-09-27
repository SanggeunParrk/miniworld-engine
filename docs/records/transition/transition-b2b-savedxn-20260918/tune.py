import argparse,json,socket
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture,native
from miniworld_engine.autotune.hopper_cuda_config import candidates
from miniworld_engine.autotune.native_compile import precompile
from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd,transition_b2b_fwd_saved
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=128);a=p.parse_args()
out=Path(__file__).parent;op='transition_fwd_b2b_sm90_cuda';grid=candidates('b2b',a.width)
settings.configure(run_autotune=False,autotune_miss_cap=3,compile_jobs=2,bench_rep_ms=100,bench_clear_mb=256)
capture.reset();capture.set_incremental(False);capture.set_round_cache(str(out/f'native-rounds-D{a.width}'))
report={'node':socket.gethostname(),'width':a.width,'source_identity':native.source_identity(),'candidates':grid,'rows':[]}
for length in ([384,768] if a.width==128 else [384]):
 torch.manual_seed(812)
 m=length**2;k=a.width
 x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
 g=torch.randn(k,device='cuda',dtype=x.dtype);b=torch.randn_like(g)
 wa=torch.randn(4*k,k,device='cuda',dtype=x.dtype)/k**.5;wb=torch.randn_like(wa)/k**.5
 ws=torch.randn(k,4*k,device='cuda',dtype=x.dtype)/(4*k)**.5
 rs,c1=stats_triton(x,1e-5);args=(x,rs,c1,g,b,wa,wb,ws)
 bucket=native.tensor_key(*args,extra=(True,'save_xn'))
 print('PRECOMPILE',length,len(grid),flush=True)
 comp=precompile(op,grid,bucket)
 assert all(v['status']=='ok' for v in comp.values()),comp
 expected=transition_b2b_fwd_saved(*args,config=grid[0])
 for config in grid:
  y,xn=transition_b2b_fwd_saved(*args,config=config)
  torch.testing.assert_close(y,expected[0],rtol=0,atol=0)
  torch.testing.assert_close(xn,expected[1],rtol=0,atol=0)
  y0=transition_b2b_fwd(*args,config=config)
  torch.testing.assert_close(y0,y,rtol=0,atol=0)
  print('CORRECT',length,config,flush=True)
 for saved in [False,True]:
  fn=transition_b2b_fwd_saved if saved else transition_b2b_fwd
  settings.configure(run_autotune=True)
  fn(*args)
  settings.configure(run_autotune=False)
  key=native.tensor_key(*args,extra=(True,'save_xn') if saved else (True,))
  selected=native.choose_config(op,grid,dtype=str(x.dtype),bucket=key,run=lambda c:fn(*args,config=c))
  # Until shard publication the runtime selector may return its default;
  # every measured candidate and winner is preserved in the shard below.
  report['rows'].append({'length':length,'save_xn':saved,'bucket':key,'validated_configs':len(grid)})
 capture.dump_shard(str(out/f'native-D{a.width}.shard.json'),unit_complete=True)
 (out/f'tune-D{a.width}.json').write_text(json.dumps(report,indent=2)+'\n')
print(capture.summary(),flush=True)
