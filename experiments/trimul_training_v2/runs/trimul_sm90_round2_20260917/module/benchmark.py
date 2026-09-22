import argparse,json,statistics
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import trimul_sm90_config as cfg
from benchmarks.runners import bench

p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--triton-configs',type=Path,required=True);p.add_argument('--length',type=int,default=384);p.add_argument('--configs-eight',type=Path);a=p.parse_args()
import triton
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import _bidir_front_kernel
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel
triton_configs=json.loads(a.triton_configs.read_text())
for name,kernel in [('front',_bidir_front_kernel),('f567',_output_f567_kernel),('dual_bwd',_input_dual_bwd_kernel)]:
 c=dict(triton_configs[name]);nw=c.pop('num_warps');ns=c.pop('num_stages')
 kernel.configs=[triton.Config(c,num_warps=nw,num_stages=ns)]
 kernel.cache.clear()
 kernel.early_config_prune=None
winners=json.loads(a.configs.read_text());base_winners=dict(winners);eight_winners=json.loads(a.configs_eight.read_text()) if a.configs_eight else None;original_resolve=cfg.resolve
# An explicit benchmark manifest, not a fabricated native-cache hit.
def measured_configs(op,tensors,**kwargs):
    if op in winners:
        chosen=winners[op];reason=kwargs['feasibility'](chosen)
        assert reason is None,(op,chosen,reason)
        return dict(chosen)
    return original_resolve(op,tensors,**kwargs)
cfg.resolve=measured_configs
class NoFabric:
 @staticmethod
 def setup_module(m):return m
 @staticmethod
 def backward(tensor,gradient):tensor.backward(gradient)
conf=bench.BenchConfig(target='triangle_multiplication_bidirectional',level='module',n_layers=1,mode='training',metric='time',compile=True,cudagraph='manual',precision='bf16-mixed',d_pair=128,dropout=0.0,min_seq_len=a.length,max_seq_len=a.length)
original=bench.bench_time;original_measured=bench.measured_result
calls={};owners={};rows={};arm=None

def record(fn,**kwargs):
 calls[arm]=(fn,kwargs.get('grad_to_none',[]));return original(fn,**kwargs)
def keep(**kwargs):
 owners[arm]=kwargs;return original_measured(**kwargs)
bench.bench_time=record;bench.measured_result=keep
arms=[('triton',()),('front',('front',)),('f567',('f567',)),('dual_bwd',('dual_bwd',)),('all',('front','f567','dual_bwd'))]
if eight_winners:arms.extend([('front8',('front',)),('all8',('front','f567','dual_bwd'))])
for arm,selected in arms:
 winners=eight_winners if arm in ('front8','all8') else base_winners
 torch.compiler.reset()  # Each arm owns a fresh static graph; avoid Dynamo's cross-arm recompile limit.
 settings.configure(engine_backend='triton',trimul_sm90_kernels=selected)
 with bench.forward_stream(conf):row=bench.bench_module_triangle_multiplication_bidirectional(conf,a.length,'triton',NoFabric())
 rows[arm]=row._asdict();assert row.compiled and row.cudagraph=='manual',rows[arm]
 print('HARNESS',arm,row.value,flush=True)
bench.bench_time=original
samples={k:[] for k in calls}
for _ in range(2):
 for fn,grads in calls.values():original(fn,warmup=20,rep=50,grad_to_none=grads)
for i in range(12):
 for key in list(calls)[::1 if i%2==0 else -1]:
  fn,grads=calls[key];samples[key].append(float(original(fn,warmup=10,rep=50,grad_to_none=grads)['median_ms']))
medians={k:statistics.median(v) for k,v in samples.items()}
report={'length':a.length,'config':conf.model_dump(),'explicit_sm90_configs':base_winners,'explicit_sm90_eight_configs':eight_winners,'explicit_triton_configs':triton_configs,'ms':medians,'speedup_vs_triton':{k:medians['triton']/v for k,v in medians.items()},'samples':samples,'harness':rows,'device':torch.cuda.get_device_name(),'scope':'Official module setup, static compile plus manual CUDA graph, FWD+BWD without optimizer, dropout=0 for the graph benchmark, 12 alternating rounds. Nonzero fixed-dropout correctness is checked separately. Explicit measured SM90 and Triton configs; no claim full-grid native cache complete.'}
Path('module/benchmark-L%d.json'%a.length).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'ms':medians,'speedup':report['speedup_vs_triton']}),flush=True)
