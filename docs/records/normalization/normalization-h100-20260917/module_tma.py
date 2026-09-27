"""Official TriMul fixture with an explicit stochastic manual-graph measurement adapter.

BenchConfig remains cudagraph=disabled because its stock replay checker assumes
identical outputs. This adapter replaces that checker with RNG/reset/finiteness
checks; it never disables dropout or substitutes a fixed mask in timed execution.
"""
import argparse,gc,hashlib,inspect,json,statistics
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.autotune import trimul_sm90_config as cfg
from benchmarks.runners import bench
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import _bidir_front_kernel
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel

parser=argparse.ArgumentParser();parser.add_argument('--length',type=int,required=True);parser.add_argument('--builds',type=int,default=2);a=parser.parse_args()
root=Path(__file__).parent;prior=root
sm90=json.loads((prior/f'measured-configs-L{a.length}.json').read_text());tri=json.loads((prior/f'triton-configs-L{a.length}.json').read_text())
for name,kernel in [('front',_bidir_front_kernel),('f567',_output_f567_kernel),('dual_bwd',_input_dual_bwd_kernel)]:
 c=dict(tri[name]);nw=c.pop('num_warps');ns=c.pop('num_stages');kernel.configs=[triton.Config(c,num_warps=nw,num_stages=ns)];kernel.cache.clear();kernel.early_config_prune=None
resolve=cfg.resolve
native_calls={}
def explicit(op,tensors,**kw):
 if op not in sm90:return resolve(op,tensors,**kw)
 c=dict(sm90[op]);reason=kw['feasibility'](c);assert reason is None,(op,reason);native_calls[op]=native_calls.get(op,0)+1;return c
cfg.resolve=explicit
class NoFabric:
 @staticmethod
 def setup_module(m):return m
 @staticmethod
 def backward(y,dy):y.backward(dy)
conf=bench.BenchConfig(target='triangle_multiplication_bidirectional',level='module',n_layers=1,mode='training',metric='time',compile=True,cudagraph='disabled',precision='bf16-mixed',d_pair=128,dropout=.25,min_seq_len=a.length,max_seq_len=a.length)

def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def snapshot(tensors):return [t.detach().clone() for t in tensors]
def metrics(tensors,old):
 result=[]
 for x,y in zip(tensors,old,strict=True):
  diff=x.float()-y.float();result.append({'equal':torch.equal(x,y),'max_abs':diff.abs().max().item(),'relative_l2':(diff.norm()/y.float().norm().clamp_min(1e-20)).item()})
 return result

report={'length':a.length,'batch':1,'d_pair':128,'dropout':.25,'fixture_config':conf.model_dump(),'actual_capture':'manual stochastic adapter outside stock measured_result','source':'official bench_module_triangle_multiplication_bidirectional fixture; its separate paired-dropout correctness runs unchanged','timed_scope':'static-compiled forward plus backward with fresh production torch.rand dropout on every CUDA graph replay; fresh-gradient overwrite graph; no optimizer','explicit_sm90_configs':sm90,'explicit_triton_configs':tri,'builds':[],'device':torch.cuda.get_device_name()}
runtime_chosen={}
report['arms']={'triton':'latest H100 path with updated Triton B4','h100':'same H100 path plus experimental CuTe TMA B4'}
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent
from norm_tma import prepare
orig_b4=_ln_bwd_persistent.run
def candidate_b4(*args,**kw):
 if arm!='h100':return orig_b4(*args,**kw)
 amap=dict(zip(_ln_bwd_persistent.arg_names,args));amap.update(kw)
 assert amap['N']==256 and amap['stride_r']==1
 native_calls['b4_tma']=native_calls.get('b4_tma',0)+1
 c=dict(BLOCK_M1=32,BLOCK_K=256,num_warps=8,num_stages=2)
 prepare(amap['X'],amap['DY'],amap['W'],amap['W'],amap['Mean'],amap['Rstd'],amap['DX'],amap['PART_DW'],amap['PART_DB'],c,True)()
_ln_bwd_persistent.run=candidate_b4
original_measured=bench.measured_result
for build in range(a.builds):
 handles={};arm=None
 def adapter(**kw):
  fn=kw['func'];leaves=kw['grad_to_none'];params=kw['params'];assert kw['is_train']
  training=inspect.getclosurevars(fn).nonlocals;forward=inspect.getclosurevars(training['inference_step']).nonlocals;model=forward['model'];pair=forward['pair'];mask=forward['mask'];dy=training['dy']
  for layer in model.layers:
   assert layer.training and layer.p_drop==.25
   method=layer._make_drop_row_scale
   assert getattr(method,'__func__',None) is type(layer)._make_drop_row_scale, 'paired correctness mask was not restored'
   assert torch.count_nonzero(layer.to_out.weight)>0
  provenance={'pair':digest(pair),'dy':digest(dy),'token_mask':digest(mask),'parameters':{n:digest(p) for n,p in model.named_parameters()},'output_projection_nonzero':[int(torch.count_nonzero(layer.to_out.weight)) for layer in model.layers],'dropout_method_restored':True}
  captured={}
  def fresh_step():
   # Same official fresh-gradient capture semantics: capture AccumulateGrad with
   # grad absent, creating overwrite-producing graph nodes/static output buffers.
   for leaf in leaves:leaf.grad=None
   captured['output']=fn()
   captured['gradients']=tuple(leaf.grad for leaf in leaves)
   return captured['output']
  with bench.observe_execution() as evidence:
   fresh_step()
   finite=bench.check_finite_outputs(captured)
   graph=bench.capture_cudagraph(fresh_step,params,is_train=True,warmup_iters=8)
  bench.require_compile_evidence(True,evidence)
  assert all(t is not None for t in captured['gradients'])
  tensors=[captured['output'],*captured['gradients']]
  ptrs=[t.data_ptr() for t in tensors]
  # Deterministic RNG reset is validation only; no RNG reset is used in timing.
  torch.cuda.synchronize();rng0=torch.cuda.get_rng_state()
  graph.replay();torch.cuda.synchronize();first=snapshot(tensors)
  torch.cuda.set_rng_state(rng0)
  graph.replay();torch.cuda.synchronize();repeat=metrics(tensors,first)
  assert all(r['relative_l2']<1e-5 for r in repeat),('gradient overwrite/RNG repeat failed',repeat)
  assert ptrs==[t.data_ptr() for t in tensors]
  # Now leave the RNG alone: fresh masks must change output AND input gradient.
  state_a=torch.cuda.get_rng_state();second=snapshot(tensors)
  graph.replay();torch.cuda.synchronize();state_b=torch.cuda.get_rng_state();fresh=metrics(tensors,second)
  assert not fresh[0]['equal'] and not fresh[1]['equal'], 'captured dropout did not advance'
  assert not torch.equal(state_a,state_b), 'CUDA RNG state did not advance'
  assert all(torch.isfinite(t).all() for t in tensors)
  checks={'finite_tensors':finite,'all_gradients_present':len(captured['gradients']),'same_rng_repeat':repeat,'stable_buffer_pointers':True,'fresh_rng_output_changed':not fresh[0]['equal'],'fresh_rng_dx_changed':not fresh[1]['equal'],'fresh_rng_output_relative_l2':fresh[0]['relative_l2'],'fresh_rng_dx_relative_l2':fresh[1]['relative_l2'],'cuda_rng_state_advanced':True,'timed_rng_reseed':False}
  handles[arm]={'graph':graph,'owners':kw,'captured':captured,'provenance':provenance,'checks':checks}
  return bench.BenchResult(value=float('nan'),compiled=evidence.compiled,cudagraph='manual_stochastic_adapter',compile_scope='+'.join(sorted(evidence.scopes)),compiled_graphs=len(evidence.graphs),measurement_scope='forward_backward',execution_path=kw['execution_path'],input_dtype=kw['input_dtype'],parameter_dtype=kw['parameter_dtype'],reference=kw['reference'],execution_validation=json.dumps(checks))
 bench.measured_result=adapter;rows={}
 for arm,kernels in [('triton',('front','f567','dual_bwd')),('h100',('front','f567','dual_bwd'))]:
  torch.compiler.reset();settings.configure(engine_backend='triton',trimul_sm90_kernels=kernels)
  # The fixture's compiled forwards, warmup, backward and capture all use the
  # same side stream, even though its validated BenchConfig says disabled.
  side=bench.capture_stream();side.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side):row=bench.bench_module_triangle_multiplication_bidirectional(conf,a.length,'triton',NoFabric())
  torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
  rows[arm]=row._asdict();print('CAPTURED',build,arm,json.dumps(handles[arm]['checks']),flush=True)

 for other in ('h100',):
  assert handles[other]['provenance']==handles['triton']['provenance'],'different fixture'
 torch.cuda.manual_seed(319);rng=torch.cuda.get_rng_state()
 values={}
 for tag,h in handles.items():
  torch.cuda.set_rng_state(rng);h['graph'].replay();torch.cuda.synchronize()
  values[tag]=snapshot([h['captured']['output'],*h['captured']['gradients']])
 errors={tag:metrics(values[tag],values['triton']) for tag in ('h100',)}
 assert all(v['relative_l2']<.002 for err in errors.values() for v in err),errors
 samples={tag:[] for tag in handles}
 for rnd in range(12):
  order=list(handles);order=order[rnd%len(handles):]+order[:rnd%len(handles)]
  if rnd%2:order.reverse()
  for tag in order:samples[tag].append(float(bench.bench_time(handles[tag]['graph'].replay,warmup=10,rep=50)['median_ms']))
 medians={tag:statistics.median(v) for tag,v in samples.items()}
 report['builds'].append({'build':build,'ms':medians,'samples_ms':samples,'errors_vs_triton':errors,'checks':{tag:h['checks'] for tag,h in handles.items()},'runtime_chosen':runtime_chosen,'provenance':handles['triton']['provenance']})
 print('RESULT',json.dumps(report['builds'][-1]),flush=True)
 report['passed']=True
 report['native_calls']=dict(native_calls)
 (root/f'module-tma-L{a.length}.json').write_text(json.dumps(report,indent=2)+'\n')
