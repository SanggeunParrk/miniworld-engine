"""Fresh-process paired full and BWD timings of the split4 candidate."""
import pathlib,sys
# Reuse only definitions, not the exploratory search.
source=pathlib.Path(__file__).with_name('pilot.py');scope={'__file__':str(source)}
exec(compile(source.read_text().split('leaves,dy,mask,ds,ref,triton,names=setup')[0],str(source),'exec'),scope)
globals().update({k:v for k,v in scope.items() if k!='__file__'})
from input_split import attach
leaves,dy,mask,ds,ref,triton,names=setup(a.width,a.length)
r={'width':a.width,'length':a.length,'candidate':'input_split4_index0','complete':False,'repeats':[]}
physical=os.environ.get('CUDA_VISIBLE_DEVICES','unknown');dest=out/f'repeat-input-split-gpu{physical}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);y,g=plan();expected=[x.clone() for x in (y,*g)]
 full0,_=capture(plan);bwd0,_=capture(plan.backward)
 attach(plan,4,0);y,g=plan();torch.cuda.synchronize()
 r['errors']={n:error(u,v) for n,u,v in zip(names,(y,*g),expected)}
 assert all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in r['errors'].items())
 full1,_=capture(plan);bwd1,_=capture(plan.backward)
 for repeat in range(3):
  times=paired({'baseline_full':full0,'candidate_full':full1,'baseline_bwd':bwd0,'candidate_bwd':bwd1})
  r['repeats'].append(times);dest.write_text(json.dumps(r,indent=2));print('REPEAT',repeat,{k:v['median_us'] for k,v in times.items()},flush=True)
 r['complete']=True;dest.write_text(json.dumps(r,indent=2))
