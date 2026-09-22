from pathlib import Path
import sys,importlib.util,torch,json
import epilogue_policy as P
R=Path(__file__).resolve().parent;D=R/'delayed';D.mkdir(exist_ok=True)
for p in [*R.glob('*.cuh'),*R.glob('*.inc'),R/'b1_fused.cu',R/'replace_plan.py']:(D/p.name).write_bytes(p.read_bytes())
p=D/'replace_plan.py';p.write_text(p.read_text().replace("OLD=R.parent/'trimul_split_bwd_20260921'","OLD=R.parent.parent/'trimul_split_bwd_20260921'"))
p=D/'b1_fused.cu';s=p.read_text();needle=' shared_role(p,sm,bars);';assert needle in s
s=s.replace(needle,needle+'\n if(blockIdx.x%17==0 && threadIdx.x==0){for(int k=0;k<20;++k)__nanosleep(50000);}allsync();\n');p.write_text(s)
spec=importlib.util.spec_from_file_location('b1_delayed_plan',D/'replace_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
records=[]
with torch.no_grad():
 for n in (384,768):
  a=P.BASE.OLD.Q.setup(n);m=P.Training(a);cfg=json.loads((R/('selected-L%d.json'%n)).read_text())['config']
  for seed in (19,47,83):
   torch.manual_seed(seed);a['dy'].normal_();_,k=m.forward();m.backward(k);ref=tuple(t.clone() for t in m.p1())
   plan=RP.Plan(dict(a['d'],x=k[-1]),a['dy'],k[1],k[3],**cfg)
   g,out=P.BASE.OLD.Q.capture_outputs(plan)
   for _ in range(20):g.replay()
   torch.cuda.synchronize();exact=[torch.equal(v,r) for v,r in zip(out,ref)];assert all(exact),exact
   records.append(dict(L=n,seed=seed,bit_exact=exact,replays=20,delayed_ctas='blockIdx % 17 == 0, 20 x nanosleep(50000ns) before Phase B'))
(R/'delayed-cta-check.json').write_text(json.dumps(records,indent=2));print('DELAY_STRESS_PASS',records,flush=True)
