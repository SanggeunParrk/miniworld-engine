from pathlib import Path
import importlib.util,argparse,json,torch
from policy import BASE
R=Path(__file__).resolve().parent
bs=importlib.util.spec_from_file_location('local_b1_check_helpers',R/'bench.py');H=importlib.util.module_from_spec(bs);bs.loader.exec_module(H)
mutate,clone,errors=H.mutate,H.clone,H.errors
sp=importlib.util.spec_from_file_location('current_training_check',R.parent/'trimul_training_current.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
 a=BASE.OLD.Q.setup(n);before=C._policy.Training(a);after=C.Training(a);checks=[]
 graph,outs=BASE.OLD.Q.capture_outputs(after)
 for case in range(2):
  if case:mutate(a)
  ref=clone(before());eager=clone(after());graph.replay();torch.cuda.synchronize()
  e=errors(eager,ref);g=errors(outs,eager)
  assert all(v['bit_exact'] for v in [*e.values(),*g.values()]),(case,e,g)
  checks.append(dict(case=case,before_after=e,graph_eager=g))
 result=dict(length=n,checks=checks,b7_unchanged=type(before.p7) is type(after.p7),b1_cubin=after.p1.k.unit.cubin_path)
 assert result['b7_unchanged']
 (R/('current-check-L%d.json'%n)).write_text(json.dumps(result,indent=2));print('CURRENT_PASS',n,flush=True)
