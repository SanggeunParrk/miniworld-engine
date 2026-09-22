from pathlib import Path
import importlib.util,sys,os,json,torch,gc
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('width_latest_fixed',R.parent/'trimul_ln_gradient_20260922/policy.py');F=importlib.util.module_from_spec(s);sys.modules[s.name]=F;s.loader.exec_module(F)
F.F=F.load('split_accuracy_plan',R/'fixed128/plan.py')
Q=F.Q;N=768;record=[]
with torch.no_grad():
 a=Q.setup(N);reg=F.P.Regression(a);ref=reg();ref=(ref[0].clone(),tuple(t.clone() for t in ref[1]));torch.cuda.synchronize()
 for clusters,consumers in [(10,10),(11,8),(9,12)]:
  F.P.SELECTION['kwargs']['clusters']=clusters;F.P.SELECTION['environment']['B7_CONSUMERS']=str(consumers)
  m=F.Fixed(a);o=m();torch.cuda.synchronize();e=F.H.errors(o,ref);ok=all(v['finite'] and v['relative_l2']<=(0 if k=='forward' else 2e-5 if k=='dx' else 5e-6 if k.startswith(('dgamma','dbeta')) else 5e-4) for k,v in e.items());g,_=Q.capture_outputs(m);t=Q.pool([Q.paired({'cuda':g},warmup=10,iterations=40) for _ in range(2)]);row=dict(clusters=clusters,consumers=consumers,valid=ok,errors=e,time=t);record.append(row);(R/'tune128.json').write_text(json.dumps(record,indent=2));print('RESULT',clusters,consumers,ok,t,flush=True);del g,m;gc.collect()
