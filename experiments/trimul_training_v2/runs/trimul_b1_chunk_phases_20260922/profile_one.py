from pathlib import Path
import sys,argparse,json,importlib.util,torch
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_wait_folding_20260921'
sys.path.insert(0,str(S));import wait_policy as B
sp=importlib.util.spec_from_file_location('chunk_plan',R/'replace_plan.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--mode',type=int,required=True);a=ap.parse_args()
with torch.no_grad():
 f=B.BASE.OLD.Q.setup(a.length);m=B.Training(f);_,k=m.forward();m.backward(k)
 if a.mode:
  cfg=json.loads((S/('selected-L%d.json'%a.length)).read_text())['config'];cfg['defines']['B1_PHASE_CHUNK']=a.mode
  obj=P.Plan(dict(f['d'],x=k[-1]),f['dy'],k[1],k[3],**cfg)
 else:obj=m.p1
 for _ in range(5):obj()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();obj();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 print('PROFILE_DONE',a.length,a.mode,obj.k.unit.cubin_path,flush=True)
