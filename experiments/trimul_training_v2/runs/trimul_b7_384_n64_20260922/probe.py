print('START probe',flush=True)
from pathlib import Path
import sys,os,importlib.util,torch,json
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
folder=R
os.environ['B7_CONSUMERS']='10'
s=importlib.util.spec_from_file_location('probe_plan',folder/'plan.py');P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
with torch.no_grad():
 a,m=H.setup(384);r=m.p7;dl,dr,dg,dy=r.inputs[:4]
 opts=dict(mode=int(os.environ['MODE'])) if folder==R else {}
 q=P.Plan(r.d,dy,dl,dr,dg,r.xn,clusters=10,**opts)
 q.mask=r.mask;q.bind(dl,dr,dg,dy,xn=r.xn)
 ref=tuple(x.clone() for x in r());out=q();torch.cuda.synchronize();errors=[H.rel(x,y) for x,y in zip(out,ref)]
 print('ERRORS',errors,flush=True);assert all(torch.isfinite(x).all().item() for x in out)
 assert all(e<=l for e,l in zip(errors,[2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6]))
 if os.environ.get('CHECK_ONLY')=='1':
  assert torch.count_nonzero(q.counts).item()==0;assert torch.count_nonzero(q.flags).item()==0;sys.exit(0)
 for _ in range(3):q()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();q();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 print('COUNTS',q.counts.tolist(),flush=True);assert torch.count_nonzero(q.counts).item()==0
