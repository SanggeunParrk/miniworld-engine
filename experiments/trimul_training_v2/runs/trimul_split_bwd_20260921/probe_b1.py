import json,torch,sys
from pathlib import Path
import role_plan as R
import saved_plans as SP
import compare_cueq_training as Q
from compare_all_training import BoundB1
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'trimul_ln_only_save_20260921'));import ln_save_core as LN
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args()
with torch.no_grad():
 a=Q.setup(args.length);d=a['d'];_,_,xn,_=LN.forward(d,1);ref=tuple(t.clone() for t in BoundB1(d,a['dy'],a['s'],132,2,'dual_ln_prefetch')())
 sd=dict(d,x=xn);p=SP.B1(sd,a['dy'],a['s'][0].saved_tensors[11],splits=24,defines=dict(GATE_SPLITS=36,TRAIN_L=args.length,B1_DWPROJ_PIPE=1,USE_SAVED_XN=1));out=p();torch.cuda.synchronize()
 names=('dg','dwg','dt','dgo','dbo','dwp');limits=(0,5e-4,2e-5,5e-6,5e-6,5e-4);err={k:dict(rel=Q.rel(x,y),limit=lim) for k,x,y,lim in zip(names,out,ref,limits)};assert all(v['rel']<=v['limit'] for v in err.values()),err
 g,_=Q.capture_outputs(p);tm=Q.pool([Q.paired({'new':g},iterations=200) for _ in range(3)])['new'];j=dict(errors=err,time=tm)
 (P/('probe-b1-L%d.json'%args.length)).write_text(json.dumps(j,indent=2));print('RESULT',err,tm['median_us'],flush=True)
