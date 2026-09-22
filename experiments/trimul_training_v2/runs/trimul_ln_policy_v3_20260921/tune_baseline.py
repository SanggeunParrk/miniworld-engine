import argparse,json,torch
from pathlib import Path
import bench as B
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args();n=args.length
with torch.no_grad():
 a=B.Q.setup(n);base=B.BASE.Training(a);m=B.BASE.Training(a);_,k=m.forward();m.backward(k);base();ref=tuple(x.clone() for x in base.p1());g0,_=B.Q.capture_outputs(base.p1);results=[]
 for count in (96,112,120,128,132):
  for unroll in (1,2,4,8):
   cfg=dict(count=count,part=2,defines=dict(GATE_PHASE=1,LOWREG=1,STREAM_LN=1,B1_STREAM_AFFINE_UNROLL=unroll))
   p=B.BASE.N.Plan(dict(m.d,x=k[-1]),a['dy'],k[1],**cfg);out=p();torch.cuda.synchronize();rel=[B.Q.rel(x,y) for x,y in zip(out,ref)];ok=torch.equal(out[0],ref[0]) and torch.equal(out[2],ref[2]) and max(rel)<=5e-4
   row=dict(config=cfg,valid=ok,rel=rel,cubin=p.k.unit.cubin_path)
   if ok:
    g,_=B.Q.capture_outputs(p);blocks=[B.Q.paired({'baseline':g0,'candidate':g},iterations=60) for _ in range(2)];pooled=B.Q.pool(blocks);row.update(blocks=blocks,times=pooled,ratio=pooled['baseline']['median_us']/pooled['candidate']['median_us']);del g
   results.append(row);(R/('baseline-sweep-L%d.json'%n)).write_text(json.dumps(results,indent=2));print('CANDIDATE',count,unroll,ok,row.get('ratio'),flush=True)
 best=max((r for r in results if r['valid']),key=lambda r:r['ratio']);(R/('baseline-selected-L%d.json'%n)).write_text(json.dumps(best,indent=2));print('SELECTED',best['config'],best['ratio'],flush=True)
