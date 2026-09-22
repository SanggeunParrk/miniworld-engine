from pathlib import Path
import argparse,sys,json,itertools,torch,gc,hashlib,importlib.util
R=Path(__file__).resolve().parent;P=R.parent/'trimul_tri_restore_20260921'
sys.path.insert(0,str(P));import tri_policy as B
spec=importlib.util.spec_from_file_location('b1_tri_opt_plan',R/'replace_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
Q=B.BASE.OLD.Q
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);args=ap.parse_args();n=args.length
records=[]
with torch.no_grad():
 a=Q.setup(n);m=B.Training(a);y,k=m.forward();m.backward(k);ref=tuple(t.clone() for t in m.p1());g0,_=Q.capture_outputs(m.p1)
 base=json.loads((B.STATS/('selected-L%d.json'%n)).read_text())['config']
 candidates=[]
 for direct,sync,pair,early,dy in itertools.product((0,1),repeat=5):
  if dy and not direct:continue
  # Isolate simple changes and combinations; same arithmetic and CTA count.
  cfg=json.loads(json.dumps(base));cfg['defines'].update(XHAT_FP32=-1,B1_DIRECT_DP=direct,B1_NO_REDUNDANT_SYNC=sync,PAIR_WP=pair,B1_EARLY_RAW=early,B1_PREFETCH_DY=dy);candidates.append(cfg)
 for cfg in candidates:
  row=dict(config=cfg)
  try:
   p=RP.Plan(dict(a['d'],x=k[-1]),a['dy'],k[1],k[3],**cfg);out=p();torch.cuda.synchronize();exact=[torch.equal(v,r) for v,r in zip(out,ref)];assert all(exact),exact
   g,_=Q.capture_outputs(p);blocks=[Q.paired({'baseline':g0,'candidate':g},iterations=100) for _ in range(2)];t=Q.pool(blocks);row.update(valid=True,bit_exact=exact,ratio=t['candidate']['median_us']/t['baseline']['median_us'],times={key:v['median_us'] for key,v in t.items()},cubin=p.k.unit.cubin_path);del g,p;gc.collect()
  except Exception as e:row.update(valid=False,error=str(e)[-1800:])
  records.append(row);(R/('tune-L%d.json'%n)).write_text(json.dumps(records,indent=2));print('TUNE',row,flush=True)
 valid=[r for r in records if r['valid']];best=min(valid,key=lambda r:r['ratio']);(R/('selected-L%d.json'%n)).write_text(json.dumps(best,indent=2));print('BEST',best,flush=True)
