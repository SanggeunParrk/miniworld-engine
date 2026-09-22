from pathlib import Path
import argparse,sys,json,torch,gc,importlib.util,platform
R=Path(__file__).resolve().parent;P=R.parent/'trimul_b1_wait_folding_20260921'
sys.path.insert(0,str(P));import wait_policy as B
spec=importlib.util.spec_from_file_location('b1_param_hybrid_plan',R/'replace_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
Q=B.BASE.OLD.Q
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--levels',default='0,1,2,3');ap.add_argument('--rounds',type=int,default=3);ap.add_argument('--iterations',type=int,default=150);args=ap.parse_args();n=args.length
records=[]
with torch.no_grad():
 a=Q.setup(n);m=B.Training(a);y,k=m.forward();m.backward(k);base=json.loads((P/('selected-L%d.json'%n)).read_text())['config']
 for level in map(int,args.levels.split(',')):
  cfg=json.loads(json.dumps(base));cfg['defines']['B1_PARAM_HYBRID']=level
  row=dict(config=cfg,host=platform.node(),checks=[])
  try:
   p=RP.Plan(dict(a['d'],x=k[-1]),a['dy'],k[1],k[3],**cfg)
   gs={};outs={}
   for label,obj in [('baseline',m.p1),('candidate',p)]:gs[label],outs[label]=Q.capture_outputs(obj)
   tensors=[p.xhat,p.rstd,a['dy'],p.d['go'],p.d['ds']];snap=[x.clone() for x in tensors]
   for case in range(3):
    if case==1:p.xhat.mul_(.97);a['dy'].mul_(.93);p.d['go'].mul_(1.07);p.d['go'][0]=0;p.d['ds'].copy_(p.d['ds'].roll(1,0))
    if case==2:p.xhat.add_(.003);a['dy'].mul_(1.02);p.d['go'].mul_(.99)
    ref=tuple(x.clone() for x in m.p1());p.partw.fill_(float('nan'));p.partln.fill_(float('nan'))
    eager=tuple(x.clone() for x in p());gs['candidate'].replay();gs['candidate'].replay();torch.cuda.synchronize()
    exact=[torch.equal(x,y) for x,y in zip(outs['candidate'],ref)];ee=[torch.equal(x,y) for x,y in zip(eager,ref)]
    valid=all(exact+ee) and all(torch.isfinite(x).all().item() for x in outs['candidate']) and torch.count_nonzero(p.counts).item()==0
    row['checks'].append(dict(case=case,valid=valid,bit_exact=exact,eager_bit_exact=ee));assert valid,row['checks'][-1]
   for x,v in zip(tensors,snap):x.copy_(v)
   blocks=[Q.paired(gs,iterations=args.iterations) for _ in range(args.rounds)];t=Q.pool(blocks)
   row.update(valid=True,blocks=blocks,ratio=t['candidate']['median_us']/t['baseline']['median_us'],times={key:v['median_us'] for key,v in t.items()},cubin=p.k.unit.cubin_path)
   del gs,outs,p;gc.collect()
  except Exception as e:row.update(valid=False,error=str(e)[-1800:])
  records.append(row);(R/('tune-L%d.json'%n)).write_text(json.dumps(records,indent=2));print('TUNE',level,{k:v for k,v in row.items() if k not in ('blocks','config')},flush=True)
 valid=[r for r in records if r['valid']]
 if valid:
  best=min(valid,key=lambda r:r['ratio']);(R/('selected-L%d.json'%n)).write_text(json.dumps(best,indent=2));print('BEST',best['config'],best['times'],flush=True)
