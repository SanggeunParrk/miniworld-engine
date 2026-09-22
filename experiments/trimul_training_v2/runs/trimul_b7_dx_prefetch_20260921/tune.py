from pathlib import Path
import argparse,importlib.util,json,sys,torch,gc,itertools
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
spec=importlib.util.spec_from_file_location('b7_dw_overlap_plan',R/'role_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
 a,m=H.setup(n);ref=tuple(x.clone() for x in m.p7());cfg=dict(m.p7.cfg);cfg.pop('saved');records=[]
 for xn_early,mask_hoist in itertools.product((0,1),repeat=2):
  mode=2
  skip,overlap=1,0
  row=dict(skip_restore=skip,overlap=overlap,ln_mode=mode,xn_early=xn_early,mask_hoist=mask_hoist)
  try:
   p=RP.Plan(m.d,*[m.p7.inputs[i] for i in (3,0,1,2)],xn=m.p7.xn,split=True,skip_restore=skip,overlap=overlap,ln_mode=mode,xn_early=xn_early,mask_hoist=mask_hoist,**cfg);p.mask=m.p7.mask;p.bind(*m.p7.inputs[:4],xn=m.p7.xn)
   out=p();torch.cuda.synchronize();e=[H.rel(x,y) for x,y in zip(out,ref)];exact=[torch.equal(x,y) for x,y in zip(out,ref)];assert all(exact),(e,exact)
   graphs={name:H.Q.capture_outputs(fn)[0] for name,fn in [('baseline',m.p7),('candidate',p)]};t=H.Q.pool([H.Q.paired(graphs,iterations=150) for _ in range(3)])
   row.update(valid=True,relative_l2=e,bit_exact=exact,config=cfg,times={k:v['median_us'] for k,v in t.items()},ratio=t['candidate']['median_us']/t['baseline']['median_us'],cubins=[k.unit.cubin_path for k,*_ in p.units]);del graphs,p;gc.collect()
  except Exception as e:row.update(valid=False,error=str(e)[-2000:])
  records.append(row);(R/('tune-L%d.json'%n)).write_text(json.dumps(records,indent=2));print('TUNE',row,flush=True)
 valid=[x for x in records if x['valid']]
 if valid:(R/('selected-L%d.json'%n)).write_text(json.dumps(min(valid,key=lambda x:x['ratio']),indent=2))
