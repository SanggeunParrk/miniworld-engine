from pathlib import Path
import argparse,json,torch,importlib.util,platform,sys
import bench as B
from training import Training
spec=importlib.util.spec_from_file_location('prior_split_bench',B.N.OLD/'bench.py');OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);a=ap.parse_args();n=a.length
with torch.no_grad():
 data=B.Q.setup(n);d=data['d'];cfg=json.loads((R/('selected-L%d.json'%n)).read_text())['config'];base=OLD.Training(data,OLD.CONFIGS['split_xn_pc1']);new=Training(data)
 models={'baseline':base,'candidate':new};checks={};ref=base();ref=(ref[0].clone(),tuple(x.clone() for x in ref[1]));checks['initial']=OLD.check(new(),ref)
 saved=OLD.P.S.A.Training(data,json.loads((OLD.P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner']);y,ss=saved.forward_saved();ind=(y,OLD.P.S.C.backward(d,ss,data['dy']));checks['independent_initial']=OLD.check(new(),ind)
 result=dict(L=n,config=cfg,checks=checks,times={},blocks={},metadata=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),dropout=.25,includes=['live weight packing','all eleven gradients','both directions','mask and residual','CUDA graph replay'],excludes=['optimizer','RNG generation','CPU dispatch','compilation']))
 kept={k:m.forward()[1] for k,m in models.items()}
 for scope,funcs in {'b1':{k:m.p1 for k,m in models.items()},'backward':{k:(lambda k=k,m=m:m.backward(kept[k])) for k,m in models.items()},'forward_backward':models}.items():
  gs={};outs={}
  for k,f in funcs.items():gs[k],outs[k]=B.Q.capture_outputs(f)
  blocks=[B.Q.paired(gs,iterations=200) for _ in range(3)];result['blocks'][scope]=blocks;result['times'][scope]=B.Q.pool(blocks);print('RESULT',scope,{k:v['median_us'] for k,v in result['times'][scope].items()},flush=True)
  if scope=='forward_backward':
   # Validate graph replay against eager and baseline after live input/weight changes.
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);data['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);data['mask'].copy_(d['mask'].reshape(-1))
   rr=base();rr=(rr[0].clone(),tuple(t.clone() for t in rr[1]));eager=new();eager=(eager[0].clone(),tuple(t.clone() for t in eager[1]));checks['mutated_vs_baseline']=OLD.check(eager,rr)
   gs['candidate'].replay();torch.cuda.synchronize();checks['graph_vs_eager']=OLD.H.errors(outs['candidate'],eager);assert all(v['bit_exact'] for v in checks['graph_vs_eager'].values())
   yy,ss=saved.forward_saved();ir=(yy,OLD.P.S.C.backward(d,ss,data['dy']));checks['independent_mutated']=OLD.H.errors(eager,ir);checks['baseline_independent_mutated']=OLD.H.errors(rr,ir)
   result['independent_mutated_pass']=all(v['finite'] and v['relative_l2']<=5e-4 for v in checks['independent_mutated'].values())
   result['baseline_independent_mutated_pass']=all(v['finite'] and v['relative_l2']<=5e-4 for v in checks['baseline_independent_mutated'].values())
  del gs,outs
  (R/('module-L%d.json'%n)).write_text(json.dumps(result,indent=2))
 result['production_ready']=False;result['cubin']=new.p1.k.unit.cubin_path
 (R/('module-L%d.json'%n)).write_text(json.dumps(result,indent=2))
 print('DONE',n,checks,flush=True)
