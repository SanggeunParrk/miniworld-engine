from pathlib import Path
import argparse,json,torch
import bench as B
R=B.ROOT;P=B.P;Q=B.Q
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--only',choices=tuple(B.CONFIGS));args=ap.parse_args();n=args.length
with torch.no_grad():
 a=Q.setup(n);d=a['d'];cfg=json.loads((B.R.PREV/('selected-L%d.json'%n)).read_text());k3=json.loads((P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner'];baseline=B.OLD.Training(a,k3,cfg['b1'],cfg['b7']);saved=P.S.A.Training(a,k3)
 models={k:B.Training(a,v) for k,v in B.CONFIGS.items() if args.only is None or k==args.only}
 models={'baseline':baseline,**models};graphs={k:Q.capture_outputs(m) for k,m in models.items()};res={}
 for state in ('original','mutated'):
  if state=='mutated':
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
  y,ss=saved.forward_saved();ref=(y,P.S.C.backward(d,ss,a['dy']));baseout=baseline();baseout=(baseout[0].clone(),tuple(x.clone() for x in baseout[1]));res[state]={}
  for name,m in models.items():
   o=m();o=(o[0].clone(),tuple(x.clone() for x in o[1]));g,go=graphs[name];g.replay();torch.cuda.synchronize()
   e=B.H.errors(o,ref);eb=B.H.errors(o,baseout);eg=B.H.errors(go,o)
   row=dict(reference=e,baseline=eb,replay=eg,reference_pass=all(v['finite'] and v['relative_l2']<=5e-4 for v in e.values()),baseline_pass=all(v['finite'] and v['relative_l2']<=5e-4 for v in eb.values()),replay_pass=all(v['bit_exact'] for v in eg.values()))
   res[state][name]=row;print(state,name,{k:v for k,v in row.items() if k.endswith('pass')},'refmax',max((v['relative_l2'],k) for k,v in e.items()),'basemax',max((v['relative_l2'],k) for k,v in eb.items()),flush=True)
 (R/('validation-%sL%d.json'%((args.only+'-') if args.only else '',n))).write_text(json.dumps(res,indent=2))
