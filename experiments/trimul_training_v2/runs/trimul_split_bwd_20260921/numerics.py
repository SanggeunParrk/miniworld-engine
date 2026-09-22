import argparse,gc,json,torch
import bench as B
p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=768);a=p.parse_args();n=a.length
with torch.no_grad():
 dta=B.Q.setup(n);d=dta['d'];k3=json.loads((B.P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner'];saved=B.P.S.A.Training(dta,k3)
 ts={'x':d['x'],'wl':d['leaves'][1],'wg':d['leaves'][5],'dy':dta['dy'],'ds':d['ds'],'mask':d['mask']};orig={k:v.clone() for k,v in ts.items()}
 def reset(mut):
  for k,v in ts.items():v.copy_(orig[k])
  if mut:
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);dta['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask'])
  dta['mask'].copy_(d['mask'].reshape(-1))
 refs=[]
 for mut in (False,True):
  reset(mut);y,ss=saved.forward_saved();r=B.P.S.C.backward(d,ss,dta['dy']);refs.append((y.clone(),tuple(t.clone() for t in r)))
 del ss,r;rows=[]
 for splits in (6,10,12,14,15,16):
  reset(False);cfg=dict(saved=True,split=True,splits=splits,pc=1,prod=32,cons=224);m=B.Training(dta,cfg);row=dict(config=cfg,errors=[])
  for mut,ref in zip((False,True),refs):
   reset(mut);o=m();e=B.H.errors(o,ref);row['errors'].append(e)
  row['valid']=all(v['finite'] and v['relative_l2']<=5e-4 for e in row['errors'] for v in e.values())
  reset(False);m();g,_=B.Q.capture_outputs(m.p7);row['b7_time']=B.Q.pool([B.Q.paired({'candidate':g},iterations=100)])['candidate']['median_us']
  print('NUMERIC',splits,row['valid'],[max((v['relative_l2'],k) for k,v in e.items()) for e in row['errors']],row['b7_time'],flush=True);rows.append(row);(B.ROOT/('numerics-L%d.json'%n)).write_text(json.dumps(rows,indent=2));del m,g,o;gc.collect()
