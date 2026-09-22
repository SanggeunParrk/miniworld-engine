"""Check FP32 accumulation segment count with unchanged numerical tolerance."""
import json,torch
from bench import P,Q,Training,H,R
with torch.no_grad():
 a=Q.setup(768);d=a['d'];cfg=json.loads((P.S.A.P/'training-k3-audit-L768.json').read_text())['winner']
 old=P.S.A.Training(a,cfg);new=Training(a,cfg,28,8)
 d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91)
 d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
 y,ss=old.forward_saved();ref=(y,P.S.C.backward(d,ss,a['dy']))
 results=[]
 for slices in (2,4,8,16):
  new.p7=P.B7(d,a['dy'],a['dl'],a['dr'],a['dg'],splits=8,slices=slices);new.p7.mask=a['mask']
  out=new();torch.cuda.synchronize();e=H.errors(out,ref)
  row=dict(slices=slices,errors=e,valid=all(v['finite'] and v['relative_l2']<=5e-4 for v in e.values()))
  results.append(row);print('ACCUMULATION',row,flush=True)
 (R/'accumulation-L768.json').write_text(json.dumps(results,indent=2))
