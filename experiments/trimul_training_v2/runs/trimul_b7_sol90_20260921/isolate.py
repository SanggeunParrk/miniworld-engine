from pathlib import Path
import argparse,json,sys,torch
import baseline as H
import front_core as FC
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
 a,m=H.setup(n);d=a['d'];k3=json.loads((H.P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner'];saved=H.P.S.A.Training(a,k3);records=[]
 for mutant in (False,True):
  if mutant:
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
  y,ss=saved.forward_saved();ctx,mu,rs=ss;xn,wl,wlg,wr,wrg,wg,_,_,pre,*_=ctx.saved_tensors
  m();out=tuple(v.clone() for v in m.p7());dl,dr,dg,dy=m.p7.inputs[:4]
  z=dict(a,dl=dl.reshape(1,256,n,n),dr=dr.reshape(1,256,n,n),dg=dg,dy=dy,xn=xn,pre=pre,wl=wl,wlg=wlg,wr=wr,wrg=wrg,wg=wg,mu=mu,rs=rs,mask=a['mask'])
  ref=FC.baseline(z);e=FC.errors(out,ref);row=dict(mutant=mutant,matched_b7_inputs=True,errors=e);records.append(row);print('ISOLATED',n,mutant,{k:v['relative_l2'] for k,v in e.items()},flush=True)
  (R/('isolate-L%d.json'%n)).write_text(json.dumps(records,indent=2))
