"""Full backward integration: B1-B6 fixed reference, B7-B12 one CUDA launch.

The prefix remains unchanged. Tensor-map binding operates on the host while
capturing; replay contains the prefix GPU kernels and one front_b7b12 launch.
This file never edits the separate Claude-owned B1-B4 experiment.
"""
from front_plan import *

def backward_full(a,p):
 d=a['d'];n=d['n'];m=n*n;ctx,_,_=a['s']
 xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mo,ro,gate,proj=ctx.saved_tensors
 dp,dg=B.gate_elem_bwd_ew(a['dy'].reshape(m,128),proj,gate,d['ds'],n)
 dwg=torch.mm(xn.reshape(m,128).t(),dg)
 dt,dgo,dbo,dwp,_=B._te_backward(dp,norm,tri.reshape(256,m).t(),mo,ro,go,wp,False,shape_key=both_key(m))
 dl,dr=B.packed_backward(dt.t().reshape_as(tri),lf,rf,128)
 p.bind(dl,dr,dg,a['dy']);dx,dwl,dwlg,dwr,dwrg,dgi,dbi=p()
 return dx.reshape_as(d['x']),dwl.t(),dwlg.t(),dwr.t(),dwrg.t(),dwg.t(),dwp,dgi,dbi,dgo,dbo

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--source',default='front_selected');ap.add_argument('--splits',type=int,default=15);ap.add_argument('--output',required=True);args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);p=Plan(a,splits=args.splits,source=args.source)
  ref=C.backward(a['d'],a['s'],a['dy']);out=backward_full(a,p)
  e=[rel(x,y) for x,y in zip(out,ref)];print('FULL_ERRORS',e,flush=True);assert max(e)<5e-4,e
  fs={'baseline':lambda:C.backward(a['d'],a['s'],a['dy']),'cuda':lambda:backward_full(a,p)}
  gs={k:capture(fn) for k,fn in fs.items()};blocks=[paired(gs) for _ in range(3)];times={}
  for k in gs:
   ts=sorted(v for b in blocks for v in b[k]['samples_us']);times[k]=dict(median_us=statistics.median(ts),p90_us=ts[int(.9*(len(ts)-1))],samples_us=ts)
  record=dict(L=args.length,source=args.source,splits=args.splits,dropout=.25,scope='full backward; identical B1-B6 reference prefix',errors=e,blocks=blocks,times=times)
  (R/args.output).write_text(json.dumps(record,indent=2));print('FULL_TIMES',{k:v['median_us'] for k,v in times.items()},flush=True)
