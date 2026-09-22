"""Controlled B1-B4 comparison: shared B1/cuBLAS plus vendor LN backward.

This is explicitly a hybrid primitive baseline, not public cuEq TMU. It
holds forward saves and BF16 boundaries fixed so B1-B4 can be compared directly.
The full vendor-composed bidirectional backward is measured by compare_cueq.py.
"""
from measure_experiment import *
from cuequivariance_ops.triton import Layout
import cuequivariance_ops_torch.fused_layer_norm_torch
import importlib.metadata

def cueq_core(d,dy,s):
 ctx,_,_=s
 xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mean,rs,gate,proj=ctx.saved_tensors
 m=d['n']**2
 dp,dg=B.gate_elem_bwd_ew(dy.reshape(m,128),proj,gate,d['ds'],d['n'])
 dwg=torch.mm(xn.reshape(m,128).t(),dg)
 dwp=torch.mm(dp.t(),norm)
 dn=torch.mm(dp,wp)
 dt,dga,dbe=torch.ops.cuequivariance.layer_norm_transpose_bwd(
  dn.reshape(1,m,256),tri.reshape(256,1,m),go,mean.reshape(1,m),rs.reshape(1,m),True,Layout.DBN_BND)
 return dg,dwg,dt.reshape_as(tri),dga.sum((0,1)).to(go.dtype),dbe.sum((0,1)).to(go.dtype),dwp

if __name__=='__main__':
 records=[]
 with torch.no_grad():
  for n in (384,768):
   d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n)
   ref=baseline(d,dy,s);plan=Experiment(d,dy,s,132,2,'dual_ln_prefetch')
   vendor=cueq_core(d,dy,s)
   errors={k:rel(a,b) for k,a,b in zip(('dg','dWg','dtri','dgamma','dbeta','dWp'),vendor,ref)}
   print('CHECK',n,errors,flush=True)
   assert all(torch.isfinite(a).all().item() for a in vendor)
   assert max(errors.values())<.001,errors
   funcs={'triton_cublas':lambda:baseline(d,dy,s),'cueq_ln_hybrid':lambda:cueq_core(d,dy,s),'cuda':plan}
   graphs={k:capture(f) for k,f in funcs.items()}
   blocks=[paired_events(graphs) for _ in range(3)]
   times={}
   for k in graphs:
    sample=sorted(t for b in blocks for t in b[k]['samples_us'])
    times[k]=dict(median_us=statistics.median(sample),p90_us=sample[int(.9*(len(sample)-1))],samples_us=sample)
   print('RESULT',n,{k:{m:v for m,v in z.items() if m!='samples_us'} for k,z in times.items()},flush=True)
   records.append(dict(L=n,scope='B1-B4; shared engine B1 and cuBLAS GEMMs, cuEq LN backward + FP32 sums',cuequivariance_ops=importlib.metadata.version('cuequivariance-ops-torch-cu12'),dropout=.25,errors=errors,blocks=blocks,times=times))
   (R/'cueq-core-comparison.json').write_text(json.dumps(records,indent=2))
