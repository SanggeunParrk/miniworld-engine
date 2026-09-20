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
