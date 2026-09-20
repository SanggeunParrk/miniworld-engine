from warp_plan import *
from check_front import LIMITS
class RingPlan(WarpPlan):
 def __init__(self,a,count=264,splits=20,part=2,source='front_ring96_cache3'):
  super().__init__(a,count,splits,part,source)
 def bind(self,dl,dr,dg,dy):
  a=self.a;d=a['d'];n=d['n'];m=n*n;L=T._launch_module();window=self.config['ring_tiles']
  # Keep the ring address stable when the full-backward adapter rebinds inputs.
  # CUDA graph replay retains the tensor-map address captured here.
  if not hasattr(self,'ring') or self.ring.shape != (1024,window*64):
   self.ring=torch.empty((1024,window*64),device=d['x'].device,dtype=torch.bfloat16)
  tm=lambda t,box,dims,strides:L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  row=lambda t:tm(t,[64,64],[128,m],[256])
  ht=self.config.get('front_hidden_tile',64)
  maps=[tm(dl,[64,ht],[m,256],[m*2]),tm(dr,[64,ht],[m,256],[m*2]),tm(a['pre'],[64,ht*2],[m,1024],[m*2]),row(a['xn']),row(dg),*[tm(a[k],[64,self.config.get('weight_tma_rows',64)],[256,128],[512]) for k in ['wlg','wl','wrg','wr']],tm(a['wg'],[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx),tm(self.ring,[64,64],[window*64,1024],[window*128])]
  extra=[self.ring] if self.config.get('ring_direct_store') else []
  self.inputs=(dl,dr,dg,dy);self.p=L.Struct([*maps,*extra,a['mask'],a['mu'],a['rs'],d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.debugdc,self.debugxn,m,n,m//64])
