from warp_plan import *
from check_front import LIMITS
class RingPlan(WarpPlan):
 def bind(self,dl,dr,dg,dy):
  a=self.a;d=a['d'];n=d['n'];m=n*n;L=T._launch_module();window=self.config['ring_tiles']
  # Keep the ring address stable when the full-backward adapter rebinds inputs.
  # CUDA graph replay retains the tensor-map address captured here.
  if not hasattr(self,'ring') or self.ring.shape != (1024,window*64):
   self.ring=torch.empty((1024,window*64),device=d['x'].device,dtype=torch.bfloat16)
  def tm(t,box,dims,strides):
   policy=self.config.get('tma_l2_promotion','128B')
   if any(t is a[k] for k in ('wlg','wl','wrg','wr','wg')):policy=self.config.get('tma_weight_l2',policy)
   if hasattr(self,'ring') and t is self.ring:policy=self.config.get('tma_ring_l2',policy)
   return L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2=policy)
  row=lambda t:tm(t,[64,64],[128,m],[256])
  ht=self.config.get('front_hidden_tile',64)
  maps=[tm(dl,[64,ht],[m,256],[m*2]),tm(dr,[64,ht],[m,256],[m*2]),tm(a['pre'],[64,ht*2],[m,1024],[m*2]),row(a['xn']),row(dg),*[tm(a[k],[64,self.config.get('weight_tma_rows',64)],[256,128],[512]) for k in ['wlg','wl','wrg','wr']],tm(a['wg'],[64,self.config.get('gate_tma_rows',64)],[128,128],[256]),row(d['x']),row(dy),row(self.dx),tm(self.ring,[64,64],[window*64,1024],[self.ring.stride(0)*2])]
  extra=[self.ring] if self.config.get('ring_direct_store') else []
  self.inputs=(dl,dr,dg,dy);self.p=L.Struct([*maps,*extra,a['mask'],a['mu'],a['rs'],d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.debugdc,self.debugxn,m,n,m//64])
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(25,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--splits',type=int,default=13);ap.add_argument('--source',default='front_ring256');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);p=RingPlan(a,count=264,splits=args.splits,source=args.source);print('LAUNCH',flush=True);p();torch.cuda.synchronize();es=errors(p.outputs,ref);print('RING_ERRORS',es,flush=True);assert all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in es.items()),es
  if args.bench:
   ts=paired({'baseline':capture(lambda:baseline(a)),'ring':capture(p)});(R/f'{args.source}-L{args.length}-s{args.splits}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('TIMES',{k:v['median_us'] for k,v in ts.items()},flush=True)
 faulthandler.cancel_dump_traceback_later()
