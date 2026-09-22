from warp_plan import *
from check_front import LIMITS
class H32Plan(WarpPlan):
 def bind(self,dl,dr,dg,dy):
  a=self.a;d=a['d'];n=d['n'];m=n*n;L=T._launch_module();assert self.count>16*self.splits
  tm=lambda t,box,dims,strides,sw='128B':L.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle=sw,l2='128B')
  row=lambda t:tm(t,[64,64],[128,m],[256])
  maps=[tm(dl,[64,32],[m,256],[m*2]),tm(dr,[64,32],[m,256],[m*2]),tm(a['pre'],[64,64],[m,1024],[m*2]),row(a['xn']),row(dg),*[tm(a[k],[32,64],[256,128],[512],'64B') for k in ['wlg','wl','wrg','wr']],tm(a['wg'],[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx)]
  self.inputs=(dl,dr,dg,dy);self.p=L.Struct([*maps,a['mask'],a['mu'],a['rs'],d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.debugdc,self.debugxn,m,n,m//64])
if __name__=='__main__':
 import argparse,faulthandler
 faulthandler.dump_traceback_later(25,repeat=True)
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--count',type=int,default=396);ap.add_argument('--splits',type=int,default=10);ap.add_argument('--source',default='front_onewg_h32');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref=baseline(a);p=H32Plan(a,args.count,args.splits,source=args.source);print('LAUNCH',flush=True);p();torch.cuda.synchronize();es=errors(p.outputs,ref);print('H32_ERRORS',es,flush=True);assert all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items()),es
  if args.bench:
   ts=paired({'baseline':capture(lambda:baseline(a)),'h32':capture(p)});(R/f'{args.source}-L{args.length}-s{args.splits}.json').write_text(json.dumps(dict(errors=es,times=ts),indent=2));print('TIMES',{k:v['median_us'] for k,v in ts.items()},flush=True)
 faulthandler.cancel_dump_traceback_later()
