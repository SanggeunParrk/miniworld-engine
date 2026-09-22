from cached_dx_plan import *
import argparse,faulthandler
faulthandler.dump_traceback_later(45,repeat=True)
ap=argparse.ArgumentParser();ap.add_argument('--sources',nargs='+',required=True);ap.add_argument('--length',type=int,default=64);ap.add_argument('--output',required=True);args=ap.parse_args();rec={}
with torch.no_grad():
 a=setup(args.length);ref=baseline(a)
 for src in args.sources:
  try:
   p=CachedDxPlan(a,80,src);print('LAUNCH',src,flush=True);p();torch.cuda.synchronize();es=errors(p.outputs,ref);es={k:v for k,v in es.items() if k in ('dx','dgamma','dbeta')};ok=all(v['finite'] and v['relative_l2']<LIMITS[k] for k,v in es.items());rec[src]=dict(status='pass' if ok else 'numerical_rejection',errors=es);print('CACHE_STATUS',src,rec[src],flush=True)
  except RuntimeError as e:rec[src]=dict(status='build_or_launch_rejection',reason=str(e))
  (R/args.output).write_text(json.dumps(rec,indent=2))
faulthandler.cancel_dump_traceback_later()
