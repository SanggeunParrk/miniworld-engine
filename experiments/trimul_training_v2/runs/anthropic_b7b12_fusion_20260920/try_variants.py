from check_front import *
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=768);ap.add_argument('--sources',nargs='+',required=True);ap.add_argument('--splits',type=int,default=15);ap.add_argument('--output',required=True);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);ref=baseline(a);plans={};records={}
 for src in args.sources:
  try:
   p=Plan(a,splits=args.splits,source=src);p();torch.cuda.synchronize();es=errors(p.outputs,ref)
   if any(v['relative_l2']>LIMITS[k] or not v['finite'] for k,v in es.items()):records[src]=dict(status='numerical_rejection',errors=es);continue
   plans[src]=p;records[src]=dict(status='initial_pass',errors=es)
  except RuntimeError as e:records[src]=dict(status='compile_rejection',reason=str(e))
 gs={'baseline':capture(lambda:baseline(a)),**{src:capture(p) for src,p in plans.items()}};ts=paired(gs)
 (R/args.output).write_text(json.dumps(dict(L=args.length,splits=args.splits,records=records,times=ts),indent=2))
 print('VARIANTS', {k:v['median_us'] for k,v in ts.items()},'REJECTED',{k:v for k,v in records.items() if v['status']!='initial_pass'},flush=True)
