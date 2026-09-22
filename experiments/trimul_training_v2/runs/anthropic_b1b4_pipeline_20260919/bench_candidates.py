"""A/B candidates with a hard compiler spill gate and exact core baseline."""
from measure_experiment import *
parser=argparse.ArgumentParser()
parser.add_argument('--candidates',nargs='+',required=True,help='source@CTA_count')
parser.add_argument('--output',required=True)
args=parser.parse_args()
records=[]
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n);ref=baseline(d,dy,s)
  plans={};errors={};rejects={}
  for candidate in args.candidates:
   name,count=candidate.rsplit('@',1)
   try:
    p=Experiment(d,dy,s,int(count),2,name);errors[candidate]=check(p(),ref);plans[candidate]=p
    print('CHECK',candidate,n,errors[candidate],flush=True)
   except RuntimeError as e:
    if 'Spill regression:' not in str(e):raise
    rejects[candidate]=str(e);print('REJECT_SPILL',candidate,n,flush=True)
  functions=dict(baseline=lambda:baseline(d,dy,s),**plans)
  times=paired_events({k:capture(f) for k,f in functions.items()})
  record=dict(L=n,dropout=.25,warmup=20,iterations=200,times=times,errors=errors,rejects=rejects)
  records.append(record);print('RESULT',n,{k:round(v['median_us'],3) for k,v in times.items()},flush=True)
  (R/args.output).write_text(json.dumps(records,indent=2))
