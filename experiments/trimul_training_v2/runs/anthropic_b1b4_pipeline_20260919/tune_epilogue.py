from measure_experiment import *
sources=['dual_maskbits']+[f'dual_epiu_b{b}_k{k}' for b in (1,2,4,8) for k in (1,2,4) if (b,k)!=(1,4)]
records=[]
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n);ref=baseline(d,dy,s)
  plans={};errors={};rejects={}
  for name in sources:
   try:
    p=Experiment(d,dy,s,132,2,name);errors[name]=check(p(),ref);plans[name]=p
    print('CHECK',name,n,errors[name],flush=True)
   except RuntimeError as e:
    if 'Spill regression:' not in str(e):raise
    rejects[name]=str(e);print('REJECT_SPILL',name,n,flush=True)
  functions=dict(baseline=lambda:baseline(d,dy,s),**plans)
  times=paired_events({k:capture(f) for k,f in functions.items()})
  record=dict(L=n,dropout=.25,times=times,errors=errors,rejects=rejects)
  records.append(record);print('RESULT',n,{k:round(v['median_us'],3) for k,v in times.items()},flush=True)
  (R/'epilogue-results.json').write_text(json.dumps(records,indent=2))
