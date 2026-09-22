from warp_plan import *
with torch.no_grad():
 for n in [384,768]:
  a=setup(n);p=WarpPlan(a,count=264,splits=13,source='front_kindmap_trace');g=capture(p)
  for _ in range(20):g.replay()
  torch.cuda.synchronize();ticks=p.counts[2:2+4*264].view(torch.int64).reshape(264,2).cpu();base=int(ticks[:,0].min());dt=(ticks[:,1]-ticks[:,0]).double()/1000
  d=dict(sm_id=p.counts[2+4*264:2+5*264].cpu().tolist(),L=n,start_offset_us=((ticks[:,0]-base)/1000).tolist(),duration_us=dt.tolist(),dw_median_us=float(dt[:104].median()),dw_max_us=float(dt[:104].max()),dx_median_us=float(dt[104:].median()),dx_max_us=float(dt[104:].max()))
  (R/f'kindmap-role-trace-L{n}.json').write_text(json.dumps(d,indent=2));print('ROLES',n,{k:v for k,v in d.items() if not isinstance(v,list)},flush=True)
