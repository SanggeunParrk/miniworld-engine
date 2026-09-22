from cluster_plan import *
with torch.no_grad():
 a=setup(768);p=ClusterPlan(a,120,'front_cluster_pipe_trace');g=capture(p)
 for _ in range(20):g.replay()
 torch.cuda.synchronize();ticks=p.counts[2:].view(torch.int64).reshape(120,2).cpu();dt=(ticks[:,1]-ticks[:,0]).double()/1000;dw=dt[torch.arange(120)%8<4];dx=dt[torch.arange(120)%8>=4];r=dict(dw_median_us=float(dw.median()),dw_max_us=float(dw.max()),dx_median_us=float(dx.median()),dx_max_us=float(dx.max()),duration_us=dt.tolist());(R/'cluster-pipe-role-trace-L768.json').write_text(json.dumps(r,indent=2));print('ROLES',{k:v for k,v in r.items() if not isinstance(v,list)},flush=True)
