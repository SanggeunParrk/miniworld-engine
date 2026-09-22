from ring_plan import *
with torch.no_grad():
 a=setup(768);ref=baseline(a);p=RingPlan(a,count=264,splits=20,source='front_ring96_wait_timing')
 p();torch.cuda.synchronize();es=errors(p.outputs,ref)
 assert all(e['finite'] and e['relative_l2']<=LIMITS[k] for k,e in es.items()),es
 g=capture(p)
 for _ in range(20):g.replay()
 torch.cuda.synchronize();tiles=768*768//64;base=2+9*96
 dw=p.counts[base:base+tiles*8].reshape(tiles,8).double().cpu()/1000
 dx=p.counts[base+tiles*8:base+tiles*9].double().cpu()/1000
 ds=[float(dw[s::20,h].sum()) for s in range(20) for h in range(8)]
 xs=[float(dx[s::104].sum()) for s in range(104)]
 out=dict(errors=es,dw_wait_per_cta_us=ds,dx_wait_per_cta_us=xs,
          dw_wait_median_us=statistics.median(ds),dx_wait_median_us=statistics.median(xs),
          dw_one_wait_median_us=float(dw.median()),dx_one_wait_median_us=float(dx.median()),
          scope='instrumented ring handoff waits, with timer/store overhead')
 (R/'ring-wait-timing-L768.json').write_text(json.dumps(out,indent=2))
 print('WAITS',{k:v for k,v in out.items() if not isinstance(v,(list,dict))},flush=True)
