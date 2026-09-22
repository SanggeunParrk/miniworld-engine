from front_plan import *
with torch.no_grad():
 for n in [384,768]:
  a=setup(n);p=Plan(a,source='front_segment_trace');g=capture(p)
  for _ in range(20):g.replay()
  torch.cuda.synchronize();t=p.counts[2:].view(torch.int64).reshape(132,16).cpu();records={}
  for role,rows,keys in [('dw',t[:60],{'body':(0,1),'tile':(2,6),'wait':(2,3),'glu':(3,4),'mma':(4,5),'refill':(5,6)}),('dx',t[60:],{'body':(0,1),'tile':(2,9),'gate_tma':(2,3),'gate_mma':(3,4),'front':(4,5),'ln_tma':(5,6),'ln_stats':(6,7),'ln_dx_params':(7,8),'store':(8,9),'group0wait':(4,10),'group0glu':(10,11),'group0mma':(11,12)})]:
   records[role]={k:float(((rows[:,b]-rows[:,a]).double()/1000).median()) for k,(a,b) in keys.items()}
  (R/f'segment-trace-L{n}.json').write_text(json.dumps(dict(L=n,medians_us=records,timestamps_ns=t.tolist()),indent=2));print('SEGMENTS',n,records,flush=True)
