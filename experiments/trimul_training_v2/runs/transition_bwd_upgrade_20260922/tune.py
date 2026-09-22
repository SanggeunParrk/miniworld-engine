from harness import *
record=dict(job=os.environ.get('SLURM_JOB_ID'),results={})
with torch.no_grad():
 for L in (384,768):
  d=fixture(L);ref=tuple(x.clone() for x in N._bwd_launch(d['dy'],d['x'],d['xn'],d['rs'],d['c1'],d['gamma'],d['wa'],d['wb'],d['ws']))
  ps={};gs={};out={}
  specs=[('baseline',8),('parallel',8),('compact_parallel',8),('parallel_transpose',8),('compact_parallel_transpose',8),('compact_parallel_transpose',7),('compact_parallel_transpose',9)]
  for name,rep in specs:
   key=name+'_r'+str(rep)
   try:
    p=Plan(d,name,rep);checks=errors(p(),ref);gs[key]=capture(p)[0];ps[key]=p
    out[key]=dict(checks=checks,regs=p.k.regs,lmem=p.k.lmem,cubin=str(p.path),sha256=hashlib.sha256(p.path.read_bytes()).hexdigest());print('PASS',L,key,flush=True)
   except Exception as e:print('FAIL',L,key,repr(e),flush=True);out[key]=dict(failure=repr(e))
  times=paired(gs,180,5)
  for k,v in times.items():out[k]['timing']=v
  record['results'][str(L)]=out;(R/'tune.json').write_text(json.dumps(record,indent=2));print('TIME',L,{k:v['median_us'] for k,v in times.items()},flush=True)
  del ps,gs,d,ref
record['complete']=True;(R/'tune.json').write_text(json.dumps(record,indent=2))
