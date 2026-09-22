from harness import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=384);p.add_argument('--variants',default='baseline,parallel,vector_parallel,compact_parallel');a=p.parse_args()
record=dict(job=os.environ.get('SLURM_JOB_ID'),L=a.length,checks={},times={},resources={})
def save():(R/('sweep-L%d.json'%a.length)).write_text(json.dumps(record,indent=2))
with torch.no_grad():
 d=fixture(a.length);plans={};graphs={}
 ref=tuple(x.clone() for x in N._bwd_launch(d['dy'],d['x'],d['xn'],d['rs'],d['c1'],d['gamma'],d['wa'],d['wb'],d['ws']))
 for name in a.variants.split(','):
  try:
   q=Plan(d,name);record['resources'][name]=dict(regs=q.k.regs,lmem=q.k.lmem,cubin=str(q.path),sha256=hashlib.sha256(q.path.read_bytes()).hexdigest());out=q();es=errors(out,ref);record['checks'][name]=es
   plans[name]=q;graphs[name]=capture(q)[0];print('PASS',name,record['resources'][name],{n:v['relative_l2'] for n,v in es.items()},flush=True)
  except Exception as e:record.setdefault('failures',{})[name]=str(e);print('FAIL',name,str(e),flush=True)
  save()
 for scope in ['full','main','reduce']:
  gs=graphs if scope=='full' else {n:capture(q.main if scope=='main' else q.reduce)[0] for n,q in plans.items()}
  record['times'][scope]=paired(gs);print('TIME',scope,{n:round(v['median_us'],3) for n,v in record['times'][scope].items()},flush=True);save()
 record['complete']=True;save()
