from pathlib import Path
import csv,hashlib,json
R=Path(__file__).resolve().parent;P=R.parent

def compact(v):
 if isinstance(v,dict):return {k:compact(x) for k,x in v.items() if k!='samples_us'}
 if isinstance(v,list):return [compact(x) for x in v]
 return v

def raw(path):
 rows=list(csv.DictReader(path.open()));units=rows.pop(0)
 return [{k:dict(value=v,unit=units[k]) for k,v in r.items() if k.startswith(('gpu__','sm__pipe_tensor','smsp__warp_issue_stalled','launch__registers','dram__bytes','l1tex__data_bank'))} for r in rows]

out=dict(scope='B7-B12, two CUDA launches',baseline='B1 v51 + split saved-xn PC1 B7',L=[384,768],C=128,H=256,dropout=.25,SoL90_achieved=False,production_ready=False,experiments={},full_module={},profiles={},sanitizers={},accuracy={})
for name in ['dw_overlap','ln_register','dx_overlap','dx_prefetch','dw_stages','pair_round','dx_occupancy','dw_mask','weight_prefetch','dx_single_safe','dx_pc','dw_pair','nextrow','dw_pair_overlap','dw_glu_overlap','dw_shared_xn','dw_stream']:
 d=P/('trimul_b7_'+name+'_20260921')
 out['experiments'][name]={}
 for n in (384,768):
  f=d/('tune-L%d.json'%n)
  if f.exists():out['experiments'][name][n]=compact(json.loads(f.read_text()))
for n in (384,768):
 d=P/'trimul_b7_nextrow_20260921';f=d/('results-L%d.json'%n)
 if f.exists():out['full_module'][n]=compact(json.loads(f.read_text()))
 for role in ('dw','dx'):
  f=P/'trimul_b7_nextrow_20260921'/('sanitizer-%s-L%d.json'%(role,n))
  out['sanitizers']['nextrow-%s-L%d'%(role,n)]=json.loads(f.read_text())
 for name in ['dx_prefetch','dw_mask','weight_prefetch']:
  f=P/('trimul_b7_'+name+'_20260921')/('sanitizer-L%d.json'%n)
  if f.exists():out['sanitizers']['%s-L%d'%(name,n)]=json.loads(f.read_text())
 for d in (R,P/'trimul_b7_ln_register_20260921',P/'trimul_b7_dx_prefetch_20260921',P/'trimul_b7_dw_mask_20260921',P/'trimul_b7_weight_prefetch_20260921',P/'trimul_b7_nextrow_20260921'):
  for f in d.glob('ncu-L%d-mode0.csv'%n):out['profiles'][str(f.relative_to(P))]=raw(f)
 for name,filename in [('pre_audit','audit'),('dc_audit','audit'),('dc_audit','truth-all')]:
  f=P/('trimul_b7_'+name+'_20260921')/('%s-L%d.json'%(filename,n))
  if f.exists():out['accuracy'][str(f.relative_to(P))]=json.loads(f.read_text())
files=[R/'README.md',R/'render.py',R/'b7-current.svg']
for name in ['dx_prefetch','ln_register','dw_mask','weight_prefetch','nextrow']:
 d=P/('trimul_b7_'+name+'_20260921')
 files += [f for pat in ('*.cu','*.cuh','*.inc','*.py') for f in d.glob(pat)]
out['source_sha256']={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(R/'report.json').write_text(json.dumps(out,indent=2))
print('Report written; completed sanitizer entries:',{k:len(v) for k,v in out['sanitizers'].items()})
