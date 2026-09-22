import csv,json,collections
from pathlib import Path
R=Path(__file__).resolve().parent
profiles=[]
for p in sorted((R/'ncu').glob('*.csv')):
 with p.open() as f:
  rows=list(csv.DictReader(f))
 if len(rows)<2:continue
 units=rows[0];rows=rows[1:]
 def val(r,k):
  try:return float(r[k].replace(',',''))
  except (ValueError,KeyError,TypeError):return 0
 keys={'hbm_pct':'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','l2_pct':'lts__throughput.avg.pct_of_peak_sustained_elapsed','sm_pct':'sm__throughput.avg.pct_of_peak_sustained_elapsed','tensor_pct':'sm__ops_path_tensor_src_bf16_dst_fp32.sum.pct_of_peak_sustained_elapsed','issue_pct':'smsp__issue_active.avg.pct_of_peak_sustained_active','eligible_warps':'smsp__warps_eligible.avg.per_cycle_active','regs':'launch__registers_per_thread'}
 ks=[]
 for row in rows:
  tm=val(row,'gpu__time_duration.sum')*{'ns':.001,'us':1,'ms':1000,'s':1e6}.get(units.get('gpu__time_duration.sum'),1)
  ks.append(dict(name=row.get('Kernel Name'),us=tm,**{k:val(row,v) for k,v in keys.items()}))
 total=sum(k['us'] for k in ks)
 for k in ks:k['time_share']=k['us']/total if total else 0
 profiles.append(dict(profile=p.stem,total_us=total,kernels=sorted(ks,key=lambda k:-k['us'])))
(R/'ncu-summary.json').write_text(json.dumps(profiles,indent=2))
rs=[json.loads(p.read_text()) for p in sorted((R/'results').glob('*.json'))]
(R/'all-results.json').write_text(json.dumps(rs,indent=2))
print('status',collections.Counter(r.get('status') for r in rs))
for family in ('trimul','transition','triattn','ln'):
 for L in (384,768):
  for C in ((128,256,384,512) if family=='transition' else (128,)):
   rr=[r for r in rs if r['family']==family and r['length']==L and r['width']==C and r.get('status')=='passed' and r.get('direction')=='outgoing' and not r.get('module')]
   up=[r for r in rr if r['row'] not in ('engine_triton','pytorch','pytorch_compile','cueq')]
   base=[r for r in rr if r['row']=='engine_triton']
   if up:
    best=min(up,key=lambda r:r['ms']);print(family,L,C,best['row'],round(best['ms'],5),'base',round(base[0]['ms'],5) if base else None,'speedup',round(base[0]['ms']/best['ms'],2) if base else None)
print('NCU hottest')
for p in profiles:
 k=p['kernels'][0]
 print(p['profile'],round(p['total_us'],1),k['name'][:90],{v:round(k[v],1) for v in ('time_share','hbm_pct','l2_pct','tensor_pct','issue_pct')})
