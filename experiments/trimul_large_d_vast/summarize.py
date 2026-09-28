"""Collect factual results without treating process success as qualification."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--results',type=Path,default=Path('.bench/vast-20260927/results/trimul-large-d'))
a=p.parse_args();r={'scope':'Explicit native checkpoint; production dispatch unchanged','baseline':[], 'candidate':{'shape':{'D':512,'L':384},'input_splits':4,'lt_index':0,'repeats':[], 'sanitizers':{}}, 'source_sha256':{}}
for f in sorted(a.results.glob('pilot-*.json')):
 x=json.loads(f.read_text())
 r['baseline'].append({k:x[k] for k in ('width','length','complete','baseline_full_us','baseline_backward_us')})
for f in sorted(a.results.glob('repeat-input-split-gpu*.json')):
 x=json.loads(f.read_text())
 for i,row in enumerate(x['repeats']):
  r['candidate']['repeats'].append({'file':f.name,'repeat':i, 'complete':x['complete'],
   **{k:v['median_us'] for k,v in row.items()},
   'full_reduction_percent':100*(1-row['candidate_full']['median_us']/row['baseline_full']['median_us']),
   'bwd_reduction_percent':100*(1-row['candidate_bwd']['median_us']/row['baseline_bwd']['median_us'])})
v=a.results/'validation-wide-checkpoint24-D512-L384-vast-split4-index0.json'
if v.exists():
 x=json.loads(v.read_text());r['candidate']['validation']={'file':v.name,'complete':x.get('complete'),'stress_count':len(x.get('stress',[])),'stress_passed':all(z.get('passed') for z in x.get('stress',[])),'configuration':x.get('configuration')}
for tool in ('memcheck','racecheck','synccheck'):
 log=a.results/f'split4-{tool}.log';status=a.results/f'split4-{tool}.exit'
 text=log.read_text() if log.exists() else ''
 expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
 r['candidate']['sanitizers'][tool]={'passed':status.exists() and status.read_text().strip()=='0' and 'SANITIZER_DONE 512 384' in text and expected in text,'log':log.name}
for name in ('input_split.py','selected_vast.py','runtime.py','qualify_input_split.py','sanitize.sh'):
 path=Path(__file__).with_name(name);r['source_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
smoke=a.results/'selected-entry-check.json'
r['candidate']['entry_check']=json.loads(smoke.read_text()) if smoke.exists() else {}
r['candidate']['sanitizer_version']=(a.results/'sanitizer-version.txt').read_text() if (a.results/'sanitizer-version.txt').exists() else 'unknown'
validated_sources=r['candidate'].get('validation',{}).get('configuration',{}).get('vast_sources',{})
r['candidate']['source_match']=all(validated_sources.get(k)==r['source_sha256'][k] for k in ('input_split.py','qualify_input_split.py','runtime.py'))
r['candidate']['qualified_explicit']=r['candidate']['source_match'] and bool(r['candidate'].get('validation',{}).get('complete')) and all(x['passed'] for x in r['candidate']['sanitizers'].values()) and len(r['candidate']['repeats'])==6 and r['candidate']['validation'].get('stress_count')==5 and r['candidate']['validation'].get('stress_passed') and r['candidate']['entry_check'].get('passed') and r['candidate']['entry_check'].get('selected_sha256')==r['source_sha256']['selected_vast.py']
(a.results/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'qualified_explicit':r['candidate']['qualified_explicit'],'sanitizers':r['candidate']['sanitizers'],'full_gain_percent':[round(x['full_reduction_percent'],3) for x in r['candidate']['repeats']]},indent=2))
