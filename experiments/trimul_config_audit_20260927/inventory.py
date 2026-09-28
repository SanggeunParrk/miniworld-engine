"""CPU-only reproducible inventory: declarations and retained evidence, not coverage certification."""
import ast
import csv
import hashlib
import itertools
import json
from functools import reduce
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA = ROOT/'src/miniworld_engine/kernels/trimul_inproj/cuda'
AUTO = ROOT/'src/miniworld_engine/autotune'

def functions(path, names):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns = {}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
    return ns

record = {'scope':'Current declarations and selected retained historical records; counts do not prove current workload coverage',
          'triton_grid':{},'inference_table':[], 'training_k1':{},'history':{},'source_sha256':{}}
for path in sorted((AUTO/'configs/grid').glob('trimul*.csv')):
    rows=list(csv.DictReader(path.open()))
    if rows and 'axis' in rows[0]:
        axes={r['axis']:r['values'].split() for r in rows}
        count=reduce(lambda a,b: a*b, (len(v) for v in axes.values()), 1)
    else: axes={};count=len(rows)
    record['triton_grid'][path.stem]={'raw_declared_count':count,'axes':axes}
# Evaluate only the table assignment, whose contents are literals and dict calls.
path=CUDA/'_h100_infer_kernel.py'
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TILE_TABLE' for t in n.targets))
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
for (arch,cz,ch,dtype),v in ns['TILE_TABLE'].items():
    if arch!='sm_90a' or dtype!='b':continue
    a=v.get('k1_variants',[v['k1']]);b=v.get('k3_variants',[v['k3']])
    record['inference_table'].append(dict(cz=cz,ch=ch,k1_count=len(a),k3_count=len(b),cartesian_count=len(a)*len(b),k1=a,k3=b))
ns=functions(CUDA/'h100_native.py',{'k1_smem','configs'})
for width in (64,128,256,384,512):
    configs=ns['configs'](width)
    record['training_k1'][width]={'existing_count':len(configs),'configs':configs}
selection=json.loads((CUDA/'h100_sources/wide/selection.json').read_text())
record['training_selection']=selection
r=ROOT/'experiments/trimul_training_v2/runs/trimul_cuda_widths_opt_20260923'
for path in sorted(r.glob('*tune*.json')):
    data=json.loads(path.read_text())
    if not isinstance(data,list):continue
    axes={k:sorted({json.dumps(row[k],sort_keys=True) for row in data if k in row})
          for k in ('cfg','group','splits','sk','slots','minb','shared','clusters','consumers')}
    record['history'][str(path.relative_to(ROOT))]={'rows':len(data),'observed_axes':{k:v for k,v in axes.items() if v}}
record['historical_k1'] = {}
for path in sorted((ROOT/'experiments/trimul_training_v2/runs/trimul_widths_20260922').glob('native-D*.json')):
    data=json.loads(path.read_text())
    record['historical_k1'][str(path.relative_to(ROOT))] = {
        length: {'recorded_k1':len(v.get('k1',[])),
                 'timed_k1':sum('us' in row for row in v.get('k1',[])),
                 'recorded_k3':len(v.get('k3',[])),
                 'selected_k1':v.get('selected_k1')}
        for length,v in data.get('results',{}).items()}
record['b1_gate_demote_scan']={'levels':[0,1,2,3],'kind':'one-axis scan around earlier selected baseline','evidence':'experiments/trimul_training_v2/runs/trimul_b1_gate_demote_20260922/tune.py'}
for base in (CUDA,AUTO/'configs/grid'):
    for path in base.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.cu','.cuh','.inc','.json','.csv'):
            if base.name=='grid' and not path.name.startswith('trimul'):continue
            record['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
dest=Path(__file__).with_name('inventory.json');dest.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'K1_counts':{k:v['existing_count'] for k,v in record['training_k1'].items()},'inference_counts':[{k:v for k,v in r.items() if k not in ('k1','k3')} for r in record['inference_table']],'history_files':len(record['history'])},indent=2))
