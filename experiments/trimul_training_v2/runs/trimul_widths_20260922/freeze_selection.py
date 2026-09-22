from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;out={}
for D in (64,256,384,512):
 suffix='-nosave' if D==64 else '-separate' if D==512 else ''
 src=R/f'native-D{D}{suffix}.json';d=json.loads(src.read_text());assert d['complete']
 for L,row in d['results'].items():
  front=next(z for z in row['k1'] if z['config']==row['selected_k1']);k3=min([z for z in row['k3'] if 'us' in z],key=lambda z:z['us']) if row['k3_route']=='Anthropic' else None
  paths=[front['cubin']]+([k3['cubin']] if k3 else [])
  out[f'{D}-{L}']=dict(D=D,L=int(L),k1=row['selected_k1'],k3=k3['config'] if k3 else None,input_ln='separate' if D==512 else 'fused',emit_xn=D in (256,384),evidence=src.name,cubins={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
(R/'selection.json').write_text(json.dumps(out,indent=2))
