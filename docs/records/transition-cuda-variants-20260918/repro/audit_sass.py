import json,subprocess,re
from pathlib import Path
root=Path(__file__).parent;rows=[]
for p in sorted(root.glob('tune-*-D*.json')):
 data=json.loads(p.read_text())
 for direction,key in [('forward','best_forward'),('backward','best_backward')]:
  if key not in data:continue
  so=data[key]['extension'];stem=f"{data['variant']}-D{data['D']}-{direction}"
  sass=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',so],text=True)
  (root/f'{stem}.sass').write_text(sass)
  resources=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-resource-usage',so],text=True)
  (root/f'{stem}.resources.txt').write_text(resources)
  functions=[]
  for block in re.split(r'Function : ',sass)[1:]:
   name=block.splitlines()[0];isa=block.splitlines()[1:];body='\n'.join(isa)
   if 'variant_kernel' not in name:continue
   functions.append(dict(symbol=name,tma_load=len(re.findall(r'UTMALDG',body)),wgmma=len(re.findall(r'HGMMA',body)),local_load=len(re.findall(r'\bLDL\b',body)),local_store=len(re.findall(r'\bSTL\b',body))))
  rows.append(dict(variant=data['variant'],D=data['D'],direction=direction,config=data[key]['config'],extension=so,kernels=functions))
(root/'sass-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
