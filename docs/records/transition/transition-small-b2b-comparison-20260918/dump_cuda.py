import json,subprocess
from pathlib import Path
root=Path(__file__).parent
for d in (128,256):
 path=json.loads((root/f'metadata-cuda-D{d}.json').read_text())['extension_path']
 for option,suffix in (('--dump-sass','sass'),('--dump-resource-usage','resources')):
  with (root/f'cuda-D{d}.{suffix}').open('w') as f:
   subprocess.run(['/usr/local/cuda-12.9/bin/cuobjdump',option,path],stdout=f,check=True)
 for backend in ('triton','cuda'):
  data=(root/f'{backend}-D{d}.sass').read_text()
  print(d,backend,{s:data.count(s) for s in ('HGMMA','UTMALDG','LDGSTS','LDL','STL')})
