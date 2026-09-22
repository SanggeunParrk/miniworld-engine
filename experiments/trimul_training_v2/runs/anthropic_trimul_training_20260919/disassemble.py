from pathlib import Path
import json,hashlib,subprocess,re
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_training import build,default_config,_kernel,candidates
root=Path('/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919');rows=[]
for h in (128,256):
 cfg=default_config(h);path=build(128,h,cfg)
 sass=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',str(path)],text=True)
 (root/f'k3-training-h{h}.sass').write_text(sass)
 opcodes={k:len(re.findall(r'\b'+k+r'\b',sass)) for k in ('UTMALDG','UTMASTG','HGMMA','LDL','STL')}
 row=dict(h=h,config=cfg,config_count=len(list(candidates(128,h))),cubin=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),opcodes_static=opcodes,attrs=_kernel(128,h,cfg,0).attrs(),manifest=json.loads(path.with_suffix('.json').read_text()),ptxas=path.with_suffix('.ptxas.log').read_text())
 assert all(opcodes[k]>0 for k in ('UTMALDG','UTMASTG','HGMMA'))
 assert row['attrs']['local_bytes']==0
 rows.append(row);print(h,opcodes,row['attrs'],flush=True)
(root/'binary-evidence.json').write_text(json.dumps(rows,indent=2))
