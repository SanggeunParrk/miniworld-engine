import datetime,hashlib,json,os,socket
from pathlib import Path
root=Path(__file__).parent;engine=root.parent/'trimul_sm90_parity_20260917/engine'
files=['src/miniworld_engine/modules/transition/module.py','src/miniworld_engine/kernels/transition/cuda/variants.py','src/miniworld_engine/kernels/transition/cuda/transition_variants_kernel.cu','src/miniworld_engine/kernels/transition/cuda/transition_variant_norm.cu','tests/numerics/test_transition_cuda_variants_gpu.py']
d=dict(started=datetime.datetime.now().astimezone().isoformat(),node=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),sha256={f:hashlib.sha256((engine/f).read_bytes()).hexdigest() for f in files})
(root/'validation-sources.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
