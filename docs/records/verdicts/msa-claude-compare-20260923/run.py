import os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
for name in ('infer','pwa','opm'):
 env=dict(os.environ);env['PYTHONPATH']='/home/psk6950/miniworld-engine-msa/src';env['MINIWORLD_ENGINE_JIT_ROOT']='/home/psk6950/MiniWorld/runs/msa_bench_20260921/cuda/engine_jit'
 for k in ('OPT_CORE_DIR','MINIWORLD_PWA_INFER','MINIWORLD_PWA_TRAIN','MINIWORLD_OPM_TRAIN'):env.pop(k,None)
 print('historical',name,flush=True)
 with (R/f'old-{name}.log').open('w') as out:
  subprocess.run([sys.executable,f'/home/psk6950/MiniWorld/runs/msa_bench_20260921/time_engine_{name}.py'],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=1200)
print('same GPU OPM route comparison',flush=True)
subprocess.run([sys.executable,str(R/'opm_paths.py')],check=True,timeout=1200)
