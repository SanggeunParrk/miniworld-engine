import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
for L in (384,768):
 for f,row in [('atom_window','tf32rn'),('opm_core','carried'),('pwa','carried')]:
  name=f'{f}-L{L}-{row}'
  with (R/'logs'/f'{name}.log').open('w') as log:
   r=subprocess.run([sys.executable,str(R/'bench.py'),'--family',f,'--length',str(L),'--row',row,'--output',str(R/'results'/f'{name}.json')],stdout=log,stderr=subprocess.STDOUT,timeout=600)
   print(name,r.returncode,flush=True)
