import subprocess,sys,json
from pathlib import Path
R=Path(__file__).resolve().parent
jobs=[]
for L in (384,768):
 for f in ('adaln','swiglu','msa_ln','ln_linear','opm','gather'):
  jobs.append((f,L,128,'carried',[],f'{f}-L{L}'))
 jobs.append(('triattn',L,128,'block:triattn_native',['--module'],f'triattn-block-L{L}'))
 for C in (128,256,384,512):
  jobs.append(('ln',L,C,'engine_triton',[],f'ln-outgoing-L{L}-C{C}-engine_triton'))
jobs.append(('trimul',384,128,'native_rebuilt',['--direction','incoming'],'trimul-incoming-L384-C128-native_rebuilt'))
for f,L,C,row,extra,name in jobs:
 with (R/'logs'/f'{name}.log').open('w') as log:
  try:
   r=subprocess.run([sys.executable,str(R/'bench.py'),'--family',f,'--length',str(L),'--width',str(C),'--row',row,'--output',str(R/'results'/f'{name}.json'),*extra],stdout=log,stderr=subprocess.STDOUT,timeout=360)
   print(name,r.returncode,flush=True)
  except subprocess.TimeoutExpired:print(name,'TIMEOUT',flush=True)
