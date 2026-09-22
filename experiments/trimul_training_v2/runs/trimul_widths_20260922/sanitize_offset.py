import subprocess,json,sys,os
from pathlib import Path
R=Path(__file__).resolve().parent;rows=[]
for variant in ('original','fixed'):
 pth=R/('offset-'+variant+'.log');cmd=['compute-sanitizer','--tool','memcheck','--error-exitcode','99','--kernel-name','kns=_input_dual_bwd_kernel',sys.executable,'-B',str(R/'probe_offset.py'),variant]
 with pth.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=480)
 rows.append(dict(variant=variant,returncode=p.returncode,log=str(pth),job=os.environ.get('SLURM_JOB_ID')));(R/'offset-sanitizer.json').write_text(json.dumps(rows,indent=2));print(variant,p.returncode,flush=True)
assert rows[0]['returncode']!=0 and rows[1]['returncode']==0
