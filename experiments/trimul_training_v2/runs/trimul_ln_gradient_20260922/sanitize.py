from pathlib import Path
import subprocess,sys,json,os
R=Path(__file__).resolve().parent;rows=[]
for tool in ('memcheck','racecheck'):
 log=R/(tool+'.log');cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=b7_joint',sys.executable,'-B',str(R/'probe.py')]
 with log.open('w') as out:p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=480)
 rows.append(dict(tool=tool,returncode=p.returncode,log=str(log),job=os.environ.get('SLURM_JOB_ID')));(R/'sanitizer.json').write_text(json.dumps(rows,indent=2));print(tool,p.returncode,flush=True);assert p.returncode==0
