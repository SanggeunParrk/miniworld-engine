from pathlib import Path
import subprocess,sys,json,os
R=Path(__file__).resolve().parent;D=int(sys.argv[1]);separate=len(sys.argv)>2;tag="-separate" if separate else "";rows=[]
for tool in ('memcheck','racecheck'):
 log=R/f'native-{tool}-D{D}{tag}.log';cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','regex=width_k1|width_k3',sys.executable,'-B',str(R/'native_probe.py'),str(D)]+(['separate'] if separate else [])
 with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append(dict(tool=tool,returncode=p.returncode,log=str(log),job=os.environ.get('SLURM_JOB_ID')));(R/f'native-sanitizer-D{D}{tag}.json').write_text(json.dumps(rows,indent=2));print(D,tool,p.returncode,flush=True);assert p.returncode==0
