from pathlib import Path
import subprocess,sys,json,os
R=Path(__file__).resolve().parent;results=[]
for tool in ('memcheck','racecheck'):
 log=R/(tool+'.log')
 cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','regex=transition_bwd_fused|reduce_partials',sys.executable,'-B',str(R/'sanitize_probe.py')]
 with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 results.append(dict(tool=tool,returncode=p.returncode,log=str(log),job=os.environ.get('SLURM_JOB_ID')));(R/'sanitizer.json').write_text(json.dumps(results,indent=2));print(tool,p.returncode,flush=True);assert p.returncode==0
