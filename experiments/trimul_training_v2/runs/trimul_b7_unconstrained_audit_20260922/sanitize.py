from pathlib import Path
import subprocess,sys,json,os
R=Path(__file__).resolve().parent
os.environ['SANITIZE']='1'
tag=os.environ.get('VALIDATE_TAG','')
records=[]
for tool in ('memcheck','racecheck'):
 log=R/(tool+'-L384'+tag+'.log');cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','regex=front_b7b12_(dw|dx)',sys.executable,'-B',str(R/'probe.py')]
 with log.open('w') as f:
  try:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=420);rc=p.returncode
  except subprocess.TimeoutExpired:rc=124
 records.append(dict(tool=tool,L=384,returncode=rc,log=str(log)));(R/('sanitizer-L384'+tag+'.json')).write_text(json.dumps(records,indent=2));print(tool,rc,flush=True)
 if rc:break
