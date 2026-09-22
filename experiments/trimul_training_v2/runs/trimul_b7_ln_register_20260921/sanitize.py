from pathlib import Path
import argparse,subprocess,os,sys,json
R=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
records=[]
for tool in ('memcheck','racecheck'):
 log=R/('%s-L%d.log'%(tool,n));cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=front_b7b12',sys.executable,'-B',str(R/'stress.py'),'--length',str(n)]
 with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1000)
 records.append(dict(tool=tool,returncode=p.returncode,log=str(log)));print('SANITIZER',n,tool,p.returncode,flush=True);(R/('sanitizer-L%d.json'%n)).write_text(json.dumps(records,indent=2));assert p.returncode==0
